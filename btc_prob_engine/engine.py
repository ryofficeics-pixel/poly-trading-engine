"""
btc_prob_engine/src/engine.py
==============================
Top-level BTC Probability Engine orchestrator.

Wires together:
  Data Layer    (feed.py)        → DataRing
  Feature Layer (engineer.py)    → FeatureDict
  Model Layer   (probability.py) → ProbabilityOutput
  Risk Layer    (risk/engine.py) → RiskScore
  Backtest      (backtest/)      → Stats

Event loop:
  1. Binance WS tick → DataRing
  2. Every closed 1m candle → FeatureEngineer.extract()
  3. FeatureDict → BTCProbabilityEngine.predict()
  4. ProbabilityOutput → RiskEngine.check()
  5. If approved → paper_fill() (wired to poly-trading-engine)
  6. Broadcast state to all connected dashboards
"""

import asyncio
import json
import time
from dataclasses import asdict
from typing import Optional

from loguru import logger

from .data.feed import DataRing, BinanceFeed, Trade, Candle, OrderBook, Liquidation
from .features.engineer import FeatureEngineer
from .models.probability import BTCProbabilityEngine, ProbabilityOutput
from .risk.engine import RiskEngine, RiskParams


class BTCProbEngine:
    """
    Full engine pipeline: feed → features → probability → risk → signal.
    Designed to run as async background task alongside the paper trading engine.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 signal_threshold: float = 0.60,
                 max_kelly: float = 0.20,
                 risk_params: Optional[RiskParams] = None,
                 on_signal=None):
        """
        on_signal: async coroutine called with (direction, size_pct, prob_output)
                   → wire to paper_fill() in ws_server.py
        """
        self.ring       = DataRing(maxlen=5000)
        self.feed       = BinanceFeed(self.ring)
        self.engineer   = FeatureEngineer(self.ring)
        self.model      = BTCProbabilityEngine(
            model_path=model_path,
            signal_threshold=signal_threshold,
            max_kelly=max_kelly,
        )
        self.risk       = RiskEngine(risk_params or RiskParams())
        self.on_signal  = on_signal

        self._last_candle_ts   = 0.0
        self._last_prob_output = ProbabilityOutput()
        self._last_risk_score  = None
        self._last_features    = {}
        self._running          = False
        self._candle_count     = 0

    async def start(self, equity_fn=None, exposure_fn=None):
        """
        Start the engine.
        equity_fn:   callable → float (current paper account equity)
        exposure_fn: callable → float (current paper account exposure)
        """
        self._equity_fn   = equity_fn or (lambda: 1000.0)
        self._exposure_fn = exposure_fn or (lambda: 0.0)
        self._running     = True

        # Register feed callbacks
        @self.feed.on_candle
        async def handle_candle(c: Candle, ts: float):
            if c.closed and c.interval == '1m':
                await self._on_closed_candle(c, ts)

        @self.feed.on_trade
        async def handle_trade(t: Trade, ts: float):
            pass  # DataRing already updated by feed parser

        @self.feed.on_book
        async def handle_book(b: OrderBook, ts: float):
            pass  # DataRing already updated

        @self.feed.on_liquidation
        async def handle_liq(l: Liquidation, ts: float):
            logger.debug(f"liq: {l.side} qty={l.quantity:.4f} @ {l.price:.2f}")

        logger.info("BTC Probability Engine started")
        await self.feed.start()

    async def _on_closed_candle(self, candle: Candle, ts: float):
        """
        Main inference pipeline — runs on every closed 1m candle.
        Entire pipeline: <5ms on commodity hardware.
        """
        self._candle_count += 1

        # ── 1. Extract features ──────────────────────────────────────────
        t0       = time.perf_counter()
        features = self.engineer.extract()
        t_feat   = (time.perf_counter() - t0) * 1000
        self._last_features = features

        # ── 2. Probability inference ─────────────────────────────────────
        prob_out = self.model.predict(features)
        self._last_prob_output = prob_out

        # ── 3. Risk check ────────────────────────────────────────────────
        equity   = self._equity_fn()
        exposure = self._exposure_fn()
        vol      = features.get("garch_vol_1m", 50.0)

        # FIX: kelly_fraction lives on ProbabilityOutput (computed by predict),
        # NOT in the features dict. Reading from features always hit the 0.05
        # fallback, under-sizing every trade.
        kelly = prob_out.feature_snap.get("kelly_fraction", 0.05)

        risk = self.risk.check(
            model_confidence = prob_out.confidence,
            current_vol      = vol,
            equity           = equity,
            exposure         = exposure,
            long_prob        = prob_out.long_prob,
            kelly_fraction   = kelly,
        )
        self._last_risk_score = risk

        t_total = (time.perf_counter() - t0) * 1000

        logger.debug(
            f"candle #{self._candle_count}  "
            f"price={candle.close:.2f}  "
            f"prob={prob_out.long_prob:.3f}  "
            f"signal={prob_out.signal}  "
            f"conf={prob_out.confidence:.3f}  "
            f"risk_ok={risk.approved}  "
            f"pipeline={t_total:.1f}ms"
        )

        # ── 4. Signal emission ───────────────────────────────────────────
        if risk.approved and prob_out.signal != "FLAT" and self.on_signal:
            direction = "UP" if prob_out.signal == "LONG" else "DOWN"
            size_pct  = risk.recommended_size
            try:
                await self.on_signal(direction, size_pct, prob_out)
                logger.info(
                    f"SIGNAL EMITTED  {direction}  size={size_pct:.1f}%  "
                    f"prob={prob_out.long_prob:.3f}  conf={prob_out.confidence:.3f}"
                )
            except Exception as e:
                logger.error(f"signal callback error: {e}")

    def state_dict(self) -> dict:
        """Full engine state for dashboard broadcast."""
        prob   = self._last_prob_output
        risk   = self._last_risk_score
        feats  = self._last_features

        return {
            # Probability outputs
            "long_prob":        prob.long_prob,
            "short_prob":       prob.short_prob,
            "confidence":       prob.confidence,
            "edge":             prob.edge,
            "signal":           prob.signal,
            "model_name":       prob.model_name,

            # Key features
            "rsi14_1h":         feats.get("rsi14_1h", 0),
            "rsi14_5m":         feats.get("rsi14_5m", 0),
            "rsi14_1m":         feats.get("rsi14_1m", 0),
            "book_imbalance":   feats.get("order_book_imbalance", 0.5),
            "buy_sell_ratio":   feats.get("buy_sell_ratio_200", 0.5),
            "garch_vol":        feats.get("garch_vol_1m", 0),
            "hurst":            feats.get("hurst_1m", 0.5),
            "tf_alignment":     feats.get("tf_alignment", 0),
            "liq_pressure":     feats.get("liq_net_pressure", 0),
            "regime":           feats.get("regime_trend_score", 0.5),
            "atr_pct":          feats.get("atr_pct_1m", 0),
            "bb_pct_b":         feats.get("bb_pct_b_1m", 0.5),
            "vp_above_poc":     feats.get("vp_above_poc", 0),

            # Risk
            "risk_approved":    risk.approved if risk else False,
            "risk_score":       risk.risk_score if risk else 1.0,
            "vol_regime":       risk.vol_regime if risk else "NORMAL",
            "recommended_size": risk.recommended_size if risk else 0.0,
            "risk_reason":      risk.reason if risk else "",

            # Engine stats
            "candle_count":     self._candle_count,
            "btc_price":        self.ring.latest_price(),
            "vwap":             self.ring.vwap(200),
            "buy_sell_ratio":   self.ring.buy_sell_ratio(200),
        }

    def stop(self):
        self._running = False
        self.feed.stop()
