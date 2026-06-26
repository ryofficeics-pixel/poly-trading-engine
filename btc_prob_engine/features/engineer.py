"""
btc_prob_engine/src/features/engineer.py
=========================================
Feature engineering pipeline for BTC probability engine.

Architecture borrowed from:
  • vectorbt/vectorbt    : vectorized indicator factory, IndicatorBase pattern,
                           params_to_list, walk-forward splits
  • pandas-ta            : indicator chaining, extension pattern
  • statsmodels          : ADF stationarity test, ACF/PACF, regime detection
  • arch (bashtage/arch) : GARCH variance forecasting, HARX model for realized vol
  • HFTBacktest          : microstructure features: imbalance, spread, fill prob

All features are computed from raw DataRing → return flat dict[str, float]
ready for XGBoost / LightGBM feature matrix.
NO look-ahead bias: every calculation only uses data[:-1] or closes candle data.
"""

import math
import statistics
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data.feed import DataRing  # type: ignore


# ── Type alias ───────────────────────────────────────────────────────────────
FeatureDict = Dict[str, float]

NaN = float('nan')


def _safe(v: float, default: float = 0.0) -> float:
    """Replace nan/inf with default — never let bad floats reach the model."""
    if math.isnan(v) or math.isinf(v):
        return default
    return v


# ── Price series helpers ─────────────────────────────────────────────────────

def _closes(candles) -> List[float]:
    return [c.close for c in candles]


def _volumes(candles) -> List[float]:
    return [c.volume for c in candles]


def _returns(prices: List[float]) -> List[float]:
    if len(prices) < 2:
        return []
    return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]


# ── EMA / SMA (vectorbt-style: O(1) incremental) ────────────────────────────

def ema(prices: List[float], period: int) -> float:
    """Exponential moving average — final value only."""
    if len(prices) < period:
        return NaN
    k = 2 / (period + 1)
    val = sum(prices[:period]) / period
    for p in prices[period:]:
        val = p * k + val * (1 - k)
    return val


def sma(prices: List[float], period: int) -> float:
    if len(prices) < period:
        return NaN
    return sum(prices[-period:]) / period


def stdev(prices: List[float], period: int) -> float:
    if len(prices) < period:
        return NaN
    sample = prices[-period:]
    mean   = sum(sample) / period
    var    = sum((p - mean) ** 2 for p in sample) / (period - 1)
    return math.sqrt(var)


# ── RSI (Wilder smoothed) ────────────────────────────────────────────────────

def rsi(prices: List[float], period: int = 14) -> float:
    """
    Wilder RSI. Returns [0, 100].
    Returns probability-compatible value: RSI/100 = crude long probability.
    """
    if len(prices) < period + 1:
        return NaN
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains   = [max(c, 0) for c in changes]
    losses  = [abs(min(c, 0)) for c in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── MACD ─────────────────────────────────────────────────────────────────────

def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9
         ) -> Tuple[float, float, float]:
    """Returns (macd_line, signal_line, histogram)."""
    if len(prices) < slow + signal:
        return NaN, NaN, NaN
    e_fast = ema(prices, fast)
    e_slow = ema(prices, slow)
    macd_line = e_fast - e_slow

    # signal: EMA of macd values over last `signal` bars
    # Approximate: use last signal windows of macd diffs
    macd_vals = []
    for i in range(len(prices) - signal - slow + 1, len(prices) - slow + 2):
        if i >= 0:
            ef = ema(prices[:i+slow-1], fast) if i+slow-1 <= len(prices) else NaN
            es = ema(prices[:i+slow-1], slow) if i+slow-1 <= len(prices) else NaN
            if not math.isnan(ef) and not math.isnan(es):
                macd_vals.append(ef - es)

    sig_line  = ema(macd_vals, signal) if len(macd_vals) >= signal else NaN
    histogram = macd_line - sig_line if not math.isnan(sig_line) else NaN
    return macd_line, sig_line, histogram


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def bollinger(prices: List[float], period: int = 20, std_mult: float = 2.0
              ) -> Tuple[float, float, float, float]:
    """Returns (upper, mid, lower, %B). %B in [0,1] = position in bands."""
    if len(prices) < period:
        return NaN, NaN, NaN, NaN
    mid   = sma(prices, period)
    sd    = stdev(prices, period)
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    price = prices[-1]
    pct_b = (price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
    return upper, mid, lower, pct_b


# ── ATR (Average True Range) ─────────────────────────────────────────────────

def atr(candles, period: int = 14) -> float:
    """Normalized ATR as % of price — volatility proxy."""
    if len(candles) < period + 1:
        return NaN
    trs = []
    cl   = list(candles)
    for i in range(1, len(cl)):
        h, l, pc = cl[i].high, cl[i].low, cl[i-1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return NaN
    atr_val = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val / candles[-1].close * 100  # normalized %


# ── GARCH-inspired realized volatility (arch/HARX pattern) ──────────────────

def realized_vol(returns: List[float], window: int = 20) -> float:
    """
    Annualized realized volatility — arch HARX-style.
    HARX uses: RV_d = f(RV_1d, RV_5d, RV_22d) — HAR model.
    We compute raw RV here; HAR weighting done in model layer.
    """
    if len(returns) < window:
        return NaN
    sample = returns[-window:]
    mean   = sum(sample) / window
    var    = sum((r - mean) ** 2 for r in sample) / (window - 1)
    # Annualize: sqrt(var * 365 * 24 * 60) for minute returns
    return math.sqrt(var * 365 * 24 * 60) * 100


def har_vol_components(returns: List[float]) -> Tuple[float, float, float]:
    """
    HAR-RV components from arch/HARX model:
      RV_1d  (daily)   = last 1440 minutes
      RV_5d  (weekly)  = last 7200 minutes
      RV_22d (monthly) = last 31680 minutes
    Returns (rv_1d, rv_5d, rv_22d) as annualized % vol.
    Pattern: arch.univariate.mean.HARX
    """
    rv_1d  = realized_vol(returns, min(len(returns), 1440))
    rv_5d  = realized_vol(returns, min(len(returns), 7200))
    rv_22d = realized_vol(returns, min(len(returns), 31680))
    return rv_1d, rv_5d, rv_22d


def garch_vol_proxy(returns: List[float], omega: float = 0.000001,
                    alpha: float = 0.1, beta: float = 0.85) -> float:
    """
    GARCH(1,1) variance recursion without scipy/arch dependency.
    Formula: sigma2_t = omega + alpha*eps2_{t-1} + beta*sigma2_{t-1}
    Default params typical for crypto daily returns.
    Returns annualized vol estimate %.
    Pattern: arch.univariate.volatility.GARCH
    """
    if len(returns) < 2:
        return NaN
    var = statistics.variance(returns) if len(returns) > 1 else 0.0001
    for r in returns[1:]:
        var = omega + alpha * (r ** 2) + beta * var
    return math.sqrt(var * 365 * 24 * 60) * 100  # annualized minute vol


# ── Hurst Exponent (regime detection) ────────────────────────────────────────

def hurst_exponent(prices: List[float], min_n: int = 10) -> float:
    """
    Hurst exponent H:
      H < 0.5 = mean-reverting (stat arb favorable)
      H = 0.5 = random walk
      H > 0.5 = trending (momentum favorable)
    Classic R/S analysis. Used in regime detection layer.
    """
    if len(prices) < 20:
        return 0.5
    try:
        lags    = range(2, min(len(prices) // 2, 50))
        tau_vals = []
        for lag in lags:
            segments = [prices[i:i+lag] for i in range(0, len(prices)-lag, lag)]
            if not segments:
                continue
            rs_vals = []
            for seg in segments:
                if len(seg) < 2:
                    continue
                mean_s   = sum(seg) / len(seg)
                dev      = [x - mean_s for x in seg]
                cumdev   = [sum(dev[:i+1]) for i in range(len(dev))]
                R        = max(cumdev) - min(cumdev)
                std_s    = math.sqrt(sum(d**2 for d in dev) / len(dev))
                if std_s > 0:
                    rs_vals.append(R / std_s)
            if rs_vals:
                tau_vals.append((math.log(lag), math.log(sum(rs_vals) / len(rs_vals))))

        if len(tau_vals) < 2:
            return 0.5
        xs  = [t[0] for t in tau_vals]
        ys  = [t[1] for t in tau_vals]
        n   = len(xs)
        sx  = sum(xs); sy  = sum(ys)
        sxy = sum(x*y for x, y in zip(xs, ys))
        sx2 = sum(x**2 for x in xs)
        H   = (n * sxy - sx * sy) / (n * sx2 - sx**2)
        return max(0.0, min(1.0, H))
    except Exception:
        return 0.5


# ── Market Regime Detection ───────────────────────────────────────────────────

def detect_regime(closes: List[float], vol: float, hurst: float) -> Dict[str, float]:
    """
    Regime scoring inspired by:
      vectorbt: from_signals strategy classification
      statsmodels: ADF test logic (simplified inline)

    Outputs:
      trend_score  [0,1]: 1 = strong uptrend
      mean_rev_score [0,1]: 1 = strong mean reversion
      vol_regime   [0,1]: 1 = high vol (expansion)
    """
    if len(closes) < 50:
        return {"trend_score": 0.5, "mean_rev_score": 0.5, "vol_regime": 0.5}

    price = closes[-1]
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    ema200_val = ema(closes, min(len(closes), 200))

    # Simplified ADF-style test: how often does price cross the mean?
    # (stationary = lots of crossings = mean-reverting)
    mean_50 = sma(closes, 50)
    crossings = sum(
        1 for i in range(1, min(50, len(closes)))
        if (closes[-i] - mean_50) * (closes[-i-1] - mean_50) < 0
    )
    mean_rev_score = min(1.0, crossings / 10.0)

    # Trend: EMA alignment (20 > 50 > 200) = uptrend
    aligned_up   = (not math.isnan(ema20) and not math.isnan(ema50)
                    and ema20 > ema50)
    aligned_down = (not math.isnan(ema20) and not math.isnan(ema50)
                    and ema20 < ema50)
    ema_trend = 0.7 if aligned_up else (0.3 if aligned_down else 0.5)

    # Hurst contribution: H>0.55 → trending
    hurst_trend = min(1.0, max(0.0, (hurst - 0.5) * 4 + 0.5))

    trend_score = 0.5 * ema_trend + 0.5 * hurst_trend

    # Vol regime: compare current ATR vs 20-bar ATR mean
    # Proxy: vol > median(vol_history) = expansion
    vol_regime = 0.5  # placeholder — set from rolling vol comparison in caller

    return {
        "trend_score":    round(trend_score, 4),
        "mean_rev_score": round(mean_rev_score, 4),
        "vol_regime":     round(vol_regime, 4),
        "hurst":          round(hurst, 4),
        "ema_aligned_up": 1.0 if aligned_up else 0.0,
    }


# ── Volume Profile ────────────────────────────────────────────────────────────

def volume_profile(candles, n_bins: int = 20) -> Dict[str, float]:
    """
    Simplified volume profile — Point of Control + value area.
    Inspired by py-market-profile architecture.
    Identifies high-volume nodes as S/R levels.
    """
    if len(candles) < 10:
        return {"poc_price": 0.0, "va_low": 0.0, "va_high": 0.0,
                "price_at_poc": 0.5}
    cl = list(candles)
    lo  = min(c.low  for c in cl)
    hi  = max(c.high for c in cl)
    if hi == lo:
        return {"poc_price": lo, "va_low": lo, "va_high": hi, "price_at_poc": 0.5}

    step     = (hi - lo) / n_bins
    buckets  = [0.0] * n_bins
    for c in cl:
        mid_idx = min(int((c.close - lo) / step), n_bins - 1)
        buckets[mid_idx] += c.volume

    poc_idx   = buckets.index(max(buckets))
    poc_price = lo + (poc_idx + 0.5) * step

    total_vol = sum(buckets)
    va_target = total_vol * 0.70
    va_vol    = buckets[poc_idx]
    lo_idx    = poc_idx
    hi_idx    = poc_idx
    while va_vol < va_target and (lo_idx > 0 or hi_idx < n_bins - 1):
        add_lo = buckets[lo_idx - 1] if lo_idx > 0 else 0
        add_hi = buckets[hi_idx + 1] if hi_idx < n_bins - 1 else 0
        if add_lo >= add_hi and lo_idx > 0:
            lo_idx -= 1; va_vol += add_lo
        elif hi_idx < n_bins - 1:
            hi_idx += 1; va_vol += add_hi
        else:
            break

    va_low  = lo + lo_idx * step
    va_high = lo + (hi_idx + 1) * step
    current = cl[-1].close
    price_at_poc = (current - lo) / (hi - lo)

    return {
        "poc_price":    round(poc_price, 2),
        "va_low":       round(va_low, 2),
        "va_high":      round(va_high, 2),
        "price_at_poc": round(price_at_poc, 4),
        "above_poc":    1.0 if current > poc_price else 0.0,
        "in_va":        1.0 if va_low <= current <= va_high else 0.0,
    }


# ── Main Feature Extractor ────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Unified feature extraction from DataRing.
    Produces flat FeatureDict for ML models.

    Pattern: vectorbt IndicatorFactory — each feature has a name, inputs, outputs.
    All outputs are floats in interpretable ranges.
    Missing data → 0.0 (safe default for tree models, explicitly flagged).
    """

    def __init__(self, ring: DataRing):
        self.ring          = ring
        self._vol_history  = deque(maxlen=100)  # rolling vol for regime
        self._last_ts      = 0.0

    def extract(self) -> FeatureDict:
        """
        Extract ALL features from current DataRing state.
        Call after each closed 1m candle or every N ticks.
        """
        features: FeatureDict = {}
        features["extract_ts"] = time.time()

        # ── Price / candle availability ──────────────────────────────────
        c1m  = list(self.ring.candles_1m)
        c5m  = list(self.ring.candles_5m)
        c1h  = list(self.ring.candles_1h)

        has_1m = len(c1m) >= 30
        has_5m = len(c5m) >= 30
        has_1h = len(c1h) >= 20

        features["n_candles_1m"] = float(len(c1m))
        features["n_candles_5m"] = float(len(c5m))
        features["n_candles_1h"] = float(len(c1h))

        price = self.ring.latest_price()
        features["price"] = price
        if price == 0:
            return features  # no data yet

        # ── 1m features ──────────────────────────────────────────────────
        if has_1m:
            closes_1m = _closes(c1m)
            vols_1m   = _volumes(c1m)
            rets_1m   = _returns(closes_1m)

            # EMAs
            features["ema9_1m"]   = _safe(ema(closes_1m, 9))
            features["ema21_1m"]  = _safe(ema(closes_1m, 21))
            features["ema50_1m"]  = _safe(ema(closes_1m, 50))

            # EMA distance normalized
            e9  = features["ema9_1m"]
            e21 = features["ema21_1m"]
            if e21 > 0:
                features["ema9_dist_pct"] = _safe((price - e9) / e21 * 100)
                features["ema21_dist_pct"] = _safe((price - e21) / e21 * 100)
                features["ema_cross_1m"]  = 1.0 if e9 > e21 else 0.0

            # RSI
            features["rsi14_1m"]  = _safe(rsi(closes_1m, 14)) / 100
            features["rsi7_1m"]   = _safe(rsi(closes_1m, 7)) / 100

            # Bollinger
            bb_up, bb_mid, bb_lo, pct_b = bollinger(closes_1m, 20, 2)
            features["bb_pct_b_1m"] = _safe(pct_b, 0.5)
            features["bb_width_1m"] = _safe((bb_up - bb_lo) / bb_mid * 100 if bb_mid > 0 else 0)

            # ATR
            features["atr_pct_1m"] = _safe(atr(c1m, 14))

            # Vol
            garch_v = garch_vol_proxy(rets_1m) if len(rets_1m) >= 20 else 0.0
            features["garch_vol_1m"] = _safe(garch_v)
            self._vol_history.append(garch_v)

            # MACD (1m)
            ml, sl, hist = macd(closes_1m)
            features["macd_hist_1m"] = _safe(hist / price * 100 if price > 0 else 0)
            features["macd_above_signal_1m"] = 1.0 if not math.isnan(hist) and hist > 0 else 0.0

            # Candle body features
            last = c1m[-1]
            body  = abs(last.close - last.open)
            wick  = last.high - last.low
            features["candle_body_pct"] = body / wick if wick > 0 else 0.5
            features["is_bull_candle"]  = 1.0 if last.close > last.open else 0.0
            features["upper_wick_pct"]  = (last.high - max(last.open, last.close)) / wick if wick > 0 else 0.0
            features["lower_wick_pct"]  = (min(last.open, last.close) - last.low) / wick if wick > 0 else 0.0

            # Volume momentum
            vol_20 = sma(vols_1m, 20)
            features["vol_ratio_1m"] = _safe(last.volume / vol_20 if vol_20 > 0 else 1.0)

            # Hurst
            h = hurst_exponent(closes_1m)
            features["hurst_1m"] = h

            # Regime
            vol_hist = list(self._vol_history)
            regime   = detect_regime(closes_1m, garch_v, h)
            if vol_hist:
                median_vol = sorted(vol_hist)[len(vol_hist)//2]
                regime["vol_regime"] = 1.0 if garch_v > median_vol else 0.0
            features.update({f"regime_{k}": v for k, v in regime.items()})

            # Volume profile
            if len(c1m) >= 50:
                vp = volume_profile(c1m[-100:])
                features.update({f"vp_{k}": v for k, v in vp.items()})

            # HAR vol components
            if len(rets_1m) >= 100:
                rv1, rv5, rv22 = har_vol_components(rets_1m)
                features["har_rv_1d"]  = _safe(rv1)
                features["har_rv_5d"]  = _safe(rv5)
                features["har_rv_22d"] = _safe(rv22)
                features["har_vol_mean"] = _safe((rv1 + rv5 + rv22) / 3)

        # ── 5m features ──────────────────────────────────────────────────
        if has_5m:
            closes_5m = _closes(c5m)
            rets_5m   = _returns(closes_5m)

            features["rsi14_5m"]     = _safe(rsi(closes_5m, 14)) / 100
            features["ema21_5m"]     = _safe(ema(closes_5m, 21))
            features["ema50_5m"]     = _safe(ema(closes_5m, 50))
            e21_5m = features["ema21_5m"]
            if e21_5m > 0:
                features["ema21_5m_dist"] = _safe((price - e21_5m) / e21_5m * 100)
            features["garch_vol_5m"] = _safe(garch_vol_proxy(rets_5m))
            bb_up5, bb_mid5, bb_lo5, pct_b5 = bollinger(closes_5m, 20, 2)
            features["bb_pct_b_5m"]  = _safe(pct_b5, 0.5)
            features["atr_pct_5m"]   = _safe(atr(c5m, 14))
            features["hurst_5m"]     = hurst_exponent(closes_5m)

            # 5m momentum: last 3 candle direction
            if len(c5m) >= 3:
                features["mom3_5m"] = _safe(
                    (closes_5m[-1] - closes_5m[-4]) / closes_5m[-4] * 100
                    if closes_5m[-4] > 0 else 0
                )

        # ── 1h features ──────────────────────────────────────────────────
        if has_1h:
            closes_1h = _closes(c1h)
            rets_1h   = _returns(closes_1h)

            features["rsi14_1h"]    = _safe(rsi(closes_1h, 14)) / 100
            features["ema21_1h"]    = _safe(ema(closes_1h, 21))
            features["ema50_1h"]    = _safe(ema(closes_1h, 50))
            e50_1h = features["ema50_1h"]
            if e50_1h > 0:
                features["price_vs_ema50_1h"] = _safe((price - e50_1h) / e50_1h * 100)
            features["garch_vol_1h"] = _safe(garch_vol_proxy(rets_1h))
            features["atr_pct_1h"]   = _safe(atr(c1h, 14))
            features["hurst_1h"]     = hurst_exponent(closes_1h)

            # Weekly range position
            hi_20 = max(c.high for c in list(c1h)[-20:]) if len(c1h) >= 20 else price
            lo_20 = min(c.low  for c in list(c1h)[-20:]) if len(c1h) >= 20 else price
            rng   = hi_20 - lo_20
            features["range_position_1h"] = _safe((price - lo_20) / rng if rng > 0 else 0.5)

        # ── Microstructure (HFTBacktest-inspired) ─────────────────────────
        book = self.ring.orderbook
        if book:
            features["bid_ask_spread_bps"] = _safe(book.spread_bps)
            features["order_book_imbalance"] = _safe(book.imbalance(5))
            # Depth imbalance at different levels
            features["book_imbal_10"]  = _safe(book.imbalance(10))
            features["book_imbal_20"]  = _safe(book.imbalance(20))
            # Bid/ask wall detection: ratio of top1 to top5
            if len(book.bids) >= 5 and book.bids[0].size > 0:
                top5_bid = sum(b.size for b in book.bids[:5])
                features["bid_wall_ratio"] = _safe(book.bids[0].size / top5_bid)
            if len(book.asks) >= 5 and book.asks[0].size > 0:
                top5_ask = sum(a.size for a in book.asks[:5])
                features["ask_wall_ratio"] = _safe(book.asks[0].size / top5_ask)

        # ── Trade flow (cryptofeed trade.side) ───────────────────────────
        bsr = self.ring.buy_sell_ratio(200)
        features["buy_sell_ratio_200"] = _safe(bsr)
        features["buy_pressure"]       = 1.0 if bsr > 0.55 else (0.0 if bsr < 0.45 else 0.5)

        vwap = self.ring.vwap(200)
        if vwap > 0:
            features["price_vs_vwap_pct"] = _safe((price - vwap) / vwap * 100)

        # ── Liquidations ──────────────────────────────────────────────────
        liq = self.ring.liquidation_pressure(300)
        features["liq_net_pressure"]  = _safe(liq["net_pressure"])
        features["liq_long_usd"]      = _safe(math.log1p(liq["long_liq_usd"]))
        features["liq_short_usd"]     = _safe(math.log1p(liq["short_liq_usd"]))

        # ── Cross-timeframe alignment score ──────────────────────────────
        # Counts how many timeframes agree on direction
        alignment = 0.0
        if has_1m and "rsi14_1m" in features:
            alignment += 1.0 if features["rsi14_1m"] > 0.5 else -1.0
        if has_5m and "rsi14_5m" in features:
            alignment += 1.0 if features["rsi14_5m"] > 0.5 else -1.0
        if has_1h and "rsi14_1h" in features:
            alignment += 1.0 if features["rsi14_1h"] > 0.5 else -1.0
        features["tf_alignment"] = alignment / 3.0  # [-1, 1]
        features["tf_bull_count"] = max(0.0, alignment)

        return features

    def feature_names(self) -> List[str]:
        """Return list of all feature names for model column validation."""
        dummy_ring = DataRing()
        # Can't extract without data; return statically defined list
        return [
            "price", "n_candles_1m", "n_candles_5m", "n_candles_1h",
            "ema9_1m", "ema21_1m", "ema50_1m",
            "ema9_dist_pct", "ema21_dist_pct", "ema_cross_1m",
            "rsi14_1m", "rsi7_1m", "rsi14_5m", "rsi14_1h",
            "bb_pct_b_1m", "bb_width_1m", "bb_pct_b_5m",
            "atr_pct_1m", "atr_pct_5m", "atr_pct_1h",
            "garch_vol_1m", "garch_vol_5m", "garch_vol_1h",
            "macd_hist_1m", "macd_above_signal_1m",
            "hurst_1m", "hurst_5m", "hurst_1h",
            "regime_trend_score", "regime_mean_rev_score",
            "regime_vol_regime", "regime_hurst", "regime_ema_aligned_up",
            "vol_ratio_1m", "candle_body_pct", "is_bull_candle",
            "upper_wick_pct", "lower_wick_pct",
            "har_rv_1d", "har_rv_5d", "har_rv_22d", "har_vol_mean",
            "vp_poc_price", "vp_va_low", "vp_va_high",
            "vp_price_at_poc", "vp_above_poc", "vp_in_va",
            "bid_ask_spread_bps", "order_book_imbalance",
            "book_imbal_10", "book_imbal_20",
            "bid_wall_ratio", "ask_wall_ratio",
            "buy_sell_ratio_200", "buy_pressure",
            "price_vs_vwap_pct",
            "liq_net_pressure", "liq_long_usd", "liq_short_usd",
            "tf_alignment", "tf_bull_count",
        ]
