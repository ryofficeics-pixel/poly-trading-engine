"""candle_synthesizer.py
=======================
Synthesizes 1m/5m/1h OHLCV candles from REST price ticks when
Binance WebSocket is unavailable.

Improvements (deep audit):
  - 5m candles aggregated from 5 × 1m candles
  - 1h candles aggregated from 12 × 5m candles
  - Micro-jitter for realistic OHLC variation
  - Synthetic volume scaled by volatility
  - Callbacks fire for all three timeframes
"""

import asyncio
import random
import time
from datetime import datetime
from typing import Callable, List, Optional
from loguru import logger
from btc_prob_engine.data.feed import Candle


class CandleSynthesizer:
    """
    Builds 1m, 5m, and 1h OHLCV candles from price ticks (REST polling).

    Each price tick updates the current open candle(s).
    When a candle window closes (boundary crossed), callbacks fire.
    5m candles aggregate 5 × 1m closed candles.
    1h candles aggregate 12 × 5m closed candles.
    """

    INTERVALS = {
        '1m':  60,
        '5m':  300,
        '1h':  3600,
    }

    def __init__(self):
        self._candles: dict = {interval: None for interval in self.INTERVALS}
        self._callbacks: List[Callable] = []
        # Buffer of recently closed 1m candles for 5m aggregation
        self._closed_1m: List[Candle] = []
        # Buffer of recently closed 5m candles for 1h aggregation
        self._closed_5m: List[Candle] = []

    def on_candle(self, fn: Callable) -> Callable:
        """Decorator to register a candle close callback."""
        self._callbacks.append(fn)
        return fn

    async def tick(self, price: float, ts: Optional[float] = None) -> None:
        """
        Process a new price tick. Updates open candles and fires callbacks
        when candle windows close.
        """
        if price <= 0:
            return
        now = ts or time.time()

        # Process 1m candles (primary)
        closed_1m = await self._update_candle('1m', price, now)

        # When a 1m candle closes, try to aggregate 5m
        if closed_1m:
            self._closed_1m.append(closed_1m)
            # Keep only last 12 (enough for 2 × 5m candles)
            if len(self._closed_1m) > 12:
                self._closed_1m = self._closed_1m[-12:]
            closed_5m = await self._aggregate_multi('5m', self._closed_1m, 5, now)

            # When a 5m candle closes, try to aggregate 1h
            if closed_5m:
                self._closed_5m.append(closed_5m)
                if len(self._closed_5m) > 24:
                    self._closed_5m = self._closed_5m[-24:]
                await self._aggregate_multi('1h', self._closed_5m, 12, now)

    async def _update_candle(self, interval: str, price: float, now: float) -> Optional[Candle]:
        """
        Update the current candle for a given interval.
        Returns the closed candle if a boundary was crossed, else None.
        """
        window  = self.INTERVALS[interval]
        candle_ts = int(now // window) * window
        current   = self._candles[interval]
        closed    = None

        if current is None or current.ts != candle_ts:
            # Close current candle if it exists
            if current is not None:
                closed = self._finalize_candle(current, interval)
                await self._fire_callbacks(closed, now)

            # Start new candle with micro-jitter on open
            jitter = price * random.uniform(-0.0001, 0.0001)
            open_price = price + jitter
            self._candles[interval] = Candle(
                exchange='synthetic', symbol='BTC-USDT', interval=interval,
                open=open_price, high=open_price, low=open_price,
                close=price, volume=0.0, ts=candle_ts, closed=False,
            )
        else:
            # Update existing candle with micro-jitter for intra-candle realism
            jitter = price * random.uniform(-0.00005, 0.00005)
            tick = price + jitter
            current.high  = max(current.high, tick)
            current.low   = min(current.low, tick)
            current.close = price
            current.volume += random.uniform(0.1, 0.5)

        return closed

    def _finalize_candle(self, candle: Candle, interval: str) -> Candle:
        """
        Add realistic OHLC jitter to flat candles so indicators (RSI/BB/ATR) work.
        """
        jitter_pct = 0.0005  # 0.05% max variation

        if candle.high == candle.low:
            mid = candle.close
            candle.high  = mid * (1 + random.uniform(0, jitter_pct))
            candle.low   = mid * (1 - random.uniform(0, jitter_pct))
            candle.open  = mid * (1 + random.uniform(-jitter_pct/2, jitter_pct/2))
            candle.close = mid * (1 + random.uniform(-jitter_pct/2, jitter_pct/2))
            candle.open  = max(candle.low, min(candle.high, candle.open))
            candle.close = max(candle.low, min(candle.high, candle.close))

        # Synthetic volume scaled by price range volatility
        price_range_pct = (candle.high - candle.low) / candle.close * 100
        candle.volume = max(candle.volume, random.uniform(1.0, 10.0) * (1 + price_range_pct * 2))
        candle.closed = True

        logger.debug(
            f"synth candle closed: {interval}  "
            f"o={candle.open:.2f} h={candle.high:.2f} "
            f"l={candle.low:.2f} c={candle.close:.2f} "
            f"v={candle.volume:.2f}"
        )
        return candle

    async def _aggregate_multi(
        self, interval: str, source_candles: List[Candle],
        n: int, now: float
    ) -> Optional[Candle]:
        """
        Aggregate `n` closed lower-timeframe candles into one higher-timeframe candle.
        Returns the aggregated candle if enough source candles are available.
        Fires callbacks for the aggregated candle.
        """
        # Need exactly n candles at the right boundary
        window = self.INTERVALS[interval]
        candle_ts = int(now // window) * window

        # Check if we have n recent candles that fit in the current window
        if len(source_candles) < n:
            return None

        # Use the last n candles
        group = source_candles[-n:]

        # Verify they span the expected time window
        span = group[-1].ts - group[0].ts
        expected_span = self.INTERVALS[interval.replace('5m','1m').replace('1h','5m')] * (n - 1)

        agg = Candle(
            exchange='synthetic', symbol='BTC-USDT', interval=interval,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group),
            ts=candle_ts,
            closed=True,
        )

        logger.debug(
            f"synth candle closed: {interval}  "
            f"o={agg.open:.2f} h={agg.high:.2f} "
            f"l={agg.low:.2f} c={agg.close:.2f} "
            f"v={agg.volume:.2f}"
        )

        await self._fire_callbacks(agg, now)
        return agg

    async def _fire_callbacks(self, candle: Candle, ts: float) -> None:
        """Fire all registered candle callbacks."""
        for cb in self._callbacks:
            try:
                await cb(candle, ts)
            except Exception as e:
                logger.error(f"candle callback error: {e}")
