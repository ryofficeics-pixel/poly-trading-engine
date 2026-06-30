"""candle_synthesizer.py
=======================
Synthesizes 1-minute OHLCV candles from REST price ticks when
Binance WebSocket is unavailable.

Injects synthetic candles into btc_prob_engine DataRing so the
probability model can run even with only REST polling.
"""

import asyncio
import random
import time
from datetime import datetime
from typing import Optional
from loguru import logger
from btc_prob_engine.data.feed import Candle


class CandleSynthesizer:
    """
    Builds 1m/5m/1h OHLCV candles from price ticks.
    When a candle closes, triggers callbacks with closed=True.
    """
    
    def __init__(self):
        self._candles = {
            '1m': None,
            '5m': None,
            '1h': None,
        }
        self._callbacks = []
        self._last_price = 0.0
        
    def on_candle(self, callback):
        """Register callback for closed candles: callback(candle, ts)"""
        self._callbacks.append(callback)
        return callback
    
    def _get_candle_ts(self, interval: str, now: float) -> float:
        """Get the start timestamp for the current candle."""
        intervals = {'1m': 60, '5m': 300, '1h': 3600}
        period = intervals[interval]
        return int(now / period) * period
    
    async def tick(self, price: float):
        """Process a new price tick and update/close candles."""
        if price <= 0:
            return
            
        self._last_price = price
        now = time.time()
        
        for interval in ['1m', '5m', '1h']:
            candle_ts = self._get_candle_ts(interval, now)
            current = self._candles[interval]
            
            # New candle needed?
            if current is None or current.ts != candle_ts:
                # Close previous candle if exists
                if current is not None:
                    # Add micro-jitter to create realistic OHLC variation
                    # This helps technical indicators (RSI, BB, ATR) function properly
                    jitter_pct = 0.0005  # 0.05% max variation (~$30 on $60k BTC)
                    
                    if current.high == current.low:  # Flat candle, add jitter
                        mid = current.close
                        current.high = mid * (1 + random.uniform(0, jitter_pct))
                        current.low = mid * (1 - random.uniform(0, jitter_pct))
                        # Ensure open/close are within high/low
                        current.open = mid * (1 + random.uniform(-jitter_pct/2, jitter_pct/2))
                        current.close = mid * (1 + random.uniform(-jitter_pct/2, jitter_pct/2))
                        current.open = max(current.low, min(current.high, current.open))
                        current.close = max(current.low, min(current.high, current.close))
                    
                    # Add synthetic volume based on price range
                    price_range_pct = (current.high - current.low) / current.close * 100
                    # Typical BTC 1m volume: 1-50 BTC, scale by volatility
                    current.volume = random.uniform(1.0, 10.0) * (1 + price_range_pct * 2)
                    
                    current.closed = True
                    logger.debug(f"synth candle closed: {interval}  "
                               f"o={current.open:.2f} h={current.high:.2f} "
                               f"l={current.low:.2f} c={current.close:.2f} "
                               f"v={current.volume:.2f}")
                    # Fire callbacks
                    for cb in self._callbacks:
                        try:
                            await cb(current, now)
                        except Exception as e:
                            logger.error(f"candle callback error: {e}")
                
                # Start new candle with slight jitter
                jitter = price * random.uniform(-0.0001, 0.0001)  # ±0.01%
                open_price = price + jitter
                
                self._candles[interval] = Candle(
                    exchange='synthetic',
                    symbol='BTC-USDT',
                    interval=interval,
                    open=open_price,
                    high=open_price,
                    low=open_price,
                    close=price,
                    volume=0.0,
                    ts=candle_ts,
                    closed=False,
                )
            else:
                # Update current candle with micro-jitter for realistic movement
                jitter = price * random.uniform(-0.00005, 0.00005)  # ±0.005%
                tick_price = price + jitter
                
                current.high = max(current.high, tick_price)
                current.low = min(current.low, tick_price)
                current.close = price
                
                # Accumulate volume on each tick
                current.volume += random.uniform(0.1, 0.5)
    
    def get_current_candles(self) -> dict:
        """Return current in-progress candles."""
        return {k: v for k, v in self._candles.items() if v is not None}
