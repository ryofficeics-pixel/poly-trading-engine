"""candle_synthesizer.py
=======================
Synthesizes 1-minute OHLCV candles from REST price ticks when
Binance WebSocket is unavailable.

Injects synthetic candles into btc_prob_engine DataRing so the
probability model can run even with only REST polling.
"""

import asyncio
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
                    current.closed = True
                    logger.debug(f"synth candle closed: {interval}  "
                               f"o={current.open:.2f} h={current.high:.2f} "
                               f"l={current.low:.2f} c={current.close:.2f}")
                    # Fire callbacks
                    for cb in self._callbacks:
                        try:
                            await cb(current, now)
                        except Exception as e:
                            logger.error(f"candle callback error: {e}")
                
                # Start new candle
                self._candles[interval] = Candle(
                    exchange='synthetic',
                    symbol='BTC-USDT',
                    interval=interval,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0.0,  # synthetic candles have no volume data
                    ts=candle_ts,
                    closed=False,
                )
            else:
                # Update current candle
                current.high = max(current.high, price)
                current.low = min(current.low, price)
                current.close = price
    
    def get_current_candles(self) -> dict:
        """Return current in-progress candles."""
        return {k: v for k, v in self._candles.items() if v is not None}
