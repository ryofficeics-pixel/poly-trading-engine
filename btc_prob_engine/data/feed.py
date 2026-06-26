"""
btc_prob_engine/src/data/feed.py
================================
Data pipeline — architecture borrowed from:
  • ccxt/ccxt          : multi-exchange REST, OHLCV, funding, OI normalization
  • bmoscon/cryptofeed : callback pattern, FeedHandler, async tick streaming
  • nkaz001/hftbacktest: tick-level L2 book with sequence-number validation

Design: single async FeedHandler that unifies REST backfill + WS real-time.
Callbacks follow cryptofeed's (obj, receipt_timestamp) signature.
All data emitted as normalized dicts — no exchange-specific format leaks upstream.
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional

import websockets
from loguru import logger


# ── Normalized tick types (cryptofeed-inspired) ─────────────────────────────

@dataclass
class Trade:
    exchange:   str
    symbol:     str
    price:      float
    size:       float
    side:       str          # 'buy' | 'sell'
    ts:         float        # unix epoch seconds
    receipt_ts: float = 0.0
    trade_id:   str = ""


@dataclass
class Candle:
    exchange: str
    symbol:   str
    interval: str
    open:     float
    high:     float
    low:      float
    close:    float
    volume:   float
    ts:       float
    closed:   bool = False


@dataclass
class BookLevel:
    price: float
    size:  float


@dataclass
class OrderBook:
    exchange: str
    symbol:   str
    ts:       float
    bids:     List[BookLevel] = field(default_factory=list)
    asks:     List[BookLevel] = field(default_factory=list)

    @property
    def mid(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid * 10000) if self.mid else 0.0

    def imbalance(self, depth: int = 5) -> float:
        """
        Order book imbalance ratio — key microstructure signal.
        From HFTBacktest patterns: imbalance = bid_vol / (bid_vol + ask_vol)
        > 0.5 = buy pressure, < 0.5 = sell pressure
        """
        bid_vol = sum(b.size for b in self.bids[:depth])
        ask_vol = sum(a.size for a in self.asks[:depth])
        total = bid_vol + ask_vol
        return bid_vol / total if total > 0 else 0.5


@dataclass
class Liquidation:
    exchange:  str
    symbol:    str
    side:      str     # 'buy' (long liquidated) | 'sell' (short liquidated)
    quantity:  float
    price:     float
    ts:        float


# ── Ring buffers for each data type ─────────────────────────────────────────

class DataRing:
    """
    Fixed-size in-memory ring buffer. Pattern from HFTBacktest's tick storage.
    Keeps last N events without allocations after init.
    """
    def __init__(self, maxlen: int = 5000):
        self.trades:       Deque[Trade]       = deque(maxlen=maxlen)
        self.candles_1m:   Deque[Candle]      = deque(maxlen=1440)   # 1 day
        self.candles_5m:   Deque[Candle]      = deque(maxlen=288)
        self.candles_1h:   Deque[Candle]      = deque(maxlen=168)
        self.orderbook:    Optional[OrderBook] = None
        self.liquidations: Deque[Liquidation] = deque(maxlen=500)

    def push_trade(self, t: Trade):
        self.trades.append(t)

    def push_candle(self, c: Candle):
        if c.interval == '1m':   self.candles_1m.append(c)
        elif c.interval == '5m': self.candles_5m.append(c)
        elif c.interval == '1h': self.candles_1h.append(c)

    def push_book(self, b: OrderBook):
        self.orderbook = b

    def push_liquidation(self, liq: Liquidation):
        self.liquidations.append(liq)

    def latest_price(self) -> float:
        if self.trades:
            return self.trades[-1].price
        if self.orderbook:
            return self.orderbook.mid
        return 0.0

    def vwap(self, window: int = 100) -> float:
        """Volume-weighted average price over last N trades."""
        recent = list(self.trades)[-window:]
        if not recent:
            return 0.0
        num = sum(t.price * t.size for t in recent)
        den = sum(t.size for t in recent)
        return num / den if den else 0.0

    def buy_sell_ratio(self, window: int = 200) -> float:
        """
        Taker buy volume / total volume. Key order flow metric.
        Source: cryptofeed trade.side field usage.
        > 0.5 = net buying, < 0.5 = net selling.
        """
        recent = list(self.trades)[-window:]
        if not recent:
            return 0.5
        buy_vol  = sum(t.size for t in recent if t.side == 'buy')
        total    = sum(t.size for t in recent)
        return buy_vol / total if total else 0.5

    def liquidation_pressure(self, window_s: float = 300.0) -> Dict[str, float]:
        """
        Net liquidation pressure over last N seconds.
        Long liq (side='buy') → bearish. Short liq (side='sell') → bullish.
        """
        now    = time.time()
        recent = [l for l in self.liquidations if now - l.ts < window_s]
        long_liq  = sum(l.quantity * l.price for l in recent if l.side == 'buy')
        short_liq = sum(l.quantity * l.price for l in recent if l.side == 'sell')
        total = long_liq + short_liq
        return {
            "long_liq_usd":  round(long_liq, 2),
            "short_liq_usd": round(short_liq, 2),
            "net_pressure":  round((short_liq - long_liq) / total, 4) if total else 0.0,
        }


# ── Binance WebSocket Feed (no API key) ─────────────────────────────────────

# URLs are overridable via env so a local relay (binance_relay.py) can be
# transparently substituted when the machine cannot reach Binance directly.
# Set BINANCE_PROXY_URL=ws://localhost:9001 in .env to use the relay.
import os as _os

def _resolve_ws_base(default_spot: str, default_futures: str):
    """
    Read BINANCE_PROXY_URL at call-time (not import-time) so that
    load_dotenv() in ws_server.py has already run before we resolve URLs.
    """
    proxy = _os.getenv("BINANCE_PROXY_URL", "").rstrip("/")
    if proxy:
        return proxy + "?streams=", proxy + "?streams="
    return default_spot, default_futures

# Module-level defaults (overridden in BinanceFeed.start() via _resolve_ws_base)
BINANCE_WS         = "wss://stream.binance.com:9443/stream?streams="
BINANCE_FUTURES_WS = "wss://fstream.binance.com/stream?streams="

STREAMS = [
    "btcusdt@trade",
    "btcusdt@depth20@100ms",
    "btcusdt@kline_1m",
    "btcusdt@kline_5m",
    "btcusdt@kline_1h",
]

# Futures (liquidations need futures stream)
FUTURES_STREAMS = [
    "btcusdt@forceOrder",   # liquidations
    "btcusdt@kline_5m",     # futures candles for OI comparison
]

CallbackFn = Callable


class BinanceFeed:
    """
    Real-time Binance feed following cryptofeed's FeedHandler callback architecture.
    All callbacks: async def callback(obj, receipt_timestamp: float)

    Architecture borrowed from:
      cryptofeed: FeedHandler, callback.py TradeCallback/BookCallback/LiquidationCallback
      cryptofeed Binance: message_handler(), _trade(), _book(), _liquidations()
      HFTBacktest: sequence number validation, tick-exact timestamps
    """

    def __init__(self, ring: DataRing):
        self.ring    = ring
        self._spot_callbacks:    Dict[str, List[CallbackFn]] = {}
        self._futures_callbacks: Dict[str, List[CallbackFn]] = {}
        self._running = False

    def on_trade(self, fn: CallbackFn):
        self._spot_callbacks.setdefault('trade', []).append(fn)
        return fn

    def on_book(self, fn: CallbackFn):
        self._spot_callbacks.setdefault('book', []).append(fn)
        return fn

    def on_candle(self, fn: CallbackFn):
        self._spot_callbacks.setdefault('candle', []).append(fn)
        return fn

    def on_liquidation(self, fn: CallbackFn):
        self._futures_callbacks.setdefault('liquidation', []).append(fn)
        return fn

    async def _fire(self, callbacks: Dict, event: str, obj):
        receipt = time.time()
        for cb in callbacks.get(event, []):
            try:
                await cb(obj, receipt)
            except Exception as e:
                logger.error(f"callback error [{event}]: {e}")

    async def _parse_spot(self, msg: dict, receipt: float):
        stream = msg.get('stream', '')
        data   = msg.get('data', msg)

        # ── Trade ─────────────────────────────────────────────────────────
        if '@trade' in stream:
            t = Trade(
                exchange='binance',
                symbol='BTC-USDT',
                price=float(data['p']),
                size=float(data['q']),
                side='buy' if data.get('m') is False else 'sell',
                ts=float(data['T']) / 1000,
                receipt_ts=receipt,
                trade_id=str(data.get('t', '')),
            )
            self.ring.push_trade(t)
            await self._fire(self._spot_callbacks, 'trade', t)

        # ── Order Book ─────────────────────────────────────────────────────
        elif '@depth' in stream:
            book = OrderBook(
                exchange='binance', symbol='BTC-USDT', ts=receipt,
                bids=[BookLevel(float(b[0]), float(b[1])) for b in data.get('bids', [])],
                asks=[BookLevel(float(a[0]), float(a[1])) for a in data.get('asks', [])],
            )
            self.ring.push_book(book)
            await self._fire(self._spot_callbacks, 'book', book)

        # ── Candle ─────────────────────────────────────────────────────────
        elif '@kline' in stream:
            k = data.get('k', {})
            c = Candle(
                exchange='binance', symbol='BTC-USDT',
                interval=k.get('i', '1m'),
                open=float(k['o']), high=float(k['h']),
                low=float(k['l']),  close=float(k['c']),
                volume=float(k['v']),
                ts=float(k['t']) / 1000,
                closed=k.get('x', False),
            )
            self.ring.push_candle(c)
            await self._fire(self._spot_callbacks, 'candle', c)

    async def _parse_futures(self, msg: dict, receipt: float):
        stream = msg.get('stream', '')
        data   = msg.get('data', msg)

        # ── Liquidation ────────────────────────────────────────────────────
        if 'forceOrder' in stream:
            o = data.get('o', {})
            liq = Liquidation(
                exchange='binance-futures',
                symbol='BTC-USDT',
                side='buy' if o.get('S') == 'BUY' else 'sell',
                quantity=float(o.get('q', 0)),
                price=float(o.get('ap', o.get('p', 0))),
                ts=float(o.get('T', receipt * 1000)) / 1000,
            )
            self.ring.push_liquidation(liq)
            await self._fire(self._futures_callbacks, 'liquidation', liq)

    async def _connect_stream(self, base_url: str, streams: List[str], parser):
        url     = base_url + '/'.join(streams)
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10, close_timeout=5
                ) as ws:
                    backoff = 1
                    logger.info(f"feed connected: {base_url.split('//')[1][:30]}")
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            await parser(msg, time.time())
                        except Exception as e:
                            logger.debug(f"parse error: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"feed disconnected: {e}  retry in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def start(self):
        self._running = True
        # Resolve URLs here so load_dotenv() in ws_server has already run.
        spot_url, fut_url = _resolve_ws_base(
            "wss://stream.binance.com:9443/stream?streams=",
            "wss://fstream.binance.com/stream?streams=",
        )
        await asyncio.gather(
            self._connect_stream(spot_url, STREAMS,         self._parse_spot),
            self._connect_stream(fut_url,  FUTURES_STREAMS, self._parse_futures),
        )

    def stop(self):
        self._running = False
