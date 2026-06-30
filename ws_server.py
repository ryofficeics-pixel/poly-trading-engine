"""
Poly Trading Engine — WebSocket Bridge v3
==========================================
Production-ready paper trading platform.

Features:
  - Paper capital system (user-defined starting balance)
  - Real live BTC price via Binance WS (no API key)
  - Full PnL / equity / drawdown tracking
  - SQLite persistence — survives restarts
  - Watchdog + auto-reconnect on all WS connections
  - Heartbeat + equity curve snapshots every 5s
  - State recovery on startup from DB
  - Paper execution engine (no real orders)
  - Analytics endpoint

Install:
    pip install fastapi uvicorn loguru websockets python-dotenv

Run:
    python ws_server.py

Dashboard:
    Open poly-trading-dashboard.html in browser
    (or deploy both to a VPS / Railway / Render)
"""

import asyncio
import json
import os
import sys
import time
import uuid
import urllib.request
import urllib.error
import random
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from candle_synthesizer import CandleSynthesizer

from src.database.db import (
    init_db, get_or_create_account, update_account_capital,
    create_session, close_session,
    save_position, update_position_pnl, close_position_db, load_open_positions,
    save_trade, load_trades, load_daily_pnl_today,
    save_equity_snapshot, load_equity_curve,
    upsert_daily_stats, load_daily_stats,
    archive_log, compute_analytics
)
from src.portfolio.account import PaperAccount, PaperPosition
from btc_prob_engine import BTCProbEngine
from btc_prob_engine.risk import RiskParams
from btc_prob_engine.data.feed import Trade as FeedTrade

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────

DEFAULT_CAPITAL       = float(os.getenv("DEFAULT_CAPITAL", "1000.0"))
MAX_POSITION_PCT      = float(os.getenv("MAX_POSITION_PCT", "10.0"))   # % of equity per trade
# Use BINANCE_PROXY_URL (relay) when direct Binance access is blocked.
_BINANCE_PROXY        = os.getenv("BINANCE_PROXY_URL", "").rstrip("/")
BINANCE_WS_URL        = (_BINANCE_PROXY + "?streams=btcusdt@trade"
                         if _BINANCE_PROXY
                         else "wss://stream.binance.com:9443/ws/btcusdt@trade")
EQUITY_SNAPSHOT_EVERY = int(os.getenv("EQUITY_SNAPSHOT_EVERY", "10"))  # heartbeats
HEARTBEAT_INTERVAL    = int(os.getenv("HEARTBEAT_INTERVAL", "5"))       # seconds
LOG_ARCHIVE_LEVELS    = {"TRADE", "ERROR", "WARN", "PNL"}

# ── Strategy / risk config (auto-trading) ──────────────────────────────────
SIGNAL_THRESHOLD      = float(os.getenv("SIGNAL_THRESHOLD", "0.52"))  # prob → LONG/SHORT (heuristic proxy trades at lower conviction)
MAX_KELLY             = float(os.getenv("MAX_KELLY", "0.20"))         # quarter-Kelly cap
TAKE_PROFIT_PCT       = float(os.getenv("TAKE_PROFIT_PCT", "1.5"))    # per-position TP
STOP_LOSS_PCT         = float(os.getenv("STOP_LOSS_PCT", "1.0"))      # per-position SL
MAX_HOLD_MINUTES      = float(os.getenv("MAX_HOLD_MINUTES", "60"))    # time-based exit
MAX_OPEN_POSITIONS    = int(os.getenv("MAX_OPEN_POSITIONS", "3"))     # concurrent slots
EXIT_CHECK_INTERVAL   = float(os.getenv("EXIT_CHECK_INTERVAL", "2"))  # seconds between exit sweeps

# ── Logging ────────────────────────────────────────────────────────────────

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {name} | {message}")
logger.add("logs/poly_engine.log", rotation="50 MB", retention="30 days",
           level="DEBUG", enqueue=True)
# Errors + warnings only — fast triage without grepping the full log.
logger.add("logs/error.log",
           rotation="20 MB", retention="30 days",
           level="WARNING",
           enqueue=True,
           format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}")

# ── Global state ───────────────────────────────────────────────────────────

account = PaperAccount()

class EngineStatus:
    STOPPED  = "stopped"
    RUNNING  = "running"
    STOPPING = "stopping"

class MarketState:
    btc_price: float = 0.0
    feed_connected: bool = False
    feed_latency_ms: int = 0
    last_tick_ts: float = 0.0

engine_status: str   = EngineStatus.STOPPED
kill_switch: bool    = False
market = MarketState()
_engine_task: Optional[asyncio.Task] = None
_price_task:  Optional[asyncio.Task] = None
_heartbeat_count: int = 0

# ── BTC probability engine (auto-trading pipeline) ─────────────────────────
# Tracks the wall-clock open time of each position id for time-based exits.
_position_open_ts: dict = {}

# Callbacks fired (pnl: float) after each settled trade — risk engine feeds here.
account_callbacks_on_settle: list = []

# Lazily constructed in start_engine() so it inherits current account state.
btc_engine: Optional[BTCProbEngine] = None

# Candle synthesizer for REST-only mode (when Binance WS unavailable)
candle_synth: Optional[CandleSynthesizer] = None

# ── In-memory log ring ─────────────────────────────────────────────────────
from collections import deque
log_ring: deque = deque(maxlen=400)


# ── Connection Manager ─────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"dashboard connected  total={len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info(f"dashboard disconnected  total={len(self.active)}")

    async def broadcast(self, msg: dict):
        dead = []
        payload = json.dumps(msg)
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_snapshot(self, ws: WebSocket):
        equity_data = load_equity_curve(account.id, limit=500)
        analytics   = compute_analytics(account.id)
        daily       = load_daily_stats(account.id, days=30)
        recent_trades = load_trades(account.id, limit=50)
        engine_state = btc_engine.state_dict() if btc_engine is not None else {}
        await ws.send_json({
            "type":        "snapshot",
            "state":       _full_state_dict(),
            "positions":   [p.to_dict() for p in account.positions.values()],
            "logs":        list(log_ring),
            "equity_curve": equity_data,
            "analytics":   analytics,
            "daily_stats": daily,
            "recent_trades": recent_trades,
            "engine_state": engine_state,
        })

manager = ConnectionManager()


# ── Loguru WS sink ─────────────────────────────────────────────────────────

def _parse_level(record) -> str:
    lvl = record["level"].name.upper()
    return {"SUCCESS": "TRADE", "CRITICAL": "ERROR", "WARNING": "WARN"}.get(lvl, lvl)

async def _ws_sink_async(entry: dict):
    log_ring.append(entry)
    if entry["level"] in LOG_ARCHIVE_LEVELS:
        archive_log(entry["level"], entry["mod"], entry["msg"])
    await manager.broadcast({"type": "log", "entry": entry})

def loguru_ws_sink(message):
    record = message.record
    entry = {
        "ts":    record["time"].strftime("%H:%M:%S"),
        "level": _parse_level(record),
        "mod":   record["name"].split(".")[-1][:10],
        "msg":   record["message"],
    }
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_ws_sink_async(entry))
    except RuntimeError:
        pass


# ── Price tick helper ──────────────────────────────────────────────────

async def _apply_price_tick(price: float):
    """Apply a new BTC price to account state and broadcast to dashboards."""
    global candle_synth
    now = time.time()
    market.btc_price       = price
    market.feed_latency_ms = int((now - market.last_tick_ts) * 1000)
    market.last_tick_ts    = now

    account.update_unrealized(price)

    # ✅ FIX: Push synthetic trade tick so DataRing.latest_price() works in REST mode
    # Without this, latest_price() returns 0 → feature extraction exits early → no RSI
    if btc_engine is not None and price > 0:
        try:
            btc_engine.ring.push_trade(FeedTrade(
                exchange='synthetic', symbol='BTC-USDT',
                price=price, size=0.0, side='buy', ts=now,
            ))
        except Exception:
            pass

    # Feed candle synthesizer in REST-only mode
    if candle_synth is not None and price > 0:
        try:
            await candle_synth.tick(price)
        except Exception as e:
            logger.debug(f"candle synth error: {e}")

    # Persist PnL updates periodically; never crash the feed loop on DB error.
    if int(now) % 10 == 0 and account.positions:
        try:
            for pos in list(account.positions.values()):
                update_position_pnl(pos.id, pos.unrealized_pnl, pos.roi_pct)
        except Exception as _db_err:
            logger.debug(f"pnl persist skipped: {_db_err}")

    await manager.broadcast({
        "type":           "price",
        "price":          price,
        "unrealized_pnl": account.unrealized_pnl,
        "equity":         round(account.equity, 2),
        "drawdown_pct":   account.drawdown_pct,
        "exposure":       round(account.exposure, 2),
    })


# ── Binance primary WS feed ───────────────────────────────────────────

async def _binance_ws_feed():
    """
    Primary feed: Binance public trade stream (no API key).
    Raises ConnectionError after 3 consecutive failures so the orchestrator
    can hand off to the REST fallback.
    """
    global market
    backoff = 1
    failed_attempts = 0
    while True:
        try:
            async with websockets.connect(
                BINANCE_WS_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
                open_timeout=8,
            ) as ws:
                market.feed_connected = True
                market.last_tick_ts   = time.time()
                backoff         = 1
                failed_attempts = 0
                logger.info("price feed connected  BTCUSDT@trade [binance]")
                await manager.broadcast({"type": "state", "data": _full_state_dict()})

                async for raw in ws:
                    tick  = json.loads(raw)
                    price = float(tick["p"])
                    await _apply_price_tick(price)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            market.feed_connected = False
            failed_attempts += 1
            logger.warning(f"binance feed lost: {e}  retry in {backoff}s")
            await manager.broadcast({"type": "state", "data": _full_state_dict()})
            if failed_attempts >= 3:
                raise ConnectionError(f"binance unreachable after {failed_attempts} attempts")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ── CoinGecko / fallback REST polling feed ──────────────────────────────

_COINGECKO_URL   = ("https://api.coingecko.com/api/v3/simple/price"
                    "?ids=bitcoin&vs_currencies=usd")
_COINPAPRIKA_URL = "https://api.coinpaprika.com/v1/tickers/btc-bitcoin?quotes=USD"
FALLBACK_POLL_INTERVAL        = float(os.getenv("FALLBACK_POLL_INTERVAL", "10"))  # seconds
_FALLBACK_BINANCE_RETRY_EVERY = int(os.getenv("FALLBACK_BINANCE_RETRY_EVERY", "30"))
SKIP_BINANCE_WS               = os.getenv("SKIP_BINANCE_WS", "false").lower() == "true"


async def _fetch_rest(url: str, extract_fn) -> float:
    """Run a blocking REST call in a thread executor."""
    import urllib.request
    loop = asyncio.get_event_loop()
    def _get():
        req = urllib.request.Request(url, headers={"User-Agent": "poly-trading-engine/3"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return extract_fn(json.loads(r.read()))
    return await loop.run_in_executor(None, _get)


async def _fallback_poll_feed():
    """
    Secondary feed: REST polling (CoinGecko -> CoinPaprika).
    Activated when Binance WS is unreachable.
    Retries Binance every _FALLBACK_BINANCE_RETRY_EVERY polls and hands back
    to _binance_ws_feed once Binance becomes reachable again.
    """
    global market
    poll_count = 0
    logger.warning("switching to REST fallback feed (CoinGecko/CoinPaprika)")
    await manager.broadcast({"type": "log", "entry": {
        "ts": datetime.now().strftime("%H:%M:%S"), "level": "WARN",
        "mod": "feed", "msg": "Binance blocked - using REST fallback (CoinGecko)",
    }})

    while True:
        poll_count += 1

        # Periodically retry Binance primary feed (unless REST-only mode)
        if not SKIP_BINANCE_WS and poll_count % _FALLBACK_BINANCE_RETRY_EVERY == 0:
            logger.info("fallback: retrying Binance WS...")
            try:
                await _binance_ws_feed()
                logger.info("fallback: Binance reconnected - returning to primary feed")
                return   # binance alive again; exit fallback loop
            except asyncio.CancelledError:
                raise
            except ConnectionError:
                logger.warning("fallback: Binance still unreachable")

        # Try REST sources in priority order.
        price = None
        sources = [
            ("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", 
             lambda d: float(d["result"]["XXBTZUSD"]["c"][0]), "kraken"),
            ("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
             lambda d: float(d["price"]), "binance-rest"),
            ("https://api.coinbase.com/v2/prices/BTC-USD/spot",
             lambda d: float(d["data"]["amount"]), "coinbase"),
            (_COINGECKO_URL,   lambda d: float(d["bitcoin"]["usd"]),         "coingecko"),
            (_COINPAPRIKA_URL, lambda d: float(d["quotes"]["USD"]["price"]), "coinpaprika"),
        ]
        for url, extract, source in sources:
            try:
                price = await _fetch_rest(url, extract)
                if price and price > 0:
                    market.feed_connected = True
                    logger.debug(f"fallback price  ${price:,.2f}  [{source}]")
                    break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"fallback {source} error: {e}")

        if price and price > 0:
            await _apply_price_tick(price)
        else:
            market.feed_connected = False
            logger.warning("all fallback sources failed")
            await manager.broadcast({"type": "state", "data": _full_state_dict()})

        await asyncio.sleep(FALLBACK_POLL_INTERVAL)


async def binance_price_feed():
    """
    Price feed orchestrator.
    Primary:   Binance WebSocket (real-time, sub-second)
    Fallback:  CoinGecko -> CoinPaprika REST polling (~10s cadence)
    Auto-promotes back to Binance once it becomes reachable again.
    """
    global market
    # REST-only mode: skip Binance WS entirely if configured
    if SKIP_BINANCE_WS:
        logger.info("REST-only mode enabled - skipping Binance WS")
        try:
            await _fallback_poll_feed()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"REST feed fatal: {e}")
        return

    # Normal mode: try Binance WS first, fallback to REST on failure
    while True:
        try:
            await _binance_ws_feed()
        except asyncio.CancelledError:
            break
        except ConnectionError:
            try:
                await _fallback_poll_feed()
                # _fallback_poll_feed returns when Binance is back;
                # loop around to restart primary.
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"fallback feed fatal: {e}")
                await asyncio.sleep(5)
        except Exception as e:
            market.feed_connected = False
            logger.error(f"price feed unexpected error: {e}")
            await asyncio.sleep(5)

# ── Paper Execution Hooks ──────────────────────────────────────────────────

async def paper_fill(direction: str, size: float, price: float,
                     oid: str = None, symbol: str = "BTC-USDT"):
    """Simulate an order fill at current market price."""
    if kill_switch:
        logger.warning(f"kill switch active — fill rejected")
        return None
    if engine_status != EngineStatus.RUNNING:
        logger.warning("engine not running — fill rejected")
        return None

    # ✅ FIX #4: Generate truly unique IDs (nanosecond + UUID)
    if not oid:
        oid = f"p{int(time.time()*1e9)}_{uuid.uuid4().hex[:8]}"

    # ✅ FIX #4: Clamp size to reasonable range (prevent NaN/inf)
    size = float(max(0.0001, min(size, account.equity / price * 0.1))) if price > 0 else size
    if size <= 0:
        logger.warning(f"fill rejected: size={size} must be > 0")
        return None

    if price <= 0:
        logger.warning(f"fill rejected: invalid price={price}")
        return None

    notional = round(size * price, 6)
    if notional < 1.0:
        logger.warning(f"fill rejected: notional=${notional:.4f} < $1")
        return None

    # ✅ FIX #4: Relax cash check for paper trading (allow 10% buffer)
    available = account.available_cash
    max_size = min(account.equity * (MAX_POSITION_PCT / 100), available * 1.1)
    if notional > max_size:
        logger.warning(f"position too large: {notional:.2f} > max {max_size:.2f}")
        return None

    pos = PaperPosition(
        id=oid, symbol=symbol, direction=direction,
        size=size, entry_price=price, notional=notional,
        opened_at=datetime.now().strftime("%H:%M:%S"),
        account_id=account.id, session_id=account.session_id,
    )
    # BUG-1+3 FIX: open_position checks available_cash and guards duplicates
    opened = account.open_position(pos)
    if not opened:
        logger.warning(f"fill rejected: duplicate id={oid} or insufficient cash")
        return None

    # Persist
    save_position({
        **pos.to_dict(),
        "account_id": account.id,
        "session_id": account.session_id,
        "status":     "open",
    })

    tag = "[PAPER]"
    logger.success(f"{tag} {oid} {symbol}  fill {direction} @ {price}  sz {size}  notional ${notional}")
    await manager.broadcast({
        "type":      "positions",
        "data":      [p.to_dict() for p in account.positions.values()],
        "state":     _full_state_dict(),
    })
    return oid


async def paper_settle(oid: str, exit_price: float = None):
    """Settle a paper position at current (or given) price."""
    exit_price = exit_price or market.btc_price
    if not exit_price:
        logger.warning("no price available for settlement")
        return

    trade_data = account.close_position(oid, exit_price)
    if not trade_data:
        logger.warning(f"position {oid} not found")
        return

    # Persist trade + close position record
    close_position_db(oid)
    save_trade({**trade_data, "closed_at": datetime.now().isoformat()})
    # BUG-1 FIX: balance is derived; persist starting_balance only when reset

    # Notify risk engine + any other subscribers of the realized PnL.
    _position_open_ts.pop(oid, None)
    for cb in list(account_callbacks_on_settle):
        try:
            cb(trade_data["pnl"])
        except Exception as e:
            logger.debug(f"settle callback error: {e}")

    # Update daily stats
    today = datetime.now().strftime("%Y-%m-%d")
    upsert_daily_stats(
        account.id, today,
        account.equity, account.realized_pnl,
        account.trade_count, account.win_count
    )

    sign = "+" if trade_data["pnl"] >= 0 else ""
    tag  = "[PAPER]"
    logger.success(
        f"{tag} {oid} {trade_data['direction']} settled "
        f"{trade_data['entry_price']:.4f}→{exit_price:.4f}  "
        f"{sign}${trade_data['pnl']:.2f}  roi {sign}{trade_data['roi_pct']:.1f}%"
    )
    await manager.broadcast({
        "type":      "positions",
        "data":      [p.to_dict() for p in account.positions.values()],
        "state":     _full_state_dict(),
    })


async def on_signal(direction: str, score: float, strategy: str = "MANUAL"):
    logger.info(f"signal {strategy}  {direction}  score {score:.2f}")
    await manager.broadcast({
        "type":  "signal",
        "direction": direction,
        "score": score,
        "strategy": strategy,
    })


# ── Engine: live BTC probability pipeline ──────────────────────────────────

async def _on_engine_signal(direction: str, size_pct: float, prob_output):
    """
    BTCProbEngine signal callback.
    Converts risk-engine recommended size (% of equity) → fractional BTC
    notional and submits a paper fill. Guards against over-filling slots.
    """
    global _position_open_ts

    # ✅ DEBUG: Log ALL signals (even if rejected later)
    logger.info(
        f"[SIGNAL_CALLBACK] direction={direction}  size_pct={size_pct:.1f}%  "
        f"long_prob={prob_output.long_prob:.3f}  conf={prob_output.confidence:.3f}  "
        f"kill_switch={kill_switch}  engine={engine_status == EngineStatus.RUNNING}  "
        f"open_positions={len(account.positions)}/{MAX_OPEN_POSITIONS}"
    )

    # Respect concurrency cap — don't stack too many positions at once.
    if len(account.positions) >= MAX_OPEN_POSITIONS:
        logger.warning(f"[SIGNAL_REJECTED] max {MAX_OPEN_POSITIONS} positions reached")
        return
    if market.btc_price <= 0:
        logger.warning(f"[SIGNAL_REJECTED] btc_price={market.btc_price} <= 0")
        return

    # Clamp size to a sane per-trade max (defense-in-depth alongside risk layer).
    size_pct = max(0.5, min(size_pct, MAX_POSITION_PCT))
    notional_usd = account.equity * (size_pct / 100.0)
    btc_size     = round(notional_usd / market.btc_price, 8)

    oid = await paper_fill(
        direction=direction,
        size=btc_size,
        price=market.btc_price,
        symbol="BTC-USDT",
    )
    if oid:
        _position_open_ts[oid] = time.time()
        await manager.broadcast({
            "type":       "engine_signal",
            "direction":  direction,
            "size_pct":   round(size_pct, 2),
            "long_prob":  prob_output.long_prob,
            "confidence": prob_output.confidence,
            "signal":     prob_output.signal,
            "oid":        oid,
        })


async def _exit_sweep():
    """
    Walk all open positions and settle any that hit TP / SL / max-hold,
    or whose model signal has flipped against them.
    """
    if not account.positions or market.btc_price <= 0:
        return

    price = market.btc_price
    now   = time.time()
    # Snapshot current model bias (if available) for signal-flip exits.
    bias = None
    if btc_engine is not None:
        sig = btc_engine._last_prob_output.signal
        bias = {"LONG": "UP", "SHORT": "DOWN"}.get(sig)

    for oid, pos in list(account.positions.items()):
        move = price - pos.entry_price
        if pos.direction == "DOWN":
            move = -move
        ret_pct = (move / pos.entry_price * 100) if pos.entry_price else 0.0
        hold_min = (now - _position_open_ts.get(oid, now)) / 60.0

        reason = None
        if ret_pct >= TAKE_PROFIT_PCT:
            reason = f"TP +{ret_pct:.2f}%"
        elif ret_pct <= -STOP_LOSS_PCT:
            reason = f"SL {ret_pct:.2f}%"
        elif hold_min >= MAX_HOLD_MINUTES:
            reason = f"time {hold_min:.1f}min"
        elif bias and bias != pos.direction:
            reason = f"signal flip →{bias}"

        if reason:
            logger.info(f"[EXIT] {oid} {pos.direction}  {reason}  "
                        f"roi={ret_pct:.2f}%  hold={hold_min:.1f}min")
            await paper_settle(oid, price)


def _aggregate_candles(candles_1m, n: int, interval: str, source: str):
    """
    Aggregate n × 1m FeedCandles into one higher-timeframe candle.
    Returns list of aggregated candles.
    """
    from btc_prob_engine.data.feed import Candle as FeedCandle
    result = []
    for i in range(0, len(candles_1m) - n + 1, n):
        group = candles_1m[i:i + n]
        if len(group) < n:
            break
        agg = FeedCandle(
            exchange=source, symbol='BTC-USDT', interval=interval,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group),
            ts=group[0].ts,
            closed=True,
        )
        result.append(agg)
    return result


async def _seed_data_ring(ring) -> None:
    """
    Seed the DataRing with 60 historical 1m candles + derived 5m/1h candles
    so RSI/BB/EMA50/MACD all work immediately on engine start.
    Uses CoinGecko OHLC API (4h candles), interpolates to 1m.
    Falls back to synthetic random-walk candles if API fails.
    """
    from btc_prob_engine.data.feed import Candle as FeedCandle, Trade as FeedTrade

    candles_1m_list = []

    try:
        # CoinGecko returns 4h candles for days=1 (48 candles)
        # Use days=2 to get more history for 5m/1h aggregation
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        # Use all available candles (up to 60)
        raw = data[-48:] if len(data) >= 48 else data
        now  = time.time()
        n    = len(raw)

        for i, entry in enumerate(raw):
            _, o, h, l, c = entry
            ts = now - (n - i) * 60  # Space 60s apart to simulate 1m candles

            # Add micro-jitter to create realistic OHLC variation
            jitter = float(c) * random.uniform(-0.0005, 0.0005)
            candle = FeedCandle(
                exchange='coingecko_seed', symbol='BTC-USDT', interval='1m',
                open=float(o),
                high=max(float(h), float(c) + abs(jitter)),
                low=min(float(l), float(c) - abs(jitter)),
                close=float(c),
                volume=random.uniform(3.0, 12.0),
                ts=ts, closed=True,
            )
            ring.push_candle(candle)
            ring.push_trade(FeedTrade(
                exchange='coingecko_seed', symbol='BTC-USDT',
                price=float(c), size=random.uniform(0.5, 3.0),
                side='buy' if float(c) >= float(o) else 'sell', ts=ts,
            ))
            candles_1m_list.append(candle)

        # Derive 5m candles from 5 × 1m
        candles_5m = _aggregate_candles(candles_1m_list, 5, '5m', 'coingecko_seed')
        for c5 in candles_5m:
            ring.push_candle(c5)

        # Derive 1h candles from 12 × 5m
        candles_1h = _aggregate_candles(candles_5m, 12, '1h', 'coingecko_seed')
        for c1h in candles_1h:
            ring.push_candle(c1h)

        logger.info(
            f"[ENGINE] DataRing seeded: {len(candles_1m_list)} × 1m, "
            f"{len(candles_5m)} × 5m, {len(candles_1h)} × 1h (CoinGecko)"
        )
    except Exception as e:
        logger.warning(f"[ENGINE] CoinGecko seed failed ({e}) — using synthetic candles")
        price = market.btc_price or 60000.0
        now   = time.time()
        for i in range(60):
            price = price * (1 + random.uniform(-0.003, 0.003))
            ts    = now - (60 - i) * 60
            candle = FeedCandle(
                exchange='synthetic_seed', symbol='BTC-USDT', interval='1m',
                open=price * (1 + random.uniform(-0.001, 0.001)),
                high=price * (1 + random.uniform(0.0002, 0.002)),
                low=price * (1 - random.uniform(0.0002, 0.002)),
                close=price,
                volume=random.uniform(2.0, 10.0), ts=ts, closed=True,
            )
            ring.push_candle(candle)
            ring.push_trade(FeedTrade(
                exchange='synthetic_seed', symbol='BTC-USDT',
                price=price, size=random.uniform(0.5, 3.0),
                side='buy' if random.random() > 0.5 else 'sell', ts=ts,
            ))
            candles_1m_list.append(candle)

        # Derive 5m and 1h from synthetic 1m
        candles_5m = _aggregate_candles(candles_1m_list, 5, '5m', 'synthetic_seed')
        for c5 in candles_5m:
            ring.push_candle(c5)
        candles_1h = _aggregate_candles(candles_5m, 12, '1h', 'synthetic_seed')
        for c1h in candles_1h:
            ring.push_candle(c1h)

        logger.info(
            f"[ENGINE] synthetic seed: 60 × 1m, {len(candles_5m)} × 5m, "
            f"{len(candles_1h)} × 1h"
        )


async def _trigger_engine_pipeline(engine) -> None:
    """
    After seeding DataRing, run one inference cycle so _last_features,
    _last_prob_output, and _last_risk_score are populated immediately.
    Also ensures market.btc_price is set from seeded data before first signal.
    """
    try:
        from btc_prob_engine.data.feed import Candle as FeedCandle
        import time as _time

        # Ensure market price is set from seeded candles before pipeline runs
        if engine.ring.candles_1m and market.btc_price <= 0:
            market.btc_price = engine.ring.candles_1m[-1].close
            logger.info(f"[ENGINE] market price set from seed: ${market.btc_price:,.2f}")

        if engine.ring.candles_1m:
            last = engine.ring.candles_1m[-1]
            await engine._on_closed_candle(last, _time.time())
            logger.info("[ENGINE] pipeline primed from seeded candles")
    except Exception as e:
        logger.debug(f"pipeline prime failed: {e}")


async def _broadcast_engine_state():
    """Push probability / risk / feature snapshot to dashboards."""
    if btc_engine is None:
        return
    try:
        payload = {"type": "engine_state", "data": btc_engine.state_dict()}
        payload["data"]["equity"]      = round(account.equity, 2)
        payload["data"]["exposure"]    = round(account.exposure, 2)
        payload["data"]["open_count"]  = len(account.positions)
        await manager.broadcast(payload)
    except Exception as e:
        logger.debug(f"engine_state broadcast failed: {e}")


async def _run_engine():
    """
    Live engine loop. Starts the BTCProbEngine feed (which runs the full
    pipeline on every closed 1m candle: features → probability → risk → signal),
    then drives a fast exit-sweep and periodic state broadcasts.
    """
    global engine_status, btc_engine, candle_synth
    logger.info(f"[ENGINE] started  capital=${account.starting_balance}  "
                f"threshold={SIGNAL_THRESHOLD}  TP={TAKE_PROFIT_PCT}%  "
                f"SL={STOP_LOSS_PCT}%  max_hold={MAX_HOLD_MINUTES}min")
    await manager.broadcast({"type": "state", "data": _full_state_dict()})

    # Build a fresh engine bound to the live account.
    btc_engine = BTCProbEngine(
        signal_threshold=SIGNAL_THRESHOLD,
        max_kelly=MAX_KELLY,
        risk_params=RiskParams(max_single_pos_pct=MAX_POSITION_PCT),
        on_signal=_on_engine_signal,
    )

    feed_task = None
    exit_task = None
    try:
        # In REST-only mode, don't start the engine's built-in Binance feed.
        # Instead, wire the candle synthesizer to inject candles into the engine's DataRing.
        if SKIP_BINANCE_WS:
            # Initialize equity/exposure functions (normally set in engine.start())
            btc_engine._equity_fn = lambda: account.equity
            btc_engine._exposure_fn = lambda: account.exposure
            btc_engine._running = True
            
            candle_synth = CandleSynthesizer()
            @candle_synth.on_candle
            async def handle_synth_candle(candle, ts):
                if candle.closed:
                    # Push candle to engine's DataRing
                    btc_engine.ring.push_candle(candle)
                    # Trigger engine's candle handler directly
                    await btc_engine._on_closed_candle(candle, ts)
            logger.info("[ENGINE] using candle synthesizer (REST-only mode)")

            # ✅ Seed DataRing with 30 historical 1m candles so RSI/BB work immediately
            await _seed_data_ring(btc_engine.ring)
            # ✅ Prime the pipeline so features/prob/risk are populated immediately
            await _trigger_engine_pipeline(btc_engine)
        else:
            # Normal mode: start engine's built-in Binance feed
            feed_task = asyncio.create_task(
                btc_engine.start(
                    equity_fn=lambda: account.equity,
                    exposure_fn=lambda: account.exposure,
                )
            )
        
        # Feed risk analytics into the risk engine after every settle.
        account_callbacks_on_settle.append(_record_trade_risk)

        last_state_ts = 0.0
        while engine_status == EngineStatus.RUNNING:
            if kill_switch:
                logger.warning("[ENGINE] kill switch active — halting")
                break
            # Exit sweep on the price tick cadence.
            await _exit_sweep()

            # Throttle engine-state broadcasts to ~once per EXIT_CHECK_INTERVAL.
            now = time.time()
            if now - last_state_ts >= EXIT_CHECK_INTERVAL:
                await _broadcast_engine_state()
                last_state_ts = now

            await asyncio.sleep(EXIT_CHECK_INTERVAL)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[ENGINE] fatal: {e}")
    finally:
        # Tear down feed + exit tasks.
        try:
            btc_engine.stop()
        except Exception:
            pass
        for t in (feed_task, exit_task):
            if t:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        if account_callbacks_on_settle and _record_trade_risk in account_callbacks_on_settle:
            account_callbacks_on_settle.remove(_record_trade_risk)
        # Clean up candle synthesizer
        candle_synth = None
        engine_status = EngineStatus.STOPPED
        logger.info("[ENGINE] stopped")
        await manager.broadcast({"type": "state", "data": _full_state_dict()})


def _record_trade_risk(pnl: float):
    """Feed realized PnL + equity into the risk engine after each settle.
    If the risk layer trips its kill switch (drawdown / daily loss), propagate
    to the global kill_switch so the engine loop halts on the next tick."""
    global kill_switch
    if btc_engine is not None:
        try:
            btc_engine.risk.record(
                pnl, account.equity,
                btc_engine._last_features.get("garch_vol_1m", 0.0),
            )
            if btc_engine.risk._kill_switch and not kill_switch:
                kill_switch = True
                logger.warning("[RISK] risk engine triggered kill switch "
                               "(drawdown / daily limit breached)")
        except Exception:
            pass


async def start_engine():
    global _engine_task, engine_status, kill_switch
    if engine_status == EngineStatus.RUNNING:
        return
    engine_status = EngineStatus.RUNNING
    kill_switch   = False
    _engine_task  = asyncio.create_task(_run_engine())


async def stop_engine():
    global _engine_task, engine_status
    if engine_status != EngineStatus.RUNNING:
        return
    engine_status = EngineStatus.STOPPING
    logger.info("[ENGINE] stopping...")
    await manager.broadcast({"type": "state", "data": _full_state_dict()})
    if _engine_task:
        _engine_task.cancel()
        try:
            await _engine_task
        except asyncio.CancelledError:
            pass


# ── Heartbeat + equity snapshots ───────────────────────────────────────────

async def heartbeat_loop():
    global _heartbeat_count
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        _heartbeat_count += 1
        ts = datetime.now().strftime("%H:%M:%S")

        account.check_new_day()

        # Persist equity snapshot periodically
        if _heartbeat_count % EQUITY_SNAPSHOT_EVERY == 0:
            save_equity_snapshot(
                account.id,
                account.equity,
                account.realized_pnl,
                account.unrealized_pnl,
                market.btc_price,
                account.drawdown_pct,
            )

        await manager.broadcast({
            "type":  "heartbeat",
            "ts":    ts,
            "state": _full_state_dict(),
        })


# ── State dict ─────────────────────────────────────────────────────────────

def _full_state_dict() -> dict:
    s = account.to_state_dict(market.btc_price)
    s.update({
        "engine_status":  engine_status,
        "kill_switch":    kill_switch,
        "feed_connected": market.feed_connected,
        "feed_latency_ms": market.feed_latency_ms,
        "ws_clients":     len(manager.active),
    })
    return s


# ── State recovery ─────────────────────────────────────────────────────────

def restore_state():
    """Load persistent state from DB on startup."""
    acc_row = get_or_create_account("default", DEFAULT_CAPITAL)
    account.id               = acc_row["id"]
    account.name             = acc_row["name"]
    account.starting_balance = acc_row["starting_balance"]
    # BUG-1 FIX: no current_balance field — equity is derived from realized+unrealized
    # peak_equity set after trades are loaded below

    # Create a new session for this run
    session_id = create_session(account.id, account.equity)
    account.session_id = session_id
    account.session_started = time.time()

    # ✅ FIX: Close stale open positions from previous sessions before restoring
    # Paper positions are only valid for the current run; stale ones block new trades.
    import sqlite3 as _sqlite3
    try:
        _conn = _sqlite3.connect("poly_engine.db")
        _conn.execute("UPDATE positions SET status='closed' WHERE status='open'")
        _conn.commit()
        stale = _conn.execute("SELECT changes()").fetchone()[0]
        _conn.close()
        if stale:
            logger.info(f"restore_state: closed {stale} stale positions from previous session")
    except Exception as _e:
        logger.debug(f"stale position cleanup failed: {_e}")

    # Restore open positions
    open_positions = load_open_positions(account.id)
    for p in open_positions:
        pos = PaperPosition(
            id=p["id"], symbol=p["symbol"], direction=p["direction"],
            size=p["size"], entry_price=p["entry_price"], notional=p["notional"],
            unrealized_pnl=p["unrealized_pnl"], roi_pct=p["roi_pct"],
            opened_at=p["opened_at"], account_id=account.id, session_id=account.session_id,
        )
        account.positions[pos.id] = pos
        # ✅ FIX: Register restored positions in _position_open_ts
        # Use a very old timestamp so time-based exit fires quickly if stale.
        # Positions from the current session will be refreshed when found in logs.
        if pos.id not in _position_open_ts:
            _position_open_ts[pos.id] = time.time() - MAX_HOLD_MINUTES * 60

    # Restore counters from trades table
    trades = load_trades(account.id, limit=10000)
    account.trade_count   = len(trades)
    account.win_count     = sum(1 for t in trades if t["pnl"] > 0)
    account.loss_count    = sum(1 for t in trades if t["pnl"] <= 0)
    account.realized_pnl  = round(sum(t["pnl"] for t in trades), 6)

    # BUG-5 FIX: restore daily PnL from DB — was always 0 after restart
    account.daily_realized_pnl = load_daily_pnl_today(account.id)

    # Restore unrealized from open positions (price unknown until first tick)
    # Will auto-update on first Binance price tick
    account.unrealized_pnl = round(
        sum(p.unrealized_pnl for p in account.positions.values()), 6
    )

    # Restore peak_equity from starting + realized (conservative floor)
    account.peak_equity = max(
        acc_row["starting_balance"] + account.realized_pnl,
        acc_row["starting_balance"]
    )

    logger.info(
        f"state restored  account={account.name}  "
        f"equity=${account.equity:.2f}  "
        f"realized_pnl=${account.realized_pnl:.2f}  "
        f"daily_pnl=${account.daily_realized_pnl:.2f}  "
        f"open_positions={len(account.positions)}  "
        f"trades={account.trade_count}"
    )


# ── Command handler ─────────────────────────────────────────────────────────

async def handle_command(data: dict):
    global kill_switch

    cmd = data.get("cmd")
    logger.debug(f"cmd received: {cmd}")

    # ── Engine control ──────────────────────────────────────────────
    if cmd == "start":
        await start_engine()

    elif cmd == "stop":
        await stop_engine()

    elif cmd == "kill_switch":
        kill_switch = True
        logger.warning("KILL SWITCH activated")
        await stop_engine()
        await manager.broadcast({"type": "state", "data": _full_state_dict()})

    elif cmd == "reset_kill":
        kill_switch = False
        logger.info("kill switch reset")
        await manager.broadcast({"type": "state", "data": _full_state_dict()})

    elif cmd == "ping":
        await manager.broadcast({
            "type": "pong",
            "ts":   datetime.now().strftime("%H:%M:%S")
        })

    # ── Capital management ──────────────────────────────────────────
    elif cmd == "set_capital":
        if engine_status == EngineStatus.RUNNING:
            await manager.broadcast({"type": "error", "msg": "Stop engine before changing capital"})
            return
        if account.positions:
            await manager.broadcast({"type": "error", "msg": "Close all positions before changing capital"})
            return
        # BUG-7 FIX: validate input before float() conversion
        try:
            new_cap = float(data.get("amount", DEFAULT_CAPITAL))
        except (TypeError, ValueError):
            await manager.broadcast({"type": "error", "msg": "Invalid capital amount"})
            return
        if new_cap < 10 or new_cap > 100_000_000:
            await manager.broadcast({"type": "error", "msg": "Capital must be between $10 and $100M"})
            return
        account.reset(new_cap)
        update_account_capital(account.id, new_cap)
        logger.info(f"paper capital set to ${new_cap:,.2f}")
        await manager.broadcast({"type": "state", "data": _full_state_dict()})

    # ── Manual paper trades ─────────────────────────────────────────
    elif cmd == "manual_buy":
        direction    = data.get("direction", "UP")
        pct          = float(data.get("size_pct", 5.0))      # % of equity
        price        = market.btc_price
        if price <= 0:
            await manager.broadcast({"type": "error", "msg": "No live price yet"})
            return
        # BUG-2 FIX: USD-denominated fractional sizing (not int BTC units)
        notional_usd = account.equity * (pct / 100.0)
        size         = round(notional_usd / price, 8)        # fractional BTC
        await paper_fill(direction, size, price)

    elif cmd == "manual_sell":
        oid = data.get("position_id")
        if oid:
            await paper_settle(oid)
        else:
            # Close all positions
            for pid in list(account.positions.keys()):
                await paper_settle(pid)

    elif cmd == "manual_fill":
        # BUG-7 FIX: validate all float inputs
        try:
            fill_size  = float(data.get("size", 0))
            fill_price = float(data.get("price", market.btc_price) or market.btc_price)
        except (TypeError, ValueError):
            await manager.broadcast({"type": "error", "msg": "Invalid size or price"})
            return
        await paper_fill(
            direction = data.get("direction", "UP"),
            size      = fill_size,
            price     = fill_price,
            oid       = data.get("oid"),
            symbol    = data.get("symbol", "BTC-USDT"),
        )

    elif cmd == "manual_settle":
        try:
            ep = float(data.get("exit_price", 0) or 0)
        except (TypeError, ValueError):
            ep = 0.0
        await paper_settle(
            oid        = data.get("position_id"),
            exit_price = ep if ep > 0 else None,
        )

    # ── Data requests ───────────────────────────────────────────────
    elif cmd == "get_equity_curve":
        curve = load_equity_curve(account.id, limit=data.get("limit", 1000))
        await manager.broadcast({"type": "equity_curve", "data": curve})

    elif cmd == "get_analytics":
        analytics = compute_analytics(account.id)
        await manager.broadcast({"type": "analytics", "data": analytics})

    elif cmd == "get_trades":
        trades = load_trades(account.id, limit=data.get("limit", 100))
        await manager.broadcast({"type": "trades", "data": trades})

    elif cmd == "get_daily_stats":
        stats = load_daily_stats(account.id, days=data.get("days", 30))
        await manager.broadcast({"type": "daily_stats", "data": stats})


# ── FastAPI ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Path("logs").mkdir(exist_ok=True)
    init_db()
    restore_state()
    logger.add(loguru_ws_sink, format="{message}", level="DEBUG", enqueue=False)
    logger.info("Poly Trading Engine v3 — starting up")

    hb_task    = asyncio.create_task(heartbeat_loop())
    price_task = asyncio.create_task(binance_price_feed())

    yield

    # Shutdown
    hb_task.cancel()
    price_task.cancel()
    if _engine_task:
        _engine_task.cancel()
    close_session(
        account.session_id, account.equity,
        account.realized_pnl, account.trade_count, account.win_count
    )
    logger.info("Poly Trading Engine — shutdown complete")


app = FastAPI(title="Poly Trading Engine v3", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static dashboard if file exists
_dashboard = Path("poly-trading-dashboard.html")
if _dashboard.exists():
    @app.get("/")
    async def serve_dashboard():
        from fastapi.responses import FileResponse
        return FileResponse("poly-trading-dashboard.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    await manager.send_snapshot(ws)
    try:
        while True:
            data = await ws.receive_json()
            await handle_command(data)
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        logger.error(f"ws error: {e}")
        manager.disconnect(ws)


@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "engine":          engine_status,
        "kill_switch":     kill_switch,
        "capital":         account.starting_balance,
        "equity":          round(account.equity, 2),
        "realized_pnl":    round(account.realized_pnl, 2),
        "unrealized_pnl":  round(account.unrealized_pnl, 2),
        "roi_pct":         account.roi_pct,
        "win_rate":        account.win_rate,
        "trade_count":     account.trade_count,
        "open_positions":  len(account.positions),
        "btc_price":       market.btc_price,
        "feed_connected":  market.feed_connected,
        "ws_clients":      len(manager.active),
        "drawdown_pct":    account.drawdown_pct,
    }


@app.get("/analytics")
async def analytics():
    return compute_analytics(account.id)


@app.get("/equity-curve")
async def equity_curve(limit: int = 500):
    return load_equity_curve(account.id, limit=limit)


@app.get("/trades")
async def trades(limit: int = 100):
    return load_trades(account.id, limit=limit)


@app.get("/daily-stats")
async def daily_stats(days: int = 30):
    return load_daily_stats(account.id, days=days)


@app.get("/engine-state")
async def engine_state():
    """Live probability/risk/feature snapshot from the BTC engine."""
    if btc_engine is None:
        return {"running": False, "message": "engine not started"}
    state = btc_engine.state_dict()
    state["running"]      = engine_status == EngineStatus.RUNNING
    state["equity"]       = round(account.equity, 2)
    state["exposure"]     = round(account.exposure, 2)
    state["open_count"]   = len(account.positions)
    state["risk_analytics"] = btc_engine.risk.analytics(account.equity)
    return state


if __name__ == "__main__":
    uvicorn.run(
        "ws_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="warning",
    )
