"""
binance_relay.py
================
Lightweight WebSocket relay. Connects to Binance public streams and
re-broadcasts every message to all local subscribers.

Run this on any machine that can reach stream.binance.com, then point
the trading engine at it via BINANCE_PROXY_URL.

Install:  pip install websockets
Run:      python binance_relay.py
          (default: listens on ws://0.0.0.0:9001)

Then in your .env on the engine machine:
  BINANCE_PROXY_URL=ws://<relay-ip>:9001

Streams relayed (combined multi-stream):
  btcusdt@trade            spot trade ticks
  btcusdt@depth20@100ms    order book snapshots
  btcusdt@kline_1m/5m/1h   spot candles
  btcusdt@forceOrder       futures liquidations (from fstream)
"""

import asyncio
import json
import os
import time
from collections import deque
from typing import Set

import websockets
try:
    from websockets.server import WebSocketServerProtocol
except ImportError:
    from websockets.legacy.server import WebSocketServerProtocol  # type: ignore

# ── Config ────────────────────────────────────────────────────────────

HOST       = os.getenv("RELAY_HOST", "0.0.0.0")
PORT       = int(os.getenv("RELAY_PORT", "9001"))

SPOT_STREAMS = [
    "btcusdt@trade",
    "btcusdt@depth20@100ms",
    "btcusdt@kline_1m",
    "btcusdt@kline_5m",
    "btcusdt@kline_1h",
]
FUTURES_STREAMS = [
    "btcusdt@forceOrder",
    "btcusdt@kline_5m",
]

SPOT_URL    = "wss://stream.binance.com:9443/stream?streams=" + "/".join(SPOT_STREAMS)
FUTURES_URL = "wss://fstream.binance.com/stream?streams=" + "/".join(FUTURES_STREAMS)

# ── Subscriber registry + replay buffer ──────────────────────────────────

subscribers: Set[WebSocketServerProtocol] = set()
replay: deque = deque(maxlen=50)   # last 50 messages for late joiners


async def broadcast(msg: str):
    replay.append(msg)
    dead = set()
    for ws in list(subscribers):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    subscribers.difference_update(dead)


# ── Binance upstream pumps ──────────────────────────────────────────────

async def pump(url: str, label: str):
    """Connect to Binance, re-broadcast every message to all subscribers."""
    backoff = 1
    while True:
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=10, close_timeout=5
            ) as ws:
                backoff = 1
                print(f"[relay] {label} connected", flush=True)
                async for raw in ws:
                    await broadcast(raw)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[relay] {label} lost: {e}  retry in {backoff}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ── Subscriber server ─────────────────────────────────────────────────────────

async def handle_subscriber(ws: WebSocketServerProtocol):
    """Accept a new subscriber and replay recent messages."""
    subscribers.add(ws)
    print(f"[relay] subscriber connected  total={len(subscribers)}", flush=True)
    # Replay last N messages so the new client gets context immediately
    for msg in list(replay):
        try:
            await ws.send(msg)
        except Exception:
            break
    try:
        await ws.wait_closed()
    finally:
        subscribers.discard(ws)
        print(f"[relay] subscriber disconnected  total={len(subscribers)}", flush=True)


# ── Entrypoint ────────────────────────────────────────────────────────────────

async def main():
    print(f"[relay] starting  listen={HOST}:{PORT}", flush=True)
    print(f"[relay] spot    → {SPOT_URL[:60]}...", flush=True)
    print(f"[relay] futures → {FUTURES_URL[:60]}...", flush=True)

    server = await websockets.serve(handle_subscriber, HOST, PORT)
    print(f"[relay] ready  ws://{HOST}:{PORT}", flush=True)

    await asyncio.gather(
        pump(SPOT_URL,    "spot"),
        pump(FUTURES_URL, "futures"),
        server.wait_closed(),
    )


if __name__ == "__main__":
    asyncio.run(main())
