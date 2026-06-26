"""
Poly Trading Engine — Database Layer v2
========================================
FIXED BUGS (audit 2026-06-26):
  BUG-5: Added load_daily_pnl() to restore daily_realized_pnl on restart
  BUG-6: compute_analytics() now uses ONE connection for all queries
         (was two separate db_conn() calls — inconsistent under load)
  NEW:   load_daily_pnl_today() for daily reset restoration
"""

import sqlite3
import os
import statistics
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from loguru import logger

DB_PATH = os.getenv("DB_PATH", "poly_engine.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")  # wait up to 5s on lock
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_conn():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    with db_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL DEFAULT 'default',
            starting_balance REAL NOT NULL DEFAULT 1000.0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL REFERENCES accounts(id),
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            ended_at TEXT,
            starting_equity REAL NOT NULL DEFAULT 0.0,
            ending_equity REAL,
            realized_pnl REAL NOT NULL DEFAULT 0.0,
            trade_count INTEGER NOT NULL DEFAULT 0,
            win_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS positions (
            id TEXT PRIMARY KEY,
            account_id INTEGER NOT NULL REFERENCES accounts(id),
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('UP','DOWN')),
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            notional REAL NOT NULL,
            unrealized_pnl REAL NOT NULL DEFAULT 0.0,
            roi_pct REAL NOT NULL DEFAULT 0.0,
            opened_at TEXT NOT NULL DEFAULT (datetime('now')),
            status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','closed'))
        );

        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id TEXT NOT NULL,
            account_id INTEGER NOT NULL REFERENCES accounts(id),
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            notional REAL NOT NULL,
            pnl REAL NOT NULL,
            roi_pct REAL NOT NULL,
            opened_at TEXT NOT NULL,
            closed_at TEXT NOT NULL DEFAULT (datetime('now')),
            hold_minutes REAL NOT NULL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS equity_curve (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL REFERENCES accounts(id),
            ts TEXT NOT NULL DEFAULT (datetime('now')),
            equity REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0.0,
            unrealized_pnl REAL NOT NULL DEFAULT 0.0,
            btc_price REAL NOT NULL DEFAULT 0.0,
            drawdown_pct REAL NOT NULL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL REFERENCES accounts(id),
            date TEXT NOT NULL,
            starting_equity REAL NOT NULL DEFAULT 0.0,
            ending_equity REAL NOT NULL DEFAULT 0.0,
            realized_pnl REAL NOT NULL DEFAULT 0.0,
            trade_count INTEGER NOT NULL DEFAULT 0,
            win_count INTEGER NOT NULL DEFAULT 0,
            UNIQUE(account_id, date)
        );

        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL DEFAULT (datetime('now')),
            level TEXT NOT NULL,
            module TEXT NOT NULL,
            message TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_equity_curve_account_ts ON equity_curve(account_id, ts);
        CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account_id);
        CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at);
        CREATE INDEX IF NOT EXISTS idx_trades_account_date ON trades(account_id, closed_at);
        CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
        """)
    logger.info(f"database initialized  path={DB_PATH}")


# ── Account ────────────────────────────────────────────────────────────────

def get_or_create_account(name: str = "default", starting_balance: float = 1000.0) -> dict:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM accounts WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return dict(row)
        conn.execute(
            "INSERT INTO accounts (name, starting_balance) VALUES (?,?)",
            (name, starting_balance)
        )
        row = conn.execute("SELECT * FROM accounts WHERE name = ?", (name,)).fetchone()
        return dict(row)


def update_account_capital(account_id: int, starting_balance: float):
    with db_conn() as conn:
        conn.execute(
            "UPDATE accounts SET starting_balance=?, updated_at=datetime('now') WHERE id=?",
            (starting_balance, account_id)
        )


# ── Session ────────────────────────────────────────────────────────────────

def create_session(account_id: int, starting_equity: float) -> int:
    with db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (account_id, starting_equity) VALUES (?,?)",
            (account_id, starting_equity)
        )
        return cur.lastrowid


def close_session(session_id: int, ending_equity: float, realized_pnl: float,
                  trade_count: int, win_count: int):
    with db_conn() as conn:
        conn.execute(
            """UPDATE sessions SET ended_at=datetime('now'), ending_equity=?,
               realized_pnl=?, trade_count=?, win_count=? WHERE id=?""",
            (ending_equity, realized_pnl, trade_count, win_count, session_id)
        )


# ── Positions ──────────────────────────────────────────────────────────────

def save_position(pos: dict):
    with db_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO positions
               (id, account_id, session_id, symbol, direction, size, entry_price,
                notional, unrealized_pnl, roi_pct, opened_at, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pos["id"], pos["account_id"], pos["session_id"], pos["symbol"],
             pos["direction"], pos["size"], pos["entry_price"], pos["notional"],
             pos.get("unrealized_pnl", 0.0), pos.get("roi_pct", 0.0),
             pos.get("opened_at", datetime.now().isoformat()), pos.get("status", "open"))
        )


def update_position_pnl(pos_id: str, unrealized_pnl: float, roi_pct: float):
    try:
        with db_conn() as conn:
            conn.execute(
                "UPDATE positions SET unrealized_pnl=?, roi_pct=? WHERE id=?",
                (round(unrealized_pnl, 6), round(roi_pct, 6), pos_id)
            )
    except Exception as e:
        # BUG-9 FIX: never let a DB error crash the price feed
        logger.debug(f"update_position_pnl failed silently: {e}")


def close_position_db(pos_id: str):
    with db_conn() as conn:
        conn.execute(
            "UPDATE positions SET status='closed' WHERE id=?", (pos_id,)
        )


def load_open_positions(account_id: int) -> list:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE account_id=? AND status='open'",
            (account_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Trades ─────────────────────────────────────────────────────────────────

def save_trade(trade: dict):
    with db_conn() as conn:
        conn.execute(
            """INSERT INTO trades
               (position_id, account_id, session_id, symbol, direction, size,
                entry_price, exit_price, notional, pnl, roi_pct, opened_at,
                closed_at, hold_minutes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (trade["position_id"], trade["account_id"], trade["session_id"],
             trade["symbol"], trade["direction"], trade["size"],
             trade["entry_price"], trade["exit_price"], trade["notional"],
             trade["pnl"], trade["roi_pct"], trade["opened_at"],
             trade.get("closed_at", datetime.now().isoformat()),
             trade.get("hold_minutes", 0.0))
        )


def load_trades(account_id: int, limit: int = 500) -> list:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE account_id=? ORDER BY closed_at DESC LIMIT ?",
            (account_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]


def load_daily_pnl_today(account_id: int) -> float:
    """
    BUG-5 FIX: Restore today's realized PnL on server restart.
    Sums all trades closed today for this account.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    with db_conn() as conn:
        row = conn.execute(
            """SELECT COALESCE(SUM(pnl), 0.0) as daily_pnl
               FROM trades
               WHERE account_id=? AND date(closed_at)=?""",
            (account_id, today)
        ).fetchone()
        return float(row["daily_pnl"]) if row else 0.0


# ── Equity Curve ───────────────────────────────────────────────────────────

def save_equity_snapshot(account_id: int, equity: float, realized_pnl: float,
                          unrealized_pnl: float, btc_price: float, drawdown_pct: float):
    with db_conn() as conn:
        conn.execute(
            """INSERT INTO equity_curve
               (account_id, equity, realized_pnl, unrealized_pnl, btc_price, drawdown_pct)
               VALUES (?,?,?,?,?,?)""",
            (account_id, round(equity, 6), round(realized_pnl, 6),
             round(unrealized_pnl, 6), btc_price, round(drawdown_pct, 6))
        )


def load_equity_curve(account_id: int, limit: int = 2000) -> list:
    with db_conn() as conn:
        rows = conn.execute(
            """SELECT ts, equity, realized_pnl, unrealized_pnl, btc_price, drawdown_pct
               FROM equity_curve WHERE account_id=?
               ORDER BY ts DESC LIMIT ?""",
            (account_id, limit)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


# ── Daily Stats ────────────────────────────────────────────────────────────

def upsert_daily_stats(account_id: int, date: str, ending_equity: float,
                        realized_pnl: float, trade_count: int, win_count: int):
    with db_conn() as conn:
        existing = conn.execute(
            "SELECT id, starting_equity FROM daily_stats WHERE account_id=? AND date=?",
            (account_id, date)
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE daily_stats SET ending_equity=?, realized_pnl=?,
                   trade_count=?, win_count=? WHERE id=?""",
                (ending_equity, realized_pnl, trade_count, win_count, existing["id"])
            )
        else:
            conn.execute(
                """INSERT INTO daily_stats
                   (account_id, date, starting_equity, ending_equity, realized_pnl,
                    trade_count, win_count)
                   VALUES (?,?,?,?,?,?,?)""",
                (account_id, date, ending_equity, ending_equity, realized_pnl,
                 trade_count, win_count)
            )


def load_daily_stats(account_id: int, days: int = 30) -> list:
    with db_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM daily_stats WHERE account_id=?
               ORDER BY date DESC LIMIT ?""",
            (account_id, days)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


# ── Log Archive ────────────────────────────────────────────────────────────

def archive_log(level: str, module: str, message: str):
    try:
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO logs (level, module, message) VALUES (?,?,?)",
                (level, module[:16], message[:512])
            )
    except Exception:
        pass  # log archival must never crash the engine


# ── Analytics (BUG-6 FIX: single connection for all queries) ──────────────

def compute_analytics(account_id: int) -> dict:
    """
    BUG-6 FIX: Uses ONE db_conn() for both trades and equity_curve queries.
    Previously opened two separate connections — inconsistent reads under load.
    """
    with db_conn() as conn:
        trade_rows = conn.execute(
            "SELECT pnl, roi_pct FROM trades WHERE account_id=?", (account_id,)
        ).fetchall()

        if not trade_rows:
            return _empty_analytics()

        pnls  = [r["pnl"] for r in trade_rows]
        total = len(pnls)
        wins  = [p for p in pnls if p > 0]
        losses= [p for p in pnls if p <= 0]

        avg_win  = sum(wins)  / len(wins)   if wins   else 0.0
        avg_loss = sum(losses)/ len(losses) if losses else 0.0
        pf_denom = abs(sum(losses))
        profit_factor = abs(sum(wins) / pf_denom) if pf_denom != 0 else 99.0

        # Sharpe (simplified)
        sharpe = 0.0
        if len(pnls) > 1:
            mean  = sum(pnls) / len(pnls)
            var   = sum((p - mean)**2 for p in pnls) / (len(pnls) - 1)
            stdev = var ** 0.5
            sharpe = (mean / stdev * (252 ** 0.5)) if stdev > 0 else 0.0

        # Max drawdown — SAME connection (BUG-6 fix)
        curve_rows = conn.execute(
            "SELECT equity FROM equity_curve WHERE account_id=? ORDER BY ts",
            (account_id,)
        ).fetchall()

        max_dd = 0.0
        if curve_rows:
            equities = [r["equity"] for r in curve_rows]
            peak     = equities[0]
            for e in equities:
                peak   = max(peak, e)
                dd     = (peak - e) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)

    return {
        "total_trades":  total,
        "win_count":     len(wins),
        "loss_count":    len(losses),
        "win_rate":      round(len(wins) / total * 100, 2) if total else 0.0,
        "total_pnl":     round(sum(pnls), 2),
        "avg_trade":     round(sum(pnls) / total, 2) if total else 0.0,
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
        "profit_factor": round(min(profit_factor, 99.0), 2),
        "sharpe_ratio":  round(sharpe, 2),
        "max_drawdown":  round(max_dd, 2),
    }


def _empty_analytics() -> dict:
    return {
        "total_trades": 0, "win_count": 0, "loss_count": 0,
        "win_rate": 0.0, "total_pnl": 0.0, "avg_trade": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
        "sharpe_ratio": 0.0, "max_drawdown": 0.0,
    }
