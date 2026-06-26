"""
Poly Trading Engine — Paper Account v2
=======================================
FIXED BUGS (audit 2026-06-26):
  BUG-1 [CRITICAL] equity model was wrong — open_position() deducted notional
         from current_balance, making equity = cash only (not cash + position value).
         CORRECT model: equity = starting_balance + realized_pnl + unrealized_pnl
         current_balance tracks AVAILABLE CASH (not total equity).

  BUG-3 [MAJOR] open_position() had no duplicate-ID guard — calling twice
         deducted notional twice from current_balance.

Correct accounting model (matches every real exchange):
  available_cash   = starting_balance + realized_pnl - sum(notional_of_open_positions)
  unrealized_pnl   = sum(mark_to_market gain/loss on each open position)
  equity           = starting_balance + realized_pnl + unrealized_pnl
  ROI              = (equity - starting_balance) / starting_balance * 100
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import time


@dataclass
class PaperPosition:
    id:            str
    symbol:        str
    direction:     str       # UP | DOWN
    size:          float     # units (e.g. BTC)
    entry_price:   float
    notional:      float     # size * entry_price (USD committed)
    unrealized_pnl: float = 0.0
    roi_pct:       float  = 0.0
    opened_at:     str    = ""
    account_id:    int    = 1
    session_id:    int    = 1

    def update_pnl(self, current_price: float):
        """
        Unrealized PnL = price move * size (direction-aware).
        UP:   profit when price rises.
        DOWN: profit when price falls.
        """
        move = current_price - self.entry_price
        if self.direction == "DOWN":
            move = -move
        self.unrealized_pnl = round(move * self.size, 6)
        self.roi_pct = round(move / self.entry_price * 100, 6) if self.entry_price else 0.0

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "symbol":         self.symbol,
            "direction":      self.direction,
            "size":           self.size,
            "entry_price":    self.entry_price,
            "notional":       self.notional,
            "unrealized_pnl": self.unrealized_pnl,
            "roi_pct":        self.roi_pct,
            "opened_at":      self.opened_at,
        }


@dataclass
class PaperAccount:
    id:               int   = 1
    name:             str   = "default"
    starting_balance: float = 1000.0
    realized_pnl:     float = 0.0     # cumulative settled PnL
    unrealized_pnl:   float = 0.0     # live mark-to-market
    peak_equity:      float = 1000.0  # for drawdown calculation
    max_drawdown_pct: float = 0.0
    trade_count:      int   = 0
    win_count:        int   = 0
    loss_count:       int   = 0
    session_id:       int   = 1
    session_started:  float = field(default_factory=time.time)
    daily_realized_pnl: float = 0.0
    _today:           str   = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    # Open positions (in-memory)
    positions: Dict[str, PaperPosition] = field(default_factory=dict)

    # ── Core accounting ────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        """
        FIX BUG-1: equity = starting_balance + realized_pnl + unrealized_pnl
        This is always correct regardless of how much is in open positions.
        The notional in open positions is still OUR capital — just deployed.
        """
        return round(self.starting_balance + self.realized_pnl + self.unrealized_pnl, 6)

    @property
    def available_cash(self) -> float:
        """
        Cash not committed to open positions.
        = equity - sum(notional of open positions)
        Cannot go below 0.
        """
        committed = sum(p.notional for p in self.positions.values())
        return max(0.0, round(self.equity - committed, 6))

    @property
    def roi_pct(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return round((self.equity - self.starting_balance) / self.starting_balance * 100, 4)

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return round(self.win_count / self.trade_count * 100, 2)

    @property
    def exposure(self) -> float:
        """Total notional in open positions."""
        return round(sum(p.notional for p in self.positions.values()), 6)

    @property
    def exposure_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        return round(self.exposure / self.equity * 100, 2)

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak equity."""
        if self.peak_equity <= 0:
            return 0.0
        dd = (self.peak_equity - self.equity) / self.peak_equity * 100
        return round(max(0.0, dd), 4)

    @property
    def session_runtime_s(self) -> int:
        return int(time.time() - self.session_started)

    # ── Position management ────────────────────────────────────────────────

    def can_open(self, notional: float) -> tuple:
        """
        Returns (ok: bool, reason: str).
        Checks available cash, not equity — prevents over-commitment.
        """
        if notional <= 0:
            return False, "notional must be > 0"
        if notional > self.available_cash:
            return False, (f"insufficient cash: need ${notional:.2f} "
                           f"have ${self.available_cash:.2f}")
        return True, "ok"

    def open_position(self, pos: PaperPosition) -> bool:
        """
        FIX BUG-1: Do NOT touch any balance counter.
                   Equity is derived from realized_pnl + unrealized_pnl.
        FIX BUG-3: Guard against duplicate position ID — silently ignore.
        Returns True if opened, False if duplicate.
        """
        if pos.id in self.positions:
            return False   # duplicate — already open
        # Pre-check cash availability
        ok, reason = self.can_open(pos.notional)
        if not ok:
            return False
        self.positions[pos.id] = pos
        return True

    def close_position(self, pos_id: str, exit_price: float) -> Optional[dict]:
        """
        Settle a position at exit_price.
        Updates realized_pnl in-place.
        Returns trade record dict or None if position not found.
        """
        pos = self.positions.pop(pos_id, None)
        if not pos:
            return None  # already closed or never existed

        pos.update_pnl(exit_price)
        pnl = pos.unrealized_pnl   # locked in at exit

        # Update realized — this flows back into equity via property
        self.realized_pnl       = round(self.realized_pnl + pnl, 6)
        self.daily_realized_pnl = round(self.daily_realized_pnl + pnl, 6)

        # Recompute unrealized (position removed from dict above)
        self.unrealized_pnl = round(
            sum(p.unrealized_pnl for p in self.positions.values()), 6
        )

        # Stats
        self.trade_count += 1
        if pnl > 0:
            self.win_count  += 1
        else:
            self.loss_count += 1

        return {
            "position_id": pos.id,
            "account_id":  pos.account_id,
            "session_id":  pos.session_id,
            "symbol":      pos.symbol,
            "direction":   pos.direction,
            "size":        pos.size,
            "entry_price": pos.entry_price,
            "exit_price":  exit_price,
            "notional":    pos.notional,
            "pnl":         round(pnl, 6),
            "roi_pct":     round(pos.roi_pct, 6),
            "opened_at":   pos.opened_at,
        }

    def update_unrealized(self, current_price: float):
        """Recalculate unrealized PnL from all open positions and update peak."""
        for pos in self.positions.values():
            pos.update_pnl(current_price)
        self.unrealized_pnl = round(
            sum(p.unrealized_pnl for p in self.positions.values()), 6
        )
        # Update peak equity and max drawdown
        eq = self.equity
        if eq > self.peak_equity:
            self.peak_equity = eq
        dd = self.drawdown_pct
        if dd > self.max_drawdown_pct:
            self.max_drawdown_pct = dd

    def check_new_day(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._today:
            self.daily_realized_pnl = 0.0
            self._today = today

    def reset(self, new_balance: float):
        """Reset account with new paper capital. Requires no open positions."""
        self.starting_balance   = new_balance
        self.realized_pnl       = 0.0
        self.unrealized_pnl     = 0.0
        self.peak_equity        = new_balance
        self.max_drawdown_pct   = 0.0
        self.trade_count        = 0
        self.win_count          = 0
        self.loss_count         = 0
        self.daily_realized_pnl = 0.0
        self.positions.clear()

    def to_state_dict(self, btc_price: float = 0.0) -> dict:
        return {
            "account_id":        self.id,
            "starting_balance":  self.starting_balance,
            "available_cash":    round(self.available_cash, 2),
            "equity":            round(self.equity, 2),
            "realized_pnl":      round(self.realized_pnl, 2),
            "unrealized_pnl":    round(self.unrealized_pnl, 2),
            "roi_pct":           self.roi_pct,
            "win_rate":          self.win_rate,
            "win_count":         self.win_count,
            "loss_count":        self.loss_count,
            "trade_count":       self.trade_count,
            "exposure":          round(self.exposure, 2),
            "exposure_pct":      self.exposure_pct,
            "drawdown_pct":      self.drawdown_pct,
            "max_drawdown_pct":  round(self.max_drawdown_pct, 2),
            "daily_pnl":         round(self.daily_realized_pnl, 2),
            "open_positions":    len(self.positions),
            "btc_price":         btc_price,
            "session_runtime_s": self.session_runtime_s,
        }
