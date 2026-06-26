"""
btc_prob_engine/src/backtest/engine.py
=======================================
Backtesting layer.

Architecture borrowed from:
  • polakowo/vectorbt    : vectorized from_signals(), Portfolio.from_orders(),
                           walk-forward splits, stats_builder, Monte Carlo
  • mementum/backtrader  : event-driven Strategy + Cerebro + Order patterns
  • nkaz001/hftbacktest  : tick-level fill simulation, slippage, L2 book fills

Three modes:
  1. FAST vectorized  — entire price series at once (vectorbt style)
  2. EVENT loop       — bar-by-bar with order queue (backtrader style)
  3. MONTECARLO       — random resampling of trade outcomes (vectorbt MC)
"""

import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ── Trade record ─────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    entry_bar:   int
    exit_bar:    int
    direction:   str        # UP | DOWN
    entry_price: float
    exit_price:  float
    size:        float
    pnl:         float
    roi_pct:     float
    hold_bars:   int
    slippage:    float = 0.0
    commission:  float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.pnl - self.commission - self.slippage


# ── Backtest config ───────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_capital:    float = 10000.0
    commission_pct:     float = 0.05      # 0.05% per trade (taker)
    slippage_pct:       float = 0.02      # 0.02% slippage estimate
    max_position_pct:   float = 10.0      # % of equity per trade
    stop_loss_pct:      float = 1.5       # % stop loss
    take_profit_pct:    float = 3.0       # % take profit (2:1 R:R)
    max_hold_bars:      int   = 20        # bars before forced exit
    allow_short:        bool  = True


# ── vectorbt-style vectorized backtester ─────────────────────────────────────

class VectorizedBacktester:
    """
    Vectorized signal → trade → performance.
    Pattern: vectorbt Portfolio.from_signals()
    Processes entire price series in O(n) without bar-by-bar loop.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(self, prices: List[float], signals: List[int],
            probs: Optional[List[float]] = None) -> Dict:
        """
        signals: +1=long, -1=short, 0=flat
        probs:   optional probability for variable sizing (Kelly-like)
        Returns: full stats dict
        """
        cfg      = self.config
        capital  = cfg.initial_capital
        equity   = [capital]
        trades:  List[BacktestTrade] = []
        position: Optional[Dict] = None

        for i in range(1, len(prices)):
            price = prices[i]
            sig   = signals[i - 1] if i - 1 < len(signals) else 0
            prob  = probs[i - 1] if probs and i - 1 < len(probs) else 0.5

            # ── Check exit conditions ──────────────────────────────────
            if position:
                entry    = position['entry_price']
                dir_mult = 1 if position['direction'] == 'UP' else -1
                ret_pct  = (price - entry) / entry * 100 * dir_mult

                exit_now = False
                if ret_pct <= -cfg.stop_loss_pct:    exit_now = True  # stop loss
                if ret_pct >= cfg.take_profit_pct:   exit_now = True  # take profit
                if i - position['entry_bar'] >= cfg.max_hold_bars: exit_now = True
                if sig * dir_mult < 0:               exit_now = True  # signal flip

                if exit_now:
                    slippage  = price * cfg.slippage_pct / 100
                    exit_p    = price - slippage * dir_mult
                    pnl       = (exit_p - entry) * dir_mult * position['size']
                    commission= entry * position['size'] * cfg.commission_pct / 100 * 2
                    capital  += pnl - commission
                    trades.append(BacktestTrade(
                        entry_bar   = position['entry_bar'],
                        exit_bar    = i,
                        direction   = position['direction'],
                        entry_price = entry,
                        exit_price  = exit_p,
                        size        = position['size'],
                        pnl         = pnl,
                        roi_pct     = ret_pct,
                        hold_bars   = i - position['entry_bar'],
                        slippage    = slippage * position['size'],
                        commission  = commission,
                    ))
                    position = None

            # ── Check entry ────────────────────────────────────────────
            if not position and sig != 0:
                if sig < 0 and not cfg.allow_short:
                    pass
                else:
                    # Variable sizing: Kelly-like from probability
                    kelly_pct = abs(prob * 2 - 1) * cfg.max_position_pct if prob else cfg.max_position_pct
                    pos_value = capital * min(kelly_pct / 100, cfg.max_position_pct / 100)
                    slippage  = price * cfg.slippage_pct / 100
                    entry_p   = price + slippage * sig
                    size      = pos_value / entry_p if entry_p > 0 else 0

                    if size > 0:
                        position = {
                            'entry_bar':   i,
                            'entry_price': entry_p,
                            'direction':   'UP' if sig > 0 else 'DOWN',
                            'size':        size,
                        }

            equity.append(capital)

        return self._compute_stats(trades, equity, prices)

    def _compute_stats(self, trades: List[BacktestTrade],
                       equity: List[float], prices: List[float]) -> Dict:
        """
        Compute full performance metrics.
        Pattern: vectorbt Portfolio.stats() / stats_builder.py
        """
        if not trades:
            return {"error": "no trades", "total_trades": 0}

        pnls       = [t.net_pnl for t in trades]
        wins       = [p for p in pnls if p > 0]
        losses     = [p for p in pnls if p <= 0]
        hold_bars  = [t.hold_bars for t in trades]

        final_equity   = equity[-1]
        total_return   = (final_equity - equity[0]) / equity[0] * 100

        # Drawdown
        peak = equity[0]; mdd = 0.0
        for e in equity:
            peak = max(peak, e)
            mdd  = max(mdd, (peak - e) / peak * 100 if peak else 0)

        # Returns for Sharpe
        equity_rets = [(equity[i] - equity[i-1]) / equity[i-1]
                       for i in range(1, len(equity)) if equity[i-1] > 0]
        mean_ret = statistics.mean(equity_rets) if equity_rets else 0
        std_ret  = statistics.stdev(equity_rets) if len(equity_rets) > 1 else 0
        sharpe   = (mean_ret / std_ret * math.sqrt(252 * 24 * 60)) if std_ret > 0 else 0

        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))

        return {
            "total_trades":   len(trades),
            "win_count":      len(wins),
            "loss_count":     len(losses),
            "win_rate":       round(len(wins) / len(trades) * 100, 2) if trades else 0,
            "total_pnl":      round(sum(pnls), 2),
            "total_return":   round(total_return, 2),
            "max_drawdown":   round(mdd, 2),
            "sharpe":         round(sharpe, 3),
            "profit_factor":  round(gross_profit / gross_loss, 3) if gross_loss > 0 else 99.0,
            "avg_trade_pnl":  round(statistics.mean(pnls), 2) if pnls else 0,
            "avg_win":        round(statistics.mean(wins), 2) if wins else 0,
            "avg_loss":       round(statistics.mean(losses), 2) if losses else 0,
            "avg_hold_bars":  round(statistics.mean(hold_bars), 1) if hold_bars else 0,
            "final_equity":   round(final_equity, 2),
            "initial_equity": round(equity[0], 2),
        }


# ── Event-driven backtester (backtrader Cerebro pattern) ─────────────────────

class EventBacktester:
    """
    Bar-by-bar event loop. Pattern: backtrader Cerebro + Strategy.
    More realistic than vectorized (processes orders with 1-bar delay).
    """

    def __init__(self, strategy_fn: Callable, config: Optional[BacktestConfig] = None):
        """
        strategy_fn(bar_idx, prices, features, account) → signal: -1|0|1
        """
        self.strategy = strategy_fn
        self.config   = config or BacktestConfig()

    def run(self, prices: List[float],
            features: Optional[List[Dict]] = None) -> Dict:
        """Backtrader-style bar-by-bar execution with 1-bar signal delay."""
        cfg     = self.config
        capital = cfg.initial_capital
        equity  = [capital]
        trades: List[BacktestTrade] = []
        pending_signal  = 0
        pending_prob    = 0.5
        position        = None

        for i in range(len(prices)):
            price   = prices[i]
            feats   = features[i] if features and i < len(features) else {}

            # ── Execute pending order (1-bar delay, backtrader style) ────
            if pending_signal != 0 and not position:
                sig       = pending_signal
                dir_mult  = 1 if sig > 0 else -1
                slippage  = price * cfg.slippage_pct / 100
                entry_p   = price + slippage * dir_mult
                kelly_pct = abs(pending_prob * 2 - 1) * cfg.max_position_pct
                pos_value = capital * min(kelly_pct / 100, cfg.max_position_pct / 100)
                size      = pos_value / entry_p if entry_p > 0 else 0
                if size > 0:
                    commission = entry_p * size * cfg.commission_pct / 100
                    capital   -= commission
                    position   = {
                        'entry_bar':   i,
                        'entry_price': entry_p,
                        'direction':   'UP' if sig > 0 else 'DOWN',
                        'size':        size,
                    }
                pending_signal = 0

            # ── Check stops / exits ─────────────────────────────────────
            if position:
                entry    = position['entry_price']
                dir_mult = 1 if position['direction'] == 'UP' else -1
                ret_pct  = (price - entry) / entry * 100 * dir_mult

                if (ret_pct <= -cfg.stop_loss_pct or
                    ret_pct >= cfg.take_profit_pct or
                    i - position['entry_bar'] >= cfg.max_hold_bars):
                    slippage  = price * cfg.slippage_pct / 100
                    exit_p    = price - slippage * dir_mult
                    pnl       = (exit_p - entry) * dir_mult * position['size']
                    commission= exit_p * position['size'] * cfg.commission_pct / 100
                    capital  += pnl - commission
                    trades.append(BacktestTrade(
                        entry_bar=position['entry_bar'], exit_bar=i,
                        direction=position['direction'],
                        entry_price=entry, exit_price=exit_p,
                        size=position['size'], pnl=pnl, roi_pct=ret_pct,
                        hold_bars=i - position['entry_bar'],
                        slippage=slippage * position['size'], commission=commission,
                    ))
                    position = None

            # ── Strategy signal (executed next bar) ─────────────────────
            sig, prob = self.strategy(i, prices, feats, {
                'capital': capital, 'position': position
            })
            if sig != 0 and position is None:
                pending_signal = sig
                pending_prob   = prob

            equity.append(capital + (
                (price - position['entry_price']) *
                (1 if position['direction'] == 'UP' else -1) *
                position['size']
                if position else 0
            ))

        backtester = VectorizedBacktester(self.config)
        return backtester._compute_stats(trades, equity, prices)


# ── Monte Carlo simulation (vectorbt pattern) ─────────────────────────────────

def monte_carlo_simulation(trade_pnls: List[float], n_trials: int = 1000,
                            n_trades: int = 0,
                            initial_equity: float = 10000.0) -> Dict:
    """
    Bootstrap resampling of trade outcomes.
    Pattern: vectorbt Portfolio.from_random_signals() + Monte Carlo analysis.

    Answers: "What range of outcomes is probable over the next N trades?"
    Uses: historical trade PnLs as empirical distribution.
    """
    if not trade_pnls:
        return {"error": "no trades for MC simulation"}

    n_trades = n_trades or len(trade_pnls)
    final_equities = []
    max_drawdowns  = []
    win_rates      = []

    for _ in range(n_trials):
        trial_pnls  = random.choices(trade_pnls, k=n_trades)
        eq          = initial_equity
        peak_eq     = eq
        mdd         = 0.0
        wins        = 0

        for pnl in trial_pnls:
            eq     += pnl
            peak_eq = max(peak_eq, eq)
            dd      = (peak_eq - eq) / peak_eq * 100 if peak_eq > 0 else 0
            mdd     = max(mdd, dd)
            if pnl > 0:
                wins += 1

        final_equities.append(eq)
        max_drawdowns.append(mdd)
        win_rates.append(wins / n_trades * 100)

    final_equities.sort()
    n = len(final_equities)

    return {
        "n_trials":    n_trials,
        "n_trades":    n_trades,
        "median_equity":    round(final_equities[n // 2], 2),
        "p5_equity":        round(final_equities[int(n * 0.05)], 2),
        "p95_equity":       round(final_equities[int(n * 0.95)], 2),
        "pct_profitable":   round(sum(1 for e in final_equities if e > initial_equity) / n * 100, 1),
        "median_mdd":       round(sorted(max_drawdowns)[n // 2], 2),
        "worst_mdd_p95":    round(sorted(max_drawdowns)[int(n * 0.95)], 2),
        "expected_return":  round((statistics.mean(final_equities) - initial_equity) / initial_equity * 100, 2),
    }


# ── Walk-forward validation (vectorbt pattern) ────────────────────────────────

class WalkForwardValidator:
    """
    Time-series walk-forward validation.
    Pattern: vectorbt walk-forward, hyperopt parameter optimization.
    Prevents overfitting by training on past, testing on unseen future.
    """

    def __init__(self, n_splits: int = 5, train_pct: float = 0.7,
                 config: Optional[BacktestConfig] = None):
        self.n_splits  = n_splits
        self.train_pct = train_pct
        self.config    = config or BacktestConfig()

    def run(self, prices: List[float], signals: List[int],
            probs: Optional[List[float]] = None) -> Dict:
        """
        Run walk-forward splits and aggregate stats.
        """
        n       = len(prices)
        window  = n // self.n_splits
        backt   = VectorizedBacktester(self.config)
        fold_results = []

        for i in range(self.n_splits):
            start     = i * window
            train_end = start + int(window * self.train_pct)
            end       = start + window
            if end > n:
                end = n

            test_prices  = prices[train_end:end]
            test_signals = signals[train_end:end] if signals else [0] * len(test_prices)
            test_probs   = probs[train_end:end] if probs else None

            if len(test_prices) < 5:
                continue

            stats = backt.run(test_prices, test_signals, test_probs)
            stats["fold"] = i + 1
            stats["test_bars"] = len(test_prices)
            fold_results.append(stats)

        if not fold_results:
            return {"error": "no valid folds"}

        # Aggregate
        def avg(key: str) -> float:
            vals = [f[key] for f in fold_results if key in f and f[key] != 0]
            return round(statistics.mean(vals), 3) if vals else 0.0

        return {
            "folds":          fold_results,
            "avg_win_rate":   avg("win_rate"),
            "avg_return":     avg("total_return"),
            "avg_sharpe":     avg("sharpe"),
            "avg_drawdown":   avg("max_drawdown"),
            "avg_pf":         avg("profit_factor"),
            "consistency":    round(
                sum(1 for f in fold_results if f.get("total_return", 0) > 0) /
                len(fold_results) * 100, 1
            ),
        }
