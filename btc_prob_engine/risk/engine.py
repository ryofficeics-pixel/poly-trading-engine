"""
btc_prob_engine/src/risk/engine.py
====================================
Risk management layer.

Architecture borrowed from:
  • robertmartin8/PyPortfolioOpt : risk_models.py — covariance, VaR, CVaR
  • polakowo/vectorbt            : portfolio stats, sharpe, sortino, max drawdown
  • bashtage/arch                : GARCH VaR / CVaR via conditional vol
  • empyrical (quantopian)       : returns analysis, risk metrics

Responsibilities:
  1. Real-time position risk (exposure, VaR, drawdown)
  2. Pre-trade checks (max position, kill conditions)
  3. Post-trade analytics (realized risk metrics)
  4. Volatility-adjusted sizing
"""

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Risk Parameters ──────────────────────────────────────────────────────────

@dataclass
class RiskParams:
    max_exposure_pct:    float = 80.0    # % of equity max in positions
    max_single_pos_pct:  float = 20.0   # % of equity per trade (hard cap)
    max_drawdown_pct:    float = 15.0   # session kill-switch level
    daily_loss_limit_pct: float = 5.0   # daily stop-out %
    var_confidence:      float = 0.95   # VaR confidence level
    vol_scale_cap:       float = 3.0    # max vol-scaling multiplier
    min_confidence:      float = 0.03   # ✅ lowered from 0.05 → 0.03 (heuristic proxy trades at lower conviction)


# ── Risk Score ───────────────────────────────────────────────────────────────

@dataclass
class RiskScore:
    """
    Pre-trade risk assessment output.
    Pattern: PyPortfolioOpt efficient_frontier constraints + empyrical risk checks.
    """
    approved:         bool  = False
    risk_score:       float = 1.0      # 0=safe, 1=max risk
    reason:           str   = ""
    recommended_size: float = 0.0     # % of equity to risk
    var_1pct:         float = 0.0     # 1% VaR in USD
    vol_regime:       str   = "NORMAL" # LOW | NORMAL | HIGH | EXTREME
    checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "approved":         self.approved,
            "risk_score":       round(self.risk_score, 4),
            "reason":           self.reason,
            "recommended_size": round(self.recommended_size, 4),
            "var_1pct":         round(self.var_1pct, 2),
            "vol_regime":       self.vol_regime,
            "checks":           self.checks,
        }


# ── Metrics (empyrical / vectorbt patterns) ──────────────────────────────────

def sharpe_ratio(returns: List[float], risk_free: float = 0.0,
                 periods_per_year: int = 365 * 24 * 60) -> float:
    """
    Annualized Sharpe ratio.
    Pattern: empyrical.sharpe_ratio(), vectorbt Portfolio.stats()
    """
    if len(returns) < 2:
        return 0.0
    mean = statistics.mean(returns) - risk_free / periods_per_year
    std  = statistics.stdev(returns)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(periods_per_year)


def sortino_ratio(returns: List[float], risk_free: float = 0.0,
                  periods_per_year: int = 365 * 24 * 60) -> float:
    """
    Sortino ratio: penalizes only downside volatility.
    Pattern: empyrical.sortino_ratio()
    """
    if len(returns) < 2:
        return 0.0
    mean      = statistics.mean(returns) - risk_free / periods_per_year
    neg_rets  = [r for r in returns if r < 0]
    if not neg_rets:
        return float('inf')
    downside  = math.sqrt(sum(r**2 for r in neg_rets) / len(neg_rets))
    return (mean / downside * math.sqrt(periods_per_year)) if downside > 0 else 0.0


def max_drawdown(equity_curve: List[float]) -> float:
    """
    Maximum peak-to-trough drawdown %.
    Pattern: vectorbt Portfolio.max_drawdown, empyrical.max_drawdown
    """
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    mdd  = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd   = (peak - v) / peak * 100 if peak > 0 else 0
        mdd  = max(mdd, dd)
    return mdd


def calmar_ratio(returns: List[float], equity_curve: List[float]) -> float:
    """
    Calmar = annualized return / max drawdown.
    Pattern: empyrical.calmar_ratio()
    """
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    ann_ret = statistics.mean(returns) * 365 * 24 * 60 * 100 if returns else 0
    return ann_ret / mdd


def var_historical(returns: List[float], confidence: float = 0.95,
                   portfolio_value: float = 1000.0) -> float:
    """
    Historical VaR at given confidence level.
    Pattern: PyPortfolioOpt risk models, arch GARCH-based VaR.
    Returns VaR in USD (positive = loss amount).
    """
    if not returns:
        return 0.0
    sorted_rets = sorted(returns)
    idx = int((1 - confidence) * len(sorted_rets))
    return abs(sorted_rets[min(idx, len(sorted_rets)-1)]) * portfolio_value


def cvar_historical(returns: List[float], confidence: float = 0.95,
                    portfolio_value: float = 1000.0) -> float:
    """
    Conditional VaR (Expected Shortfall) — expected loss beyond VaR.
    Pattern: PyPortfolioOpt efficient_semivariance
    More robust tail risk estimate than VaR.
    """
    if not returns:
        return 0.0
    sorted_rets = sorted(returns)
    cutoff = int((1 - confidence) * len(sorted_rets))
    tail   = sorted_rets[:max(1, cutoff)]
    return abs(statistics.mean(tail)) * portfolio_value


def garch_var(returns: List[float], confidence: float = 0.95,
              portfolio_value: float = 1000.0) -> float:
    """
    GARCH(1,1) parametric VaR.
    Pattern: arch GARCH VaR computation (arch.univariate.volatility)
    Assumes normal distribution for simplicity; use t-dist for fat tails in prod.
    """
    if len(returns) < 10:
        return var_historical(returns, confidence, portfolio_value)

    # GARCH variance recursion
    omega, alpha, beta = 1e-6, 0.1, 0.85
    var_t = statistics.variance(returns)
    for r in returns[1:]:
        var_t = omega + alpha * r**2 + beta * var_t

    sigma = math.sqrt(var_t)
    # Normal quantile for confidence level
    z_map = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z     = z_map.get(confidence, 1.645)
    return sigma * z * portfolio_value


def profit_factor(pnls: List[float]) -> float:
    """Gross profit / gross loss. >1 = profitable."""
    gross_win  = sum(p for p in pnls if p > 0)
    gross_loss = sum(abs(p) for p in pnls if p < 0)
    return gross_win / gross_loss if gross_loss > 0 else float('inf')


# ── Volatility regime classifier ─────────────────────────────────────────────

def classify_vol_regime(current_vol: float, vol_history: List[float]) -> str:
    """
    Compare current GARCH vol vs rolling distribution.
    Pattern: arch volatility forecast → regime classification.
    Returns: LOW | NORMAL | HIGH | EXTREME
    
    ✅ FIX #3: Relaxed thresholds for paper trading (BTC often 100-200% vol)
    """
    if not vol_history or len(vol_history) < 10:
        return "NORMAL"
    sorted_vols = sorted(vol_history)
    n           = len(sorted_vols)
    p25  = sorted_vols[n // 4]
    p75  = sorted_vols[3 * n // 4]
    p99  = sorted_vols[int(n * 0.99)]  # Changed from p95 to p99

    if current_vol < p25:     return "LOW"
    elif current_vol < p75:   return "NORMAL"
    elif current_vol < p99:   return "HIGH"      # More lenient
    else:                     return "EXTREME"


# ── Volatility-scaled position sizing ────────────────────────────────────────

def vol_adjusted_size(base_size_pct: float, current_vol: float,
                      target_vol: float = 50.0, cap: float = 3.0) -> float:
    """
    Scale position size inversely with volatility.
    Pattern: PyPortfolioOpt volatility-targeting, arch GARCH-based sizing.

    If vol is half of target → double size (but cap at `cap`).
    If vol is double target  → halve size.
    """
    if current_vol <= 0 or target_vol <= 0:
        return base_size_pct
    scalar = min(target_vol / current_vol, cap)
    return base_size_pct * scalar


# ── Pre-trade Risk Engine ─────────────────────────────────────────────────────

class RiskEngine:
    """
    Real-time pre-trade risk checker + position sizing.

    Checks (all must pass for approval):
      1. Kill switch off
      2. Max drawdown not breached
      3. Daily loss limit not breached
      4. Exposure under limit
      5. Model confidence above threshold
      6. Vol regime not EXTREME (configurable)
    """

    def __init__(self, params: Optional[RiskParams] = None):
        self.params          = params or RiskParams()
        self._returns:       List[float] = []
        self._equity_curve:  List[float] = []
        self._vol_history:   List[float] = []
        self._pnls:          List[float] = []
        self._daily_pnl:     float = 0.0
        self._daily_reset_ts: float = time.time()
        self._kill_switch:   bool = False

    def record(self, pnl: float, equity: float, vol: float = 0.0):
        """Called after each trade settlement or heartbeat."""
        # Daily reset
        now = time.time()
        if now - self._daily_reset_ts > 86400:
            self._daily_pnl      = 0.0
            self._daily_reset_ts = now

        self._pnls.append(pnl)
        self._daily_pnl += pnl
        self._equity_curve.append(equity)

        # BUG-4 FIX: equity is float; len(float) crashes — use list length check
        if len(self._equity_curve) >= 2 and self._equity_curve[-2] > 0:
            ret = (equity - self._equity_curve[-2]) / self._equity_curve[-2]
            self._returns.append(ret)

        if vol > 0:
            self._vol_history.append(vol)

        # Prune history
        for lst in [self._pnls, self._returns, self._equity_curve, self._vol_history]:
            if len(lst) > 10000:
                lst.pop(0)

    def check(self, model_confidence: float, current_vol: float,
              equity: float, exposure: float, long_prob: float,
              kelly_fraction: float) -> RiskScore:
        """
        Full pre-trade risk assessment.
        Returns RiskScore with approval, recommended_size, and per-check results.
        """
        params = self.params
        checks = {}

        # ── Kill switch ──────────────────────────────────────────────────
        checks["kill_switch_off"] = not self._kill_switch
        if self._kill_switch:
            return RiskScore(approved=False, reason="kill switch active", checks=checks)

        # ── Drawdown check ───────────────────────────────────────────────
        mdd = max_drawdown(self._equity_curve[-500:]) if self._equity_curve else 0
        checks["drawdown_ok"] = mdd < params.max_drawdown_pct
        if mdd >= params.max_drawdown_pct:
            self._kill_switch = True
            return RiskScore(approved=False,
                             reason=f"max drawdown {mdd:.1f}% >= {params.max_drawdown_pct}%",
                             checks=checks, risk_score=1.0)

        # ── Daily loss limit ─────────────────────────────────────────────
        if equity > 0:
            daily_loss_pct = (-self._daily_pnl / equity * 100) if self._daily_pnl < 0 else 0
            checks["daily_loss_ok"] = daily_loss_pct < params.daily_loss_limit_pct
            if daily_loss_pct >= params.daily_loss_limit_pct:
                return RiskScore(approved=False,
                                 reason=f"daily loss {daily_loss_pct:.1f}% >= {params.daily_loss_limit_pct}%",
                                 checks=checks, risk_score=0.9)
        else:
            checks["daily_loss_ok"] = True

        # ── Exposure check ────────────────────────────────────────────────
        exposure_pct = exposure / equity * 100 if equity > 0 else 100
        checks["exposure_ok"] = exposure_pct < params.max_exposure_pct
        if exposure_pct >= params.max_exposure_pct:
            return RiskScore(approved=False,
                             reason=f"exposure {exposure_pct:.0f}% >= {params.max_exposure_pct}%",
                             checks=checks, risk_score=0.7)

        # ── Model confidence check ────────────────────────────────────────
        checks["confidence_ok"] = model_confidence >= params.min_confidence
        if model_confidence < params.min_confidence:
            return RiskScore(approved=False,
                             reason=f"confidence {model_confidence:.2f} < min {params.min_confidence}",
                             checks=checks, risk_score=0.5)

        # ── Volatility regime ─────────────────────────────────────────────
        vol_regime  = classify_vol_regime(current_vol, self._vol_history)
        checks["vol_regime_ok"] = vol_regime != "EXTREME"
        if vol_regime == "EXTREME":
            return RiskScore(approved=False,
                             reason="vol regime EXTREME — no new positions",
                             checks=checks, vol_regime=vol_regime, risk_score=0.8)

        # ── All checks passed ─────────────────────────────────────────────

        # VaR calculation (arch-inspired)
        ret_sample = self._returns[-100:] if self._returns else []
        var_95     = garch_var(ret_sample, 0.95, equity) if len(ret_sample) >= 10 \
                     else var_historical(ret_sample, 0.95, equity)

        # Vol-adjusted Kelly size
        base_size  = kelly_fraction * 100  # % of equity
        adj_size   = vol_adjusted_size(base_size, current_vol)
        final_size = min(adj_size, params.max_single_pos_pct)
        final_size = max(0.0, final_size)

        # Composite risk score [0=safe, 1=dangerous]
        risk_score = (
            (mdd / params.max_drawdown_pct) * 0.3 +
            (exposure_pct / params.max_exposure_pct) * 0.3 +
            ((current_vol / 200) if current_vol < 200 else 1.0) * 0.2 +
            (1 - model_confidence) * 0.2
        )

        return RiskScore(
            approved=True,
            risk_score=round(min(1.0, risk_score), 4),
            reason="all checks passed",
            recommended_size=round(final_size, 4),
            var_1pct=round(var_95 / 100, 2) if var_95 else 0.0,
            vol_regime=vol_regime,
            checks=checks,
        )

    def analytics(self, equity: float) -> Dict:
        """Full risk analytics snapshot."""
        rets    = self._returns[-1000:]
        pnls    = self._pnls[-1000:]
        eq_curve = self._equity_curve[-1000:]

        return {
            "sharpe":        round(sharpe_ratio(rets), 3) if rets else 0.0,
            "sortino":       round(sortino_ratio(rets), 3) if rets else 0.0,
            "max_drawdown":  round(max_drawdown(eq_curve), 3) if eq_curve else 0.0,
            "calmar":        round(calmar_ratio(rets, eq_curve), 3) if (rets and eq_curve) else 0.0,
            "var_95":        round(var_historical(rets, 0.95, equity), 2) if rets else 0.0,
            "cvar_95":       round(cvar_historical(rets, 0.95, equity), 2) if rets else 0.0,
            "garch_var_95":  round(garch_var(rets, 0.95, equity), 2) if len(rets) >= 10 else 0.0,
            "profit_factor": round(profit_factor(pnls), 3) if pnls else 0.0,
            "daily_pnl":     round(self._daily_pnl, 2),
            "kill_switch":   self._kill_switch,
        }

    def trigger_kill(self):
        self._kill_switch = True

    def reset_kill(self):
        self._kill_switch = False
