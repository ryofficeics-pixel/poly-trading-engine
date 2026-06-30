"""
btc_prob_engine/src/models/probability.py
==========================================
BTC direction probability models.

Architecture borrowed from:
  • dmlc/xgboost        : XGBClassifier sklearn API, predict_proba(), feature_importances_
  • microsoft/LightGBM  : LGBMClassifier, LGBMModel._process_params(), early stopping
  • pycaret/pycaret      : AutoML pipeline pattern, model comparison, calibration
  • vectorbt/vectorbt    : walk-forward validation, monte carlo simulation
  • PyPortfolioOpt       : Kelly sizing from probability estimates

Outputs:
  ProbabilityOutput.long_prob  ∈ [0, 1]
  ProbabilityOutput.short_prob ∈ [0, 1]
  ProbabilityOutput.confidence ∈ [0, 1]
  ProbabilityOutput.signal     ∈ {'LONG', 'SHORT', 'FLAT'}
"""

import json
import math
import os
import pickle
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ── Output contract ────────────────────────────────────────────────────────

@dataclass
class ProbabilityOutput:
    long_prob:    float = 0.5
    short_prob:   float = 0.5
    confidence:   float = 0.0     # |long_prob - 0.5| * 2 → [0, 1]
    edge:         float = 0.0     # expected value estimate
    signal:       str   = "FLAT"  # LONG | SHORT | FLAT
    model_name:   str   = ""
    feature_snap: Dict  = field(default_factory=dict)
    ts:           float = 0.0

    def to_dict(self) -> dict:
        return {
            "long_prob":  round(self.long_prob, 4),
            "short_prob": round(self.short_prob, 4),
            "confidence": round(self.confidence, 4),
            "edge":       round(self.edge, 4),
            "signal":     self.signal,
            "model":      self.model_name,
            "ts":         self.ts,
        }


def _compute_confidence(prob: float) -> float:
    """Distance from 0.5, normalized to [0, 1]."""
    return abs(prob - 0.5) * 2


def _signal(prob: float, threshold: float = 0.60) -> str:
    if prob >= threshold:
        return "LONG"
    elif prob <= (1 - threshold):
        return "SHORT"
    return "FLAT"


# ── Feature matrix builder ───────────────────────────────────────────────────

class FeatureMatrix:
    """
    Rolling feature store — keeps last N observations.
    Pattern: vectorbt walk-forward split + PyCaret feature pipeline.
    Stores feature dicts in order; builds numpy arrays for model input.
    """

    def __init__(self, maxlen: int = 10000, label_horizon: int = 5):
        """
        label_horizon: how many future candles to look ahead for labeling.
        For live inference: horizon=0 (no label needed).
        """
        self.maxlen        = maxlen
        self.label_horizon = label_horizon
        self._rows: List[Dict] = []
        self._labels: List[int] = []   # 1=UP, 0=DOWN
        self._col_order: List[str] = []

    def push(self, features: Dict[str, float], label: Optional[int] = None):
        self._rows.append(features)
        if label is not None:
            self._labels.append(label)
        if len(self._rows) > self.maxlen:
            self._rows.pop(0)
            if self._labels:
                self._labels.pop(0)

    def _sync_columns(self, rows: List[Dict]) -> List[str]:
        """Consistent column ordering — critical for XGBoost/LGBM."""
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        # Exclude metadata cols
        exclude = {"extract_ts", "price", "n_candles_1m", "n_candles_5m", "n_candles_1h"}
        return sorted(all_keys - exclude)

    def to_numpy(self, last_n: Optional[int] = None
                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns (X, y, col_names).
        X shape: (n_samples, n_features)
        y shape: (n_samples,)
        Handles missing values via 0.0 fill (tree-model safe, XGBoost handles nan natively).
        """
        rows = self._rows[-last_n:] if last_n else self._rows
        if len(rows) < 2:
            return np.zeros((0, 1)), np.zeros(0), []

        cols  = self._sync_columns(rows)
        self._col_order = cols

        n = min(len(rows), len(self._labels)) if self._labels else 0
        if n == 0:
            return np.zeros((0, len(cols))), np.zeros(0), cols

        X = np.array(
            [[r.get(c, 0.0) for c in cols] for r in rows[:n]],
            dtype=np.float32
        )
        y = np.array(self._labels[:n], dtype=np.int32)

        # NaN guard (tree models handle NaN, but clean is safer)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        return X, y, cols

    def latest_x(self) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Single-row inference vector."""
        if not self._rows or not self._col_order:
            return None
        row  = self._rows[-1]
        cols = self._col_order
        x    = np.array([[row.get(c, 0.0) for c in cols]], dtype=np.float32)
        x    = np.nan_to_num(x, nan=0.0)
        return x, cols

    def add_price_labels(self, prices: List[float], horizon: int = 5,
                         threshold_pct: float = 0.1):
        """
        Retrospective labeling: 1 if price rises >threshold% in `horizon` bars.
        Applied offline during training. Pattern from vectorbt signal generation.
        """
        self._labels = []
        for i in range(len(prices) - horizon):
            ret = (prices[i + horizon] - prices[i]) / prices[i] * 100
            self._labels.append(1 if ret > threshold_pct else 0)


# ── Gradient boosting models (XGBoost / LightGBM-inspired) ──────────────────
# Note: pure-numpy implementation for zero-dependency inference.
# Wire in actual xgboost/lightgbm when training offline.

class GradientBoostProxy:
    """
    Pure-Python gradient boosting inference proxy.
    Architecture: XGBClassifier / LGBMClassifier sklearn API.
    For production training, serialize trained XGB/LGBM model and load here.

    Implements predict_proba() compatible interface.
    Falls back to weighted feature ensemble if no trained model is loaded.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model       = None
        self._feature_imp: Dict[str, float] = {}
        self._col_order:   List[str] = []
        self._is_loaded    = False
        self._model_type   = "proxy"

        if model_path and os.path.exists(model_path):
            self._load(model_path)

    def _load(self, path: str):
        """Load pickled XGBoost / LightGBM model."""
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)
            self._model      = payload.get('model')
            self._col_order  = payload.get('columns', [])
            self._feature_imp = payload.get('feature_importance', {})
            self._model_type = payload.get('model_type', 'xgb')
            self._is_loaded  = True
            logger.info(f"model loaded: {path}  type={self._model_type}  "
                        f"features={len(self._col_order)}")
        except Exception as e:
            logger.error(f"model load failed: {e}")
            self._is_loaded = False

    def predict_proba(self, X: np.ndarray, cols: List[str]) -> float:
        """
        Returns long probability ∈ [0, 1].
        XGBoost / LightGBM pattern: predict_proba(X)[:, 1]
        """
        if self._is_loaded and self._model is not None:
            try:
                # Reorder columns to match training
                if self._col_order:
                    col_map = {c: i for i, c in enumerate(cols)}
                    X_ordered = np.zeros((X.shape[0], len(self._col_order)), dtype=np.float32)
                    for j, c in enumerate(self._col_order):
                        if c in col_map:
                            X_ordered[:, j] = X[:, col_map[c]]
                    X = X_ordered

                proba = self._model.predict_proba(X)
                return float(proba[0, 1])
            except Exception as e:
                logger.debug(f"model predict error: {e}, falling back to proxy")

        # ── Proxy ensemble (no trained model) ─────────────────────────────
        return self._proxy_predict(X, cols)

    def _proxy_predict(self, X: np.ndarray, cols: List[str]) -> float:
        """
        Multi-factor heuristic model using RSI + Bollinger Bands + trend.
        Designed to generate signals in normal market conditions (not just extremes).
        ✅ Lowered thresholds: fires on RSI 40/60 range instead of 30/70 only.
        """
        if X.shape[1] == 0:
            return 0.5

        feat = {cols[i]: float(X[0, i]) for i in range(len(cols))}

        rsi_1m  = feat.get("rsi14_1m", 0.5)
        rsi_5m  = feat.get("rsi14_5m", 0.5)
        rsi_1h  = feat.get("rsi14_1h", 0.5)
        bb_pct  = feat.get("bb_pct_b_1m", 0.5)
        regime  = feat.get("regime_trend_score", 0.5)

        long_score  = 0.0
        short_score = 0.0

        # RSI signals — fire on 40/60 range (normal conditions) not just 30/70
        # 1m RSI (weight 3.0)
        if rsi_1m < 0.40:    long_score  += 3.0 * (0.40 - rsi_1m) / 0.40
        elif rsi_1m > 0.60:  short_score += 3.0 * (rsi_1m - 0.60) / 0.40

        # 5m RSI (weight 2.5)
        if rsi_5m < 0.42:    long_score  += 2.5 * (0.42 - rsi_5m) / 0.42
        elif rsi_5m > 0.58:  short_score += 2.5 * (rsi_5m - 0.58) / 0.42

        # 1h RSI (weight 2.0)
        if rsi_1h < 0.45:    long_score  += 2.0 * (0.45 - rsi_1h) / 0.45
        elif rsi_1h > 0.55:  short_score += 2.0 * (rsi_1h - 0.55) / 0.45

        # Bollinger Band %B (weight 2.0 — mean reversion)
        if bb_pct < 0.35:    long_score  += 2.0 * (0.35 - bb_pct) / 0.35
        elif bb_pct > 0.65:  short_score += 2.0 * (bb_pct - 0.65) / 0.35

        # Regime / trend (weight 1.5)
        if regime > 0.55:    long_score  += 1.5 * (regime - 0.55) / 0.45
        elif regime < 0.45:  short_score += 1.5 * (0.45 - regime) / 0.45

        total = long_score + short_score
        if total < 0.01:
            return 0.5  # Genuinely neutral, no signals

        long_prob = long_score / total

        # Amplify signal strength away from 0.5
        bias = long_prob - 0.5
        long_prob = 0.5 + bias * 1.5

        # Clamp to [0.25, 0.75] — heuristic shouldn't be too extreme
        return max(0.25, min(0.75, long_prob))

    def feature_importance(self) -> Dict[str, float]:
        if self._feature_imp:
            return self._feature_imp
        # Default proxy importances (from domain knowledge)
        return {
            "order_book_imbalance":  0.12,
            "buy_sell_ratio_200":    0.11,
            "rsi14_1h":              0.10,
            "rsi14_5m":              0.09,
            "regime_trend_score":    0.08,
            "tf_alignment":          0.07,
            "liq_net_pressure":      0.07,
            "garch_vol_1m":          0.06,
            "ema_cross_1m":          0.06,
            "bb_pct_b_5m":           0.05,
            "hurst_1m":              0.05,
            "atr_pct_1m":            0.04,
            "har_rv_1d":             0.04,
            "vp_above_poc":          0.03,
            "regime_hurst":          0.03,
        }


# ── Calibration (Platt scaling pattern from PyCaret) ─────────────────────────

class IsotonicCalibrator:
    """
    Post-hoc probability calibration.
    In production: use sklearn.calibration.CalibratedClassifierCV.
    Here: linear interpolation from observed frequency tables.
    Pattern: PyCaret calibrate_model() → isotonic regression.
    """

    def __init__(self):
        self._bins   = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0
        self._cal    = list(self._bins)  # identity mapping initially
        self._fitted = False

    def fit(self, raw_probs: List[float], actuals: List[int]):
        """
        Fit calibration mapping from raw model output → calibrated probability.
        Uses observed frequency in each bin.
        """
        bin_counts = [0] * 10
        bin_pos    = [0] * 10
        for p, a in zip(raw_probs, actuals):
            b = min(int(p * 10), 9)
            bin_counts[b] += 1
            bin_pos[b]    += a

        cal = []
        for i in range(10):
            if bin_counts[i] > 0:
                cal.append(bin_pos[i] / bin_counts[i])
            else:
                cal.append((i + 0.5) / 10)  # prior

        self._cal    = [0.0] + cal + [1.0]
        self._fitted = True
        logger.info(f"calibrator fitted on {len(raw_probs)} samples")

    def calibrate(self, raw_prob: float) -> float:
        """Linear interpolation in calibration table."""
        if not self._fitted:
            return raw_prob
        p  = max(0.0, min(1.0, raw_prob))
        idx = p * 10
        lo  = int(idx)
        hi  = min(lo + 1, 10)
        frac = idx - lo
        return self._cal[lo] * (1 - frac) + self._cal[hi] * frac


# ── Kelly sizing (PyPortfolioOpt pattern) ────────────────────────────────────

def kelly_fraction(prob: float, win_mult: float = 1.0, loss_mult: float = 1.0,
                   max_fraction: float = 0.25) -> float:
    """
    Kelly criterion for position sizing.
    Pattern: PyPortfolioOpt risk-based allocation.

    f* = (p * b - q) / b
    where p=win_prob, q=loss_prob, b=win/loss ratio

    Kelly is aggressive; we cap at max_fraction (quarter-Kelly is common).
    """
    q = 1 - prob
    b = win_mult / loss_mult if loss_mult > 0 else 1.0
    if b <= 0:
        return 0.0
    k = (prob * b - q) / b
    k = max(0.0, k)        # no negative sizing
    return min(k, max_fraction)


# ── Ensemble Probability Engine ───────────────────────────────────────────────

class BTCProbabilityEngine:
    """
    Top-level engine: features → probability → signal → position size.

    Models:
      1. GradientBoostProxy (XGBoost / LGBM pattern)
      2. Rule-based regime filter (reduces false signals in choppy markets)
      3. Microstructure confirmer (order flow / liquidation)

    Ensemble: weighted average with confidence gating.
    """

    def __init__(self, model_path: Optional[str] = None,
                 signal_threshold: float = 0.60,
                 max_kelly: float = 0.20):
        self.model      = GradientBoostProxy(model_path)
        self.calibrator = IsotonicCalibrator()
        self.matrix     = FeatureMatrix(maxlen=5000, label_horizon=5)

        self.signal_threshold = signal_threshold
        self.max_kelly        = max_kelly

        self._history: List[ProbabilityOutput] = []

    def predict(self, features: Dict[str, float]) -> ProbabilityOutput:
        """
        Main inference path. Call after each feature extraction.
        Returns full ProbabilityOutput including signal and Kelly size.
        """
        self.matrix.push(features)

        # ── No trained model: run heuristic directly on feature dict ─────
        # latest_x() returns None until _col_order is populated by to_numpy()
        # which requires labels (offline training). Bypass it when no model loaded.
        if not self.model._is_loaded:
            cols = sorted(features.keys())
            X = np.array([[features.get(c, 0.0) for c in cols]], dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0)
            raw_prob = self.model._proxy_predict(X, cols)
            cal_prob = self.calibrator.calibrate(raw_prob)
        else:
            latest = self.matrix.latest_x()
            if latest is None:
                return ProbabilityOutput(ts=time.time())
            X, cols = latest
            raw_prob = self.model.predict_proba(X, cols)
            cal_prob = self.calibrator.calibrate(raw_prob)

        # ── Microstructure override ───────────────────────────────────────
        # Strong imbalance + liquidation = amplify signal
        micro_adj = 0.0
        if "order_book_imbalance" in features:
            imbal = features["order_book_imbalance"]
            if imbal > 0.65:   micro_adj += 0.03
            elif imbal < 0.35: micro_adj -= 0.03
        if "liq_net_pressure" in features:
            liq = features["liq_net_pressure"]
            micro_adj += liq * 0.02  # liq ∈ [-1,1]

        # ── Regime gate ───────────────────────────────────────────────────
        # In high-vol / low-hurst regime, compress toward 0.5 (reduce conviction)
        # ✅ FIX: Relaxed gates - hurst 0.5 default was crushing all signals
        regime_mult = 1.0
        if "garch_vol_1m" in features:
            if features["garch_vol_1m"] > 300:  # raised from 200 → 300
                regime_mult = 0.85  # reduced penalty: was 0.7
        if "hurst_1m" in features:
            h = features["hurst_1m"]
            # Only apply near-random-walk penalty when hurst is explicitly computed
            # (non-default value). Default 0.5 = not computed, not a signal.
            if h != 0.5 and 0.45 <= h <= 0.55:
                regime_mult *= 0.9  # reduced penalty: was 0.8

        # ── Final probability ─────────────────────────────────────────────
        final_prob = cal_prob + micro_adj
        final_prob = 0.5 + (final_prob - 0.5) * regime_mult
        final_prob = max(0.01, min(0.99, final_prob))

        short_prob  = 1 - final_prob
        confidence  = _compute_confidence(final_prob)
        signal      = _signal(final_prob, self.signal_threshold)

        # ── Kelly edge estimate ───────────────────────────────────────────
        kelly = kelly_fraction(final_prob, max_fraction=self.max_kelly)
        # Edge = expected value: p*1 - (1-p)*1 = 2p - 1
        edge  = round(2 * final_prob - 1, 4)

        out = ProbabilityOutput(
            long_prob=round(final_prob, 4),
            short_prob=round(short_prob, 4),
            confidence=round(confidence, 4),
            edge=round(edge, 4),
            signal=signal,
            model_name=self.model._model_type,
            feature_snap={
                "rsi14_1h":              features.get("rsi14_1h", 0),
                "order_book_imbalance":  features.get("order_book_imbalance", 0.5),
                "buy_sell_ratio_200":    features.get("buy_sell_ratio_200", 0.5),
                "regime_trend_score":    features.get("regime_trend_score", 0.5),
                "garch_vol_1m":          features.get("garch_vol_1m", 0),
                "hurst_1m":              features.get("hurst_1m", 0.5),
                "tf_alignment":          features.get("tf_alignment", 0),
                "liq_net_pressure":      features.get("liq_net_pressure", 0),
                "kelly_fraction":        round(kelly, 4),
            },
            ts=time.time(),
        )

        self._history.append(out)
        if len(self._history) > 1000:
            self._history.pop(0)

        return out

    def recent_outputs(self, n: int = 50) -> List[Dict]:
        return [o.to_dict() for o in self._history[-n:]]

    def feature_importance(self) -> Dict[str, float]:
        return self.model.feature_importance()


# ── Walk-forward validator (vectorbt pattern) ─────────────────────────────────

def walk_forward_split(n: int, n_splits: int = 5, train_pct: float = 0.7
                       ) -> List[Tuple[List[int], List[int]]]:
    """
    Time-series walk-forward splits. Pattern from vectorbt backtesting.
    NO random shuffle — respects temporal order.
    Returns list of (train_indices, test_indices) tuples.
    """
    window = n // n_splits
    splits = []
    for i in range(n_splits):
        start  = i * window
        train_end = start + int(window * train_pct)
        test_end  = start + window
        if test_end > n:
            test_end = n
        train = list(range(start, train_end))
        test  = list(range(train_end, test_end))
        if train and test:
            splits.append((train, test))
    return splits
