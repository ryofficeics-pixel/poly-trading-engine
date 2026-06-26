"""
btc_prob_engine
================
BTC direction-probability engine.

Pipeline:
  Data Layer    (data.feed)        → DataRing
  Feature Layer (features.engineer) → FeatureDict
  Model Layer   (models.probability)→ ProbabilityOutput
  Risk Layer    (risk.engine)       → RiskScore
  Backtest      (backtest.engine)   → Stats
"""
from .engine import BTCProbEngine

__all__ = ["BTCProbEngine"]
