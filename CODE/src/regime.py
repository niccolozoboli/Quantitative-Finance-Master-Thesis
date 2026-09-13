"""
regime.py
---------
Market regime classification for metal commodity series.

Regimes defined by realized volatility (rolling 21-day std of log-returns):
  - STABLE:   rolling_vol <= 33rd percentile
  - NORMAL:   33rd–66th percentile
  - VOLATILE: rolling_vol > 66th percentile

Percentile thresholds are estimated causally via an expanding quantile
(only observations up to and including day t), not on the full series —
using future observations to set the threshold for day t would leak
information not available at classification time. A minimum warm-up of
REGIME_WARMUP observations is required before any day is classified;
before that, the day defaults to "normal".
"""

import numpy as np
import pandas as pd


REGIME_WINDOW = 21
REGIME_WARMUP = 63  # ~3 trading months, 3x REGIME_WINDOW


def compute_rolling_vol(df: pd.DataFrame,
                         window: int = REGIME_WINDOW) -> pd.Series:
    return df["log_return"].rolling(window).std()


def classify_regimes(df: pd.DataFrame,
                      window: int = REGIME_WINDOW,
                      warmup: int = REGIME_WARMUP) -> pd.Series:
    rolling_vol = compute_rolling_vol(df, window)
    p33 = rolling_vol.expanding(min_periods=warmup).quantile(0.33)
    p66 = rolling_vol.expanding(min_periods=warmup).quantile(0.66)

    regimes = pd.Series("normal", index=df.index)
    valid = p33.notna()
    regimes[valid & (rolling_vol <= p33)] = "stable"
    regimes[valid & (rolling_vol >  p66)] = "volatile"
    return regimes


def split_by_regime(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     dates: pd.DatetimeIndex,
                     regimes: pd.Series) -> dict:
    """
    Splits predictions into regime buckets.
    Aligns all inputs to shortest length — safe against any off-by-one.
    """
    n = min(len(y_true), len(y_pred), len(dates))
    y_true = np.array(y_true)[-n:]
    y_pred = np.array(y_pred)[-n:]
    dates  = dates[-n:]

    result = {}
    for regime in ["stable", "normal", "volatile"]:
        mask = regimes.reindex(dates).values == regime
        if mask.sum() > 0:
            result[regime] = (y_true[mask], y_pred[mask])
    return result


def regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    regimes   = classify_regimes(df)
    counts    = regimes.value_counts()
    rv        = compute_rolling_vol(df)
    vol_stats = {}

    for r in ["stable", "normal", "volatile"]:
        mask = regimes == r
        vol_stats[r] = {
            "count":    int(counts.get(r, 0)),
            "pct":      round(counts.get(r, 0) / len(regimes) * 100, 1),
            "mean_vol": round(float(rv[mask].mean()), 6),
            "max_vol":  round(float(rv[mask].max()),  6),
        }

    return pd.DataFrame(vol_stats).T