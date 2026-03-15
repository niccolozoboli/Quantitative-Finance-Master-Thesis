"""
statistical_tests.py
--------------------
Statistical validation tools for forecasting comparisons.

1. Random Walk (Naive) Baseline
2. Diebold-Mariano Test (Diebold & Mariano, 1995)
"""

import numpy as np
import pandas as pd
from scipy import stats


def _align(y_true, y_pred):
    """Trim both arrays to the same length. Always safe."""
    n = min(len(y_true), len(y_pred))
    return np.array(y_true)[-n:], np.array(y_pred)[-n:]


# ── 1. Random Walk Baseline ───────────────────────────────────────────────────

def random_walk_forecast(y_true: np.ndarray) -> np.ndarray:
    """
    Random Walk (Naive) forecast: predicts zero log-return at every step.
    Under the EMH, the best short-horizon forecast of log-returns is zero.
    This is the canonical baseline in financial forecasting.
    """
    return np.zeros_like(y_true)


def compute_random_walk_metrics(y_true: np.ndarray) -> dict:
    from src.utils import compute_metrics
    y_pred = random_walk_forecast(y_true)
    return compute_metrics(y_true, y_pred)


# ── 2. Diebold-Mariano Test ───────────────────────────────────────────────────

def diebold_mariano_test(y_true: np.ndarray,
                          y_pred1: np.ndarray,
                          y_pred2: np.ndarray,
                          h: int = 1,
                          loss: str = "mse") -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.
    H0: equal predictive accuracy between model1 and model2.
    """
    y_true  = np.array(y_true).flatten()
    y_pred1 = np.array(y_pred1).flatten()
    y_pred2 = np.array(y_pred2).flatten()

    # Align all three to shortest length
    n = min(len(y_true), len(y_pred1), len(y_pred2))
    y_true  = y_true[-n:]
    y_pred1 = y_pred1[-n:]
    y_pred2 = y_pred2[-n:]

    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    if loss == "mse":
        d = e1**2 - e2**2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    n      = len(d)
    d_mean = np.mean(d)

    gamma0 = np.var(d, ddof=1)
    gammas = [np.cov(d[j:], d[:-j])[0, 1] for j in range(1, h)]
    hac_var = gamma0 + 2 * sum(gammas) if gammas else gamma0
    hac_var = max(hac_var, 1e-12)

    dm_stat = d_mean / np.sqrt(hac_var / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    if p_value < 0.05:
        better = "model1" if dm_stat < 0 else "model2"
    else:
        better = "equal"

    return {
        "dm_statistic": round(float(dm_stat), 4),
        "p_value":      round(float(p_value), 4),
        "significant":  p_value < 0.05,
        "better":       better,
        "interpretation": (
            "Model 1 significantly better" if better == "model1" else
            "Model 2 significantly better" if better == "model2" else
            "No significant difference (p={:.3f})".format(p_value)
        )
    }


def dm_vs_random_walk(model_name: str,
                       y_true: np.ndarray,
                       y_pred_model: np.ndarray) -> dict:
    """Tests model against the random walk baseline."""
    y_true, y_pred_model = _align(y_true, y_pred_model)
    y_rw = random_walk_forecast(y_true)
    result = diebold_mariano_test(y_true, y_pred_model, y_rw, loss="mse")
    result["model"]     = model_name
    result["benchmark"] = "Random Walk"
    return result


# ── 3. Summary table ──────────────────────────────────────────────────────────

def build_dm_table(y_true: np.ndarray,
                    predictions: dict) -> pd.DataFrame:
    """
    DM test summary table for all models vs random walk.
    Automatically aligns lengths — safe against ARIMA off-by-one.
    """
    rows = []
    for model_name, y_pred in predictions.items():
        result = dm_vs_random_walk(model_name, y_true, y_pred)
        rows.append({
            "Model":      model_name,
            "DM Stat":    result["dm_statistic"],
            "p-value":    result["p_value"],
            "Sig. (5%)":  "Yes" if result["significant"] else "No",
            "Result":     result["interpretation"]
        })
    return pd.DataFrame(rows).set_index("Model")