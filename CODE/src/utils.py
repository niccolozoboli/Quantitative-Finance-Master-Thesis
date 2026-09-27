"""
utils.py
--------
Evaluation metrics — all computed on log_return original scale.

Metrics:
  - RMSE, MAE : standard statistical accuracy
  - MAPE      : percentage error (unstable near zero — noted as limitation)
  - Hit Rate  : % of days where predicted direction matches actual direction
                This is the key QF metric: does the model know if the market
                goes up or down? A random model scores ~50%.

D4 (report del relatore, 7/9/2026): lo Sharpe "lordo" simulato senza costi
è stato rimosso da qui — l'unico Sharpe riportato in tesi è quello netto
di run_backtest/metrics_econ (costi di transazione + roll inclusi).
"""

import numpy as np
import pandas as pd
import os
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error


def set_global_seed(seed: int = 42) -> None:
    """
    Seed centralizzato — Python random, NumPy, TensorFlow.
    Va chiamato una sola volta, prima di qualsiasi training
    (sklearn, XGBoost e Keras/TensorFlow).

    enable_op_determinism() forza anche i kernel GPU/CPU non
    deterministici di TensorFlow (es. alcune riduzioni in floating point)
    a un comportamento riproducibile bit-per-bit a parità di seed — senza,
    tf.random.set_seed da solo non basta a garantire risultati identici
    tra run diverse sugli stessi dati.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Full metric suite on original log-return scale.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    n = min(len(y_true), len(y_pred))
    y_true = y_true[-n:]
    y_pred = y_pred[-n:]

    # ── Statistical metrics ───────────────────────────────────────────────────
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))

    mask = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs(
        (y_true[mask] - y_pred[mask]) / y_true[mask]
    )) * 100) if mask.sum() > 0 else None

    # ── Hit Rate ──────────────────────────────────────────────────────────────
    # Fraction of days where sign(predicted) == sign(actual)
    # Excludes days where actual return is exactly zero (rare but possible)
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        hit_rate = float(np.mean(
            np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero])
        ) * 100)
    else:
        hit_rate = None

    return {
        "RMSE":       round(rmse,     8),
        "MAE":        round(mae,      8),
        "MAPE":       round(mape, 4)  if mape     is not None else None,
        "Hit_Rate":   round(hit_rate, 4) if hit_rate is not None else None,
    }


def save_results(results: list, path: str = "results/metrics.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"Results saved → {path}")
    return df