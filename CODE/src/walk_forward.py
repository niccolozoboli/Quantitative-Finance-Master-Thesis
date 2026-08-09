"""
walk_forward.py
---------------
Walk-forward validation cronologica: 10 fold, finestra di training
espandibile, 21 giorni di test per fold (210 osservazioni OOS totali
per asset).
"""

import numpy as np

N_FOLDS   = 10
TEST_DAYS = 21   # ~1 mese trading per fold


def get_walk_forward_folds(n: int,
                            n_folds: int = N_FOLDS,
                            test_days: int = TEST_DAYS):
    total_test        = n_folds * test_days
    initial_train_end = n - total_test

    if initial_train_end < 252:
        raise ValueError(
            f"Training iniziale troppo corto: {initial_train_end} osservazioni. "
            f"Servono almeno 252 (1 anno)."
        )

    folds = []
    for fold in range(n_folds):
        train_end  = initial_train_end + fold * test_days
        test_start = train_end
        test_end   = test_start + test_days

        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(test_start, test_end)
        folds.append((train_idx, test_idx))

    return folds


def aggregate_fold_results(fold_metrics: list) -> dict:
    keys = ["RMSE", "MAE", "MAPE", "Hit_Rate", "Sharpe"]
    agg  = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics
                if isinstance(m, dict) and m.get(k) is not None]
        agg[k] = round(float(np.mean(vals)), 6) if vals else None
    return agg