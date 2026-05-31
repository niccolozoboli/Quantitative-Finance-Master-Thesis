"""
walk_forward.py
---------------
10-fold chronological walk-forward validation.
VERSIONE 2.0 — 10 fold invece di 5.

Con 10 fold → 210 osservazioni OOS per asset.
Maggiore potere statistico per il DM test.
"""

import numpy as np

N_FOLDS   = 10   # aumentato da 5
TEST_DAYS = 21   # ~1 mese trading per fold (invariato)


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


def fold_summary(folds: list, df_index) -> None:
    print(f"\n  Walk-forward: {len(folds)} fold × 21 giorni")
    print(f"  {'Fold':<6} {'Train start':<14} {'Train end':<14} "
          f"{'Test start':<14} {'Test end':<14} {'Train size':<12}")
    print(f"  {'-'*74}")
    for i, (tr, te) in enumerate(folds):
        print(f"  {i+1:<6} "
              f"{str(df_index[tr[0]].date()):<14} "
              f"{str(df_index[tr[-1]].date()):<14} "
              f"{str(df_index[te[0]].date()):<14} "
              f"{str(df_index[te[-1]].date()):<14} "
              f"{len(tr):<12}")
    print(f"\n  Totale giorni OOS: {len(folds) * 21} per asset\n")