"""
walk_forward.py
---------------
5-fold chronological walk-forward validation.

As described in the thesis (Section 3.3):
- Expanding training window
- Fixed test window of 21 trading days per fold
- 5 folds → 105 total out-of-sample days
- NO information from the future enters training at any stage

Fold structure:
  Fold 1: [---- train ----][test 21]
  Fold 2: [------ train ------][test 21]
  Fold 3: [-------- train --------][test 21]
  Fold 4: [---------- train ----------][test 21]
  Fold 5: [------------ train ------------][test 21]
"""

import numpy as np


N_FOLDS    = 5
TEST_DAYS  = 21    # ~1 trading month per fold


def get_walk_forward_folds(n: int,
                            n_folds: int = N_FOLDS,
                            test_days: int = TEST_DAYS):
    """
    Generates (train_indices, test_indices) for each fold.

    The initial training window covers everything except the
    last n_folds * test_days observations, which are reserved
    for evaluation. At each fold the training window expands
    by test_days.

    Returns a list of (train_idx, test_idx) tuples.
    """
    total_test = n_folds * test_days
    initial_train_end = n - total_test

    if initial_train_end < 50:
        raise ValueError(
            f"Not enough data: need at least {50 + total_test} observations, "
            f"got {n}."
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
    """
    Averages RMSE, MAE, MAPE, Hit_Rate, Sharpe across all folds.
    fold_metrics: list of dicts — keys vary by metric availability.
    """
    keys = ["RMSE", "MAE", "MAPE", "Hit_Rate", "Sharpe"]
    agg  = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics
                if isinstance(m, dict) and m.get(k) is not None]
        agg[k] = round(float(np.mean(vals)), 6) if vals else None
    return agg