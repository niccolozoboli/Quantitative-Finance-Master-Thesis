"""
walk_forward.py
---------------
Walk-forward validation cronologica: 10 fold, finestra di training
espandibile, 21 giorni di test per fold (210 osservazioni OOS totali
per asset).
"""

import os
import numpy as np
import pandas as pd

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


def validate_folds(folds: list, n: int,
                    n_folds: int = N_FOLDS, test_days: int = TEST_DAYS) -> None:
    """
    B1 — asserzioni esplicite sulla struttura del walk-forward:
      - max(T_k) < min(E_k)               (nessun leakage temporale)
      - |E_k| = test_days                  (21 giorni per fold)
      - E_k contigui e disgiunti, che coprono esattamente gli ultimi
        n_folds*test_days giorni della serie.
    """
    assert len(folds) == n_folds, f"Attesi {n_folds} fold, trovati {len(folds)}"

    prev_test_end = None
    for k, (train_idx, test_idx) in enumerate(folds):
        assert len(test_idx) == test_days, (
            f"Fold {k}: |E_k|={len(test_idx)}, atteso {test_days}"
        )
        assert train_idx.max() < test_idx.min(), (
            f"Fold {k}: max(T_k)={train_idx.max()} >= min(E_k)={test_idx.min()}"
        )
        assert np.array_equal(test_idx, np.arange(test_idx[0], test_idx[0] + test_days)), (
            f"Fold {k}: E_k non è un blocco contiguo di indici"
        )
        if prev_test_end is not None:
            assert test_idx[0] == prev_test_end, (
                f"Fold {k}: E_k non è disgiunto/contiguo al fold precedente "
                f"(inizia a {test_idx[0]}, il fold precedente finiva a {prev_test_end})"
            )
        prev_test_end = test_idx[-1] + 1

    assert folds[0][1][0] == n - n_folds * test_days, (
        "I fold di test non coprono esattamente gli ultimi "
        f"{n_folds * test_days} giorni della serie (n={n})"
    )
    assert folds[-1][1][-1] == n - 1, (
        "L'ultimo fold di test non arriva fino all'ultima osservazione"
    )


def export_folds_csv(folds: list, dates: pd.DatetimeIndex, asset_name: str,
                      path: str = None) -> pd.DataFrame:
    """
    B1 — esporta, per ogni fold, gli estremi di T_k ed E_k (indici e date)
    in results/folds.csv (append/merge tra asset diversi se il file esiste
    già, così un CSV unico copre tutti gli asset).
    """
    rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        rows.append({
            "Asset":        asset_name,
            "Fold":         k + 1,
            "Train Start":  dates[train_idx[0]],
            "Train End":    dates[train_idx[-1]],
            "N Train":      len(train_idx),
            "Test Start":   dates[test_idx[0]],
            "Test End":     dates[test_idx[-1]],
            "N Test":       len(test_idx),
        })
    df = pd.DataFrame(rows)

    if path is None:
        path = os.path.join(os.environ.get("RESULTS_DIR", "results"), "folds.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[existing["Asset"] != asset_name]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def aggregate_fold_results(fold_metrics: list) -> dict:
    keys = ["RMSE", "MAE", "MAPE", "Hit_Rate"]
    agg  = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics
                if isinstance(m, dict) and m.get(k) is not None]
        agg[k] = round(float(np.mean(vals)), 6) if vals else None
    return agg