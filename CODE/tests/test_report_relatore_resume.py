"""
tests/test_report_relatore_resume.py
--------------------------------------
Fix richiesto dall'utente (2026-09-27, dopo il run --smoke): il marker di
"asset già completato" per --resume deve essere SOLO
results/predictions_<asset>.csv (scritto a fine asset dal percorso live),
non backtest_<asset>.csv / backtest_rcs_<asset>.csv — questi ultimi
vengono scritti solo a fine run, dopo la correzione di Holm su tutti gli
asset (B9), quindi non esistono ancora quando un singolo asset è
"completato" a metà di una run interrotta.

Test end-to-end (dati sintetici, nessuna rete): con solo
predictions_<asset>.csv presente su disco e --resume, l'asset viene
ricostruito da _reconstruct_asset_from_disk e MAI riaddestrato (il percorso
di training, get_walk_forward_folds, non viene mai chiamato).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import main


def _synthetic_df(n=150, seed=0):
    dates = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame({"Close": close, "log_return": r}, index=dates)


def test_resume_marker_is_predictions_csv_only(tmp_path, monkeypatch):
    df = _synthetic_df()
    n_fold_days = 21
    test_dates = df.index[-n_fold_days:]

    # ── Setup: solo predictions_<asset>.csv su disco, nessun backtest_*.csv ──
    monkeypatch.setattr(main, "METALS", {"Gold": "GC=F"})
    monkeypatch.setattr(main, "load_and_preprocess",
                         lambda name, ticker, start=None, end=None: df.copy())
    monkeypatch.setattr(main, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(main, "_git_commit", lambda: "testcommit123")

    manifest = {"commit": "testcommit123", "python_version": "x",
                "packages": {}, "gpu": "none", "seed": 42,
                "date": "2026-09-27T00:00:00", "smoke": False}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    rng = np.random.default_rng(1)
    rows = []
    for model_label in ["Random Walk", "ARMA"]:
        y_true = rng.normal(0, 0.01, n_fold_days)
        y_pred = (np.zeros(n_fold_days) if model_label == "Random Walk"
                  else y_true * 0.3 + rng.normal(0, 0.005, n_fold_days))
        for d, yt, yp in zip(test_dates, y_true, y_pred):
            rows.append({"Model": model_label, "Fold": 1, "Date": d,
                        "y_true": yt, "y_pred": yp})
    pd.DataFrame(rows).to_csv(tmp_path / "predictions_Gold.csv", index=False)

    assert not (tmp_path / "backtest_Gold.csv").exists()
    assert not (tmp_path / "backtest_rcs_Gold.csv").exists()

    # ── Tripwire: get_walk_forward_folds non deve MAI essere chiamata —
    # se lo fosse, vorrebbe dire che il resume non ha riconosciuto l'asset
    # come già completato e sta rifacendo il training ──────────────────────
    def _tripwire(*args, **kwargs):
        raise AssertionError(
            "get_walk_forward_folds chiamata: l'asset è stato riaddestrato "
            "invece di essere ricostruito da predictions_Gold.csv"
        )
    monkeypatch.setattr(main, "get_walk_forward_folds", _tripwire)

    results_df = main.run_pipeline(resume=True, smoke=False)

    # ── L'asset è stato ricostruito (non saltato silenziosamente) ──────────
    assert "Gold" in set(results_df["Asset"])
    assert set(results_df.loc[results_df["Asset"] == "Gold", "Model"]) == \
        {"Random Walk", "ARMA"}
    assert (tmp_path / "backtest_Gold.csv").exists()       # scritto ora, post-Holm
    assert (tmp_path / "backtest_rcs_Gold.csv").exists()
    assert (tmp_path / "strategy_returns_Gold.csv").exists()
