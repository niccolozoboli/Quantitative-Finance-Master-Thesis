"""
tests/test_report_relatore_fase0.py
------------------------------------
FASE 0 del report del relatore (7 settembre 2026): verifica sul codice
ATTUALE (branch fix/metodologia-prof, dopo il commit 0a80b26) dei rilievi
R1-R10, senza correggere nulla.

Convenzione per i test dove serve identificare la provenienza di un
valore: log_return del giorno i = i / 1000 (asse sintetico, nessuna rete).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.feature_engine import build_ml_features, prepare_ml_fold
from src.models.arima import run_arima_fold
from src.regime import classify_regimes


# ─────────────────────────────────────────────────────────────────────────
# R1 — target ML allineato al giorno t (non t+1), 21 osservazioni per fold
# ─────────────────────────────────────────────────────────────────────────

def test_r1_last_training_target_alignment():
    n = 379
    dates = pd.bdate_range("2020-01-01", periods=n)
    r = np.arange(n, dtype=float) / 1000.0   # log_return del giorno i = i/1000
    df = pd.DataFrame({"log_return": r}, index=dates)

    train_idx = np.arange(0, 358)     # 0..357
    test_idx  = np.arange(358, 379)   # 358..378

    feat = build_ml_features(df)
    last_train_date   = df.index[357]
    last_train_target = feat.loc[last_train_date, "target"]

    # Codice ATTUALE (feature_engine.py:160, target = r, stesso giorno):
    # l'ultimo target di training è il rendimento del giorno 357, MAI
    # quello del giorno 358 (primo giorno di E_k) — con il bug originale
    # (target = r.shift(-1)) questo stesso valore sarebbe stato 358/1000.
    assert last_train_target == pytest.approx(357 / 1000)
    assert last_train_target != pytest.approx(358 / 1000)

    # Anche il conteggio delle osservazioni di test deve essere 21, non 20
    # (il bug originale perdeva l'ultima riga dell'ultimo fold per il dropna
    # di r.shift(-1) sull'ultima data della serie).
    _, X_te, _, y_te, dates_te, _ = prepare_ml_fold(df, train_idx, test_idx)
    assert len(y_te) == 21
    pd.testing.assert_index_equal(pd.DatetimeIndex(dates_te), df.index[test_idx])


# ─────────────────────────────────────────────────────────────────────────
# R2 — ARIMA: previsioni one-step r_hat_{s|s-1}, non forecast statico
# ─────────────────────────────────────────────────────────────────────────

def test_r2_arima_is_one_step_not_static_forecast():
    n = 171
    dates = pd.bdate_range("2020-01-01", periods=n)
    r = np.random.default_rng(7).normal(0, 0.01, n)
    df1 = pd.DataFrame({"log_return": r}, index=dates)

    train_idx = np.arange(0, 150)
    test_idx  = np.arange(150, 171)

    _, _, y_pred1, params1 = run_arima_fold(df1, train_idx, test_idx)

    df2 = df1.copy()
    perturb_pos = 8
    df2.iloc[150 + perturb_pos, df2.columns.get_loc("log_return")] += 0.05
    _, _, y_pred2, params2 = run_arima_fold(df2, train_idx, test_idx)

    # Stesso train → stessi parametri stimati (src/models/arima.py:35-36:
    # order/aic vengono da select_order+fit SOLO sul train).
    assert params1["order"] == params2["order"]

    # src/models/arima.py:47-50: model_fit.apply(full_s, refit=False) +
    # get_prediction(..., dynamic=False) → ogni previsione di test usa i
    # rendimenti REALIZZATI fino al giorno prima, non un forecast a 21
    # passi indipendente dai dati di test (bug originale: .forecast(steps=21)).
    np.testing.assert_allclose(y_pred1[:perturb_pos + 1], y_pred2[:perturb_pos + 1])
    assert not np.allclose(y_pred1[perturb_pos + 1:], y_pred2[perturb_pos + 1:])


# ─────────────────────────────────────────────────────────────────────────
# R3 — Regime usato per decidere su r_s è quello di s-1, non quello di s
# ─────────────────────────────────────────────────────────────────────────

def test_r3_regime_used_for_rs_is_s_minus_1():
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    r = np.full(n, 0.0001, dtype=float)
    t_shock = 150
    r[t_shock] = 0.5
    df = pd.DataFrame({"log_return": r}, index=dates)

    regimes = classify_regimes(df)
    # main.py:230 (percorso live) e main.py:102 (_reconstruct_asset_from_disk):
    # regimes_origin = regimes.shift(1).fillna("normal") — passato a
    # regime_conditional_strategy (main.py:431), split_by_regime (main.py:385)
    # e alla statistica D (main.py:333), MAI `regimes` grezzo.
    regimes_origin = regimes.shift(1).fillna("normal")

    date_t         = df.index[t_shock]
    date_t_minus_1 = df.index[t_shock - 1]

    assert regimes.loc[date_t] == "volatile"          # etichetta istantanea: vede lo shock
    assert regimes_origin.loc[date_t] == regimes.loc[date_t_minus_1]
    assert regimes_origin.loc[date_t] != "volatile"    # la decisione su r_t non lo vede


# ─────────────────────────────────────────────────────────────────────────
# R4 — CORRETTO (C7, report del relatore 7/9/2026): la TCN ora usa
# x[:, -1, :] invece di GlobalAveragePooling1D — la copertura di questo
# fix (dipendenza SOLO dagli ultimi 15 passi) è in
# tests/test_report_relatore.py::test_c7_tcn_last_step_depends_only_on_last_15.
# Il test originale (che confermava il bug pre-fix) è stato rimosso qui:
# asseriva esplicitamente "if this assert ever fails, R4 has been fixed".
# ─────────────────────────────────────────────────────────────────────────
