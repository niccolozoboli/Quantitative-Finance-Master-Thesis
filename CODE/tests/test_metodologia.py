"""
tests/test_metodologia.py
--------------------------
Verifica delle correzioni metodologiche richieste dal relatore (branch
fix/metodologia-prof): previsioni a un passo, allineamento feature/target,
regime osservabile al momento della decisione, MDD/Calmar sulla ricchezza
composta, snapshot dei dati locali.

Tutti i test lavorano su dati sintetici, generati in memoria — nessuna
chiamata a yfinance / rete.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.models.arima import run_arima_fold
from src.models.ml_models import run_dt_fold
from src.models import dl_models
from src.feature_engine import build_ml_features, get_feature_cols, prepare_dl_fold
from src.regime import classify_regimes
from src.backtesting import run_backtest
from src import preprocessing
from src import metrics_econ


def _synthetic_df(n=171, seed=None, base_value=None):
    dates = pd.bdate_range("2020-01-01", periods=n)
    if base_value is not None:
        r = np.full(n, base_value, dtype=float)
    elif seed is not None:
        r = np.random.default_rng(seed).normal(0, 0.01, n)
    else:
        r = np.arange(n, dtype=float) * 1e-4
    return pd.DataFrame({"log_return": r}, index=dates)


# ─────────────────────────────────────────────────────────────────────────
# (a) ARIMA, ML, DL restituiscono previsioni esattamente per df.index[test_idx]
# ─────────────────────────────────────────────────────────────────────────

def test_a_predictions_cover_exactly_test_dates(monkeypatch):
    df = _synthetic_df(n=171)   # r_t = t * 1e-4
    train_idx = np.arange(0, 150)
    test_idx  = np.arange(150, 171)
    expected_dates = df.index[test_idx]

    # ARIMA
    dates_a, y_true_a, y_pred_a, _ = run_arima_fold(df, train_idx, test_idx)
    assert len(y_pred_a) == 21
    pd.testing.assert_index_equal(pd.DatetimeIndex(dates_a), expected_dates)

    # ML — Decision Tree
    dates_dt, y_true_dt, y_pred_dt, _ = run_dt_fold(df, train_idx, test_idx)
    assert len(y_pred_dt) == 21
    pd.testing.assert_index_equal(pd.DatetimeIndex(dates_dt), expected_dates)

    # DL — LSTM, epochs=1 per velocità
    monkeypatch.setattr(dl_models, "EPOCHS", 1)
    dates_lstm, y_true_lstm, y_pred_lstm, _ = dl_models.run_lstm_fold(
        df, train_idx, test_idx)
    assert len(y_pred_lstm) == 21
    pd.testing.assert_index_equal(pd.DatetimeIndex(dates_lstm), expected_dates)


# ─────────────────────────────────────────────────────────────────────────
# (b) Nessuna feature ML della riga t contiene r_t o valori successivi;
#     nessuna sequenza DL per il target r_i contiene righe >= i.
# ─────────────────────────────────────────────────────────────────────────

def test_b_ml_features_no_leakage_and_target_alignment():
    df1 = _synthetic_df(n=200)
    feat1 = build_ml_features(df1)

    df2 = df1.copy()
    t_perturb = 150
    date_t = df1.index[t_perturb]
    df2.loc[date_t, "log_return"] += 999.0   # shock enorme SOLO al giorno t

    feat2 = build_ml_features(df2)

    assert date_t in feat1.index and date_t in feat2.index
    # Solo le colonne effettivamente passate al modello (get_feature_cols) —
    # non "log_return" grezzo, che resta nel DataFrame come passthrough e
    # DEVE cambiare (è r_t stesso, non è una feature usata dal modello).
    feat_cols = get_feature_cols()

    # Le feature della riga t (costruite da info fino a t-1) NON devono
    # cambiare quando si perturba r_t.
    pd.testing.assert_series_equal(
        feat1.loc[date_t, feat_cols], feat2.loc[date_t, feat_cols])

    # Il target della riga t DEVE cambiare: target = r_t (item 2), non
    # r_{t+1} — se cambiasse una riga diversa da t, l'allineamento sarebbe
    # ancora sbagliato.
    assert feat1.loc[date_t, "target"] != feat2.loc[date_t, "target"]


def test_b_dl_sequences_no_leakage():
    df1 = _synthetic_df(n=200)
    train_idx = np.arange(0, 150)
    test_idx  = np.arange(150, 171)

    _, X_te1, _, y_te1, dates1, _ = prepare_dl_fold(df1, train_idx, test_idx)

    df2 = df1.copy()
    t_perturb = 160   # riga di test interna, non l'ultima
    date_t = df1.index[t_perturb]
    df2.loc[date_t, "log_return"] += 999.0

    _, X_te2, _, y_te2, dates2, _ = prepare_dl_fold(df2, train_idx, test_idx)

    pos = list(dates1).index(date_t)

    # Il target di quel giorno (= r_t) deve cambiare.
    assert y_te1[pos] != y_te2[pos]
    # La sequenza usata per PREVEDERE r_t (righe fino a t-1) non deve
    # contenere la riga t: non deve cambiare.
    np.testing.assert_array_equal(X_te1[pos], X_te2[pos])
    # Una sequenza successiva (entro seq_len giorni) INCLUDE la riga t
    # nella propria finestra: deve cambiare.
    later_pos = pos + 5
    assert not np.array_equal(X_te1[later_pos], X_te2[later_pos])


# ─────────────────────────────────────────────────────────────────────────
# (c) L'etichetta di regime usata per il rendimento di t è quella di t-1
# ─────────────────────────────────────────────────────────────────────────

def test_c_regime_label_for_rt_is_t_minus_1():
    n = 200
    df = _synthetic_df(n=n, base_value=0.0001)   # volatilità rolling = 0 ovunque
    t_shock = 150
    df.iloc[t_shock, df.columns.get_loc("log_return")] = 0.5   # shock isolato

    regimes = classify_regimes(df)
    regimes_origin = regimes.shift(1).fillna("normal")

    date_t          = df.index[t_shock]
    date_t_minus_1  = df.index[t_shock - 1]

    # L'etichetta ISTANTANEA di t (rolling vol include r_t) diventa "volatile"
    assert regimes.loc[date_t] == "volatile"
    # Ma la decisione sul rendimento di t usa l'etichetta di t-1, immune
    # allo shock dello stesso giorno t.
    assert regimes_origin.loc[date_t] == regimes.loc[date_t_minus_1]
    assert regimes_origin.loc[date_t] != "volatile"


# ─────────────────────────────────────────────────────────────────────────
# (d) Le previsioni ARIMA di test cambiano con i dati realizzati nel test,
#     i parametri stimati restano identici.
# ─────────────────────────────────────────────────────────────────────────

def test_d_arima_one_step_uses_realized_test_data():
    df1 = _synthetic_df(n=171, seed=1)
    train_idx = np.arange(0, 150)
    test_idx  = np.arange(150, 171)

    _, _, y_pred1, params1 = run_arima_fold(df1, train_idx, test_idx)

    df2 = df1.copy()
    perturb_pos = 5   # sesto giorno di test (indice 0-based)
    df2.iloc[150 + perturb_pos, df2.columns.get_loc("log_return")] += 0.05

    _, _, y_pred2, params2 = run_arima_fold(df2, train_idx, test_idx)

    # Stesso train in df1/df2 → stesso ordine e stessi parametri stimati.
    assert params1["order"] == params2["order"]
    assert params1["aic"] == pytest.approx(params2["aic"])

    # Previsioni fino al giorno perturbato incluso: dipendono solo da dati
    # fino a t-1, quindi identiche.
    np.testing.assert_allclose(y_pred1[:perturb_pos + 1], y_pred2[:perturb_pos + 1])

    # Previsioni successive: condizionano sul rendimento REALIZZATO del
    # giorno perturbato → devono cambiare (prova che sono a un passo e
    # non un forecast statico indipendente dai dati di test).
    assert not np.allclose(y_pred1[perturb_pos + 1:], y_pred2[perturb_pos + 1:])


# ─────────────────────────────────────────────────────────────────────────
# (e) metrics_econ: MDD e Calmar sulla ricchezza composta (D4).
#
# NOTA (D1, report del relatore 7/9/2026): run_backtest ora costruisce il
# P&L da R = exp(r)-1, non più dal log-return grezzo passato come y_true —
# questo test storico passava [0.1,-0.5,0.2] come se fosse già la serie
# netta y, il che non è più equivalente sotto run_backtest con w=1 fisso
# (gross_pnl = R, non r). Il caso "MDD su y=[0.1,-0.5,0.2] = 0.5" resta
# valido com'era pensato, ma va testato direttamente su metrics_econ (che
# è la fonte unica di MDD/Calmar per backtest, bootstrap e statistica D —
# vedi D4/test_report_relatore.py::test_d4_max_drawdown_known_series). La
# copertura di run_backtest con la trasformazione R=exp(r)-1 e i costi è in
# test_report_relatore.py::test_d1_run_backtest_manual_three_days.
# ─────────────────────────────────────────────────────────────────────────

def test_e_metrics_econ_mdd_calmar_compounded_wealth():
    y = np.array([0.1, -0.5, 0.2])

    # W = [1.1, 0.55, 0.66], picco 1.1 → MDD = 1 - 0.55/1.1 = 0.5
    mdd = metrics_econ.max_drawdown(y)
    assert mdd == pytest.approx(0.5, abs=1e-6)

    # Calmar = mean(y)*252 / MDD
    calmar = metrics_econ.calmar_ratio(y, mdd)
    assert calmar == pytest.approx(np.mean(y) * 252 / 0.5, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────
# (f) load_and_preprocess non chiama yfinance se lo snapshot esiste già
# ─────────────────────────────────────────────────────────────────────────

def test_f_load_and_preprocess_uses_snapshot_no_download(tmp_path, monkeypatch):
    ticker = "FAKE=T"
    dates  = pd.bdate_range("2020-01-01", periods=30)
    close  = pd.Series(np.linspace(100, 110, 30), index=dates, name="Close")
    snapshot_path = tmp_path / f"{ticker}.csv"
    close.to_frame().to_csv(snapshot_path)

    monkeypatch.setattr(preprocessing, "_data_path", lambda t: str(snapshot_path))

    def _boom(*args, **kwargs):
        raise AssertionError(
            "download_data chiamata: lo snapshot esisteva già, non doveva "
            "riscaricare da yfinance")
    monkeypatch.setattr(preprocessing, "download_data", _boom)

    df = preprocessing.load_and_preprocess("Fake", ticker, refresh=False)
    assert "log_return" in df.columns
    assert len(df) == 29   # 30 giorni - 1 per il dropna del log_return iniziale
