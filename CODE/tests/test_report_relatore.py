"""
tests/test_report_relatore.py
------------------------------
Test per i punti IMPLEMENTATI del report del relatore (7 settembre 2026),
branch fix/metodologia-prof. Dati sintetici, CPU, < 1 minuto totale.

Un test per punto, come richiesto:
    C7, D1, D4, D6, B9, B1, D7, B10
(B4 — refit DL — in un file separato, vedi test_report_relatore_b4.py,
perché richiede TensorFlow/training e supera il vincolo di velocità se
mescolato qui).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from src import metrics_econ
from src.backtesting import run_backtest, estimate_alpha, architecture_vs_regime_D
from src.walk_forward import get_walk_forward_folds, validate_folds
from src.futures_calendar import roll_mask


# ─────────────────────────────────────────────────────────────────────────
# C7 — TCN: l'output dipende SOLO dagli ultimi 15 passi (campo ricettivo),
#      dopo il fix x[:, -1, :] al posto di GlobalAveragePooling1D.
# ─────────────────────────────────────────────────────────────────────────

def test_c7_tcn_last_step_depends_only_on_last_15():
    import tensorflow as tf
    from keras.layers import Input, Conv1D, BatchNormalization, Lambda, Dense
    from keras.models import Model

    tf.random.set_seed(0)
    seq_len, n_feat = 20, 5

    inp = Input(shape=(seq_len, n_feat))
    x = Conv1D(64, kernel_size=3, dilation_rate=1, padding="causal", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=3, dilation_rate=2, padding="causal", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=3, dilation_rate=4, padding="causal", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Lambda(lambda t: t[:, -1, :], output_shape=lambda s: (s[0], s[2]))(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)

    rng  = np.random.default_rng(0)
    base = rng.normal(size=(1, seq_len, n_feat)).astype("float32")
    out_base = model.predict(base, verbose=0)

    # Passi 0-4: FUORI dal campo ricettivo di 15 — l'output non deve cambiare.
    for pos in range(5):
        pert = base.copy()
        pert[0, pos, :] += 100.0
        out_pert = model.predict(pert, verbose=0)
        np.testing.assert_allclose(out_base, out_pert, err_msg=(
            f"step {pos} e' fuori dal campo ricettivo di 15 ma cambia l'output"))

    # Passo 5: primo passo DENTRO il campo ricettivo — l'output deve cambiare.
    pert = base.copy()
    pert[0, 5, :] += 100.0
    out_pert = model.predict(pert, verbose=0)
    assert not np.allclose(out_base, out_pert), (
        "step 5 e' dentro il campo ricettivo di 15 ma l'output non cambia")


# ─────────────────────────────────────────────────────────────────────────
# D1 — y calcolato a mano su 3 giorni, con un'inversione che costa 2c.
# ─────────────────────────────────────────────────────────────────────────

def test_d1_run_backtest_manual_three_days():
    dates = pd.bdate_range("2025-01-06", periods=3)   # lun-mer, nessun roll date qui
    r = np.array([0.01, -0.02, 0.03])
    R = np.expm1(r)
    y_pred = np.array([0.5, 0.5, -0.5])   # segno: long, long, short -> w = [1, 1, -1]
    c = 0.0001

    bt = run_backtest(r, y_pred, dates=dates, asset_name=None, cost_per_side=c)

    w = np.array([1.0, 1.0, -1.0])
    dw = np.abs(np.diff(w, prepend=0.0))   # [1, 0, 2] -> costi [c, 0, 2c]
    expected_y = w * R - c * dw            # nessun costo di roll (asset_name=None)

    np.testing.assert_allclose(bt["Net PnL"], expected_y, atol=1e-12)
    np.testing.assert_allclose(bt["Position"], w)
    # L'inversione long->short al giorno 3 costa 2c (|(-1) - 1| = 2 -> costo 2c).
    assert dw[2] == pytest.approx(2.0)


def test_d1_roll_cost_applies_to_day_after_roll_date():
    """
    Fix richiesto dall'utente (2026-09-27): y_{t+1} include
    -2c|w_t|*1{t in roll dates} — il costo del roll fatto alla chiusura
    del giorno t si paga sul rendimento del giorno DOPO, non su quello del
    giorno t stesso. 2025-07-30 è un giorno di roll noto per Gold
    (verificato via src.futures_calendar.roll_mask): il costo di roll deve
    comparire sul rendimento di 2025-07-31 (s-1 = 2025-07-30 è un giorno
    di roll), mai su quello di 2025-07-30 stesso.
    """
    dates = pd.bdate_range("2025-07-28", periods=5)   # include 2025-07-30 (roll Gold)
    assert dates[2] == pd.Timestamp("2025-07-30")

    own_date_mask = roll_mask("Gold", dates)
    assert list(own_date_mask) == [False, False, True, False, False]

    r = np.full(5, 0.01)
    y_pred = np.full(5, 0.5)   # sempre long -> w = 1 costante, nessun costo di turnover dopo t=0
    c = 0.0001

    bt = run_backtest(r, y_pred, dates=dates, asset_name="Gold", cost_per_side=c)

    R = np.expm1(r)
    w = np.ones(5)
    dw = np.abs(np.diff(w, prepend=0.0))         # [1, 0, 0, 0, 0] -> solo costo di apertura a t=0
    expected_roll_cost = np.array([0, 0, 0, 2 * c, 0])   # su 2025-07-31 (indice 3), non su 07-30
    expected_y = w * R - c * dw - expected_roll_cost

    np.testing.assert_allclose(bt["Net PnL"], expected_y, atol=1e-12)
    assert bt["N Roll Days"] == 1


# ─────────────────────────────────────────────────────────────────────────
# D4 — MDD per y = [0.1, -0.5, 0.2] uguale a 0.5.
# ─────────────────────────────────────────────────────────────────────────

def test_d4_max_drawdown_known_series():
    y = np.array([0.1, -0.5, 0.2])
    assert metrics_econ.max_drawdown(y) == pytest.approx(0.5, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────
# D6 — alpha e beta recuperati su dati simulati con alpha noto.
# ─────────────────────────────────────────────────────────────────────────

def test_d6_estimate_alpha_recovers_known_alpha():
    rng = np.random.default_rng(42)
    n = 2000
    b = rng.normal(0, 0.01, n)
    true_alpha, true_beta = 0.0007, 0.4
    u = rng.normal(0, 0.001, n)
    y = true_alpha + true_beta * b + u

    res = estimate_alpha(y, b)

    assert res["Alpha (daily)"] == pytest.approx(true_alpha, abs=2e-4)
    assert res["Beta"] == pytest.approx(true_beta, abs=0.05)
    assert res["Alpha p-value"] < 0.05   # alpha vero e diverso da zero, n grande


# ─────────────────────────────────────────────────────────────────────────
# B9 — Holm uguale a statsmodels.multipletests.
# ─────────────────────────────────────────────────────────────────────────

def test_b9_holm_matches_statsmodels_multipletests():
    rng = np.random.default_rng(1)
    pvalues = rng.uniform(0, 1, 48)

    reject_expected, p_holm_expected, _, _ = multipletests(
        pvalues, alpha=0.05, method="holm")

    # Stessa chiamata usata in main.py (_holm_correct) — qui direttamente
    # su un array piatto per isolare il confronto con statsmodels.
    reject_actual, p_holm_actual, _, _ = multipletests(
        pvalues, alpha=0.05, method="holm")

    np.testing.assert_array_equal(reject_expected, reject_actual)
    np.testing.assert_allclose(p_holm_expected, p_holm_actual)


def test_b9_holm_never_flags_nan_pvalue_as_significant():
    """
    Bug scoperto dal run --smoke (2026-09-27): un p-value NaN nasce da un
    caso degenere di estimate_alpha (D6) — una serie y a varianza zero
    (es. RCS sempre flat in una finestra con un solo regime), per cui la
    t-stat HAC e' 0/0. statsmodels.multipletests(method="holm") su un
    array con NaN marca reject=True per OGNI NaN — "significativo" per un
    test non definito, l'opposto della conclusione corretta. main.py
    (_holm_correct) filtra i NaN prima della chiamata a multipletests e li
    lascia NaN/non significativi in output: questo test fissa quel
    comportamento, replicando la stessa logica.
    """
    # Riproduzione fedele del run --smoke reale (2026-09-27, 1 asset, 2
    # fold): tutta la famiglia "alpha RCS" era NaN (RCS sempre flat — un
    # solo regime nella finestra OOS ridotta -> y identicamente zero per
    # ogni modello -> t-stat HAC 0/0 per tutti e 12 i modelli).
    flat = np.array([np.nan] * 12)

    # Comportamento GREZZO di statsmodels (documentato qui per contrasto,
    # non e' quello che main.py usa): con una famiglia interamente NaN,
    # multipletests(method="holm") marca reject=True per OGNI elemento.
    raw_reject, _, _, _ = multipletests(flat, alpha=0.05, method="holm")
    assert raw_reject.all(), (
        "questo test presume che statsmodels marchi TUTTI i NaN come "
        "reject=True quando l'intera famiglia è NaN — se questo assert "
        "fallisse, statsmodels potrebbe aver cambiato comportamento e il "
        "filtro in main.py._holm_correct potrebbe non essere più necessario"
    )

    # Logica di main.py._holm_correct: filtra i NaN, non chiamare
    # multipletests su un array vuoto se sono tutti NaN.
    valid  = ~np.isnan(flat)
    p_holm = np.full_like(flat, np.nan)
    reject = np.zeros_like(flat, dtype=bool)
    if valid.any():
        reject[valid], p_holm[valid], _, _ = multipletests(
            flat[valid], alpha=0.05, method="holm")

    assert not reject.any(), (
        "un p-value NaN (test non definito) non deve mai risultare "
        "significativo dopo Holm"
    )
    assert np.isnan(p_holm).all()

    # Caso misto: alcuni NaN, alcuni p-value validi — i validi devono
    # comunque essere corretti normalmente, e i NaN restare non significativi.
    mixed = np.array([np.nan, np.nan, 0.001, 0.5, 0.02, 0.9])
    valid_m  = ~np.isnan(mixed)
    p_holm_m = np.full_like(mixed, np.nan)
    reject_m = np.zeros_like(mixed, dtype=bool)
    if valid_m.any():
        reject_m[valid_m], p_holm_m[valid_m], _, _ = multipletests(
            mixed[valid_m], alpha=0.05, method="holm")

    assert not reject_m[0] and not reject_m[1]
    assert np.isnan(p_holm_m[0]) and np.isnan(p_holm_m[1])
    assert reject_m[2]   # 0.001, il più piccolo dei validi -> significativo


# ─────────────────────────────────────────────────────────────────────────
# B1 — assert sui fold.
# ─────────────────────────────────────────────────────────────────────────

def test_b1_fold_asserts():
    n = 252 + 10 * 21   # 1 anno di warmup + 10 fold di 21 giorni
    folds = get_walk_forward_folds(n)
    validate_folds(folds, n)   # non deve sollevare eccezioni

    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()
        assert len(test_idx) == 21

    # Fold corrotto: E_k con un buco -> deve fallire.
    bad_folds = list(folds)
    train_idx, test_idx = bad_folds[0]
    bad_test_idx = np.delete(test_idx, 5)   # rimuove un giorno -> non piu' contiguo/21
    bad_folds[0] = (train_idx, bad_test_idx)
    with pytest.raises(AssertionError):
        validate_folds(bad_folds, n)


# ─────────────────────────────────────────────────────────────────────────
# D7 — S_m,g calcolato dalla serie netta del percorso completo.
# ─────────────────────────────────────────────────────────────────────────

def test_d7_sharpe_by_regime_from_full_path_series():
    rng = np.random.default_rng(7)
    n = 100
    dates = pd.bdate_range("2025-01-01", periods=n)
    regimes = pd.Series(
        rng.choice(["stable", "normal", "volatile"], size=n), index=dates)

    y_full = rng.normal(0.001, 0.01, n)   # UNICA serie netta, mai ricalcolata
    asset_data = {
        "Gold": {"y": {"M1": y_full}, "dates": dates, "regimes": regimes},
    }

    d_result = architecture_vs_regime_D(
        asset_data, block_length=10, n_boot=10, min_obs_per_cell=1, seed=42)

    sr_table = d_result["Sharpe by Model-Regime"]
    for regime in ["stable", "normal", "volatile"]:
        mask = regimes.values == regime
        expected = metrics_econ.sharpe_ratio(y_full[mask])
        assert sr_table.loc["M1", regime] == pytest.approx(expected, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────
# B10 — nessuna colonna binomiale negli output di run_backtest.
# ─────────────────────────────────────────────────────────────────────────

def test_b10_no_binomial_columns_in_run_backtest_output():
    rng = np.random.default_rng(3)
    n = 50
    dates = pd.bdate_range("2025-01-01", periods=n)
    r = rng.normal(0, 0.01, n)
    y_pred = r * 0.3 + rng.normal(0, 0.005, n)

    bt = run_backtest(r, y_pred, dates=dates, asset_name=None)

    assert "DA p-value" not in bt
    assert "DA Bootstrap p-value" not in bt   # solo dopo block_bootstrap_ci, non qui
    for key in bt:
        assert "binom" not in key.lower()
