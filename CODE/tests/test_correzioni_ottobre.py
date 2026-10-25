"""
tests/test_correzioni_ottobre.py
--------------------------------
Test delle modifiche richieste dal relatore nella tesi annotata di ottobre
2026 (commenti K04, K13, K14, K15, K02, K06, K08, K11). Dati sintetici in
memoria, nessuna chiamata di rete.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (run_backtest, lag_positions, architecture_vs_regime_D,
                             _prepare_D_inputs, _D_operator, two_way_decomposition,
                             exposure_benchmarks, regime_conditional_strategy)
from src.statistical_tests import (da_test_hac, da_test_centred_bootstrap,
                                   newey_west_lrv, dm_test_loss, MIN_ACTIVE_DA)
from src.preprocessing import validate_ohlc, open_to_open_returns
from src import metrics_econ


# ─────────────────────────────────────────────────────────────────────────
# K04 — test della DA sotto H0: E[h_t] = 1/2
# ─────────────────────────────────────────────────────────────────────────

def _dependent_hits_under_null(rng, n, rho=0.4):
    # hit dipendenti con media 1/2: soglia a 0 di un AR(1) gaussiano
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return (x > 0).astype(float)


def test_k04_da_hac_has_correct_size_under_dependent_null():
    rng = np.random.default_rng(0)
    n, reps = 210, 2000
    rej = 0
    for _ in range(reps):
        h = _dependent_hits_under_null(rng, n)
        rej += da_test_hac(h, np.ones(n))["DA p-value (HAC)"] < 0.05
    assert 0.03 <= rej / reps <= 0.08


def test_k04_da_centred_bootstrap_has_correct_size_under_dependent_null():
    rng = np.random.default_rng(1)
    n, reps = 210, 300
    rej = 0
    for _ in range(reps):
        h = _dependent_hits_under_null(rng, n)
        rej += da_test_centred_bootstrap(h, np.ones(n), n_boot=400, seed=1)["DA p-value (Boot H0)"] < 0.05
    assert 0.02 <= rej / reps <= 0.10


def test_k04_da_tests_detect_strong_signal():
    h = np.r_[np.ones(150), np.zeros(60)]
    np.random.default_rng(2).shuffle(h)
    assert da_test_hac(h, np.ones(210))["DA p-value (HAC)"] < 0.01
    assert da_test_centred_bootstrap(h, np.ones(210))["DA p-value (Boot H0)"] < 0.01


def test_k04_da_not_tested_below_min_active_days():
    n_act = MIN_ACTIVE_DA - 1
    active = np.r_[np.ones(n_act), np.zeros(210 - n_act)]
    hit = np.ones(210)
    assert np.isnan(da_test_hac(hit, active)["DA p-value (HAC)"])
    assert np.isnan(da_test_centred_bootstrap(hit, active)["DA p-value (Boot H0)"])


def test_k04_bootstrap_ci_has_no_uncentred_pvalue():
    from src.backtesting import block_bootstrap_ci
    rng = np.random.default_rng(3)
    out = block_bootstrap_ci(rng.normal(0, 0.01, 60), rng.integers(0, 2, 60),
                             np.ones(60), n_boot=50)
    assert not any("p-value" in k for k in out)
    assert "DA CI Lower (95%)" in out


def test_newey_west_lrv_matches_textbook_formula():
    rng = np.random.default_rng(4)
    x = rng.normal(size=100)
    xc = x - x.mean()
    L = 4
    expected = xc @ xc / 100 + 2 * sum((1 - j / (L + 1)) * (xc[j:] @ xc[:-j]) / 100
                                       for j in range(1, L + 1))
    assert newey_west_lrv(x, L) == pytest.approx(expected, rel=1e-12)


def test_dm_test_loss_sign():
    rng = np.random.default_rng(5)
    l2 = rng.uniform(1, 2, 300)
    out = dm_test_loss(l2 - 0.5, l2)
    assert out["DM Stat"] < 0 and out["p-value"] < 0.01


# ─────────────────────────────────────────────────────────────────────────
# K13 — esecuzione con un giorno di ritardo e all'apertura
# ─────────────────────────────────────────────────────────────────────────

def test_k13_lag_positions():
    assert np.array_equal(lag_positions(np.array([1., -1., 1.]), 1), [0., 1., -1.])


def test_k13_lag1_backtest_manual_three_days():
    c = 1e-4
    r = np.array([0.01, -0.02, 0.03])
    y_pred = np.array([1.0, 1.0, -1.0])      # posizioni decise: +1, +1, -1
    bt = run_backtest(r, y_pred, cost_per_side=c, execution="lag1")
    R = np.expm1(r)
    # eseguite: 0, +1, +1
    expected = np.array([0.0, R[1] - c, R[2]])
    assert np.allclose(bt["Net PnL"], expected, atol=1e-15)
    assert bt["Position"][0] == 0.0
    # DA sul segno della previsione eseguita (giorno 1: +1 vs R<0; giorno 2: +1 vs R>0)
    assert bt["Directional Acc."] == pytest.approx(50.0)


def test_k13_close_execution_unchanged_vs_default():
    rng = np.random.default_rng(6)
    r = rng.normal(0, 0.01, 40); yp = rng.normal(0, 0.01, 40)
    a = run_backtest(r, yp)
    b = run_backtest(r, yp, execution="close")
    assert np.array_equal(a["Net PnL"], b["Net PnL"])


def test_k13_open_execution_uses_open_to_open_returns():
    c = 1e-4
    r = np.array([0.01, -0.02, 0.03])
    y_pred = np.array([1.0, -1.0, -1.0])
    R_oo = np.array([0.005, -0.01, np.nan])  # ultimo giorno non disponibile
    bt = run_backtest(r, y_pred, cost_per_side=c, execution="open", open_returns=R_oo)
    expected = np.array([0.005 - c, 0.01 - 2 * c])
    assert np.allclose(bt["Net PnL"], expected, atol=1e-15)
    assert bt["N Days"] == 2


def test_k13_open_to_open_returns_and_validation():
    idx = pd.bdate_range("2026-01-01", periods=6)
    ohlc = pd.DataFrame({"Open": [10, 11, 12, 11, 13, 14.],
                         "High": [11, 12, 13, 12, 14, 15.],
                         "Low":  [9, 10, 11, 10, 12, 13.],
                         "Close": [10.5, 11.5, 12.5, 11.5, 13.5, 14.5]}, index=idx)
    oos = idx[2:5]
    r = open_to_open_returns(ohlc, oos)
    assert np.allclose(r, [11 / 12 - 1, 13 / 11 - 1, 14 / 13 - 1])
    close_df = ohlc[["Close"]]
    assert validate_ohlc(ohlc, close_df, oos)["ok"]
    bad = ohlc.copy(); bad.loc[idx[3], "Open"] = 20.0
    assert not validate_ohlc(bad, close_df, oos)["ok"]
    assert not validate_ohlc(ohlc, close_df, idx[3:])["ok"]   # manca il giorno dopo
    assert not validate_ohlc(None, close_df, oos)["ok"]


# ─────────────────────────────────────────────────────────────────────────
# K15 / K02 — statistica D: stesso operatore in stima e bootstrap
# ─────────────────────────────────────────────────────────────────────────

def _synthetic_asset_data(regime_effect=0.0, model_effect=0.0, seed=0, n=210):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-07-01", periods=n)
    out = {}
    for ai, a in enumerate(["Gold", "Silver", "Copper", "Platinum"]):
        reg = pd.Series(rng.choice(["stable", "normal", "volatile"], size=n), index=dates)
        shift = reg.map({"stable": regime_effect, "normal": 0.0,
                         "volatile": -regime_effect}).values
        y = {f"M{m}": rng.normal(0, 0.01, n) + 0.01 * (shift + model_effect * (m - 5.5) / 5.5)
             for m in range(12)}
        out[a] = {"y": y, "dates": dates, "regimes": reg}
    return out


@pytest.mark.parametrize("aggregation", ["portfolio", "per_asset"])
def test_k15_point_estimate_equals_operator_on_identity_index(aggregation):
    data = _synthetic_asset_data(seed=1)
    res = architecture_vs_regime_D(data, aggregation=aggregation, n_boot=20)
    models, assets, common, Y, RG = _prepare_D_inputs(data)
    op = _D_operator(Y, RG, models, assets, np.arange(len(common)), aggregation, 5)
    assert res["D"] == pytest.approx(op["D"], abs=1e-6)


def test_k15_portfolio_cell_is_mean_of_assets_in_regime():
    dates = pd.bdate_range("2025-07-01", periods=6)
    regA = pd.Series(["stable"] * 3 + ["volatile"] * 3, index=dates)
    regB = pd.Series(["stable", "volatile"] * 3, index=dates)
    yA = np.array([1, 2, 3, 4, 5, 6.]) / 100
    yB = np.array([6, 5, 4, 3, 2, 1.]) / 100
    data = {"A": {"y": {"M": yA}, "dates": dates, "regimes": regA},
            "B": {"y": {"M": yB}, "dates": dates, "regimes": regB}}
    models, assets, common, Y, RG = _prepare_D_inputs(data)
    op = _D_operator(Y, RG, models, assets, np.arange(6), "portfolio", 1)
    # stabile: giorno0 (A,B) -> 3.5; giorno1 (A) -> 2; giorno2 (A,B) -> 3.5; giorno4 (B) -> 2
    expected = metrics_econ.sharpe_ratio(np.array([3.5, 2, 3.5, 2]) / 100)
    assert op["S"][0, 0] == pytest.approx(expected)
    assert op["counts"]["stable"] == 4 and op["counts"]["volatile"] == 4
    assert op["counts"]["normal"] == 0 and not op["valid"]


def test_k15_sign_of_D_follows_dominant_dispersion():
    d_reg = architecture_vs_regime_D(_synthetic_asset_data(regime_effect=0.5, seed=2), n_boot=10)
    d_mod = architecture_vs_regime_D(_synthetic_asset_data(model_effect=0.5, seed=2), n_boot=10)
    assert d_reg["D"] > 0
    assert d_mod["D"] < 0


def test_k15_discarded_fraction_reported():
    res = architecture_vs_regime_D(_synthetic_asset_data(seed=3), n_boot=50)
    assert res["N Bootstrap Used"] + round(res["Frac. Discarded"] * 50) == 50


def test_k02_two_way_shares_sum_to_one_and_detect_regime_effect():
    tw = two_way_decomposition(_synthetic_asset_data(regime_effect=0.5, seed=4), n_boot=20)
    total = tw.loc[["Share Model", "Share Regime", "Share Interaction", "Share Residual"], "Share"].sum()
    assert total == pytest.approx(1.0, abs=1e-6)
    assert tw.loc["Share Regime", "Share"] > tw.loc["Share Model", "Share"]


# ─────────────────────────────────────────────────────────────────────────
# K14 — benchmark della regola di esposizione della RCS
# ─────────────────────────────────────────────────────────────────────────

def test_k14_exposure_benchmarks():
    rng = np.random.default_rng(7)
    n = 120
    dates = pd.bdate_range("2025-07-01", periods=n)
    reg = pd.Series(rng.choice(["stable", "normal", "volatile"], size=n), index=dates)
    r = rng.normal(0.001, 0.01, n)
    long_pos = regime_conditional_strategy(np.ones(n), dates, reg)
    rcs_long = run_backtest(r, np.ones(n), position=long_pos)
    summary, per_model, a = exposure_benchmarks(
        r, dates, reg, asset_name=None, rcs_results={"Always long": rcs_long}, n_rand=50)
    assert np.array_equal(a, long_pos)
    assert per_model.loc["Always long", "Identical to Long Filtered"]
    assert summary["Sharpe Long Filtered"] == pytest.approx(rcs_long["Sharpe Ratio"])
    assert 0 < per_model.loc["Always long", "p_emp vs Random Filtered"] <= 1
