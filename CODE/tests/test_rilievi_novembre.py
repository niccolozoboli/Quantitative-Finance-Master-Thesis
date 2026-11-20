"""
tests/test_rilievi_novembre.py
------------------------------
Test delle correzioni ai rilievi della revisione del codice (novembre
2026): R4 (date delle previsioni), R5 (GARCH dalle previsioni salvate),
R11 (p-value DM non arrotondati), R14 (stessi asset nelle repliche del
controllo per asset), R19 (commit nelle previsioni). Dati sintetici,
nessuna rete.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd
import pytest

import main
from src.backtesting import _D_operator, architecture_vs_regime_D
from src.models.garch import garch_fold_losses, qlike_loss, run_garch_fold
from src.statistical_tests import build_dm_table, diebold_mariano_test


# ── R11 ──────────────────────────────────────────────────────────────────────

def test_dm_raw_pvalue_positive_and_unrounded():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.01, 210)
    good = y + rng.normal(0, 1e-4, 210)
    bad = np.zeros(210)
    r = diebold_mariano_test(y, good, bad)
    assert abs(r["dm_statistic"]) > 9           # 1 - cdf varrebbe 0 qui
    assert r["p_value_raw"] > 0.0
    assert r["p_value"] == round(r["p_value_raw"], 4)

    t = build_dm_table(y, {"buono": good})
    assert "p-value (raw)" in t.columns
    assert t.loc["buono", "p-value (raw)"] > 0.0


def test_holm_uses_raw_pvalues():
    # Arrotondati a 4 decimali entrambi varrebbero 0.0125 e il p Holm
    # sarebbe 0.025; con i valori esatti è 2 x 0.01249 = 0.02498.
    out, n = main._holm_correct({"A": {"m1": 0.01249, "m2": 0.01251}})
    assert n == 2
    assert out["A"]["m1"][0] == pytest.approx(0.02498)
    assert out["A"]["m2"][0] == pytest.approx(0.02498)
    out_nan, n_nan = main._holm_correct({"A": {"m1": np.nan, "m2": 0.001}})
    assert np.isnan(out_nan["A"]["m1"][0]) and not out_nan["A"]["m1"][1]


# ── R14 ──────────────────────────────────────────────────────────────────────

def _d_inputs(n=60, seed=1):
    rng = np.random.default_rng(seed)
    assets = ["A", "B", "C"]
    models = ["m1", "m2"]
    RG = np.empty((3, n), dtype=object)
    cycle = np.array(["stable", "normal", "volatile"] * (n // 3))
    RG[0] = cycle
    RG[1] = np.roll(cycle, 1)
    RG[2] = "normal"
    RG[2, :3] = "stable"          # C: solo 3 giorni stabili, cella non valida
    RG[2, 3:30] = "volatile"
    Y = {m: rng.normal(0, 0.01, (3, n)) for m in models}
    return Y, RG, models, assets


def test_per_asset_point_estimate_excludes_invalid_asset():
    Y, RG, models, assets = _d_inputs()
    pt = _D_operator(Y, RG, models, assets, np.arange(RG.shape[1]),
                     "per_asset", min_obs=5)
    assert pt["valid"] and pt["assets_used"] == ["A", "B"]
    assert pt["D"] == pytest.approx(np.mean([pt["D_i"]["A"], pt["D_i"]["B"]]))


def test_per_asset_replicate_keeps_point_estimate_assets():
    Y, RG, models, assets = _d_inputs()
    n = RG.shape[1]
    # Replica in cui i 3 giorni stabili di C sono ripetuti: C diventerebbe
    # valido, ma la replica deve restare la media su A e B.
    idx = np.concatenate([np.tile(np.arange(3), 4), np.arange(3, n)])[:n]
    free = _D_operator(Y, RG, models, assets, idx, "per_asset", min_obs=5)
    assert "C" in free["assets_used"]                 # il vecchio comportamento
    fixed = _D_operator(Y, RG, models, assets, idx, "per_asset", min_obs=5,
                        required_assets=["A", "B"])
    assert fixed["valid"] and fixed["assets_used"] == ["A", "B"]
    assert fixed["D"] == pytest.approx(np.mean([fixed["D_i"]["A"], fixed["D_i"]["B"]]))


def test_per_asset_replicate_discarded_if_required_asset_invalid():
    Y, RG, models, assets = _d_inputs()
    # Solo date in cui B è "normal": le celle di B non sono tutte valide.
    idx = np.where(RG[1] == "normal")[0]
    rep = _D_operator(Y, RG, models, assets, idx, "per_asset", min_obs=5,
                      required_assets=["A", "B"])
    assert not rep["valid"] and np.isnan(rep["D"])


def test_bootstrap_per_asset_reports_point_assets():
    dates = pd.bdate_range("2025-01-01", periods=90)
    rng = np.random.default_rng(3)
    cyc = ["stable", "normal", "volatile"] * 30
    asset_data = {}
    for k, a in enumerate(["A", "B"]):
        asset_data[a] = {
            "y": {m: pd.Series(rng.normal(0, 0.01, 90), index=dates)
                  for m in ("m1", "m2")},
            "dates": dates,
            "regimes": pd.Series(np.roll(cyc, k), index=dates),
        }
    res = architecture_vs_regime_D(asset_data, aggregation="per_asset",
                                   block_length=5, n_boot=50, seed=0)
    assert res["Assets Used"] == "A,B"
    assert 0.0 <= res["Frac. Discarded"] <= 1.0


# ── R4 ───────────────────────────────────────────────────────────────────────

def test_align_fold_output_accepts_exact_match():
    d = pd.bdate_range("2025-03-03", periods=21)
    y = np.linspace(-0.01, 0.01, 21)
    out = main._align_fold_output("m", d, y, y * 0.5, d, y)
    assert out.equals(d)


@pytest.mark.parametrize("case", ["short", "shifted", "wrong_y", "tiny_relative"])
def test_align_fold_output_rejects_mismatch(case):
    d = pd.bdate_range("2025-03-03", periods=21)
    y = np.linspace(-0.01, 0.01, 21)
    if case == "short":
        args = (d[1:], y[1:], y[1:])
    elif case == "shifted":
        args = (d + pd.offsets.BDay(1), y, y)
    elif case == "tiny_relative":
        # differenza relativa 1e-7: accettata da una tolleranza relativa di
        # default, rifiutata dalla tolleranza assoluta 1e-12 (RC10)
        args = (d, y * (1 + 1e-7) + 1e-9, y)
    else:
        args = (d, y + 0.001, y)
    with pytest.raises(ValueError):
        main._align_fold_output("m", *args, d, y)


# ── R19 ──────────────────────────────────────────────────────────────────────

def test_predictions_commit_check(monkeypatch, capsys):
    monkeypatch.setattr(main, "_git_commit", lambda: "abc")
    monkeypatch.setattr(main, "_git_dirty", lambda: False)
    main._check_predictions_commit(pd.DataFrame({"Commit": ["abc"] * 3}), "p.csv")
    with pytest.raises(RuntimeError):
        main._check_predictions_commit(pd.DataFrame({"Commit": ["abc", "def"]}), "p.csv")
    with pytest.raises(RuntimeError):
        main._check_predictions_commit(pd.DataFrame({"Commit": ["def"]}), "p.csv")
    main._check_predictions_commit(pd.DataFrame({"y": [1.0]}), "old.csv")
    assert "non ha la colonna Commit" in capsys.readouterr().out


# ── R5 ───────────────────────────────────────────────────────────────────────

def test_garch_rebuilt_from_saved_forecasts_matches_live(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    n = 400
    dates = pd.bdate_range("2023-01-02", periods=n)
    r = rng.standard_t(6, n) * 0.008
    df = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(r)), "log_return": r},
                      index=dates)
    folds = main.get_walk_forward_folds(len(df), n_folds=2)

    live, fc_rows = [], []
    for dist in main.GARCH_DISTS:
        for fi, (tr, te) in enumerate(folds):
            gd, gr, gf, gp = run_garch_fold(df, tr, te, dist=dist)
            r_fold = df["log_return"].values[te]
            live.append({"Dist": dist, "Fold": fi + 1, **gp,
                         **garch_fold_losses(gf, r_fold)})
            fc_rows += [{"Asset": "X", "Dist": dist, "Fold": fi + 1, "Date": d,
                         "sigma_hat": s, "r": rr}
                        for d, s, rr in zip(gd, gf, r_fold)]
    pd.DataFrame(fc_rows).to_csv(tmp_path / "garch_forecasts_X.csv", index=False)
    monkeypatch.setattr(main, "RESULTS_DIR", str(tmp_path))

    rows, dm = main._garch_from_disk("X", df, n_folds=2)
    rebuilt = pd.DataFrame(rows).set_index(["Dist", "Fold"])
    live = pd.DataFrame(live).set_index(["Dist", "Fold"])
    for c in ["alpha", "beta", "persistence", "loss_RMSE",
              "loss_RMSE_scaled_proxy", "QLIKE"]:
        np.testing.assert_allclose(rebuilt[c].values, live[c].values, atol=1e-5)
    assert dm is not None and dm["N"] == sum(len(te) for _, te in folds)

    fc = pd.DataFrame(fc_rows)
    ql_n = qlike_loss(fc[fc.Dist == "normal"].sigma_hat.values, fc[fc.Dist == "normal"].r.values)
    assert dm["Mean QLIKE normal"] == pytest.approx(float(np.mean(ql_n)))


# ── R7, R8 con --resume ──────────────────────────────────────────────────────

def test_model_params_rebuilt_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(main, "_git_commit", lambda: "abc")
    monkeypatch.setattr(main, "_git_dirty", lambda: False)
    pd.DataFrame([
        {"Asset": "X", "Model": "ARMA", "Fold": 1,
         "Params": '{"aic": -1.0, "order": [1, 0, 1]}', "Commit": "abc"},
        {"Asset": "X", "Model": "CNN LSTM", "Fold": 1,
         "Params": '{"best_epoch": 12, "seed": 42}', "Commit": "abc"},
    ]).to_csv(tmp_path / "model_params_X.csv", index=False)
    p_rows, e_rows = main._model_params_from_disk("X")
    assert len(p_rows) == 2 and "Commit" not in p_rows[0]
    assert e_rows == [{"Asset": "X", "Model": "CNN_LSTM", "Fold": 1, "Best Epoch": 12}]

    monkeypatch.setattr(main, "_git_commit", lambda: "zzz")
    with pytest.raises(RuntimeError):
        main._model_params_from_disk("X")
    assert main._model_params_from_disk("Y") == ([], [])


# ── Integrazione: le correzioni sono usate dalla pipeline (rilievo RC1) ─────
# Pipeline vera su dati sintetici, con due modelli veloci al posto dei 13 e
# il GARCH sostituito da una previsione costante. Le mutazioni usate per
# verificare che questi test proteggano le correzioni sono elencate in
# PIANO_CORREZIONI.md (rilettura finale).

def _pipeline_df(n=520, seed=11):
    dates = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.012, n)
    return pd.DataFrame({"Close": 100 * np.exp(np.cumsum(r)), "log_return": r},
                        index=dates)


def _fake_arma(df, train_idx, test_idx):
    y = df["log_return"].values[test_idx]
    prev = df["log_return"].values[np.asarray(test_idx) - 1]
    return df.index[test_idx], y, 0.3 * prev, {"order": (1, 0, 0), "aic": -1.0}


def _fake_garch_params(df, train_idx, dist="normal"):
    return {"alpha": 0.05, "beta": 0.9, "persistence": 0.95,
            "nu": 6.0 if dist == "t" else None}


def _fake_garch(df, train_idx, test_idx, dist="normal"):
    r = df["log_return"].values[test_idx]
    sd = float(np.std(df["log_return"].values[train_idx]))
    return (df.index[test_idx], np.abs(r), np.full(len(r), sd),
            _fake_garch_params(df, train_idx, dist))


def _setup_pipeline(tmp_path, monkeypatch, models, commit="c0ffee"):
    df = _pipeline_df()
    monkeypatch.setattr(main, "METALS", {"Gold": "GC=F"})
    monkeypatch.setattr(main, "load_and_preprocess",
                        lambda name, ticker, start=None, end=None: df.copy())
    monkeypatch.setattr(main, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(main, "_git_commit", lambda: commit)
    monkeypatch.setattr(main, "_git_dirty", lambda: False)
    monkeypatch.setattr(main, "MODELS", models)
    monkeypatch.setattr(main, "run_garch_fold", _fake_garch)
    monkeypatch.setattr(main, "fit_garch_params", _fake_garch_params)
    for fn in ("plot_returns_forecast", "plot_price_comparison",
               "plot_all_models_prices", "plot_garch_volatility"):
        if hasattr(main, fn):
            monkeypatch.setattr(main, fn, lambda *a, **k: None)
    return df


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Una esecuzione completa condivisa dai test di integrazione."""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("pipeline")
    _setup_pipeline(tmp, mp, [("Random_Walk", None), ("ARMA", _fake_arma)])
    try:
        main.run_pipeline(resume=False, smoke=False)
    finally:
        mp.undo()
    return tmp


def test_pipeline_writes_commit_params_and_raw_holm(pipeline_run):
    pred = pd.read_csv(pipeline_run / "predictions_Gold.csv")
    assert set(pred["Commit"]) == {"c0ffee"}                         # R19
    mp_ = pd.read_csv(pipeline_run / "model_params.csv")
    assert set(mp_["Model"]) == {"ARMA"} and len(mp_) == main.N_FOLDS   # R7, R8
    assert json.loads(mp_["Params"].iloc[0])["order"] == [1, 0, 0]
    dm = pd.read_csv(pipeline_run / "DM_test_Gold.csv", index_col=0)
    # una sola ipotesi: p Holm = p grezzo NON arrotondato (R11)
    assert dm.loc["ARMA", "p Holm"] == pytest.approx(dm.loc["ARMA", "p-value (raw)"], abs=1e-15)
    assert dm.loc["ARMA", "p Holm"] != round(dm.loc["ARMA", "p Holm"], 4)


def test_pipeline_executable_benchmarks_and_d(pipeline_run):
    b = pd.read_csv(pipeline_run / "rcs_benchmarks.csv")
    assert set(b["Execution"]) >= {"close", "lag1"}                    # K13
    bm = pd.read_csv(pipeline_run / "rcs_benchmarks_by_model.csv")
    assert set(bm["Execution"]) >= {"close", "lag1"}
    assert (pipeline_run / "architecture_vs_regime_D_exec.csv").exists()
    assert (pipeline_run / "two_way_decomposition_exec.csv").exists()


def test_resume_rebuilds_garch_params_and_checks_commit(pipeline_run, tmp_path, monkeypatch):
    import shutil
    for f in ("run_manifest.json", "predictions_Gold.csv", "model_params_Gold.csv",
              "garch_forecasts_Gold.csv"):
        shutil.copy(pipeline_run / f, tmp_path / f)
    _setup_pipeline(tmp_path, monkeypatch, [("Random_Walk", None), ("ARMA", _fake_arma)])

    def _no_training(*a, **k):
        raise AssertionError("modello riaddestrato durante --resume")
    monkeypatch.setattr(main, "MODELS", [("Random_Walk", None), ("ARMA", _no_training)])
    main.run_pipeline(resume=True, smoke=False)
    for f in ("garch_volatility.csv", "model_params.csv"):          # R5, R7, R8
        a = pd.read_csv(pipeline_run / f).sort_values(list(pd.read_csv(pipeline_run / f).columns[:3]))
        b = pd.read_csv(tmp_path / f).sort_values(list(a.columns[:3]))
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                      check_exact=False, atol=1e-12)

    # file prodotti da un altro commit: ripresa rifiutata (R19)
    monkeypatch.setattr(main, "_git_commit", lambda: "deadbeef")
    m = json.loads((tmp_path / "run_manifest.json").read_text())
    m["commit"] = "deadbeef"
    (tmp_path / "run_manifest.json").write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="rifiutato"):
        main.run_pipeline(resume=True, smoke=False)


def test_pipeline_stops_on_model_error(tmp_path, monkeypatch):            # R18
    def _broken(df, train_idx, test_idx):
        raise RuntimeError("modello rotto")
    _setup_pipeline(tmp_path, monkeypatch, [("Random_Walk", None), ("ARMA", _broken)])
    with pytest.raises(RuntimeError, match="modello rotto"):
        main.run_pipeline(resume=False, smoke=False)


def test_pipeline_rejects_misaligned_forecasts(tmp_path, monkeypatch):    # R4
    def _shifted(df, train_idx, test_idx):
        d, y, p, prm = _fake_arma(df, train_idx, test_idx)
        return d + pd.offsets.BDay(1), y, p, prm
    _setup_pipeline(tmp_path, monkeypatch, [("Random_Walk", None), ("ARMA", _shifted)])
    with pytest.raises(ValueError, match="date"):
        main.run_pipeline(resume=False, smoke=False)


def test_bootstrap_replicates_use_point_assets(monkeypatch):              # R14
    from src import backtesting as bk
    calls = []
    real = bk._D_operator

    def spy(*args, **kwargs):
        calls.append(kwargs.get("required_assets"))
        return real(*args, **kwargs)
    monkeypatch.setattr(bk, "_D_operator", spy)
    dates = pd.bdate_range("2025-01-01", periods=90)
    rng = np.random.default_rng(3)
    cyc = ["stable", "normal", "volatile"] * 30
    asset_data = {a: {"y": {m: pd.Series(rng.normal(0, 0.01, 90), index=dates)
                            for m in ("m1", "m2")},
                      "dates": dates,
                      "regimes": pd.Series(np.roll(cyc, k), index=dates)}
                  for k, a in enumerate(["A", "B"])}
    bk.architecture_vs_regime_D(asset_data, aggregation="per_asset",
                                block_length=5, n_boot=20, seed=0)
    assert calls[0] is None                       # stima puntuale
    assert all(c == ["A", "B"] for c in calls[1:]) and len(calls) == 21


# ── K13: benchmark della RCS con la stessa tempistica della strategia ──────

@pytest.mark.parametrize("execution", ["lag1", "open"])
def test_exposure_benchmarks_follow_execution(execution):
    from src.backtesting import (exposure_benchmarks, regime_conditional_strategy,
                                 run_backtest)
    rng = np.random.default_rng(7)
    n = 120
    dates = pd.bdate_range("2025-07-01", periods=n)
    reg = pd.Series(rng.choice(["stable", "normal", "volatile"], size=n), index=dates)
    r = rng.normal(0.001, 0.01, n)
    alt = rng.normal(0.001, 0.01, n) if execution == "open" else None
    a = regime_conditional_strategy(np.ones(n), dates, reg)
    rcs_long = run_backtest(r, np.ones(n), position=a, execution=execution, alt_returns=alt)
    summary, per_model, _ = exposure_benchmarks(
        r, dates, reg, asset_name=None, rcs_results={"Always long": rcs_long},
        n_rand=30, execution=execution, alt_returns=alt)
    # il long filtrato è eseguito come la RCS: stessa serie, stessa identità
    assert summary["Execution"] == execution
    assert summary["Sharpe Long Filtered"] == pytest.approx(rcs_long["Sharpe Ratio"])
    assert per_model.loc["Always long", "Identical to Long Filtered"]
    close = exposure_benchmarks(r, dates, reg, asset_name=None,
                                rcs_results={"Always long": rcs_long}, n_rand=30)[0]
    assert close["Sharpe Long Filtered"] != pytest.approx(summary["Sharpe Long Filtered"])


def test_git_dirty_sees_untracked_code(tmp_path, monkeypatch):
    import subprocess
    out = {"value": ""}
    monkeypatch.setattr(subprocess, "check_output",
                        lambda *a, **k: out["value"].encode())
    out["value"] = "?? CODE/results/metrics.csv\n"
    assert main._git_dirty() is False
    out["value"] = "?? CODE/tools/recompute_post_run.py\n"
    assert main._git_dirty() is True
    out["value"] = " M CODE/main.py\n"
    assert main._git_dirty() is True



# ── Rilettura finale: test che proteggono le correzioni K13 e R4/R8/R9/R19 ──

def _concat_two_models(n=120, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-07-01", periods=n)
    y = rng.normal(0.0005, 0.01, n)
    pred = 0.4 * np.r_[0.0, y[:-1]] + rng.normal(0, 0.002, n)
    reg = pd.Series(rng.choice(["stable", "normal", "volatile"], size=n), index=dates)
    concat = {"Random_Walk": {"y_true": y, "y_pred": np.zeros(n), "dates": dates},
              "ARMA": {"y_true": y, "y_pred": pred, "dates": dates}}
    alt = {"open": rng.normal(0.0005, 0.011, n)}
    return concat, reg, alt


def test_alpha_vs_filtered_long_uses_same_execution(monkeypatch):
    """L'intercetta della RCS è stimata sul long FILTRATO con la stessa
    esecuzione della strategia (K13, K14): ricalcolata qui senza le
    funzioni di main."""
    from src.backtesting import estimate_alpha, regime_conditional_strategy
    monkeypatch.setattr(main, "block_bootstrap_ci", lambda *a, **k: {})
    concat, reg, alt = _concat_two_models()
    out = main._run_backtests_for_asset(concat, reg, "Gold", alt)
    c = concat["ARMA"]
    a = regime_conditional_strategy(np.ones(len(c["y_true"])), c["dates"], reg)
    for execution in ("close", "lag1", "open"):
        bt_rcs = out["by_exec"][execution][1]["ARMA"]
        a_exec = np.r_[0.0, a[:-1]] if execution == "lag1" else a
        R = alt["open"] if execution == "open" else np.expm1(c["y_true"])
        fl = (a_exec * R)[bt_rcs["Kept"]]
        exp = estimate_alpha(bt_rcs["Net PnL"], fl)
        assert bt_rcs["Alpha vs Long Filtered (daily)"] == pytest.approx(exp["Alpha (daily)"])
        assert bt_rcs["Alpha vs Long Filtered p-value"] == pytest.approx(exp["Alpha p-value"])
        passive = estimate_alpha(bt_rcs["Net PnL"], R[bt_rcs["Kept"]])
        assert bt_rcs["Alpha vs Long Filtered (daily)"] != pytest.approx(passive["Alpha (daily)"])
    assert set(out["benchmarks"]) == {"close", "lag1", "open"}


@pytest.mark.parametrize("execution", ["close", "lag1", "open"])
def test_random_signal_and_empirical_p_follow_execution(execution):
    """Segnale casuale eseguito come la RCS e p empirico (1 + #)/(n + 1)."""
    from src.backtesting import (exposure_benchmarks, regime_conditional_strategy,
                                 run_backtest)
    concat, reg, alt = _concat_two_models(n=80, seed=9)
    y, dates = concat["ARMA"]["y_true"], concat["ARMA"]["dates"]
    a = regime_conditional_strategy(np.ones(len(y)), dates, reg)
    altr = alt["open"] if execution == "open" else None
    # strategia con previsione perfetta: batte ogni segnale casuale
    perfect = run_backtest(y, y, position=a * np.sign(y), execution=execution,
                           alt_returns=altr if altr is None else np.sign(y) * np.abs(altr))
    n_rand, seed = 25, 3
    summary, per_model, _ = exposure_benchmarks(
        y, dates, reg, None, {"perfect": perfect}, n_rand=n_rand, seed=seed,
        execution=execution, alt_returns=altr)
    rng = np.random.default_rng(seed)
    sr = np.array([run_backtest(y, e, dates=dates, cost_per_side=0.0001, position=e * a,
                                execution=execution, alt_returns=altr)["Sharpe Ratio"]
                   for e in rng.choice([-1.0, 1.0], size=(n_rand, len(y)))])
    assert summary["Sharpe Random Filtered Mean"] == pytest.approx(round(float(sr.mean()), 4))
    k = int(np.sum(sr >= perfect["Sharpe Ratio"]))
    assert per_model.loc["perfect", "p_emp vs Random Filtered"] == \
        pytest.approx(round((1 + k) / (n_rand + 1), 4))
    assert per_model.loc["perfect", "p_emp vs Random Filtered"] >= 1 / (n_rand + 1) - 1e-4


def _bt_all_for_d():
    dates = pd.bdate_range("2025-07-01", periods=30)
    kept = np.ones(30, dtype=bool)
    reg = pd.Series(["stable", "normal", "volatile"] * 10, index=dates)

    def series(v):
        return {"m1": {"Net PnL": np.full(30, v), "Kept": kept}}
    bt_all = {e: {"static": {"Gold": series(k), "Platinum": series(10 + k)}}
              for k, e in enumerate(["close", "lag1", "open"])}
    asset_data = {a: {"dates": dates, "regimes": reg} for a in ("Gold", "Platinum")}
    return bt_all, asset_data, reg


def test_d_executable_uses_reference_series_and_decision_label():
    """D eseguibile: serie della variante di riferimento (open per l'oro,
    lag1 per il platino) e, con lag1, etichetta del giorno di decisione."""
    bt_all, asset_data, reg = _bt_all_for_d()
    out = main._asset_data_executable(bt_all, asset_data)
    assert out["Gold"]["execution"] == "open" and out["Platinum"]["execution"] == "lag1"
    assert np.all(out["Gold"]["y"]["m1"] == 2)            # serie open, non close
    assert np.all(out["Platinum"]["y"]["m1"] == 11)       # serie lag1, non close
    pd.testing.assert_series_equal(out["Gold"]["regimes"], reg)
    pd.testing.assert_series_equal(out["Platinum"]["regimes"],
                                   reg.shift(1).fillna("normal"))


def test_executable_fallback_is_per_asset():
    """Se l'apertura manca per un solo asset, quell'asset ripiega su lag1
    e gli altri restano su open (prima: KeyError)."""
    from src.backtesting import executable_check
    bt_all, asset_data, _ = _bt_all_for_d()
    bt_all["open"]["static"].pop("Platinum")
    bt_all["close"]["static"]["Silver"] = bt_all["close"]["static"]["Gold"]
    bt_all["lag1"]["static"]["Silver"] = bt_all["lag1"]["static"]["Gold"]
    asset_data["Silver"] = asset_data["Gold"]
    out = main._asset_data_executable(bt_all, asset_data)
    assert out["Silver"]["execution"] == "lag1" and out["Gold"]["execution"] == "open"
    for e in bt_all:
        bt_all[e]["rcs"] = bt_all[e]["static"]
    for b in bt_all.values():
        for a in b["static"].values():
            a["m1"].update({"Sharpe Ratio": 0.0, "Alpha (daily)": 0.0,
                            "Alpha p Holm": 1.0, "Alpha p-value": 1.0})
    ex = executable_check(bt_all)
    assert set(ex.loc[ex.Asset == "Silver", "Executable Variant"]) == {"lag1"}


def test_code_version_marks_uncommitted_code(monkeypatch):
    monkeypatch.setattr(main, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(main, "_git_dirty", lambda: True)
    assert main._code_version() == "abc123-dirty"
    monkeypatch.setattr(main, "_git_dirty", lambda: False)
    assert main._code_version() == "abc123"


def test_predictions_saved_before_garch_with_code_version(tmp_path, monkeypatch):
    """Previsioni e parametri sono scritti prima del GARCH (un errore del
    GARCH non fa perdere l'asset) e la colonna Commit porta il suffisso
    -dirty se il codice non è committato."""
    _setup_pipeline(tmp_path, monkeypatch, [("Random_Walk", None), ("ARMA", _fake_arma)])
    monkeypatch.setattr(main, "_git_dirty", lambda: True)

    def _garch_down(*a, **k):
        raise RuntimeError("GARCH non disponibile")
    monkeypatch.setattr(main, "run_garch_fold", _garch_down)
    with pytest.raises(RuntimeError, match="GARCH"):
        main.run_pipeline(resume=False, smoke=False)
    pred = pd.read_csv(tmp_path / "predictions_Gold.csv")
    assert set(pred["Commit"]) == {"c0ffee-dirty"}
    assert (tmp_path / "model_params_Gold.csv").exists()


def test_run_partial_rejects_misaligned_forecasts(tmp_path, monkeypatch):    # R4
    _setup_pipeline(tmp_path, monkeypatch, [])

    def _shifted(df, train_idx, test_idx, cross_asset_dfs=None, seed=None):
        d, y, p, prm = _fake_arma(df, train_idx, test_idx)
        return d + pd.offsets.BDay(1), y, p, prm
    with pytest.raises(ValueError, match="date"):
        main.run_partial("x", [("ARMA", _shifted)], seed=1)


def _robustness_dir(tmp_path, tcn31_shift=0, commit_col="c0ffee", dirty=False):
    dates = pd.bdate_range("2025-07-01", periods=40)
    rng = np.random.default_rng(2)
    y = rng.normal(0, 0.01, 40)

    def pred(model, shift=0, commit=commit_col):
        d = dates + pd.offsets.BDay(shift)
        df = pd.DataFrame({"Date": d, "Model": model, "Fold": np.repeat([0, 1], 20),
                           "y_true": y, "y_pred": 0.1 * y + rng.normal(0, 0.001, 40),
                           "Seed": 1})
        if commit is not None:
            df["Commit"] = commit
        return df
    pd.concat([pred("TCN"), pred("LSTM")]).to_csv(tmp_path / "predictions_Gold.csv", index=False)
    pred("LSTM").to_csv(tmp_path / "predictions_dl_seed1_Gold.csv", index=False)
    pred("TCN-31", shift=tcn31_shift).to_csv(tmp_path / "predictions_tcn31_Gold.csv", index=False)
    for name in ("run_manifest.json", "run_manifest_dl_seed1.json", "run_manifest_tcn31.json"):
        (tmp_path / name).write_text(json.dumps({"commit": "c0ffee", "commit_dirty": dirty}))


@pytest.mark.parametrize("case", ["ok", "tcn31_shifted", "dirty_column", "dirty_manifest"])
def test_summarize_robustness_checks(tmp_path, monkeypatch, case):           # RC4
    import tools.summarize_robustness as sr
    _robustness_dir(tmp_path, tcn31_shift=1 if case == "tcn31_shifted" else 0,
                    commit_col="c0ffee-dirty" if case == "dirty_column" else "c0ffee",
                    dirty=case == "dirty_manifest")
    monkeypatch.setattr(sr, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(sr, "METALS", {"Gold": "GC=F"})
    if case == "ok":
        sr.main()
        assert (tmp_path / "tcn31_vs_tcn.csv").exists()
    elif case == "tcn31_shifted":
        with pytest.raises(ValueError, match="TCN-31"):
            sr.main()
    else:
        with pytest.raises(SystemExit):
            sr.main()
