import os
import sys
import json
import platform
import subprocess
import datetime
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.stats.multitest import multipletests

from src.preprocessing     import (METALS, load_and_preprocess, reconstruct_prices,
                                    load_ohlc_snapshot, validate_ohlc,
                                    open_to_open_returns)
from src.walk_forward      import (get_walk_forward_folds, aggregate_fold_results,
                                    validate_folds, export_folds_csv, N_FOLDS)
from src.utils             import compute_metrics, save_results, set_global_seed
from src.regime            import classify_regimes, split_by_regime, regime_summary
from src.statistical_tests import (random_walk_forecast,
                                    build_dm_table,
                                    build_adf_table,
                                    da_test_hac, da_test_centred_bootstrap,
                                    dm_test_loss, MIN_ACTIVE_DA)
from src.backtesting       import (run_backtest, regime_conditional_strategy,
                                    backtest_summary_table, block_bootstrap_ci,
                                    architecture_vs_regime_D, estimate_alpha,
                                    two_way_decomposition, exposure_benchmarks,
                                    performance_by_regime)
from src.visualization     import (
    plot_returns_forecast, plot_price_comparison,
    plot_all_models_prices, plot_garch_volatility,
    plot_heatmap, plot_ranking, plot_dashboard,
    plot_all_models_regime, plot_dm_heatmap, plot_rolling_vol_regimes,
    plot_hit_rate_heatmap, plot_sharpe_heatmap,
)

from src.futures_calendar  import post_roll_mask, roll_window_mask
from src                   import metrics_econ
from src.models.arima     import run_arima_fold
from src.models.garch     import run_garch_fold, qlike_loss, GARCH_DISTS
from src.models.ml_models import (run_rf_fold, run_xgboost_fold,
                                   run_gbm_fold, run_svr_fold, run_dt_fold)
from src.models import dl_models
from src.models.dl_models import (run_lstm_fold, run_gru_fold,
                                   run_bilstm_fold, run_transformer_fold,
                                   run_tcn_fold, run_cnn_lstm_fold)

START = "2010-01-01"
END   = "2026-05-01"   # aggiornato — include dati fino ad oggi

# Riproducibilità/Colab: directory di output configurabile (default
# CODE/results quando lanciato da CODE/, così sia il run locale sia quello
# su Colab con Drive montato usano lo stesso codice — vedi colab_run.ipynb.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

_MANIFEST_PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost",
                       "statsmodels", "arch", "tensorflow", "keras", "yfinance"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _package_versions() -> dict:
    from importlib import metadata
    versions = {}
    for pkg in _MANIFEST_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def _gpu_info() -> str:
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return [g.name for g in gpus] if gpus else "none (CPU)"
    except Exception:
        return "unknown"


def _write_run_manifest(seed: int = 42, smoke: bool = False) -> dict:
    """
    results/run_manifest.json — commit git, versioni pacchetti, GPU,
    versione Python, seed, data. --resume confronta il commit corrente con
    quello salvato qui (vedi _check_resume_commit) per rifiutare un resume
    contro CSV prodotti da un codice diverso.
    """
    manifest = {
        "commit":          _git_commit(),
        "python_version":  sys.version,
        "platform":        platform.platform(),
        "packages":        _package_versions(),
        "gpu":             _gpu_info(),
        "seed":            seed,
        "date":            datetime.datetime.now().isoformat(),
        "smoke":           smoke,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _check_resume_commit() -> None:
    """
    --resume accetta solo CSV prodotti dallo stesso commit git salvato in
    run_manifest.json — dopo una correzione metodologica i CSV di un
    commit precedente non sono più validi (vedi CLAUDE.md/report del
    relatore). Nessun manifest trovato -> resume rifiutato (run precedente
    antecedente a questa funzionalità, provenienza non verificabile).
    """
    manifest_path = f"{RESULTS_DIR}/run_manifest.json"
    if not os.path.exists(manifest_path):
        raise RuntimeError(
            f"--resume richiesto ma {manifest_path} non esiste: non posso "
            f"verificare da quale commit provengano i CSV in {RESULTS_DIR}. "
            f"Esegui senza --resume, oppure crea un manifest valido."
        )
    with open(manifest_path) as f:
        old_manifest = json.load(f)

    current_commit = _git_commit()
    old_commit = old_manifest.get("commit")
    if current_commit is None or old_commit != current_commit:
        raise RuntimeError(
            f"--resume rifiutato: i CSV in {RESULTS_DIR} provengono dal "
            f"commit {old_commit!r}, ma HEAD è {current_commit!r}. Dopo una "
            f"correzione metodologica i CSV del commit precedente non sono "
            f"più validi — esegui senza --resume per rigenerarli."
        )

MODELS = [
    ("Random_Walk",       None),
    ("ARMA",              run_arima_fold),   # C1: label "ARMA" — classe statsmodels resta ARIMA(d=0)
    ("Decision_Tree",     run_dt_fold),
    ("Random_Forest",     run_rf_fold),
    ("Gradient_Boosting", run_gbm_fold),
    ("XGBoost",           run_xgboost_fold),
    ("SVR",               run_svr_fold),
    ("LSTM",              run_lstm_fold),
    ("GRU",               run_gru_fold),
    ("BiLSTM",            run_bilstm_fold),
    ("Transformer",       run_transformer_fold),
    ("TCN",               run_tcn_fold),
    ("CNN_LSTM",          run_cnn_lstm_fold),
]

DL_MODELS = {"LSTM", "GRU", "BiLSTM", "Transformer", "TCN", "CNN_LSTM"}
ML_MODELS = {"Decision_Tree", "Random_Forest", "Gradient_Boosting",
             "XGBoost", "SVR"}


def p(folder, filename):
    path = f"{RESULTS_DIR}/plots/{folder}/{filename}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _resume_marker_path(metal_name):
    """
    Unico marker di "asset già completato" per --resume:
    predictions_<asset>.csv, scritto a fine asset dal percorso live.
    backtest_<asset>.csv/backtest_rcs_<asset>.csv NON sono più marker
    validi — dopo B9 vengono scritti solo a fine run, dopo la correzione
    di Holm su tutti gli asset (vedi la sezione dedicata in run_pipeline),
    quindi non esistono ancora quando un singolo asset è "completato".
    _reconstruct_asset_from_disk ricalcola comunque backtest, DM e
    statistica D da predictions_<asset>.csv, senza approssimazioni.
    """
    return f"{RESULTS_DIR}/predictions_{metal_name}.csv"


# Esecuzioni del backtest: "close" è la specificazione principale
# (idealizzazione close-to-close, dichiarata in tesi). Sensitivity, con
# proprie famiglie di Holm separate: "lag1" (sempre) e "open" (commento K13),
# "rolled" (serie realmente rollata dai contratti singoli, commento K03),
# questi ultimi due solo se i dati esterni superano i controlli.
SENSITIVITY_EXECUTIONS = ("lag1", "open", "rolled")


def _open_returns_for_asset(metal_name: str, df: pd.DataFrame,
                            oos_dates: pd.DatetimeIndex) -> tuple:
    """
    Rendimenti open-to-open per la variante "open", dallo snapshot OHLC
    separato (CODE/data/<ticker>_ohlc.csv, vedi tools/fetch_ohlc.py).
    Ritorna (array allineato a oos_dates oppure None, esito dei controlli).
    Nessun download: se lo snapshot manca o non supera validate_ohlc, la
    variante "open" non viene calcolata e l'esito finisce nel manifest.
    """
    ohlc = load_ohlc_snapshot(METALS[metal_name])
    report = validate_ohlc(ohlc, df, oos_dates)
    if not report["ok"]:
        return None, report
    return open_to_open_returns(ohlc, oos_dates), report


def _rolled_returns_for_asset(metal_name: str,
                              oos_dates: pd.DatetimeIndex) -> tuple:
    """
    Rendimenti della serie realmente rollata (CODE/data/<ticker>_rolled.csv,
    scritto da tools/build_rolled_series.py). Ritorna (array allineato a
    oos_dates oppure None, esito). Richiede che ogni data OOS abbia un
    rendimento del contratto detenuto.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        f"{METALS[metal_name]}_rolled.csv")
    if not os.path.exists(path):
        return None, {"ok": False, "reason": "serie rollata assente"}
    rolled = pd.read_csv(path, index_col=0, parse_dates=True)
    r = rolled["R_rolled"].reindex(pd.DatetimeIndex(oos_dates))
    if r.isna().any():
        return None, {"ok": False,
                      "reason": f"serie rollata incompleta: {int(r.isna().sum())} date OOS mancanti"}
    return r.values.astype(float), {"ok": True, "reason": "ok"}


def _backtest_one(c: dict, metal_name: str, execution: str, alt,
                  position=None) -> dict:
    """run_backtest + intervalli bootstrap a blocchi."""
    bt = run_backtest(c["y_true"], c["y_pred"], dates=c["dates"],
                      asset_name=metal_name, cost_per_side=0.0001,
                      position=position, execution=execution,
                      alt_returns=alt.get(execution))
    bt.update(block_bootstrap_ci(bt["Net PnL"], bt["Hit"], bt["Active"],
                                 block_length=10, n_boot=2000, seed=42))
    return bt


def _benchmark_return(c: dict, execution: str, alt, kept) -> np.ndarray:
    """b_{t+1} = rendimento del long passivo con la stessa esecuzione."""
    if execution in ("open", "rolled"):
        return np.asarray(alt[execution], dtype=float)[kept]
    return np.expm1(c["y_true"])[kept]


def _run_backtests_for_asset(concat: dict, regimes_origin: pd.Series,
                              metal_name: str, alt: dict = None) -> dict:
    """
    Backtest statico + Regime-Conditional Strategy per tutti i modelli di
    un asset già concatenati (concat[model_name] = {y_true, y_pred,
    dates}) — condivisa dal percorso live e da quello di resume, così i
    risultati sono identici. Random Walk esclusa dalla RCS, dall'alpha e
    dai test DA (posizione sempre nulla).

    alt = {"open": array | None, "rolled": array | None}: rendimenti
    alternativi; un'esecuzione è calcolata solo se il suo array esiste.
    Per ogni esecuzione: bootstrap, alpha HAC rispetto al long passivo con
    la stessa esecuzione, test DA HAC e bootstrap centrato. Solo per
    "close": alpha della RCS rispetto al long filtrato a_t R_{t+1} e
    benchmark della regola di esposizione (commento K14).

    Ritorna {"by_exec": {execution: (bt_results, rcs_results)},
             "benchmarks": (summary, per_model_df), "perf_by_regime": df}.
    """
    alt = {k: v for k, v in (alt or {}).items() if v is not None}
    executions = ["close", "lag1"] + [e for e in ("open", "rolled") if e in alt]
    by_exec = {}
    for execution in executions:
        bt_results, rcs_results = {}, {}
        for model_name, c in concat.items():
            label = model_name.replace("_", " ")
            bt = _backtest_one(c, metal_name, execution, alt)
            if model_name != "Random_Walk":
                b = _benchmark_return(c, execution, alt, bt["Kept"])
                bt.update(estimate_alpha(bt["Net PnL"], b))
                bt.update(da_test_hac(bt["Hit"], bt["Active"]))
                bt.update(da_test_centred_bootstrap(bt["Hit"], bt["Active"]))
            bt_results[label] = bt

            if model_name != "Random_Walk":
                pos = regime_conditional_strategy(c["y_pred"], c["dates"], regimes_origin)
                bt_rcs = _backtest_one(c, metal_name, execution, alt, position=pos)
                b = _benchmark_return(c, execution, alt, bt_rcs["Kept"])
                bt_rcs.update(estimate_alpha(bt_rcs["Net PnL"], b))
                bt_rcs.update(da_test_hac(bt_rcs["Hit"], bt_rcs["Active"]))
                bt_rcs.update(da_test_centred_bootstrap(bt_rcs["Hit"], bt_rcs["Active"]))
                if execution == "close":
                    a_t = regime_conditional_strategy(
                        np.ones(len(c["y_true"])), c["dates"], regimes_origin)
                    al = estimate_alpha(bt_rcs["Net PnL"], a_t * np.expm1(c["y_true"]))
                    bt_rcs["Alpha vs Long Filtered (daily)"] = al["Alpha (daily)"]
                    bt_rcs["Alpha vs Long Filtered p-value"] = al["Alpha p-value"]
                rcs_results[label] = bt_rcs
        by_exec[execution] = (bt_results, rcs_results)

    c_rw = concat["Random_Walk"]
    bt_close, rcs_close = by_exec["close"]
    summary, per_model, _ = exposure_benchmarks(
        c_rw["y_true"], c_rw["dates"], regimes_origin, metal_name, rcs_close,
        n_rand=1000, seed=42)
    perf = pd.concat([
        performance_by_regime({k: v for k, v in bt_close.items() if k != "Random Walk"},
                              c_rw["dates"], regimes_origin, "static"),
        performance_by_regime(rcs_close, c_rw["dates"], regimes_origin, "rcs"),
    ], ignore_index=True)
    return {"by_exec": by_exec, "benchmarks": (summary, per_model),
            "perf_by_regime": perf}


def _alt_returns_for_asset(metal_name: str, df: pd.DataFrame,
                           oos_dates: pd.DatetimeIndex) -> tuple:
    """{"open": ..., "rolled": ...} e gli esiti dei controlli."""
    open_ret, open_rep = _open_returns_for_asset(metal_name, df, oos_dates)
    rolled_ret, rolled_rep = _rolled_returns_for_asset(metal_name, oos_dates)
    return ({"open": open_ret, "rolled": rolled_ret},
            {"open": open_rep, "rolled": rolled_rep})


def _save_strategy_returns(by_exec: dict, concat: dict, metal_name: str) -> None:
    """Serie giornaliere w (Position) e y (Net PnL), statiche e RCS, per esecuzione."""
    rows = []
    for execution, (bt_results, rcs_results) in by_exec.items():
        for strategy_name, results in (("static", bt_results), ("rcs", rcs_results)):
            for label, bt in results.items():
                model_name = label.replace(" ", "_")
                dates = pd.DatetimeIndex(concat[model_name]["dates"])[bt["Kept"]]
                for d, w, y in zip(dates, bt["Position"], bt["Net PnL"]):
                    rows.append({"Model": label, "Strategy": strategy_name,
                                 "Execution": execution, "Date": d, "w": w, "y": y})
    pd.DataFrame(rows).to_csv(
        f"{RESULTS_DIR}/strategy_returns_{metal_name}.csv", index=False)


def _reconstruct_asset_from_disk(metal_name: str, all_dfs: dict) -> dict:
    """
    Ricostruisce, da CSV già salvati su disco, tutto ciò che le sezioni
    finali della pipeline (ranking RMSE/Sharpe, correzione di Holm,
    statistica D) richiedono per un asset già completato in un run
    precedente — senza rifare training.

    Fonte unica: results/predictions_<asset>.csv (Model, Fold, Date,
    y_true, y_pred) — da lì si ricalcolano gli STESSI oggetti che il ciclo
    live costruirebbe, richiamando le stesse identiche funzioni
    (compute_metrics, aggregate_fold_results, build_dm_table,
    classify_regimes, _run_backtests_for_asset), non approssimazioni.
    Questo garantisce risultati identici a quelli di un run non
    interrotto, sotto lo stesso seed=42.

    Nota deliberata: dm_table e bt_results/rcs_results vengono RICALCOLATE
    (non rilette dai CSV di riepilogo), perché quei CSV vengono scritti
    solo dopo la correzione di Holm finale — per un asset completato ma
    interrotto prima di quella sezione potrebbero non esistere affatto o
    essere una versione stale di un run precedente. Ricalcolare da
    predictions_<asset>.csv è sempre corretto sotto il codice corrente.
    """
    df      = all_dfs[metal_name]
    regimes = classify_regimes(df)
    # Il regime del giorno t è osservabile solo a partire da t+1 (la
    # volatilità rolling usata da classify_regimes include r_t): per
    # decidere la posizione sul rendimento di t si usa l'etichetta di
    # t-1, mai quella di t stesso — vedi regimes_origin più sotto.
    regimes_origin = regimes.shift(1).fillna("normal")

    pred_df = pd.read_csv(f"{RESULTS_DIR}/predictions_{metal_name}.csv",
                          parse_dates=["Date"])

    concat         = {}
    results_list_a = []
    for model_label, g in pred_df.groupby("Model"):
        model_name = model_label.replace(" ", "_")
        g = g.sort_values(["Fold", "Date"])

        fold_metrics = [
            compute_metrics(gf["y_true"].values, gf["y_pred"].values)
            for _, gf in g.groupby("Fold")
        ]
        agg = aggregate_fold_results(fold_metrics)
        agg["RMSE pooled"] = _pooled_rmse(g["y_true"].values, g["y_pred"].values)
        results_list_a.append({"Asset": metal_name, "Model": model_label, **agg})

        concat[model_name] = {
            "y_true": g["y_true"].values,
            "y_pred": g["y_pred"].values,
            "dates":  pd.DatetimeIndex(g["Date"].values),
        }

    dm_preds = {
        mn.replace("_", " "): concat[mn]["y_pred"]
        for mn in concat if mn != "Random_Walk"
    }
    y_true_all = concat["Random_Walk"]["y_true"]
    dm_table   = build_dm_table(y_true_all, dm_preds)

    alt, open_report = _alt_returns_for_asset(
        metal_name, df, concat["Random_Walk"]["dates"])
    bt_out = _run_backtests_for_asset(concat, regimes_origin, metal_name, alt)
    _save_strategy_returns(bt_out["by_exec"], concat, metal_name)
    bt_results, _ = bt_out["by_exec"]["close"]

    asset_data_entry = {
        "y":       {m: bt["Net PnL"] for m, bt in bt_results.items()
                    if m != "Random Walk"},
        "dates":   concat["Random_Walk"]["dates"],
        "regimes": regimes_origin,
    }

    return {
        "results_list": results_list_a,
        "dm_table":     dm_table,
        "bt_out":       bt_out,
        "open_report":  open_report,
        "concat":       concat,
        "asset_data":   asset_data_entry,
    }


def _pooled_rmse(y_true, y_pred) -> float:
    """RMSE sui 210 errori aggregati (blu §3.5: diverso dalla media dei fold)."""
    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return round(float(np.sqrt(np.mean(e ** 2))), 6)


def _collect_backtests(bt_out: dict, metal_name: str, bt_all: dict,
                       bench_rows: list, bench_model_dfs: list,
                       perf_regime_dfs: list) -> None:
    """Accumula i risultati di _run_backtests_for_asset di un asset."""
    for execution, (bt_res, rcs_res) in bt_out["by_exec"].items():
        bt_all.setdefault(execution, {"static": {}, "rcs": {}})
        bt_all[execution]["static"][metal_name] = bt_res
        bt_all[execution]["rcs"][metal_name] = rcs_res
    summary, per_model = bt_out["benchmarks"]
    bench_rows.append({"Asset": metal_name, **summary})
    bench_model_dfs.append(per_model.assign(Asset=metal_name).reset_index())
    perf_regime_dfs.append(bt_out["perf_by_regime"].assign(Asset=metal_name))


def run_pipeline(resume: bool = False, smoke: bool = False):
    set_global_seed(42)

    if resume:
        _check_resume_commit()
    _write_run_manifest(seed=42, smoke=smoke)

    # --smoke: 1 asset, 2 fold, 2 epoche — solo per verificare che la
    # pipeline giri end-to-end, mai per numeri da citare in tesi.
    metals   = dict(list(METALS.items())[:1]) if smoke else METALS
    n_folds  = 2 if smoke else N_FOLDS
    if smoke:
        dl_models.EPOCHS = 2
        print(f"\n[--smoke attivo] asset={list(metals)}  n_folds={n_folds}  "
              f"epochs={dl_models.EPOCHS} — solo per verificare la pipeline, "
              f"NON per numeri da citare in tesi.\n")

    results_list    = []
    # bt_all[execution]["static" | "rcs"][asset] = {model_label: bt_dict}
    bt_all          = {}
    bench_rows      = []   # K14 — results/rcs_benchmarks.csv
    bench_model_dfs = []   # K14 — results/rcs_benchmarks_by_model.csv
    perf_regime_dfs = []   # K14 — results/performance_by_regime.csv
    open_reports    = {}   # K13 — esito dei controlli OHLC, nel manifest
    concat_all      = {}   # previsioni concatenate per asset (sensitivity roll)
    dm_tables_all   = {}   # accumula le dm_table per asset — la correzione di
                           # Holm è globale sui 48 test (12 modelli × 4 asset),
                           # quindi il salvataggio su CSV è rimandato a dopo
                           # il loop principale (vedi sezione dedicata sotto)
    asset_data      = {}   # per la statistica D (RQ3) — pooled sui 4 asset,
                           # anche questa serve solo a fine loop
    garch_rows_all  = []   # results/garch_volatility.csv (per Dist e fold)
    garch_dm_rows   = []   # K06 — results/garch_dm_qlike.csv
    dl_epoch_rows   = []   # B4 — results/dl_best_epochs.csv

    # ── Carica tutti gli asset prima del loop ─────────────────────────────────
    print("\nCaricamento dati...")
    all_dfs = {}
    for metal_name, ticker in metals.items():
        all_dfs[metal_name] = load_and_preprocess(
            metal_name, ticker, start=START, end=END)
        n = len(all_dfs[metal_name])
        print(f"  {metal_name}: {n} osservazioni")

    # ── Test di stazionarietà ADF (Capitolo 3) ────────────────────────────────
    # Diagnostica descrittiva sulla sola variabile che entra nella stima, il
    # log-rendimento; non usata per scegliere d (d = 0 fissato ex ante).
    print("\nTest ADF (diagnostica di radice unitaria sui log-rendimenti)...")
    return_series = {m: df["log_return"].values for m, df in all_dfs.items()}
    adf_table = build_adf_table(return_series)
    print(adf_table.to_string())
    adf_table.to_csv(f"{RESULTS_DIR}/adf_test.csv")

    # ── Loop principale per asset ─────────────────────────────────────────────
    for metal_name, ticker in metals.items():
        print(f"\n{'='*64}")
        print(f"  ASSET: {metal_name}  ({ticker})")
        print(f"{'='*64}")

        pred_path = _resume_marker_path(metal_name)
        if resume and os.path.exists(pred_path):
            print(f"  [--resume attivo] ATTENZIONE: sto riusando CSV di una run "
                  f"precedente — verifica che provengano dal codice corrente. "
                  f"Dopo una correzione metodologica (feature/target, ARIMA, "
                  f"regime, ecc.) i vecchi CSV NON sono più validi e vanno "
                  f"rigenerati con un run senza --resume.")
            print(f"  Asset già processato (trovato {pred_path}) — salto il "
                  f"training e ricostruisco i dati necessari dal CSV già "
                  f"salvato.")
            recon = _reconstruct_asset_from_disk(metal_name, all_dfs)
            results_list.extend(recon["results_list"])
            dm_tables_all[metal_name] = recon["dm_table"]
            _collect_backtests(recon["bt_out"], metal_name, bt_all, bench_rows,
                               bench_model_dfs, perf_regime_dfs)
            open_reports[metal_name]  = recon["open_report"]
            concat_all[metal_name]    = recon["concat"]
            asset_data[metal_name]    = recon["asset_data"]
            continue

        df = all_dfs[metal_name]
        n  = len(df)

        # Cross-asset: tutti gli asset, incluso quello corrente — necessario
        # perché gs_spread/gc_spread siano disponibili anche quando l'asset
        # modellato è esso stesso Gold/Silver/Copper. Feature laggate
        # (shift(1) in feature_engine.py), nessun leakage.
        cross_asset_dfs = all_dfs

        # ── Regime classification ─────────────────────────────────────────────
        regimes = classify_regimes(df)
        print(f"\n  Regime:\n{regime_summary(df).to_string()}")

        # Grafici della volatilità: etichette NON traslate (mostrano il
        # regime "vero" del giorno, calcolato con dati fino a quel giorno).
        plot_rolling_vol_regimes(
            df, regimes, asset_name=metal_name,
            save_path=p("04_volatility_regimes",
                        f"{metal_name}_rolling_volatility_with_regimes"))

        # Per ogni uso decisionale (RCS, split_by_regime, statistica D) il
        # regime del giorno t deve essere osservabile PRIMA di prendere
        # posizione su r_t: classify_regimes(df) etichetta t usando la
        # volatilità rolling che include r_t stesso (look-ahead se usata
        # per il rendimento dello stesso giorno) — si usa quindi
        # l'etichetta di t-1 per decidere la posizione su t.
        regimes_origin = regimes.shift(1).fillna("normal")

        folds = get_walk_forward_folds(n, n_folds=n_folds)
        validate_folds(folds, n, n_folds=n_folds)   # B1 — max(T_k) < min(E_k), |E_k|=21, E_k contigui/disgiunti
        export_folds_csv(folds, df.index, metal_name, path=f"{RESULTS_DIR}/folds.csv")
        print(f"\n  Walk-forward: {len(folds)} folds × 21 giorni "
              f"= {len(folds)*21} giorni OOS\n")

        all_true     = {m: [] for m, _ in MODELS}
        all_pred     = {m: [] for m, _ in MODELS}
        all_dates    = {m: [] for m, _ in MODELS}
        all_fmetrics = {m: [] for m, _ in MODELS}

        # ── Loop fold ─────────────────────────────────────────────────────────
        for fi, (train_idx, test_idx) in enumerate(folds):
            print(f"  Fold {fi+1}/{len(folds)}  "
                  f"({df.index[test_idx[0]].date()} → "
                  f"{df.index[test_idx[-1]].date()})")

            y_true_fold = df["log_return"].values[test_idx]
            dates_fold  = df.index[test_idx]

            for model_name, run_fn in MODELS:
                if run_fn is None:
                    y_pred = random_walk_forecast(y_true_fold)
                    dates  = dates_fold
                else:
                    try:
                        if model_name in DL_MODELS:
                            result = run_fn(df, train_idx, test_idx,
                                            cross_asset_dfs=cross_asset_dfs)
                        elif model_name in ML_MODELS:
                            result = run_fn(df, train_idx, test_idx,
                                            cross_asset_dfs=cross_asset_dfs)
                        else:
                            result = run_fn(df, train_idx, test_idx)

                        dates, y_true_fold, y_pred, run_params = result

                        if model_name in DL_MODELS and "best_epoch" in run_params:
                            dl_epoch_rows.append({          # B4
                                "Asset": metal_name, "Model": model_name,
                                "Fold": fi + 1,
                                "Best Epoch": run_params["best_epoch"],
                            })

                    except Exception as e:
                        print(f"    {model_name}: ERRORE {e}")
                        all_fmetrics[model_name].append(
                            {"RMSE": None, "MAE": None, "MAPE": None})
                        continue

                n_align    = min(len(y_true_fold), len(y_pred), len(dates_fold))
                y_true_aln = np.array(y_true_fold)[-n_align:]
                y_pred_aln = np.array(y_pred)[-n_align:]
                dates_aln  = dates_fold[-n_align:]

                metrics = compute_metrics(y_true_aln, y_pred_aln)
                all_true[model_name].append(y_true_aln)
                all_pred[model_name].append(y_pred_aln)
                all_dates[model_name].append(dates_aln)
                all_fmetrics[model_name].append(metrics)
                print(f"    {model_name:22s}  "
                      f"RMSE={metrics['RMSE']:.6f}  "
                      f"MAE={metrics['MAE']:.6f}")

        # ── GARCH (benchmark di volatilità, commenti K01, K06) ─────────────────
        # Gaussiano = specificazione di riferimento; Student-t = unico
        # competitore, per la robustezza. Le sigma_hat giornaliere sono
        # salvate in garch_forecasts_<asset>.csv.
        print(f"\n  [GARCH — Volatilità]")
        g_real_all, g_fc_all, g_dates_all = [], [], []
        g_fc_rows = []
        for dist in GARCH_DISTS:
            for fi, (train_idx, test_idx) in enumerate(folds):
                try:
                    gd, gr, gf, gp = run_garch_fold(df, train_idx, test_idx, dist=dist)
                    r_fold = df["log_return"].values[test_idx]
                    if dist == "normal":
                        g_real_all.append(gr)
                        g_fc_all.append(gf)
                        g_dates_all.append(df.index[test_idx])
                    gm = compute_metrics(gr, gf)
                    se_scaled = float(np.sqrt(np.mean((gf - gr * np.sqrt(np.pi / 2)) ** 2)))
                    ql = float(np.mean(qlike_loss(gf, r_fold)))
                    print(f"    [{dist:6s}] Fold {fi+1}: RMSE vs |r|={gm['RMSE']:.6f}  "
                          f"vs |r|sqrt(pi/2)={se_scaled:.6f}  QLIKE={ql:.4f}  "
                          f"persistence={gp['persistence']:.4f}")
                    garch_rows_all.append({
                        "Asset": metal_name, "Dist": dist, "Fold": fi + 1,
                        "alpha": gp["alpha"], "beta": gp["beta"],
                        "persistence": gp["persistence"], "nu": gp["nu"],
                        "loss_RMSE": gm["RMSE"],
                        "loss_RMSE_scaled_proxy": se_scaled,
                        "QLIKE": ql,
                    })
                    for d, sg, rr in zip(df.index[test_idx], gf, r_fold):
                        g_fc_rows.append({"Asset": metal_name, "Dist": dist,
                                          "Fold": fi + 1, "Date": d,
                                          "sigma_hat": sg, "r": rr})
                except Exception as e:
                    print(f"    [{dist}] Fold {fi+1}: GARCH ERRORE {e}")
        g_fc_df = pd.DataFrame(g_fc_rows)
        g_fc_df.to_csv(f"{RESULTS_DIR}/garch_forecasts_{metal_name}.csv", index=False)
        if set(g_fc_df.get("Dist", [])) == set(GARCH_DISTS):
            piv = g_fc_df.pivot_table(index="Date", columns="Dist",
                                      values=["sigma_hat", "r"])
            ql_n = qlike_loss(piv["sigma_hat"]["normal"].values, piv["r"]["normal"].values)
            ql_t = qlike_loss(piv["sigma_hat"]["t"].values, piv["r"]["t"].values)
            dm_q = dm_test_loss(ql_n, ql_t)
            garch_dm_rows.append({"Asset": metal_name,
                                  "Mean QLIKE normal": float(np.mean(ql_n)),
                                  "Mean QLIKE t": float(np.mean(ql_t)), **dm_q})
            print(f"    DM QLIKE normale vs t: DM={dm_q['DM Stat']:.3f}  "
                  f"p={dm_q['p-value']:.4f}  (DM < 0: normale migliore)")

        # ── Aggregazione metriche ─────────────────────────────────────────────
        print(f"\n  ── Risultati Aggregati ──")
        for model_name, _ in MODELS:
            agg = aggregate_fold_results(all_fmetrics[model_name])
            # Media dei fold (riportata nelle tabelle) e RMSE sui 210 errori
            # aggregati: diverse per la non linearità della radice (blu §3.5).
            agg["RMSE pooled"] = (_pooled_rmse(np.concatenate(all_true[model_name]),
                                               np.concatenate(all_pred[model_name]))
                                  if all_true[model_name] else None)
            print(f"    {model_name:22s}  RMSE={agg['RMSE']}  "
                  f"RMSE pooled={agg['RMSE pooled']}  MAE={agg['MAE']}")
            results_list.append({
                "Asset": metal_name,
                "Model": model_name.replace("_", " "),
                **agg
            })

        # ── Concatenazione previsioni ─────────────────────────────────────────
        concat = {}
        for model_name, _ in MODELS:
            if not all_true[model_name]:
                continue
            dates_concat = all_dates[model_name][0]
            for d in all_dates[model_name][1:]:
                dates_concat = dates_concat.append(d)
            concat[model_name] = {
                "y_true": np.concatenate(all_true[model_name]),
                "y_pred": np.concatenate(all_pred[model_name]),
                "dates":  dates_concat,
            }

        # ── Export previsioni grezze (per DM test / regime / backtest futuri) ──
        pred_rows = []
        for model_name, _ in MODELS:
            if model_name not in all_true:
                continue
            for fi in range(len(all_dates[model_name])):
                fold_dates = all_dates[model_name][fi]
                fold_true  = all_true[model_name][fi]
                fold_pred  = all_pred[model_name][fi]
                for d, yt, yp in zip(fold_dates, fold_true, fold_pred):
                    pred_rows.append({
                        "Model": model_name.replace("_", " "),
                        "Fold":  fi + 1,
                        "Date":  d,
                        "y_true": yt,
                        "y_pred": yp,
                    })
        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(f"{RESULTS_DIR}/predictions_{metal_name}.csv", index=False)

        # ── Diebold-Mariano ───────────────────────────────────────────────────
        print(f"\n  ── Diebold-Mariano ──")
        dm_preds = {
            mn.replace("_", " "): concat[mn]["y_pred"]
            for mn, _ in MODELS
            if mn in concat and mn != "Random_Walk"
        }
        y_true_all = concat["Random_Walk"]["y_true"]
        dm_table   = build_dm_table(y_true_all, dm_preds)
        print(dm_table[["DM Stat", "p-value", "HAC Lag (L)", "Sig. (5%)"]].to_string())
        dm_tables_all[metal_name] = dm_table
        # Nota: il CSV viene scritto dopo il loop principale, una volta
        # applicata la correzione di Holm globale (vedi sotto) — la colonna
        # "Sig. Holm (5%)" richiede i p-value di tutti e 4 gli asset insieme.

        plot_dm_heatmap(
            dm_table, asset_name=metal_name,
            save_path=p("06_diebold_mariano",
                        f"{metal_name}_DM_test_vs_RandomWalk"))

        # ── Regime analysis ───────────────────────────────────────────────────
        print(f"\n  ── Regime Analysis ──")
        regime_results = {}
        for model_name, _ in MODELS:
            if model_name not in concat:
                continue
            c = concat[model_name]
            splits = split_by_regime(
                c["y_true"], c["y_pred"], c["dates"], regimes_origin)
            label = model_name.replace("_", " ")
            regime_results[label] = {}
            for regime, (yt, yp) in splits.items():
                m = compute_metrics(yt, yp)
                regime_results[label][regime] = m
                print(f"    {model_name:22s} [{regime:8s}]  "
                      f"RMSE={m['RMSE']:.6f}")

        plot_all_models_regime(
            regime_results, asset_name=metal_name, metric="RMSE",
            save_path=p("05_regime_performance",
                        f"{metal_name}_RMSE_by_regime_all_models"))

        # ── BACKTESTING (statico + Regime-Conditional Strategy) ────────────────
        # y = w*R - c|Δw| - 2c|w|*1{roll}, R = exp(r)-1, c = 1e-4 per lato.
        # Esecuzione principale "close" + sensitivity "lag1" e, se lo
        # dati esterni superano i controlli, "open" (K13) e "rolled" (K03).
        print(f"\n  ── Backtesting (statico + RCS) ──")
        alt, open_reports[metal_name] = _alt_returns_for_asset(
            metal_name, df, concat["Random_Walk"]["dates"])
        print(f"  Esecuzione all'apertura: {open_reports[metal_name]['open']['reason']}  |  "
              f"serie rollata: {open_reports[metal_name]['rolled']['reason']}")
        bt_out = _run_backtests_for_asset(concat, regimes_origin, metal_name, alt)
        bt_results, rcs_results = bt_out["by_exec"]["close"]
        bt_lag, rcs_lag = bt_out["by_exec"]["lag1"]
        for model_name, _ in MODELS:
            label = model_name.replace("_", " ")
            if label not in bt_results:
                continue
            bt = bt_results[label]
            print(f"    {model_name:22s}  "
                  f"Sharpe={bt['Sharpe Ratio']:6.3f} "
                  f"[{bt['Sharpe CI Lower (95%)']:.2f}, {bt['Sharpe CI Upper (95%)']:.2f}]  "
                  f"lag1={bt_lag[label]['Sharpe Ratio']:6.3f}  "
                  f"DA={bt['Directional Acc.']:5.1f}%  "
                  f"MaxDD={bt['Max Drawdown (%)']:6.2f}%  "
                  f"RollDays={bt['N Roll Days']}")
            if label in rcs_results:
                bt_rcs = rcs_results[label]
                print(f"    {model_name:22s} [RCS]  "
                      f"Sharpe={bt_rcs['Sharpe Ratio']:6.3f}  "
                      f"lag1={rcs_lag[label]['Sharpe Ratio']:6.3f}  "
                      f"DA={bt_rcs['Directional Acc.']:5.1f}% "
                      f"[{bt_rcs['DA CI Lower (95%)']:.1f}, {bt_rcs['DA CI Upper (95%)']:.1f}]")
        summary, _ = bt_out["benchmarks"]
        print(f"    Benchmark RCS: long filtrato Sharpe={summary['Sharpe Long Filtered']:.3f}  "
              f"casuale filtrato media={summary['Sharpe Random Filtered Mean']:.3f} "
              f"p95={summary['Sharpe Random Filtered P95']:.3f}")

        # CSV di riepilogo rimandati a dopo la correzione di Holm globale.
        _collect_backtests(bt_out, metal_name, bt_all, bench_rows,
                           bench_model_dfs, perf_regime_dfs)
        _save_strategy_returns(bt_out["by_exec"], concat, metal_name)
        concat_all[metal_name] = concat

        # ── Dati per la statistica D (RQ3, D7) — serie y NETTA della
        # strategia statica, già calcolata sopra, mai ricalcolata ──────────
        asset_data[metal_name] = {
            "y":       {m: bt["Net PnL"] for m, bt in bt_results.items()
                        if m != "Random Walk"},
            "dates":   concat["Random_Walk"]["dates"],
            "regimes": regimes_origin,
        }

        # ── Plot individuali ──────────────────────────────────────────────────
        model_pred_prices = {}
        for model_name, _ in MODELS:
            if model_name not in concat:
                continue
            c     = concat[model_name]
            label = model_name.replace("_", " ")

            plot_returns_forecast(
                pd.Series(c["y_true"], index=c["dates"]),
                pd.Series(c["y_pred"], index=c["dates"]),
                model_name=label, asset_name=metal_name,
                save_path=p("01_logreturn_forecast",
                            f"{metal_name}_{model_name}_logreturn_forecast_vs_actual"))

            prices_before = df["Close"][df.index < c["dates"][0]]
            last_price    = float(prices_before.iloc[-1])
            pred_prices   = reconstruct_prices(last_price, c["y_pred"])
            pred_series   = pd.Series(pred_prices, index=c["dates"])
            real_prices   = df["Close"].reindex(c["dates"], method="nearest")
            model_pred_prices[label] = pred_series

            plot_price_comparison(
                real_prices=real_prices,
                predicted_prices=pred_series,
                model_name=label, asset_name=metal_name,
                save_path=p("02_predicted_vs_real_price",
                            f"{metal_name}_{model_name}_predicted_vs_real_price"))

        if model_pred_prices:
            ref_dates   = list(model_pred_prices.values())[0].index
            real_prices = df["Close"].reindex(ref_dates, method="nearest")
            plot_all_models_prices(
                real_prices=real_prices,
                pred_prices_dict=model_pred_prices,
                asset_name=metal_name,
                save_path=p("03_all_models_price_overlay",
                            f"{metal_name}_all_models_predicted_vs_real_price"))

        if g_real_all:
            g_dates_concat = g_dates_all[0]
            for d in g_dates_all[1:]:
                g_dates_concat = g_dates_concat.append(d)
            plot_garch_volatility(
                pd.Series(np.concatenate(g_real_all), index=g_dates_concat),
                pd.Series(np.concatenate(g_fc_all),   index=g_dates_concat),
                asset_name=metal_name,
                save_path=p("04_volatility_regimes",
                            f"{metal_name}_GARCH_volatility_forecast_vs_realized"))

    # ── GARCH: parametri e loss per fold, DM sulla QLIKE ──────────────────────
    pd.DataFrame(garch_rows_all).to_csv(f"{RESULTS_DIR}/garch_volatility.csv", index=False)
    pd.DataFrame(garch_dm_rows).to_csv(f"{RESULTS_DIR}/garch_dm_qlike.csv", index=False)

    # ── B4: epoche migliori (e*) per fold/modello/asset DL ─────────────────────
    pd.DataFrame(dl_epoch_rows).to_csv(f"{RESULTS_DIR}/dl_best_epochs.csv", index=False)

    # ── Correzione di Holm per famiglia (blu §1.3 "Multiple comparisons") ─────
    # Famiglie principali pre-specificate, ciascuna di (al più) 48 test =
    # 12 modelli × 4 asset, Random Walk esclusa: (a) DM, (b) alpha statica,
    # (c) alpha RCS, (d) DA statica, (e) DA RCS. La DA usa il test HAC sotto
    # H0: E[h_t] = 1/2 (commento K04); le strategie con meno di
    # MIN_ACTIVE_DA giorni attivi non sono testate (p-value NaN, esclusi
    # dalla famiglia). Le esecuzioni di sensitivity ("lag1", "open", "rolled") formano
    # famiglie separate, riportate come controlli. Il controllo è per
    # famiglia: non implica il 5% sull'unione di tutte le famiglie.
    print(f"\n{'='*64}")
    print("  Correzione di Holm (famiglie separate, 12 modelli x 4 asset)")
    print(f"{'='*64}")

    def _holm_correct(pvalues_by_asset: dict) -> dict:
        """
        pvalues_by_asset: {asset: {model_label: p_value}}.
        Ritorna {asset: {model_label: (p_holm, sig_bool)}} — correzione di
        Holm globale sui valori concatenati, con verifica esplicita che
        nessun valore vada perso/disallineato nello split per-asset.

        NaN esclusi dalla chiamata a multipletests: un p-value NaN nasce da
        un caso degenere (es. alpha regression su una serie y a varianza
        zero — RCS sempre flat in una finestra, tutta la varianza campionaria
        della statistica di test è nulla, t-stat = 0/0). Verificato: senza
        questo filtro, statsmodels.multipletests(method="holm") marca
        `reject=True` per OGNI NaN in input, cioè "significativo" per un
        caso in cui il test è semplicemente non definito — l'opposto della
        conclusione corretta. Qui i NaN restano NaN e mai significativi.
        """
        asset_order = list(pvalues_by_asset.keys())
        model_order = {a: list(pvalues_by_asset[a].keys()) for a in asset_order}
        flat = np.concatenate([
            [pvalues_by_asset[a][m] for m in model_order[a]] for a in asset_order
        ])

        valid = ~np.isnan(flat)
        p_holm = np.full_like(flat, np.nan)
        reject = np.zeros_like(flat, dtype=bool)
        if valid.any():
            reject[valid], p_holm[valid], _, _ = multipletests(
                flat[valid], alpha=0.05, method="holm")

        out, offset = {}, 0
        for a in asset_order:
            out[a] = {}
            for m in model_order[a]:
                out[a][m] = (float(p_holm[offset]), bool(reject[offset]))
                offset += 1
        assert offset == len(flat), (
            f"Split per-asset incompleto: consumati {offset} valori su "
            f"{len(flat)} totali dopo Holm"
        )
        return out, len(flat)

    # (a) DM
    asset_order = list(dm_tables_all.keys())
    dm_pvalues = {a: dm_tables_all[a]["p-value"].to_dict() for a in asset_order}
    dm_holm, n_dm = _holm_correct(dm_pvalues)
    n_sig_raw = n_sig_holm = 0
    for a in asset_order:
        dm_table = dm_tables_all[a]
        dm_table["p Holm"] = [dm_holm[a][m][0] for m in dm_table.index]
        dm_table["Sig. Holm (5%)"] = [
            "Yes" if dm_holm[a][m][1] else "No" for m in dm_table.index]
        n_sig_raw  += int((dm_table["p-value"] < 0.05).sum())
        n_sig_holm += int((dm_table["Sig. Holm (5%)"] == "Yes").sum())
        dm_table.to_csv(f"{RESULTS_DIR}/DM_test_{a}.csv")
    print(f"  (a) DM — significativi grezzi/Holm: {n_sig_raw}/{n_dm} -> {n_sig_holm}/{n_dm}")

    # (b)-(e) e famiglie di sensitivity
    families = []
    for execution in bt_all:
        tag = "" if execution == "close" else f"{execution} "
        for strategy in ("static", "rcs"):
            families.append((f"{tag}alpha {strategy}", "Alpha p-value", "Alpha p Holm",
                             "Alpha Sig. Holm (5%)", bt_all[execution][strategy]))
            families.append((f"{tag}DA {strategy}", "DA p-value (HAC)", "DA p Holm",
                             "DA Sig. Holm (5%)", bt_all[execution][strategy]))
    holm_summary = []
    for fam_name, src_key, holm_key, sig_key, results_all in families:
        pvalues_by_asset = {
            a: {m: bt[src_key] for m, bt in results_all[a].items()
                if m != "Random Walk"}
            for a in results_all
        }
        holm, n_fam = _holm_correct(pvalues_by_asset)
        n_valid = sum(not np.isnan(v) for d in pvalues_by_asset.values() for v in d.values())
        for a, per_model in holm.items():
            for m, (p_holm, sig) in per_model.items():
                results_all[a][m][holm_key] = p_holm
                results_all[a][m][sig_key]  = sig
        n_sig = sum(sig for per_model in holm.values() for _, sig in per_model.values())
        holm_summary.append({"Family": fam_name, "N Tests": n_valid,
                             "N Sig. Holm (5%)": int(n_sig)})
        print(f"  {fam_name:18s} — test validi {n_valid}/{n_fam}, "
              f"significativi dopo Holm: {n_sig}")
    holm_summary.insert(0, {"Family": "DM", "N Tests": n_dm, "N Sig. Holm (5%)": n_sig_holm})
    pd.DataFrame(holm_summary).to_csv(f"{RESULTS_DIR}/holm_families.csv", index=False)

    # ── Scrittura CSV di riepilogo backtest (ora con le colonne Holm) ──────────
    for execution in bt_all:
        suffix = "" if execution == "close" else f"_{execution}"
        for asset_name in bt_all[execution]["static"]:
            backtest_summary_table(bt_all[execution]["static"][asset_name]).to_csv(
                f"{RESULTS_DIR}/backtest{suffix}_{asset_name}.csv")
            backtest_summary_table(bt_all[execution]["rcs"][asset_name]).to_csv(
                f"{RESULTS_DIR}/backtest_rcs{suffix}_{asset_name}.csv")

    # ── K14: benchmark della regola di esposizione e performance per regime ───
    pd.DataFrame(bench_rows).to_csv(f"{RESULTS_DIR}/rcs_benchmarks.csv", index=False)
    pd.concat(bench_model_dfs, ignore_index=True).to_csv(
        f"{RESULTS_DIR}/rcs_benchmarks_by_model.csv", index=False)
    pd.concat(perf_regime_dfs, ignore_index=True).to_csv(
        f"{RESULTS_DIR}/performance_by_regime.csv", index=False)

    # ── K03: sensitivity sui giorni di cambio contratto ────────────────────────
    # Le serie nette già calcolate (esecuzione principale) vengono ristrette
    # escludendo i rendimenti post-roll e i giorni adiacenti (k = 1); i costi
    # non si ricalcolano. Documentazione empirica delle discontinuità: |r|
    # medio post-roll contro gli altri giorni e i 10 |r| più grandi.
    roll_rows, disc_rows = [], []
    for asset_name, concat in concat_all.items():
        dates = pd.DatetimeIndex(concat["Random_Walk"]["dates"])
        r = concat["Random_Walk"]["y_true"]
        post = post_roll_mask(asset_name, dates)
        win = roll_window_mask(asset_name, dates, k=1)
        R = np.expm1(r)
        for strategy in ("static", "rcs"):
            for label, bt in bt_all["close"][strategy][asset_name].items():
                if label == "Random Walk":
                    continue
                y = np.asarray(bt["Net PnL"])
                al = estimate_alpha(y[~win], R[~win])
                roll_rows.append({
                    "Asset": asset_name, "Model": label, "Strategy": strategy,
                    "N Excluded": int(win.sum()),
                    "Sharpe": bt["Sharpe Ratio"],
                    "Sharpe excl. roll window": round(metrics_econ.sharpe_ratio(y[~win]), 4),
                    "Alpha p-value": bt["Alpha p-value"],
                    "Alpha p-value excl. roll window": al["Alpha p-value"]})
        top = np.argsort(-np.abs(r))[:10]
        for rank, i in enumerate(top, 1):
            disc_rows.append({"Asset": asset_name, "Rank |r|": rank,
                              "Date": dates[i].date(), "r": float(r[i]),
                              "Post-roll": bool(post[i]), "In roll window": bool(win[i]),
                              "Mean |r| post-roll": float(np.abs(r[post]).mean()),
                              "Mean |r| other days": float(np.abs(r[~post]).mean()),
                              "N post-roll days": int(post.sum())})
    pd.DataFrame(roll_rows).to_csv(f"{RESULTS_DIR}/roll_sensitivity.csv", index=False)
    pd.DataFrame(disc_rows).to_csv(f"{RESULTS_DIR}/roll_discontinuities.csv", index=False)

    # ── RQ3: statistica D (dispersione tra regimi vs tra modelli) ──────────────
    # Diagnostica descrittiva, non attribuzione causale. Principale:
    # aggregazione "portfolio" (un rendimento equal-weight per data e regime,
    # stesso operatore in stima e bootstrap — commento K15); controlli: D per
    # asset aggregata e decomposizione a due fattori (commento K02).
    print(f"\n{'='*64}")
    print("  RQ3 — Statistica D (dispersione tra regimi vs tra modelli)")
    print(f"{'='*64}")
    d_rows = []
    for aggregation in ("portfolio", "per_asset"):
        d_result = architecture_vs_regime_D(asset_data, aggregation=aggregation,
                                            block_length=10, n_boot=2000,
                                            min_obs_per_cell=5, seed=42)
        print(f"  [{aggregation}] D = {d_result['D']:.6f}   "
              f"CI 95% = [{d_result['D CI Lower (95%)']}, {d_result['D CI Upper (95%)']}]   "
              f"repliche scartate = {d_result['Frac. Discarded']:.2%}   "
              f"asset usati = {d_result['Assets Used']}")
        print(f"    N osservazioni per regime = {d_result['N Obs per Regime (point estimate)']}")
        d_rows.append({k: d_result[k] for k in (
            "Aggregation", "D", "D CI Lower (95%)", "D CI Upper (95%)",
            "N Common Dates", "Block Length", "N Bootstrap Requested",
            "N Bootstrap Used", "Frac. Discarded", "Min Obs per Cell",
            "Assets Used")})
        if aggregation == "portfolio":
            d_result["Sharpe by Model-Regime"].to_csv(
                f"{RESULTS_DIR}/sharpe_by_model_regime.csv")
            pd.Series(d_result["N Obs per Regime (point estimate)"],
                      name="N Dates").to_csv(f"{RESULTS_DIR}/D_regime_counts.csv")
        else:
            d_result["Sharpe by Model-Regime"].to_csv(
                f"{RESULTS_DIR}/sharpe_by_model_regime_per_asset.csv")
            pd.Series(d_result["D by Asset"], name="D").to_csv(
                f"{RESULTS_DIR}/D_per_asset.csv")
    pd.DataFrame(d_rows).to_csv(f"{RESULTS_DIR}/architecture_vs_regime_D.csv", index=False)

    two_way = two_way_decomposition(asset_data, block_length=10, n_boot=2000, seed=42)
    print("  Decomposizione a due fattori (quote di varianza):")
    print(two_way.to_string())
    two_way.to_csv(f"{RESULTS_DIR}/two_way_decomposition.csv")

    # ── Output finale ─────────────────────────────────────────────────────────
    results_df = save_results(results_list, path=f"{RESULTS_DIR}/metrics.csv")
    valid_df   = results_df.dropna(subset=["RMSE"])

    print("\n" + "="*64)
    print("  RANKING FINALE — RMSE medio")
    print("="*64)
    ranking = (valid_df.groupby("Model")[["RMSE", "MAE"]]
               .mean().sort_values("RMSE"))
    print(ranking.to_string())

    print("\n" + "="*64)
    print("  RANKING FINALE — Sharpe medio")
    print("="*64)
    sharpe_rows = []
    for asset_name, bt_res in bt_all["close"]["static"].items():
        for model_name, bt in bt_res.items():
            sharpe_rows.append({
                "Asset": asset_name, "Model": model_name,
                "Sharpe": bt["Sharpe Ratio"],
                "Sharpe lag1": bt_all["lag1"]["static"][asset_name][model_name]["Sharpe Ratio"],
                "MaxDD":  bt["Max Drawdown (%)"],
                "DA":     bt["Directional Acc."]
            })
    sharpe_df = pd.DataFrame(sharpe_rows)
    sharpe_df.to_csv(f"{RESULTS_DIR}/backtesting_all_assets.csv", index=False)
    sharpe_ranking = (sharpe_df.groupby("Model")[["Sharpe", "Sharpe lag1", "MaxDD", "DA"]]
                      .mean().sort_values("Sharpe", ascending=False))
    print(sharpe_ranking.to_string())

    plot_sharpe_heatmap(
        sharpe_df[["Asset", "Model", "Sharpe"]],
        save_path=p("07_heatmaps", "heatmap_Sharpe_all_models_all_assets"))
    plot_hit_rate_heatmap(
        sharpe_df[["Asset", "Model", "DA"]].rename(columns={"DA": "Hit_Rate"}),
        save_path=p("07_heatmaps", "heatmap_HitRate_all_models_all_assets"))

    rw_rmse = valid_df.loc[valid_df["Model"] == "Random Walk", "RMSE"].mean()
    for metric in ["RMSE", "MAE"]:
        plot_heatmap(valid_df, metric=metric,
                      save_path=p("07_heatmaps",
                                  f"heatmap_{metric}_all_models_all_assets"))
        plot_ranking(valid_df, metric=metric,
                      baseline_value=rw_rmse if metric == "RMSE" else None,
                      baseline_label="Random Walk",
                      save_path=p("08_model_ranking",
                                  f"model_ranking_mean_{metric}_all_assets"))
    plot_dashboard(valid_df,
                    save_path=p("09_dashboard",
                                "summary_dashboard_all_models_all_assets"))

    print("\n✅  Pipeline completa.")
    print("    results/metrics.csv")
    print("    results/backtest_<asset>.csv")
    print("    results/backtest_rcs_<asset>.csv")
    print("    results/backtesting_all_assets.csv")
    print("    results/predictions_<asset>.csv")
    print("    results/strategy_returns_<asset>.csv")
    print("    results/folds.csv")
    print("    results/garch_volatility.csv")
    print("    results/dl_best_epochs.csv")
    print("    results/architecture_vs_regime_D.csv")
    print("    results/sharpe_by_model_regime.csv")

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Riusa results/predictions_<asset>.csv di una run precedente "
             "per gli asset già completati (backtest/DM/statistica D "
             "vengono ricalcolati da lì, non riletti), invece di rifare il "
             "training. Accettato solo se results/run_manifest.json esiste "
             "e riporta lo STESSO commit git di HEAD — altrimenti rifiutato "
             "(dopo una correzione metodologica quei CSV non riflettono più "
             "il codice corrente).")
    parser.add_argument(
        "--smoke", action="store_true", default=False,
        help="Run ridotto (1 asset, 2 fold, 2 epoche DL) per verificare che "
             "la pipeline giri end-to-end. Da eseguire in locale — MAI per "
             "numeri da citare in tesi.")
    args = parser.parse_args()
    run_pipeline(resume=args.resume, smoke=args.smoke)