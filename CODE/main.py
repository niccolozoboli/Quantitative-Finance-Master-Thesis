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

from src.preprocessing     import METALS, load_and_preprocess, reconstruct_prices
from src.walk_forward      import (get_walk_forward_folds, aggregate_fold_results,
                                    validate_folds, export_folds_csv, N_FOLDS)
from src.utils             import compute_metrics, save_results, set_global_seed
from src.regime            import classify_regimes, split_by_regime, regime_summary
from src.statistical_tests import (random_walk_forecast,
                                    build_dm_table,
                                    build_adf_table)
from src.backtesting       import (run_backtest, regime_conditional_strategy,
                                    backtest_summary_table, block_bootstrap_ci,
                                    architecture_vs_regime_D, estimate_alpha)
from src.visualization     import (
    plot_returns_forecast, plot_price_comparison,
    plot_all_models_prices, plot_garch_volatility,
    plot_heatmap, plot_ranking, plot_dashboard,
    plot_all_models_regime, plot_dm_heatmap, plot_rolling_vol_regimes,
    plot_hit_rate_heatmap, plot_sharpe_heatmap,
)

from src.models.arima     import run_arima_fold
from src.models.garch     import run_garch_fold
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


def _run_backtests_for_asset(concat: dict, regimes_origin: pd.Series,
                              metal_name: str) -> tuple:
    """
    Esegue backtest statico + Regime-Conditional Strategy per tutti i
    modelli di un asset già concatenati (concat[model_name] = {y_true,
    y_pred, dates}) — condivisa dal percorso live e da quello di resume,
    così i risultati sono identici indipendentemente da quali asset sono
    stati rieseguiti in questa run e quali ricostruiti da disco (D1-D2,
    D6). Random Walk esclusa dalla RCS (D2) e dall'alpha (D6, come da DM
    e statistica D).

    Ritorna (bt_results, rcs_results): dict {model_label: bt_dict}.
    """
    bt_results, rcs_results = {}, {}
    for model_name, c in concat.items():
        label = model_name.replace("_", " ")

        bt = run_backtest(c["y_true"], c["y_pred"], dates=c["dates"],
                          asset_name=metal_name, cost_per_side=0.0001)
        bt_ci = block_bootstrap_ci(bt["Net PnL"], bt["Hit"], bt["Active"],
                                   block_length=10, n_boot=2000, seed=42)
        bt.update(bt_ci)
        if model_name != "Random_Walk":
            R = np.expm1(c["y_true"])
            bt.update(estimate_alpha(bt["Net PnL"], R))
        bt_results[label] = bt

        if model_name != "Random_Walk":
            y_pred_rcs = regime_conditional_strategy(
                c["y_pred"], c["dates"], regimes_origin)
            bt_rcs = run_backtest(c["y_true"], c["y_pred"], dates=c["dates"],
                                  asset_name=metal_name, cost_per_side=0.0001,
                                  position=y_pred_rcs)
            bt_rcs_ci = block_bootstrap_ci(bt_rcs["Net PnL"], bt_rcs["Hit"],
                                           bt_rcs["Active"], block_length=10,
                                           n_boot=2000, seed=42)
            bt_rcs.update(bt_rcs_ci)
            R = np.expm1(c["y_true"])
            bt_rcs.update(estimate_alpha(bt_rcs["Net PnL"], R))
            rcs_results[label] = bt_rcs

    return bt_results, rcs_results


def _save_strategy_returns(bt_results: dict, rcs_results: dict,
                            concat: dict, metal_name: str) -> None:
    """D2 — serie giornaliere w (Position) e y (Net PnL), statiche e RCS."""
    rows = []
    for strategy_name, results in (("static", bt_results), ("rcs", rcs_results)):
        for label, bt in results.items():
            model_name = label.replace(" ", "_")
            dates = concat[model_name]["dates"]
            n = len(bt["Net PnL"])
            for d, w, y in zip(dates[-n:], bt["Position"], bt["Net PnL"]):
                rows.append({"Model": label, "Strategy": strategy_name,
                            "Date": d, "w": w, "y": y})
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

    bt_results, rcs_results = _run_backtests_for_asset(
        concat, regimes_origin, metal_name)
    _save_strategy_returns(bt_results, rcs_results, concat, metal_name)

    asset_data_entry = {
        "y":       {m: bt["Net PnL"] for m, bt in bt_results.items()
                    if m != "Random Walk"},
        "dates":   concat["Random_Walk"]["dates"],
        "regimes": regimes_origin,
    }

    return {
        "results_list": results_list_a,
        "dm_table":     dm_table,
        "bt_results":   bt_results,
        "rcs_results":  rcs_results,
        "asset_data":   asset_data_entry,
    }


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
    bt_results_all  = {}
    rcs_results_all = {}
    dm_tables_all   = {}   # accumula le dm_table per asset — la correzione di
                           # Holm è globale sui 48 test (12 modelli × 4 asset),
                           # quindi il salvataggio su CSV è rimandato a dopo
                           # il loop principale (vedi sezione dedicata sotto)
    asset_data      = {}   # per la statistica D (RQ3) — pooled sui 4 asset,
                           # anche questa serve solo a fine loop
    garch_rows_all  = []   # C5 — results/garch_volatility.csv
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
    print("\nTest ADF (stazionarietà)...")
    price_series  = {m: df["Close"].values      for m, df in all_dfs.items()}
    return_series = {m: df["log_return"].values for m, df in all_dfs.items()}
    adf_table = build_adf_table(price_series, return_series)
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
            dm_tables_all[metal_name]   = recon["dm_table"]
            bt_results_all[metal_name]  = recon["bt_results"]
            rcs_results_all[metal_name] = recon["rcs_results"]
            asset_data[metal_name]      = recon["asset_data"]
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

        # ── GARCH ─────────────────────────────────────────────────────────────
        print(f"\n  [GARCH — Volatilità]")
        g_real_all, g_fc_all, g_dates_all = [], [], []
        for fi, (train_idx, test_idx) in enumerate(folds):
            try:
                gd, gr, gf, gp = run_garch_fold(df, train_idx, test_idx)
                g_real_all.append(gr)
                g_fc_all.append(gf)
                g_dates_all.append(df.index[test_idx])
                gm = compute_metrics(gr, gf)
                print(f"    Fold {fi+1}: Vol-RMSE={gm['RMSE']:.6f}  "
                      f"persistence={gp['persistence']:.4f}")
                garch_rows_all.append({           # C5
                    "Asset": metal_name, "Fold": fi + 1,
                    "alpha": gp["alpha"], "beta": gp["beta"],
                    "persistence": gp["persistence"],
                    "loss_RMSE": gm["RMSE"],
                })
            except Exception as e:
                print(f"    Fold {fi+1}: GARCH ERRORE {e}")

        # ── Aggregazione metriche ─────────────────────────────────────────────
        print(f"\n  ── Risultati Aggregati ──")
        for model_name, _ in MODELS:
            agg = aggregate_fold_results(all_fmetrics[model_name])
            print(f"    {model_name:22s}  RMSE={agg['RMSE']}  "
                  f"MAE={agg['MAE']}")
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
        # D1-D2: y = w*R - c|Δw| - 2c|w|*1{roll}, R = exp(r)-1, cost_per_side=c.
        # D2: Random Walk esclusa dal loop RCS (posizione RW sempre nulla,
        # RCS su RW sarebbe identica alla statica) — vedi _run_backtests_for_asset.
        print(f"\n  ── Backtesting (statico + RCS) ──")
        bt_results, rcs_results = _run_backtests_for_asset(
            concat, regimes_origin, metal_name)
        for model_name, _ in MODELS:
            label = model_name.replace("_", " ")
            if label not in bt_results:
                continue
            bt = bt_results[label]
            print(f"    {model_name:22s}  "
                  f"Sharpe={bt['Sharpe Ratio']:6.3f} "
                  f"[{bt['Sharpe CI Lower (95%)']:.2f}, {bt['Sharpe CI Upper (95%)']:.2f}]  "
                  f"DA={bt['Directional Acc.']:5.1f}%  "
                  f"MaxDD={bt['Max Drawdown (%)']:6.2f}%  "
                  f"RollDays={bt['N Roll Days']}")
            if label in rcs_results:
                bt_rcs = rcs_results[label]
                print(f"    {model_name:22s} [RCS]  "
                      f"Sharpe={bt_rcs['Sharpe Ratio']:6.3f}  "
                      f"DA={bt_rcs['Directional Acc.']:5.1f}% "
                      f"[{bt_rcs['DA CI Lower (95%)']:.1f}, {bt_rcs['DA CI Upper (95%)']:.1f}]")

        # CSV di riepilogo (bt_df/rcs_df) rimandati a dopo la correzione di
        # Holm globale — vedi sezione dedicata dopo il loop principale.
        bt_results_all[metal_name]  = bt_results
        rcs_results_all[metal_name] = rcs_results
        _save_strategy_returns(bt_results, rcs_results, concat, metal_name)

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

    # ── C5: GARCH volatility CSV ────────────────────────────────────────────────
    pd.DataFrame(garch_rows_all).to_csv(f"{RESULTS_DIR}/garch_volatility.csv", index=False)

    # ── B4: epoche migliori (e*) per fold/modello/asset DL ─────────────────────
    pd.DataFrame(dl_epoch_rows).to_csv(f"{RESULTS_DIR}/dl_best_epochs.csv", index=False)

    # ── B9: Correzione di Holm — 5 famiglie indipendenti di 48 test ognuna
    # (12 modelli × 4 asset, Random Walk esclusa ovunque): (a) DM, (b) alpha
    # statica, (c) alpha RCS, (d) DA bootstrap statica, (e) DA bootstrap RCS.
    # Ogni famiglia è corretta separatamente (non un'unica famiglia da 240
    # test) — sono ipotesi scientificamente distinte (accuratezza predittiva,
    # valore economico, direzionalità), ognuna riportata come propria
    # tabella in tesi. p-value aggiustati salvati per intero (colonna
    # "p Holm"), non solo Sì/No.
    print(f"\n{'='*64}")
    print("  Correzione di Holm (5 famiglie x 48 test = 12 modelli x 4 asset)")
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

    # (b)-(e): alpha statica, alpha RCS, DA bootstrap statica, DA bootstrap RCS
    families = [
        ("b", "Alpha p-value",        "Alpha p Holm",        "Alpha Sig. Holm (5%)",        bt_results_all),
        ("c", "Alpha p-value",        "Alpha p Holm",        "Alpha Sig. Holm (5%)",        rcs_results_all),
        ("d", "DA Bootstrap p-value", "DA Bootstrap p Holm",  "DA Bootstrap Sig. Holm (5%)", bt_results_all),
        ("e", "DA Bootstrap p-value", "DA Bootstrap p Holm",  "DA Bootstrap Sig. Holm (5%)", rcs_results_all),
    ]
    for fam_id, src_key, holm_key, sig_key, results_all in families:
        pvalues_by_asset = {
            a: {m: bt[src_key] for m, bt in results_all[a].items()
                if m != "Random Walk"}
            for a in results_all
        }
        holm, n_fam = _holm_correct(pvalues_by_asset)
        for a, per_model in holm.items():
            for m, (p_holm, sig) in per_model.items():
                results_all[a][m][holm_key] = p_holm
                results_all[a][m][sig_key]  = sig
        n_sig = sum(sig for per_model in holm.values() for _, sig in per_model.values())
        print(f"  ({fam_id}) {src_key} [{'RCS' if results_all is rcs_results_all else 'static'}] "
              f"— significativi dopo Holm: {n_sig}/{n_fam}")

    # ── Scrittura CSV di riepilogo backtest (ora con le colonne Holm) ──────────
    for asset_name in bt_results_all:
        backtest_summary_table(bt_results_all[asset_name]).to_csv(
            f"{RESULTS_DIR}/backtest_{asset_name}.csv")
        backtest_summary_table(rcs_results_all[asset_name]).to_csv(
            f"{RESULTS_DIR}/backtest_rcs_{asset_name}.csv")

    # ── RQ3: statistica D (prevalenza architettura vs regime) ──────────────────
    # Richiede i dati di tutti e 4 gli asset insieme (pooled) — vedi il design
    # discusso: D positivo -> prevale la variabilità tra regimi a modello
    # fissato; D negativo -> prevale la variabilità tra modelli a regime
    # fissato. asset_data è popolato sia dal percorso live sia da quello di
    # resume, quindi funziona identicamente indipendentemente da quali asset
    # sono stati rieseguiti in questa run e quali ricostruiti da disco.
    print(f"\n{'='*64}")
    print("  RQ3 — Statistica D (architettura vs regime)")
    print(f"{'='*64}")
    d_result = architecture_vs_regime_D(asset_data, block_length=10, n_boot=2000,
                                        min_obs_per_cell=5, seed=42)
    print(f"  D = {d_result['D']:.6f}   "
          f"CI 95% = [{d_result['D CI Lower (95%)']}, {d_result['D CI Upper (95%)']}]")
    print(f"  N Date comuni ai 4 asset (bootstrap congiunto) = "
          f"{d_result['N Common Dates']}")
    print(f"  N Bootstrap Used/Requested = "
          f"{d_result['N Bootstrap Used']}/{d_result['N Bootstrap Requested']}")
    print(f"  N Obs per Regime (pooled) = {d_result['N Obs per Regime (point estimate)']}")

    pd.DataFrame([{
        "D":                     d_result["D"],
        "D CI Lower (95%)":      d_result["D CI Lower (95%)"],
        "D CI Upper (95%)":      d_result["D CI Upper (95%)"],
        "N Common Dates":        d_result["N Common Dates"],
        "Block Length":          d_result["Block Length"],
        "N Bootstrap Requested": d_result["N Bootstrap Requested"],
        "N Bootstrap Used":      d_result["N Bootstrap Used"],
        "Min Obs per Cell":      d_result["Min Obs per Cell"],
    }]).to_csv(f"{RESULTS_DIR}/architecture_vs_regime_D.csv", index=False)
    d_result["Sharpe by Model-Regime"].to_csv(f"{RESULTS_DIR}/sharpe_by_model_regime.csv")

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
    for asset_name, bt_res in bt_results_all.items():
        for model_name, bt in bt_res.items():
            sharpe_rows.append({
                "Asset": asset_name, "Model": model_name,
                "Sharpe": bt["Sharpe Ratio"],
                "MaxDD":  bt["Max Drawdown (%)"],
                "DA":     bt["Directional Acc."]
            })
    sharpe_df = pd.DataFrame(sharpe_rows)
    sharpe_df.to_csv(f"{RESULTS_DIR}/backtesting_all_assets.csv", index=False)
    sharpe_ranking = (sharpe_df.groupby("Model")[["Sharpe", "MaxDD", "DA"]]
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