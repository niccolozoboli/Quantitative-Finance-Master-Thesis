import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.stats.multitest import multipletests

from src.preprocessing     import METALS, load_and_preprocess, reconstruct_prices
from src.walk_forward      import get_walk_forward_folds, aggregate_fold_results
from src.utils             import compute_metrics, save_results, set_global_seed
from src.regime            import classify_regimes, split_by_regime, regime_summary
from src.statistical_tests import (random_walk_forecast,
                                    build_dm_table,
                                    build_adf_table)
from src.backtesting       import (run_backtest, regime_conditional_strategy,
                                    backtest_summary_table, block_bootstrap_ci)
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
from src.models.dl_models import (run_lstm_fold, run_gru_fold,
                                   run_bilstm_fold, run_transformer_fold,
                                   run_tcn_fold, run_cnn_lstm_fold)

START = "2010-01-01"
END   = "2026-05-01"   # aggiornato — include dati fino ad oggi

MODELS = [
    ("Random_Walk",       None),
    ("ARIMA",             run_arima_fold),
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
    import os
    path = f"results/plots/{folder}/{filename}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def run_pipeline():
    set_global_seed(42)

    results_list   = []
    bt_results_all = {}
    dm_tables_all  = {}   # accumula le dm_table per asset — la correzione di
                           # Holm è globale sui 48 test (12 modelli × 4 asset),
                           # quindi il salvataggio su CSV è rimandato a dopo
                           # il loop principale (vedi sezione dedicata sotto)

    # ── Carica tutti gli asset prima del loop ─────────────────────────────────
    print("\nCaricamento dati...")
    all_dfs = {}
    for metal_name, ticker in METALS.items():
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
    adf_table.to_csv("results/adf_test.csv")

    # ── Loop principale per asset ─────────────────────────────────────────────
    for metal_name, ticker in METALS.items():
        print(f"\n{'='*64}")
        print(f"  ASSET: {metal_name}  ({ticker})")
        print(f"{'='*64}")

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

        plot_rolling_vol_regimes(
            df, regimes, asset_name=metal_name,
            save_path=p("04_volatility_regimes",
                        f"{metal_name}_rolling_volatility_with_regimes"))

        folds = get_walk_forward_folds(n)
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

                        dates, y_true_fold, y_pred, _ = result

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
        pred_df.to_csv(f"results/predictions_{metal_name}.csv", index=False)

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
                c["y_true"], c["y_pred"], c["dates"], regimes)
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

        # ── BACKTESTING ───────────────────────────────────────────────────────
        print(f"\n  ── Backtesting ──")
        bt_results = {}
        for model_name, _ in MODELS:
            if model_name not in concat:
                continue
            c  = concat[model_name]
            bt = run_backtest(c["y_true"], c["y_pred"],
                              transaction_cost=0.0001)
            bt_ci = block_bootstrap_ci(c["y_true"], c["y_pred"],
                                       transaction_cost=0.0001,
                                       block_length=10, n_boot=2000, seed=42)
            bt.update(bt_ci)
            bt_results[model_name.replace("_", " ")] = bt
            print(f"    {model_name:22s}  "
                  f"Sharpe={bt['Sharpe Ratio']:6.3f} "
                  f"[{bt['Sharpe CI Lower (95%)']:.2f}, {bt['Sharpe CI Upper (95%)']:.2f}]  "
                  f"DA={bt['Directional Acc.']:5.1f}%  "
                  f"MaxDD={bt['Max Drawdown (%)']:6.2f}%")

        bt_df = backtest_summary_table(bt_results)
        bt_df.to_csv(f"results/backtest_{metal_name}.csv")
        bt_results_all[metal_name] = bt_results

        # ── REGIME-CONDITIONAL STRATEGY ───────────────────────────────────────
        print(f"\n  ── Regime-Conditional Strategy ──")
        rcs_results = {}
        for model_name, _ in MODELS:
            if model_name not in concat:
                continue
            c          = concat[model_name]
            y_pred_rcs = regime_conditional_strategy(
                c["y_pred"], c["dates"], regimes)
            bt_rcs = run_backtest(c["y_true"], y_pred_rcs,
                                  transaction_cost=0.0001)
            bt_rcs_ci = block_bootstrap_ci(c["y_true"], y_pred_rcs,
                                           transaction_cost=0.0001,
                                           block_length=10, n_boot=2000, seed=42)
            bt_rcs.update(bt_rcs_ci)
            rcs_results[model_name.replace("_", " ")] = bt_rcs
            print(f"    {model_name:22s} [RCS]  "
                  f"Sharpe={bt_rcs['Sharpe Ratio']:6.3f}  "
                  f"DA={bt_rcs['Directional Acc.']:5.1f}% "
                  f"[{bt_rcs['DA CI Lower (95%)']:.1f}, {bt_rcs['DA CI Upper (95%)']:.1f}]")

        rcs_df = backtest_summary_table(rcs_results)
        rcs_df.to_csv(f"results/backtest_rcs_{metal_name}.csv")

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

    # ── Correzione di Holm per comparazioni multiple (globale, 48 test) ────────
    # Famiglia = tutti i confronti DM del progetto (12 modelli × 4 asset),
    # coerente con la statistica aggregata "X/48" già usata per riportare i
    # rifiuti DM. Non modifica DM Stat/p-value originali — aggiunge solo la
    # colonna "Sig. Holm (5%)" prima di scrivere i CSV definitivi.
    print(f"\n{'='*64}")
    print("  Correzione di Holm (globale, 48 test = 12 modelli × 4 asset)")
    print(f"{'='*64}")
    asset_order  = list(dm_tables_all.keys())
    rows_per_asset = [len(dm_tables_all[a]) for a in asset_order]
    all_pvalues  = np.concatenate(
        [dm_tables_all[a]["p-value"].values for a in asset_order])

    # Guardia esplicita contro un disallineamento silenzioso: se in futuro
    # un asset avesse un numero diverso di modelli/righe (es. un modello
    # fallito ed escluso solo per quell'asset), il totale concatenato deve
    # comunque coincidere con la somma delle righe delle singole dm_table —
    # altrimenti lo split per-asset più sotto assegnerebbe la colonna Holm
    # alle righe sbagliate senza sollevare alcun errore.
    assert len(all_pvalues) == sum(rows_per_asset), (
        f"Disallineamento p-value/righe: {len(all_pvalues)} p-value "
        f"concatenati ma {sum(rows_per_asset)} righe totali nelle dm_table "
        f"({dict(zip(asset_order, rows_per_asset))})"
    )

    reject_holm, _, _, _ = multipletests(all_pvalues, alpha=0.05, method="holm")

    offset = 0
    n_sig_raw, n_sig_holm = 0, 0
    for asset_name, n_rows in zip(asset_order, rows_per_asset):
        dm_table = dm_tables_all[asset_name]
        dm_table["Sig. Holm (5%)"] = [
            "Yes" if r else "No"
            for r in reject_holm[offset:offset + n_rows]
        ]
        offset += n_rows
        n_sig_raw  += int((dm_table["p-value"] < 0.05).sum())
        n_sig_holm += int((dm_table["Sig. Holm (5%)"] == "Yes").sum())
        dm_table.to_csv(f"results/DM_test_{asset_name}.csv")

    assert offset == len(all_pvalues), (
        f"Split per-asset incompleto: consumati {offset} valori su "
        f"{len(all_pvalues)} totali dopo Holm — colonna Holm probabilmente "
        f"disallineata per almeno un asset"
    )

    print(f"  Significativi (p-value grezzo < 0.05): {n_sig_raw}/{len(all_pvalues)}")
    print(f"  Significativi dopo Holm (5%):          {n_sig_holm}/{len(all_pvalues)}")

    # ── Output finale ─────────────────────────────────────────────────────────
    results_df = save_results(results_list, path="results/metrics.csv")
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
    sharpe_df.to_csv("results/backtesting_all_assets.csv", index=False)
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

    return results_df


if __name__ == "__main__":
    run_pipeline()