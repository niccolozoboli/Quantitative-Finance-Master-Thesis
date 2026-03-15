"""
Quantitative Finance Master Thesis
====================================
Commodity Price Forecasting: Econometric vs ML vs DL Models
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.preprocessing     import METALS, load_and_preprocess, reconstruct_prices
from src.walk_forward      import get_walk_forward_folds, aggregate_fold_results
from src.utils             import compute_metrics, save_results
from src.regime            import classify_regimes, split_by_regime, regime_summary
from src.statistical_tests import (random_walk_forecast,
                                    compute_random_walk_metrics,
                                    build_dm_table)
from src.visualization     import (
    plot_returns_forecast, plot_price_comparison,
    plot_all_models_prices, plot_garch_volatility,
    plot_heatmap, plot_ranking, plot_dashboard,
    plot_regime_performance, plot_all_models_regime,
    plot_dm_heatmap, plot_rolling_vol_regimes,
    plot_hit_rate_heatmap, plot_sharpe_heatmap,
)

from src.models.arima     import run_arima_fold
from src.models.garch     import run_garch_fold
from src.models.ml_models import (run_rf_fold, run_xgboost_fold,
                                   run_gbm_fold, run_svr_fold, run_dt_fold)
from src.models.dl_models import (run_lstm_fold, run_gru_fold,
                                   run_bilstm_fold, run_transformer_fold)

START = "2010-01-01"
END   = "2025-01-01"

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
]

# ── Plot folder structure ──────────────────────────────────────────────────────
# results/plots/
#   01_logreturn_forecast/     → predicted vs actual log-returns + residuals
#   02_predicted_vs_real_price/→ reconstructed price vs real price (per model)
#   03_all_models_price_overlay/→ all models on same chart vs real price
#   04_volatility_regimes/     → rolling vol + regime shading
#   05_regime_performance/     → RMSE by stable/normal/volatile
#   06_diebold_mariano/        → DM test vs random walk
#   07_heatmaps/               → RMSE/MAE heatmap models x assets
#   08_model_ranking/          → final ranking bar chart
#   09_dashboard/              → summary dashboard


def p(folder, filename):
    """Builds a clean save path."""
    import os
    path = f"results/plots/{folder}/{filename}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def run_pipeline():
    results_list = []

    for metal_name, ticker in METALS.items():
        print(f"\n{'='*64}")
        print(f"  ASSET: {metal_name}  ({ticker})")
        print(f"{'='*64}")

        df = load_and_preprocess(metal_name, ticker, start=START, end=END)
        n  = len(df)
        print(f"  Observations: {n}  "
              f"({df.index[0].date()} → {df.index[-1].date()})")

        # ── Regime classification ─────────────────────────────────────
        regimes = classify_regimes(df)
        print(f"\n  Regime distribution:\n{regime_summary(df).to_string()}")

        plot_rolling_vol_regimes(
            df, regimes, asset_name=metal_name,
            save_path=p("04_volatility_regimes",
                        f"{metal_name}_rolling_volatility_with_regimes"))

        folds = get_walk_forward_folds(n)
        print(f"\n  Walk-forward: {len(folds)} folds × 21 days\n")

        all_true     = {m: [] for m, _ in MODELS}
        all_pred     = {m: [] for m, _ in MODELS}
        all_dates    = {m: [] for m, _ in MODELS}
        all_fmetrics = {m: [] for m, _ in MODELS}

        # ── Fold loop ─────────────────────────────────────────────────
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
                        dates, y_true_fold, y_pred, _ = run_fn(
                            df, train_idx, test_idx)
                    except Exception as e:
                        print(f"    {model_name}: ERROR {e}")
                        all_fmetrics[model_name].append(
                            {"RMSE": None, "MAE": None, "MAPE": None})
                        continue

                # ── Align lengths (ARIMA and some models lose 1 obs) ──
                n_align    = min(len(y_true_fold), len(y_pred), len(dates_fold))
                y_true_aln = np.array(y_true_fold)[-n_align:]
                y_pred_aln = np.array(y_pred)[-n_align:]
                dates_aln  = dates_fold[-n_align:]

                metrics = compute_metrics(y_true_aln, y_pred_aln)
                all_true[model_name].append(y_true_aln)
                all_pred[model_name].append(y_pred_aln)
                all_dates[model_name].append(dates_aln)
                all_fmetrics[model_name].append(metrics)
                print(f"    {model_name:20s}  "
                      f"RMSE={metrics['RMSE']:.6f}  "
                      f"MAE={metrics['MAE']:.6f}")

        # ── GARCH ─────────────────────────────────────────────────────
        print(f"\n  [GARCH — Volatility]")
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
                print(f"    Fold {fi+1}: GARCH ERROR {e}")

        # ── Aggregate metrics ─────────────────────────────────────────
        print(f"\n  ── Aggregated Results ──")
        for model_name, _ in MODELS:
            agg = aggregate_fold_results(all_fmetrics[model_name])
            print(f"    {model_name:20s}  RMSE={agg['RMSE']}  "
                  f"MAE={agg['MAE']}  MAPE={agg['MAPE']}")
            results_list.append({
                "Asset": metal_name,
                "Model": model_name.replace("_", " "),
                **agg
            })

        # ── Concatenate fold predictions ──────────────────────────────
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

        # ── Diebold-Mariano vs Random Walk ────────────────────────────
        print(f"\n  ── Diebold-Mariano Tests ──")
        dm_preds = {
            mn.replace("_", " "): concat[mn]["y_pred"]
            for mn, _ in MODELS
            if mn in concat and mn != "Random_Walk"
        }
        y_true_all = concat["Random_Walk"]["y_true"]
        dm_table   = build_dm_table(y_true_all, dm_preds)
        print(dm_table[["DM Stat", "p-value", "Sig. (5%)"]].to_string())
        dm_table.to_csv(f"results/DM_test_{metal_name}.csv")

        plot_dm_heatmap(
            dm_table, asset_name=metal_name,
            save_path=p("06_diebold_mariano",
                        f"{metal_name}_DM_test_vs_RandomWalk"))

        # ── Regime analysis ───────────────────────────────────────────
        print(f"\n  ── Regime Analysis ──")
        regime_results = {}
        for model_name, _ in MODELS:
            if model_name not in concat:
                continue
            c = concat[model_name]
            splits = split_by_regime(
                c["y_true"], c["y_pred"], c["dates"], regimes)
            regime_results[model_name.replace("_", " ")] = {}
            for regime, (yt, yp) in splits.items():
                m = compute_metrics(yt, yp)
                regime_results[model_name.replace("_", " ")][regime] = m
                print(f"    {model_name:20s} [{regime:8s}]  "
                      f"RMSE={m['RMSE']:.6f}")

        plot_all_models_regime(
            regime_results, asset_name=metal_name, metric="RMSE",
            save_path=p("05_regime_performance",
                        f"{metal_name}_RMSE_by_regime_all_models"))

        # ── Price reconstruction + individual plots ───────────────────
        model_pred_prices = {}

        for model_name, _ in MODELS:
            if model_name not in concat:
                continue
            c          = concat[model_name]
            dates_all  = c["dates"]
            y_pred_all = c["y_pred"]
            y_true_all_m = c["y_true"]
            label      = model_name.replace("_", " ")

            # 01 — log-return forecast + residuals
            plot_returns_forecast(
                pd.Series(y_true_all_m, index=dates_all),
                pd.Series(y_pred_all,   index=dates_all),
                model_name=label, asset_name=metal_name,
                save_path=p("01_logreturn_forecast",
                            f"{metal_name}_{model_name}_logreturn_forecast_vs_actual"))

            # Price reconstruction
            prices_before = df["Close"][df.index < dates_all[0]]
            last_price    = float(prices_before.iloc[-1])
            pred_prices   = reconstruct_prices(last_price, y_pred_all)
            pred_series   = pd.Series(pred_prices, index=dates_all)
            real_prices   = df["Close"].reindex(dates_all, method="nearest")
            model_pred_prices[label] = pred_series

            # 02 — predicted price vs real price
            plot_price_comparison(
                real_prices=real_prices,
                predicted_prices=pred_series,
                model_name=label, asset_name=metal_name,
                save_path=p("02_predicted_vs_real_price",
                            f"{metal_name}_{model_name}_predicted_vs_real_price"))

        # 03 — all models overlay vs real price
        if model_pred_prices:
            ref_dates   = list(model_pred_prices.values())[0].index
            real_prices = df["Close"].reindex(ref_dates, method="nearest")
            plot_all_models_prices(
                real_prices=real_prices,
                pred_prices_dict=model_pred_prices,
                asset_name=metal_name,
                save_path=p("03_all_models_price_overlay",
                            f"{metal_name}_all_models_predicted_vs_real_price"))

        # GARCH volatility
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

    # ── Final comparison charts ───────────────────────────────────────────────
    results_df = save_results(results_list, path="results/metrics.csv")
    valid_df   = results_df.dropna(subset=["RMSE"])

    print("\n" + "="*64)
    print("  FINAL RANKING  (mean RMSE across all assets)")
    print("="*64)
    ranking = valid_df.groupby("Model")[["RMSE","MAE"]].mean().sort_values("RMSE")
    print(ranking.to_string())

    for metric in ["RMSE", "MAE"]:
        plot_heatmap(
            valid_df, metric=metric,
            save_path=p("07_heatmaps",
                        f"heatmap_{metric}_all_models_all_assets"))
        plot_ranking(
            valid_df, metric=metric,
            save_path=p("08_model_ranking",
                        f"model_ranking_mean_{metric}_all_assets"))

    plot_dashboard(
        valid_df,
        save_path=p("09_dashboard",
                    "summary_dashboard_all_models_all_assets"))

    if "Hit_Rate" in valid_df.columns:
        plot_hit_rate_heatmap(
            valid_df,
            save_path=p("07_heatmaps",
                        "heatmap_HitRate_all_models_all_assets"))
    if "Sharpe" in valid_df.columns:
        plot_sharpe_heatmap(
            valid_df,
            save_path=p("07_heatmaps",
                        "heatmap_Sharpe_all_models_all_assets"))

    print("\n✅  Pipeline complete. Output structure:")
    print("    results/")
    print("    ├── metrics.csv                         ← tutti i numeri")
    print("    ├── DM_test_Gold.csv                    ← DM test per asset")
    print("    └── plots/")
    print("        ├── 01_logreturn_forecast/           log-return pred vs actual")
    print("        ├── 02_predicted_vs_real_price/      prezzo predetto vs reale")
    print("        ├── 03_all_models_price_overlay/     tutti i modelli vs prezzo reale")
    print("        ├── 04_volatility_regimes/           volatilità + regimi + GARCH")
    print("        ├── 05_regime_performance/           RMSE per regime")
    print("        ├── 06_diebold_mariano/              DM test vs random walk")
    print("        ├── 07_heatmaps/                     heatmap RMSE/MAE")
    print("        ├── 08_model_ranking/                classifica finale")
    print("        └── 09_dashboard/                    dashboard riassuntivo")

    return results_df


if __name__ == "__main__":
    run_pipeline()