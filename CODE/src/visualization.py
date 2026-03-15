"""
visualization.py
----------------
All thesis plots including the KEY feature: predicted vs real prices.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f9f9f9",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
})

MODEL_COLORS = {
    "ARIMA":             "#2166AC",
    "Decision Tree":     "#D73027",
    "Random Forest":     "#7B3294",
    "Gradient Boosting": "#E08214",
    "XGBoost":           "#1A9850",
    "SVR":               "#F4A582",
    "LSTM":              "#E41A1C",
    "GRU":               "#377EB8",
    "BiLSTM":            "#4DAF4A",
    "Transformer":       "#984EA3",
    "GARCH":             "#999999",
}


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 1. Log-return forecast vs actual ─────────────────────────────────────────
def plot_returns_forecast(y_true: pd.Series, y_pred: pd.Series,
                           model_name: str, asset_name: str,
                           save_path: str = None):
    """
    Two-panel plot:
    Top:    actual vs predicted log-returns
    Bottom: residuals (actual - predicted)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                              gridspec_kw={"height_ratios": [3, 1]})
    color = MODEL_COLORS.get(model_name, "tomato")

    ax = axes[0]
    ax.plot(y_true.index, y_true.values, label="Actual log-return",
            color="steelblue", linewidth=1.5)
    ax.plot(y_pred.index, y_pred.values, label=f"{model_name} forecast",
            color=color, linewidth=1.5, linestyle="--", alpha=0.85)
    ax.fill_between(y_true.index, y_true.values, y_pred.values,
                    alpha=0.08, color=color)
    ax.set_title(f"{asset_name} — {model_name}: Log-Return Forecast vs Actual",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Log-Return")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    residuals = y_true.values - y_pred.values
    ax2.bar(y_true.index, residuals, color="steelblue", alpha=0.5, width=1)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Residuals")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    _save(fig, save_path)


# ── 2. ★ KEY PLOT: Reconstructed prices vs real prices ★ ─────────────────────
def plot_price_comparison(real_prices: pd.Series,
                           predicted_prices: pd.Series,
                           model_name: str,
                           asset_name: str,
                           save_path: str = None):
    """
    Compares reconstructed predicted prices against real observed prices.
    This is the central visualization of the thesis.

    real_prices:      actual Close prices over the test period
    predicted_prices: prices reconstructed from predicted log-returns
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    color = MODEL_COLORS.get(model_name, "tomato")

    real_vals = np.array(real_prices.values).flatten()
    pred_vals = np.array(predicted_prices.values).flatten()
    # Align lengths
    n = min(len(real_vals), len(pred_vals))
    idx       = real_prices.index[-n:]
    real_vals = real_vals[-n:]
    pred_vals = pred_vals[-n:]

    ax.plot(idx, real_vals,
            label="Real Price", color="black", linewidth=2, zorder=5)
    ax.plot(idx, pred_vals,
            label=f"{model_name} Predicted Price",
            color=color, linewidth=1.8, linestyle="--", alpha=0.85)
    ax.fill_between(idx, real_vals, pred_vals,
                    alpha=0.08, color=color)

    ax.set_title(f"{asset_name} — {model_name}: Predicted vs Real Price",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, save_path)


# ── 3. All models price comparison — one asset ────────────────────────────────
def plot_all_models_prices(real_prices: pd.Series,
                            pred_prices_dict: dict,
                            asset_name: str,
                            save_path: str = None):
    """
    Overlays all models' reconstructed price paths against the real price.
    pred_prices_dict = {"LSTM": pd.Series, "ARIMA": pd.Series, ...}
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(real_prices.index, real_prices.values,
            label="Real Price", color="black", linewidth=2.5, zorder=10)

    for model_name, pred in pred_prices_dict.items():
        color = MODEL_COLORS.get(model_name, "gray")
        ax.plot(pred.index, pred.values,
                label=model_name, color=color,
                linewidth=1.3, linestyle="--", alpha=0.75)

    ax.set_title(f"{asset_name} — All Models: Predicted vs Real Price",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=8, ncol=2, loc="upper left",
              bbox_to_anchor=(1.01, 1), framealpha=0.9)
    plt.tight_layout()
    _save(fig, save_path)


# ── 4. GARCH volatility plot ──────────────────────────────────────────────────
def plot_garch_volatility(realized_vol: pd.Series,
                           forecast_vol: pd.Series,
                           asset_name: str,
                           save_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(realized_vol.index, realized_vol.values,
            label="|Log-return| (Realized Vol)", color="steelblue",
            linewidth=1.2, alpha=0.8)
    ax.plot(forecast_vol.index, forecast_vol.values,
            label="GARCH(1,1) Forecast Vol", color="#999999",
            linewidth=1.8, linestyle="--")
    ax.set_title(f"{asset_name} — GARCH(1,1): Volatility Forecast vs Realized",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (|log-return|)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)


# ── 5. Metrics heatmap ────────────────────────────────────────────────────────
def plot_heatmap(results_df: pd.DataFrame, metric: str = "RMSE",
                  save_path: str = None):
    pivot = results_df.pivot(index="Model", columns="Asset", values=metric)
    pivot = pivot.sort_values(pivot.columns[0])

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values.astype(float), cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            try:
                txt = f"{float(val):.6f}"
            except (TypeError, ValueError):
                txt = "N/A"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"{metric} — Models × Assets (log-return scale)",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, save_path)


# ── 6. Model ranking ──────────────────────────────────────────────────────────
def plot_ranking(results_df: pd.DataFrame, metric: str = "RMSE",
                  save_path: str = None):
    ranking = (results_df.groupby("Model")[metric]
               .mean().sort_values().reset_index())
    colors  = [MODEL_COLORS.get(m, "steelblue") for m in ranking["Model"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(ranking["Model"], ranking[metric],
                   color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, ranking[metric]):
        try:
            ax.text(val * 1.005, bar.get_y() + bar.get_height() / 2,
                    f"{float(val):.6f}", va="center", fontsize=9)
        except (TypeError, ValueError):
            pass

    ax.set_xlabel(f"Mean {metric} across all assets (log-return scale)")
    ax.set_title(f"Model Ranking — Mean {metric}",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, save_path)


# ── 7. Summary dashboard ──────────────────────────────────────────────────────
def plot_dashboard(results_df: pd.DataFrame,
                    save_path: str = "results/plots/comparison/dashboard.png"):
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    for col, metric in enumerate(["RMSE", "MAE"]):
        pivot = results_df.pivot(index="Model", columns="Asset",
                                  values=metric).sort_values(
                                      results_df["Asset"].iloc[0])
        ax_h = fig.add_subplot(gs[0, col])
        im = ax_h.imshow(pivot.values.astype(float), cmap="YlOrRd", aspect="auto")
        ax_h.set_xticks(range(len(pivot.columns)))
        ax_h.set_xticklabels(pivot.columns, fontsize=9)
        ax_h.set_yticks(range(len(pivot.index)))
        ax_h.set_yticklabels(pivot.index, fontsize=8)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                try:
                    ax_h.text(j, i, f"{float(pivot.values[i,j]):.5f}",
                               ha="center", va="center", fontsize=7)
                except (TypeError, ValueError):
                    pass
        plt.colorbar(im, ax=ax_h)
        ax_h.set_title(f"{metric} Heatmap", fontsize=11, fontweight="bold")

        ranking = (results_df.groupby("Model")[metric]
                   .mean().sort_values().reset_index())
        ax_r = fig.add_subplot(gs[1, col])
        colors = [MODEL_COLORS.get(m, "steelblue") for m in ranking["Model"]]
        bars = ax_r.barh(ranking["Model"], ranking[metric],
                          color=colors, edgecolor="white", height=0.6)
        for bar, val in zip(bars, ranking[metric]):
            try:
                ax_r.text(val * 1.005, bar.get_y() + bar.get_height() / 2,
                           f"{float(val):.6f}", va="center", fontsize=8)
            except (TypeError, ValueError):
                pass
        ax_r.invert_yaxis()
        ax_r.set_title(f"Ranking — {metric}", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Model Performance Summary — Metal Commodity Forecasting\n"
        "(metrics on log-return scale, averaged over 5 walk-forward folds)",
        fontsize=13, fontweight="bold", y=1.01
    )
    _save(fig, save_path)
    print(f"Dashboard saved → {save_path}")


# ── 8. Regime-conditional performance ─────────────────────────────────────────
def plot_regime_performance(regime_metrics: dict,
                             model_name: str,
                             asset_name: str,
                             save_path: str = None):
    """
    Bar chart comparing RMSE across stable / normal / volatile regimes
    for a single model and asset.

    regime_metrics: {"stable": {"RMSE": ...}, "normal": {...}, "volatile": {...}}
    """
    regimes = ["stable", "normal", "volatile"]
    colors  = {"stable": "#2166AC", "normal": "#FEE090", "volatile": "#D73027"}
    rmse_vals = [regime_metrics.get(r, {}).get("RMSE", np.nan) for r in regimes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(regimes, rmse_vals,
                  color=[colors[r] for r in regimes],
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, rmse_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val * 1.01, f"{val:.6f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{asset_name} — {model_name}: RMSE by Market Regime",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("RMSE (log-return scale)")
    ax.set_xlabel("Market Regime")
    plt.tight_layout()
    _save(fig, save_path)


def plot_all_models_regime(regime_results: dict,
                            asset_name: str,
                            metric: str = "RMSE",
                            save_path: str = None):
    """
    Grouped bar chart: models × regimes for one asset.
    regime_results: {model_name: {"stable": {RMSE:...}, "volatile": {...}}}
    """
    regimes = ["stable", "normal", "volatile"]
    models  = list(regime_results.keys())
    x       = np.arange(len(models))
    width   = 0.25
    colors  = {"stable": "#2166AC", "normal": "#FEE090", "volatile": "#D73027"}

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, regime in enumerate(regimes):
        vals = [regime_results[m].get(regime, {}).get(metric, np.nan)
                for m in models]
        ax.bar(x + i*width, vals, width=width*0.9,
               label=regime.capitalize(),
               color=colors[regime], edgecolor="white", alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(f"{metric} (log-return scale)")
    ax.set_title(f"{asset_name} — {metric} by Model and Market Regime",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Regime", fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)


# ── 9. Diebold-Mariano summary table plot ────────────────────────────────────
def plot_dm_heatmap(dm_df: pd.DataFrame,
                    asset_name: str,
                    save_path: str = None):
    """
    Visual heatmap of DM test p-values (models vs random walk).
    Green = significantly beats RW, Red = does not.
    """
    pvals = dm_df["p-value"].values.astype(float)
    models = dm_df.index.tolist()

    fig, ax = plt.subplots(figsize=(4, len(models) * 0.55 + 1.5))
    colors = ["#1A9850" if p < 0.05 else "#D73027" for p in pvals]

    bars = ax.barh(models, pvals, color=colors, edgecolor="white", height=0.6)
    ax.axvline(0.05, color="black", linewidth=1.2,
               linestyle="--", label="α = 0.05")

    for bar, val, sig in zip(bars, pvals, dm_df["Sig. (5%)"].values):
        label = f"p={val:.3f} {'✓' if sig=='Yes' else '✗'}"
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=8)

    ax.set_xlabel("DM Test p-value")
    ax.set_title(f"{asset_name} — DM Test vs Random Walk\n"
                 f"Green = beats RW (p<0.05)",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)


# ── 10. Rolling volatility with regime shading ───────────────────────────────
def plot_rolling_vol_regimes(df,
                              regimes,
                              asset_name: str,
                              save_path: str = None):
    """
    Plots rolling volatility with regime background shading.
    Useful for Section 3.1 in the thesis.
    """
    from src.regime import compute_rolling_vol
    rv = compute_rolling_vol(df)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(rv.index, rv.values, color="steelblue",
            linewidth=1.2, label="Rolling 21-day volatility")

    # Shade regimes
    regime_colors = {"stable": "#DEEBF7", "volatile": "#FCBBA1", "normal": "#F7F7F7"}
    prev_regime, start = None, None
    for date, regime in regimes.items():
        if regime != prev_regime:
            if prev_regime is not None:
                ax.axvspan(start, date,
                           color=regime_colors[prev_regime], alpha=0.4)
            start = date
            prev_regime = regime
    if prev_regime is not None:
        ax.axvspan(start, regimes.index[-1],
                   color=regime_colors[prev_regime], alpha=0.4)

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, alpha=0.4, label=r.capitalize())
               for r, c in regime_colors.items()]
    ax.legend(handles=patches + [ax.lines[0]], fontsize=9)

    ax.set_title(f"{asset_name} — Rolling Volatility with Regime Classification",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Rolling Std of Log-Return")
    ax.set_xlabel("Date")
    plt.tight_layout()
    _save(fig, save_path)


# ── 11. Hit Rate heatmap ──────────────────────────────────────────────────────
def plot_hit_rate_heatmap(results_df: pd.DataFrame,
                           save_path: str = None):
    """
    Heatmap of Hit Rate (%) — models × assets.
    50% = random, >50% = some directional skill.
    """
    pivot = results_df.pivot(index="Model", columns="Asset",
                              values="Hit_Rate")
    pivot = pivot.sort_values(pivot.columns[0], ascending=False)

    fig, ax = plt.subplots(figsize=(11, 8))
    # Center colormap on 50% (random baseline)
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn",
                   aspect="auto", vmin=40, vmax=60)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            try:
                txt = f"{float(val):.1f}%"
                # Bold if above 50%
                fw = "bold" if float(val) > 50 else "normal"
            except (TypeError, ValueError):
                txt = "N/A"; fw = "normal"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, fontweight=fw)

    plt.colorbar(im, ax=ax, label="Hit Rate (%)")
    ax.axhline(-0.5, color="white", linewidth=0)
    ax.set_title("Hit Rate — Directional Accuracy by Model and Asset\n"
                 "(>50% = better than random  |  green = skill  |  red = no skill)",
                 fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    _save(fig, save_path)


# ── 12. Sharpe Ratio heatmap ──────────────────────────────────────────────────
def plot_sharpe_heatmap(results_df: pd.DataFrame,
                         save_path: str = None):
    """
    Heatmap of annualized Sharpe Ratio from simulated long/short strategy.
    Sharpe > 0 = strategy adds value on average.
    Sharpe > 1 = practically good.
    """
    pivot = results_df.pivot(index="Model", columns="Asset",
                              values="Sharpe")
    pivot = pivot.sort_values(pivot.columns[0], ascending=False)

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn",
                   aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            try:
                txt = f"{float(val):.3f}"
                fw = "bold" if float(val) > 0 else "normal"
            except (TypeError, ValueError):
                txt = "N/A"; fw = "normal"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, fontweight=fw)

    plt.colorbar(im, ax=ax, label="Annualized Sharpe Ratio")
    ax.set_title("Simulated Long/Short Strategy — Annualized Sharpe Ratio\n"
                 "(No transaction costs  |  >0 = profitable on average  |  >1 = practically good)",
                 fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    _save(fig, save_path)