"""
regenerate_plots.py
--------------------
Rigenera/aggiunge un sottoinsieme di grafici leggendo SOLO i CSV già
salvati in results/. Nessun ricalcolo di feature, split, training o
metriche — non importa nulla da src/models, src/feature_engine,
src/walk_forward. Non modifica main.py.

Genera:
  1. heatmap_Sharpe_all_models_all_assets.png
     heatmap_HitRate_all_models_all_assets.png
     → da results/backtesting_all_assets.csv (colonne Sharpe, DA)
  2. summary_dashboard_all_models_all_assets.png
     → da results/metrics.csv, con titolo corretto (10 fold, non 5)
  3. model_ranking_mean_RMSE_all_assets.png
     → da results/metrics.csv, con linea tratteggiata sul Random Walk

Uso: da dentro CODE/
    python3 regenerate_plots.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from src.visualization import (
    plot_hit_rate_heatmap, plot_sharpe_heatmap, plot_dashboard,
    MODEL_COLORS,
)

RESULTS_DIR = "results"


def p(folder, filename):
    path = f"{RESULTS_DIR}/plots/{folder}/{filename}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 1. Heatmap Sharpe + Hit Rate (Directional Accuracy) ──────────────────────
def make_sharpe_hitrate_heatmaps():
    bt = pd.read_csv(f"{RESULTS_DIR}/backtesting_all_assets.csv")
    # colonne: Asset, Model, Sharpe, MaxDD, DA

    plot_sharpe_heatmap(
        bt[["Asset", "Model", "Sharpe"]],
        save_path=p("07_heatmaps", "heatmap_Sharpe_all_models_all_assets"))

    hitrate_df = bt[["Asset", "Model", "DA"]].rename(columns={"DA": "Hit_Rate"})
    plot_hit_rate_heatmap(
        hitrate_df,
        save_path=p("07_heatmaps", "heatmap_HitRate_all_models_all_assets"))

    print("  ✓ heatmap Sharpe + HitRate → results/plots/07_heatmaps/")


# ── 2. Dashboard (titolo corretto: 10 fold) ───────────────────────────────────
def make_dashboard():
    metrics = pd.read_csv(f"{RESULTS_DIR}/metrics.csv")
    valid = metrics.dropna(subset=["RMSE"])
    plot_dashboard(
        valid,
        save_path=p("09_dashboard", "summary_dashboard_all_models_all_assets"))


# ── 3. Ranking RMSE con baseline Random Walk ──────────────────────────────────
def plot_ranking_with_rw_baseline(results_df, metric, rw_value, save_path):
    """
    Stessa logica/stile di visualization.plot_ranking, con l'aggiunta
    di una linea verticale tratteggiata sul valore del Random Walk.
    Duplicata qui (invece di modificare plot_ranking) perché quella
    funzione è condivisa anche dal grafico MAE, che non deve avere
    la linea.
    """
    ranking = (results_df.groupby("Model")[metric]
               .mean().sort_values().reset_index())
    colors = [MODEL_COLORS.get(m, "steelblue") for m in ranking["Model"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(ranking["Model"], ranking[metric],
                    color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, ranking[metric]):
        try:
            ax.text(val * 1.005, bar.get_y() + bar.get_height() / 2,
                    f"{float(val):.6f}", va="center", fontsize=9)
        except (TypeError, ValueError):
            pass

    ax.axvline(rw_value, color="black", linewidth=1.2, linestyle="--",
               label=f"Random Walk ({rw_value:.6f})")
    ax.legend(fontsize=9, loc="lower right")

    ax.set_xlabel(f"Mean {metric} across all assets (log-return scale)")
    ax.set_title(f"Model Ranking — Mean {metric}",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, save_path)


def make_ranking_rmse():
    metrics = pd.read_csv(f"{RESULTS_DIR}/metrics.csv")
    valid = metrics.dropna(subset=["RMSE"])

    rw_value = valid.loc[valid["Model"] == "Random Walk", "RMSE"].mean()

    plot_ranking_with_rw_baseline(
        valid, "RMSE", rw_value,
        save_path=p("08_model_ranking", "model_ranking_mean_RMSE_all_assets"))

    print(f"  ✓ ranking RMSE con baseline Random Walk = {rw_value:.6f} "
          f"→ results/plots/08_model_ranking/")


if __name__ == "__main__":
    print("Rigenerazione plot da CSV esistenti in results/ (nessun ricalcolo)...")
    make_sharpe_hitrate_heatmaps()
    make_dashboard()
    make_ranking_rmse()
    print("Fatto.")
