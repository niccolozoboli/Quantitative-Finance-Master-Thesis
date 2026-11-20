"""Figure del Capitolo 4 dai CSV (commit 2586f06). Uso: python3 figure_cap4.py <risultati> <output>"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
A = ["Gold", "Silver", "Copper", "Platinum"]
ML = ["ARMA", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "SVR",
      "LSTM", "GRU", "BiLSTM", "Transformer", "TCN", "CNN LSTM"]
DL = ["LSTM", "GRU", "BiLSTM", "Transformer", "TCN", "CNN LSTM"]
lab = lambda m: "CNN-LSTM" if m == "CNN LSTM" else m
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
rd = lambda n: pd.read_csv(f"{RES}/{n}")
load = lambda k: pd.concat([rd(f"{k}_{a}.csv").assign(Asset=a) for a in A], ignore_index=True)
S, R, bm = load("backtest"), load("backtest_rcs"), rd("rcs_benchmarks.csv").set_index("Asset")
order = ML[::-1]
y = np.arange(len(order))

# ---- Figura 1: Sharpe statico con IC e long passivo
fig, axs = plt.subplots(1, 4, figsize=(7.2, 3.9), sharey=True)
for ax, a in zip(axs, A):
    d = S[S.Asset == a].set_index("Model").loc[order]
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.axvline(bm.loc[a, "Sharpe Long Passive"], color=MUTED, lw=1, ls=(0, (3, 2)), zorder=1)
    ax.hlines(y, d["Sharpe CI Lower (95%)"], d["Sharpe CI Upper (95%)"], color=BLUE, alpha=0.35, lw=2, zorder=2)
    ax.plot(d["Sharpe Ratio"], y, "o", ms=4.5, color=BLUE, mec="white", mew=0.8, zorder=3)
    ax.set_title(a); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Sharpe ratio")
    ax.tick_params(length=0)
axs[0].set_yticks(y); axs[0].set_yticklabels([lab(m) for m in order])
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=BLUE, ms=4.5), Line2D([], [], color=BLUE, alpha=0.35, lw=2),
     Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 2)))]
fig.legend(h, ["Sharpe ratio, static", "95% block-bootstrap interval", "Passive long"], loc="lower center", ncol=3,
           frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{OUT}/fig4_sharpe_static.png"); plt.close(fig)

# ---- Figura 2: statico vs RCS, con benchmark della RCS
fig, axs = plt.subplots(1, 4, figsize=(7.2, 3.9), sharey=True)
for ax, a in zip(axs, A):
    ds = S[S.Asset == a].set_index("Model").loc[order]["Sharpe Ratio"]
    dr = R[R.Asset == a].set_index("Model").loc[order]["Sharpe Ratio"]
    ax.axvspan(-10, bm.loc[a, "Sharpe Random Filtered P95"], color="#f0efec", zorder=0)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.axvline(bm.loc[a, "Sharpe Long Filtered"], color=MUTED, lw=1, ls=(0, (3, 2)), zorder=1)
    ax.hlines(y, np.minimum(ds, dr), np.maximum(ds, dr), color=GRID, lw=1.5, zorder=2)
    ax.plot(ds, y, "o", ms=4.5, color=BLUE, mec="white", mew=0.8, zorder=3, label="Static")
    ax.plot(dr, y, "D", ms=4.2, color=ORANGE, mec="white", mew=0.8, zorder=4, label="RCS")
    lo = min(ds.min(), dr.min(), 0) - 0.4; hi = max(ds.max(), dr.max(), bm.loc[a, "Sharpe Long Filtered"]) + 0.4
    ax.set_xlim(lo, hi); ax.set_title(a); ax.set_xlabel("Sharpe ratio"); ax.tick_params(length=0)
axs[0].set_yticks(y); axs[0].set_yticklabels([lab(m) for m in order])
h, l = axs[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h.append(Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 2)))); l.append("Filtered long")
h.append(plt.Rectangle((0, 0), 1, 1, color="#f0efec")); l.append("Below the 95th percentile of the random signal with the RCS rule")
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(f"{OUT}/fig4_rcs_vs_static.png"); plt.close(fig)

# ---- Figura 3: dispersione dei seed
sm = rd("dl_seed_metrics.csv")
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.9), sharey=True)
yy = np.arange(len(DL))[::-1]
for ax, a in zip(axs, A):
    ax.axvline(0, color=AXIS, lw=0.8)
    for i, mdl in zip(yy, DL):
        d = sm[(sm.Asset == a) & (sm.Model == mdl)]
        o = d[d.Seed != 42]; m42 = d[d.Seed == 42]
        ax.hlines(i, d.Sharpe.min(), d.Sharpe.max(), color=GRID, lw=1.5, zorder=1)
        ax.plot(o.Sharpe, [i] * len(o), "o", ms=4, color=BLUE, alpha=0.55, mec="white", mew=0.6, zorder=2,
                label="Seeds 1–4" if (i == yy[0]) else None)
        ax.plot(m42.Sharpe, [i], "D", ms=4.5, color=ORANGE, mec="white", mew=0.8, zorder=3,
                label="Seed 42 (main)" if (i == yy[0]) else None)
    ax.set_title(a); ax.set_xlabel("Sharpe ratio"); ax.tick_params(length=0); ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axs[0].set_yticks(yy); axs[0].set_yticklabels([lab(m) for m in DL])
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f"{OUT}/fig4_seeds.png"); plt.close(fig)
print("ok")
