"""
tools/thesis_values_cap4.py
---------------------------
Calcola tutti i valori e i corpi delle tabelle dei Capitoli 4 e 5 e
dell'Appendice A della tesi, a partire da:
  1. la cartella dei risultati dell'esecuzione completa (commit 2586f06);
  2. la cartella dei prezzi (CODE/data);
  3. l'output di tools/recompute_post_run.py (statistica D per asset,
     DM con Holm sui p non arrotondati, GARCH dell'oro, benchmark della RCS
     e statistica D con la variante eseguibile).
Scrive un JSON {"V": valori, "T": corpi delle tabelle in LaTeX}. È lo
script con cui sono stati generati i numeri del testo: nessun numero dei
Capitoli 4–5 è scritto a mano.

Uso, da CODE/:
    python tools/thesis_values_cap4.py <risultati> data <ricalcolo> <out.json>
"""

import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

R = sys.argv[1]
D = sys.argv[2]
RC = sys.argv[3]
OUT_JSON = sys.argv[4] if len(sys.argv) > 4 else "thesis_values_cap4.json"
A = ["Gold", "Silver", "Copper", "Platinum"]
TK = {"Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F", "Platinum": "PL=F"}
ORDER = ["ARMA", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "SVR",
         "LSTM", "GRU", "BiLSTM", "Transformer", "TCN", "CNN-LSTM"]
NN = ["LSTM", "GRU", "BiLSTM", "Transformer", "TCN", "CNN-LSTM"]
TREES = ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]
ENS = ["Random Forest", "Gradient Boosting", "XGBoost"]
REG = ["stable", "normal", "volatile"]
EXEC_REF = {"Gold": "open", "Silver": "open", "Copper": "open", "Platinum": "lag1"}

V, T = {}, {}


def fix(df, col="Model"):
    df = df.copy(); df[col] = df[col].replace({"CNN LSTM": "CNN-LSTM", "CNN_LSTM": "CNN-LSTM"}); return df


def n(x, d=2, sign=True):
    """Numero in LaTeX con $-$ per i negativi."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{abs(x):.{d}f}"
    if float(s) == 0:
        return s
    return ("$-$" + s) if (x < 0 and sign) else s


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def and_join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def p3(p):
    return "$<$0.001" if p < 0.0005 else f"{p:.3f}"


def sharpe(y):
    y = np.asarray(y, float); s = y.std(ddof=1) if len(y) > 1 else 0
    return np.nan if s <= 1e-12 else np.sqrt(252) * y.mean() / s


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


# ---------------------------------------------------------------- 4.1 accuratezza
met = fix(pd.read_csv(f"{R}/metrics.csv"))
def acc_table(col):
    pv = met.pivot(index="Model", columns="Asset", values=col)[A] * 100
    pv["Mean"] = pv[A].mean(axis=1); pv = pv.sort_values("Mean"); pv["Rank"] = np.arange(1, len(pv) + 1)
    return pv
for key, col in [("rmse", "RMSE"), ("mae", "MAE"), ("pooled", "RMSE pooled")]:
    pv = acc_table(col)
    T[key] = "\n".join(f"{m} & " + " & ".join(f"{pv.loc[m, a]:.3f}" for a in A + ["Mean"]) + f" & {int(pv.loc[m, 'Rank'])} \\\\" for m in pv.index)
rm = acc_table("RMSE"); rw = rm.loc["Random Walk", "Mean"]
V["rmse_first"], V["rmse_second"], V["rmse_third"] = rm.index[:3]
V["arma_vs_rw"] = f"{100 * (1 - rm.loc['ARMA', 'Mean'] / rw):.2f}"
dev = (rm["Mean"] / rw - 1) * 100
V["maxdev"] = f"{dev.drop(['SVR']).max():.1f}"
V["svr_dev"] = f"{dev['SVR']:.1f}"
mae = acc_table("MAE"); V["mae_first"], V["mae_second"] = mae.index[:2]
for a in A:
    col = rm[a]; rwa = col["Random Walk"]; mods = col.drop("Random Walk")
    beat = [m for m in ORDER if mods[m] < rwa]
    V[f"beat_{a}"] = str(len(beat)); V[f"beatlist_{a}"] = ", ".join(beat)
    V[f"best_{a}"] = mods.idxmin(); V[f"bestv_{a}"] = f"{mods.min():.3f}"; V[f"rw_{a}"] = f"{rwa:.3f}"
V["svr_cu"] = f"{rm.loc['SVR', 'Copper']:.3f}"
pc = fix(pd.read_csv(f"{R}/predictions_Copper.csv"))
V["svr_cu_pred"] = n(pc[pc.Model == "SVR"].y_pred.mean() * 100)
V["cu_real"] = n(pc[pc.Model == "Random Walk"].y_true.mean() * 100)
pool = acc_table("RMSE pooled")
V["pooled_gt_all"] = str(bool((pool[A] > rm.loc[pool.index, A]).all().all()))
V["sp_fold_pooled"] = f"{spearmanr(rm['Mean'], pool.loc[rm.index, 'Mean']).statistic:.2f}"

# ---------------------------------------------------------------- 4.2 DM
# R11: p-value non arrotondati e Holm su di essi, da tools/recompute_post_run.py
dm = {a: fix(pd.read_csv(f"{RC}/DM_test_{a}.csv")).set_index("Model") for a in A}
for a in A:
    dm[a]["p-value"] = dm[a]["p-value (raw)"]
dm_run = {a: fix(pd.read_csv(f"{R}/DM_test_{a}.csv")).set_index("Model") for a in A}
rows = []
for m in ORDER:
    cells = []
    for a in A:
        x = dm[a].loc[m]; dag = "$^{\\dagger}$" if x["Sig. Holm (5%)"] == "Yes" else ""
        st = n(x["DM Stat"]); st = f"\\textbf{{{st}}}{dag}" if dag else st
        cells += [st, f"{p3(x['p-value'])} \\riv{{/ {p3(x['p Holm'])}}}"]
    rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
T["dm"] = "\n".join(rows)
alldm = pd.concat([dm[a].assign(Asset=a) for a in A])
V["dm_neg"] = str(int((alldm["DM Stat"] < 0).sum()))
raw = alldm[alldm["p-value"] < 0.05]
V["dm_raw"] = str(len(raw)); V["dm_raw_pos"] = str(int((raw["DM Stat"] > 0).sum()))
negraw = raw[raw["DM Stat"] < 0]
assert len(negraw) == 1
r0 = negraw.iloc[0]; V["dm_negraw_model"] = r0.name; V["dm_negraw_asset"] = r0.Asset
V["dm_negraw_stat"] = n(r0["DM Stat"]); V["dm_negraw_p"] = p3(r0["p-value"]); V["dm_negraw_holm"] = p3(r0["p Holm"])
V["dm_negraw_holm_run"] = f"{dm_run[r0.Asset].loc[r0.name, 'p Holm']:.3f}"
_cmp = pd.read_csv(f"{RC}/dm_holm_comparison.csv")
V["dm_holm_maxdiff"] = f"{(_cmp['p Holm (raw)'] - _cmp['p Holm (run)']).abs().max():.4f}".replace(".", "{,}")
V["dm_holm_changes"] = str(int((_cmp["Sig. Holm (raw)"] != _cmp["Sig. Holm (run)"]).sum()))
V["dm_negraw_holm_it"] = V["dm_negraw_holm"].replace(".", "{,}")
V["dm_negraw_holm_run_it"] = V["dm_negraw_holm_run"].replace(".", "{,}")
assert all(np.isclose(_cmp["DM Stat"], _cmp["DM Stat (run)"]))
hs = alldm[alldm["Sig. Holm (5%)"] == "Yes"]; assert len(hs) == 1
V["dm_holm_model"], V["dm_holm_asset"], V["dm_holm_stat"] = hs.index[0], hs.Asset.iloc[0], n(hs["DM Stat"].iloc[0])

# ---------------------------------------------------------------- regimi
labels = {}
for a in A:
    sr = fix(pd.read_csv(f"{R}/strategy_returns_{a}.csv", parse_dates=["Date"]))
    w = sr[(sr.Model == "ARMA") & (sr.Strategy == "rcs") & (sr.Execution == "close")].set_index("Date").w.abs()
    labels[a] = w.map({1.0: "stable", 0.5: "normal", 0.0: "volatile"})
L = pd.DataFrame(labels)
rows = []
for a in A:
    c = L[a].value_counts(); amap = L[a].map({"stable": 1.0, "normal": 0.5, "volatile": 0.0})
    exposed = int((amap > 0).sum())
    V[f"st_{a}"], V[f"no_{a}"], V[f"vo_{a}"] = (str(int(c.get(g, 0))) for g in REG)
    V[f"st_{a}_phrase"] = "a single stable day" if int(c.get("stable", 0)) == 1 else f"{int(c.get('stable', 0))} stable days"
    V[f"volsh_{a}"] = f"{100 * c.get('volatile', 0) / len(L):.0f}"
    V[f"expd_{a}"] = str(exposed); V[f"expsh_{a}"] = f"{100 * exposed / len(L):.1f}"
    V[f"abar_{a}"] = f"{100 * amap.mean():.1f}"
    rows.append(f"{a} & {V[f'st_{a}']} & {V[f'no_{a}']} & {V[f'vo_{a}']} & {V[f'volsh_{a}']}\\% & \\riv{{{V[f'expd_{a}']} ({V[f'expsh_{a}']}\\%)}} & \\riv{{{V[f'abar_{a}']}\\%}} \\\\")
T["regimes"] = "\n".join(rows)
cnt = pd.read_csv(f"{R}/D_regime_counts.csv", index_col=0)["N Dates"]
for g in REG: V[f"dcell_{g}"] = str(int(cnt[g]))
assert all(int((L == g).any(axis=1).sum()) == int(cnt[g]) for g in REG)

# ---------------------------------------------------------------- accuratezza per regime
rows, ratios = [], []
for a in A:
    p = fix(pd.read_csv(f"{R}/predictions_{a}.csv", parse_dates=["Date"]))
    rwr = {}
    for g in REG:
        dts = L.index[L[a] == g]; sub = p[p.Date.isin(dts)]
        r_ = sub.groupby("Model").apply(lambda x: np.sqrt(np.mean((x.y_true - x.y_pred) ** 2)) * 100, include_groups=False)
        rwv = r_["Random Walk"]; mm = r_.drop("Random Walk"); rwr[g] = rwv
        rows.append(f"{a} & {g} & {len(dts)} & {rwv:.3f} & {mm.idxmin()} & {mm.min():.3f} & {int((mm < rwv).sum())} \\\\")
        V[f"ra_beat_{a}_{g}"] = str(int((mm < rwv).sum()))
    ratios.append(rwr["volatile"] / rwr["stable"])
T["regacc"] = "\n".join(rows)
V["ra_ratio_lo"], V["ra_ratio_hi"] = f"{min(ratios):.1f}", f"{max(ratios):.1f}"

# ---------------------------------------------------------------- GARCH
# Argento, rame, platino: CSV dell'esecuzione. Oro (ripreso con --resume, R5):
# tools/recompute_post_run.py, stessa funzione di stima della pipeline; lo
# stesso script ricostruisce gli altri tre metalli e garch_validation.csv
# riporta lo scarto massimo dai valori salvati.
g = pd.read_csv(f"{R}/garch_volatility.csv"); q = pd.read_csv(f"{R}/garch_dm_qlike.csv").set_index("Asset")
g_rc = pd.read_csv(f"{RC}/garch_volatility.csv"); q_rc = pd.read_csv(f"{RC}/garch_dm_qlike.csv").set_index("Asset")
assert "Gold" not in set(g.Asset) and "Gold" not in q.index
G = pd.concat([g_rc[g_rc.Asset == "Gold"], g], ignore_index=True)[["Asset", "Dist", "Fold", "persistence", "nu"]]
gval = pd.read_csv(f"{RC}/garch_validation.csv")
V["garch_val_pers"] = f"{gval['max |diff| persistence'].max():.0e}".replace("e-0", "\\times10^{-") + "}"
V["garch_val_dm"] = f"{gval['|diff| DM Stat'].max():.0e}".replace("e-", "\\times10^{-") + "}"
fg = pd.read_csv(f"{R}/garch_forecasts_Gold.csv")
rows = []
for a in A:
    pn = G[(G.Asset == a) & (G.Dist == "normal")].persistence; pt = G[(G.Asset == a) & (G.Dist == "t")].persistence
    nu = G[(G.Asset == a) & (G.Dist == "t")].nu
    V[f"pers_n_{a}"] = f"{pn.min():.4f}--{pn.max():.4f}"; V[f"pers_t_{a}"] = f"{pt.min():.4f}--{pt.max():.4f}"
    V[f"nu_{a}"] = f"{nu.min():.1f}--{nu.max():.1f}"
    V[f"pers_t_higher_{a}"] = str(bool(pt.min() > pn.max()))
    rows.append((a, pn, pt, nu))
# QLIKE: argento, rame, platino dal CSV dell'esecuzione; oro dal ricalcolo
QL = {a: ((q if a in q.index else q_rc).loc[a, "Mean QLIKE normal"], (q if a in q.index else q_rc).loc[a, "Mean QLIKE t"],
          (q if a in q.index else q_rc).loc[a, "DM Stat"], (q if a in q.index else q_rc).loc[a, "p-value"]) for a in A}
T["garch"] = "\n".join(f"{a} & {V[f'pers_n_{a}']} & {V[f'pers_t_{a}']} & {V[f'nu_{a}']} & {n(QL[a][0], 3)} & {n(QL[a][1], 3)} & {n(QL[a][2])} & {QL[a][3]:.3f} \\\\" for a in A)
V["ql_minp"] = f"{min(v[3] for v in QL.values()):.3f}"
V["ql_minp_asset"] = min(QL, key=lambda a: QL[a][3])
_gn = g_rc[(g_rc.Asset == "Gold") & (g_rc.Dist == "normal")]
V["scaled_gold"], V["abs_gold"] = f"{100 * _gn.loss_RMSE_scaled_proxy.mean():.2f}", f"{100 * _gn.loss_RMSE.mean():.2f}"
# tabella per fold (appendice)
rows = []
for k in range(1, 11):
    cells = []
    for a in A:
        x = G[(G.Asset == a) & (G.Fold == k)]
        cells += [f"{x[x.Dist == 'normal'].persistence.iloc[0]:.4f}", f"{x[x.Dist == 't'].persistence.iloc[0]:.4f}",
                  f"{x[x.Dist == 't'].nu.iloc[0]:.2f}"]
    rows.append(f"{k} & " + " & ".join(cells) + " \\\\")
T["garch_folds"] = "\n".join(rows)

# ---------------------------------------------------------------- backtest statico
bt = {a: fix(pd.read_csv(f"{R}/backtest_{a}.csv")).set_index("Model") for a in A}
btr = {a: fix(pd.read_csv(f"{R}/backtest_rcs_{a}.csv")).set_index("Model") for a in A}
bto = {a: fix(pd.read_csv(f"{R}/backtest_open_{a}.csv")).set_index("Model") for a in A}
btl = {a: fix(pd.read_csv(f"{R}/backtest_lag1_{a}.csv")).set_index("Model") for a in A}
btro = {a: fix(pd.read_csv(f"{R}/backtest_rcs_open_{a}.csv")).set_index("Model") for a in A}
btrl = {a: fix(pd.read_csv(f"{R}/backtest_rcs_lag1_{a}.csv")).set_index("Model") for a in A}
bm = pd.read_csv(f"{R}/rcs_benchmarks.csv").set_index("Asset")
bmm = fix(pd.read_csv(f"{R}/rcs_benchmarks_by_model.csv")).set_index(["Asset", "Model"])

rows = []
for m in ORDER:
    cells = []
    for a in A:
        x = bt[a].loc[m]
        cells += [n(x["Sharpe Ratio"]), f"{{\\scriptsize[{n(x['Sharpe CI Lower (95%)'])}, {n(x['Sharpe CI Upper (95%)'])}]}}"]
    rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
rows.append("\\midrule")
rows.append("Passive long (net) & " + " & ".join(f"{n(bm.loc[a, 'Sharpe Long Passive'])} & " for a in A) + " \\\\")
T["sharpe"] = "\n".join(rows)
allb = pd.concat([bt[a].assign(Asset=a) for a in A]).loc[ORDER]
V["sr_pos"] = str(int((allb["Sharpe Ratio"] > 0).sum()))
V["sr_gt_long"] = str(int(sum((bt[a].loc[ORDER, "Sharpe Ratio"] > bm.loc[a, "Sharpe Long Passive"]).sum() for a in A)))
avg = allb.groupby(level=0)["Sharpe Ratio"].mean().loc[ORDER].sort_values(ascending=False)
V["sr_avg_1"], V["sr_avg_1v"] = avg.index[0], n(avg.iloc[0])
V["sr_avg_2"], V["sr_avg_3"] = avg.index[1], avg.index[2]
V["sr_trees"] = n(avg.loc[TREES].mean()); V["sr_ens"] = n(avg.loc[ENS].mean())
V["sr_nn"] = n(avg.loc[NN].mean()); V["sr_last"], V["sr_lastv"] = avg.index[-1], n(avg.iloc[-1])
pos = allb[allb["Sharpe CI Lower (95%)"] > 0]; neg = allb[allb["Sharpe CI Upper (95%)"] < 0]
V["ci_excl"] = str(len(pos) + len(neg)); V["ci_pos"] = str(len(pos)); V["ci_neg"] = str(len(neg))
fmt_ci = lambda d: "; ".join(f"{m} on {r.Asset.lower()}, {n(r['Sharpe Ratio'])}, interval {n(r['Sharpe CI Lower (95%)'])} to {n(r['Sharpe CI Upper (95%)'])}" for m, r in d.iterrows())
V["ci_pos_list"] = fmt_ci(pos); V["ci_neg_list"] = fmt_ci(neg)
V["ci_excl_w"] = WORDS[len(pos) + len(neg)].capitalize(); V["ci_pos_w"] = WORDS[len(pos)]; V["ci_neg_w"] = WORDS[len(neg)]
V["ci_expected"] = f"{48 * 0.05:.1f}"
for a in A:
    V[f"long_{a}"] = n(bm.loc[a, "Sharpe Long Passive"])
# quota di giorni lunghi
sh = {}
for a in A:
    sr = fix(pd.read_csv(f"{R}/strategy_returns_{a}.csv", parse_dates=["Date"]))
    s = sr[(sr.Strategy == "static") & (sr.Execution == "close")]
    sh[a] = s.groupby("Model").w.apply(lambda w: 100 * (w > 0).mean())
three = ["Gold", "Silver", "Copper"]
V["long_dt"] = f"{min(sh[a]['Decision Tree'] for a in three):.0f}--{max(sh[a]['Decision Tree'] for a in three):.0f}"
V["long_rf"] = f"{min(sh[a]['Random Forest'] for a in three):.0f}--{max(sh[a]['Random Forest'] for a in three):.0f}"
V["long_nn3"] = f"{min(sh[a][NN].min() for a in three):.0f}\\% and {max(sh[a][NN].max() for a in three):.0f}\\%"
V["long_nn_pl"] = f"{sh['Platinum'][NN].min():.0f}\\% and {sh['Platinum'][NN].max():.0f}\\%"

# alfa
def alpha_rows(b):
    rows = []
    for m in ORDER:
        cells = []
        for a in A:
            x = b[a].loc[m]; cells += [n(x["Alpha (daily)"] * 1e4, 1), f"({p3(x['Alpha p-value'])})"]
        rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows)
T["alpha_static"] = alpha_rows(bt); T["alpha_rcs"] = alpha_rows(btr)
for key, b in [("st", bt), ("rc", btr)]:
    ab = pd.concat([b[a].assign(Asset=a) for a in A]).loc[ORDER]
    V[f"al_{key}_minholm"] = f"{ab['Alpha p Holm'].min():.3f}"
    rw_ = ab[ab["Alpha p-value"] < 0.05]
    V[f"al_{key}_nraw"] = str(len(rw_))
    rw_ = rw_.assign(_o=rw_.Asset.map({a: i for i, a in enumerate(A)})).sort_values("_o", kind="stable")
    V[f"al_{key}_rawlist"] = and_join(f"{m} on {r.Asset.lower()} ({n(r['Alpha (daily)'] * 1e4, 1)} basis points per day, $p={r['Alpha p-value']:.3f}$)" for m, r in rw_.iterrows())
    V[f"al_{key}_rawnames"] = ", ".join(f"{m} on {r.Asset.lower()}" for m, r in rw_.iterrows())
    V[f"al_{key}_nraw_w"] = WORDS[len(rw_)]
    V[f"al_{key}_rawsign"] = "negative" if (rw_["Alpha (daily)"] < 0).all() else ("positive" if (rw_["Alpha (daily)"] > 0).all() else "mixed")
ab = pd.concat([bt[a].assign(Asset=a) for a in A]).loc[ORDER]
top = ab.sort_values("Alpha (daily)", ascending=False).iloc[0]
V["al_top_model"], V["al_top_asset"] = top.name, top.Asset.lower()
V["al_top_bp"] = n(top["Alpha (daily)"] * 1e4, 1); V["al_top_ann"] = f"{top['Alpha (daily)'] * 252 * 100:.0f}"
V["al_top_p"] = f"{top['Alpha p-value']:.3f}"; V["al_top_beta"] = n(top["Beta"])
V["al_n_pos"] = str(int((ab["Alpha (daily)"] > 0).sum()))
se_top = top["Alpha (daily)"] / top["Alpha t-stat (HAC)"]
V["al_top_lo"] = n((top["Alpha (daily)"] - 1.96 * se_top) * 1e4, 1); V["al_top_hi"] = n((top["Alpha (daily)"] + 1.96 * se_top) * 1e4, 1)
V["al_top_holm"] = f"{top['Alpha p Holm']:.3f}"
# DA
rows = []
for m in ORDER:
    cells = []
    for a in A:
        x = bt[a].loc[m]; cells += [f"{x['Directional Acc.']:.1f}", f"({p3(x['DA p-value (HAC)'])})"]
    rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
T["da"] = "\n".join(rows)
V["da_min"], V["da_max"] = f"{ab['Directional Acc.'].min():.2f}", f"{ab['Directional Acc.'].max():.2f}"
td = ab.sort_values("Directional Acc.", ascending=False).iloc[0]
V["da_top_model"], V["da_top_asset"] = td.name, td.Asset.lower()
V["da_top_hac"] = f"{td['DA p-value (HAC)']:.3f}"; V["da_top_boot"] = f"{td['DA p-value (Boot H0)']:.3f}"
V["da_minholm"] = f"{ab['DA p Holm'].min():.3f}"
abr = pd.concat([btr[a].assign(Asset=a) for a in A]).loc[ORDER]
V["da_rcs_minholm"] = f"{abr['DA p Holm'].min():.3f}"; V["da_rcs_n"] = str(int(abr['DA p Holm'].notna().sum()))
V["da_active"] = str(int(ab["N Active"].min()))

# ---------------------------------------------------------------- esecuzione
for half, assets in [("a", ["Gold", "Silver"]), ("b", ["Copper", "Platinum"])]:
    rows = []
    for m in ORDER:
        cells = []
        for a in assets:
            vals = [bt[a].loc[m, "Sharpe Ratio"], bto[a].loc[m, "Sharpe Ratio"], btl[a].loc[m, "Sharpe Ratio"]]
            ref = 1 if EXEC_REF[a] == "open" else 2
            cells += [n(v) if i != ref else f"\\textbf{{{n(v)}}}" for i, v in enumerate(vals)]
        rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
    T[f"exec_{half}"] = "\n".join(rows)
ex = fix(pd.read_csv(f"{R}/executable_check.csv")).set_index(["Asset", "Model"])
C = np.array([ex.loc[(a, m), "Sharpe Static (close)"] for a in A for m in ORDER])
E = np.array([ex.loc[(a, m), "Sharpe Static (exec)"] for a in A for m in ORDER])
V["ex_corr"] = f"{np.corrcoef(C, E)[0, 1]:.2f}"; V["ex_signchg"] = str(int((np.sign(C) != np.sign(E)).sum()))
V["ex_gold_arma_o"], V["ex_gold_arma_l"] = n(bto["Gold"].loc["ARMA", "Sharpe Ratio"]), n(btl["Gold"].loc["ARMA", "Sharpe Ratio"])
V["ex_cu_arma_o"], V["ex_cu_arma_l"] = n(bto["Copper"].loc["ARMA", "Sharpe Ratio"]), n(btl["Copper"].loc["ARMA", "Sharpe Ratio"])
V["gold_arma_c"], V["cu_arma_c"] = n(bt["Gold"].loc["ARMA", "Sharpe Ratio"]), n(bt["Copper"].loc["ARMA", "Sharpe Ratio"])
for key, b in [("open", bto), ("lag1", btl), ("ropen", btro), ("rlag1", btrl)]:
    x = pd.concat([b[a].assign(Asset=a) for a in A]).loc[ORDER]
    xx = x.reset_index().sort_values("Alpha p Holm").iloc[0]
    V[f"exh_{key}"] = f"{xx['Alpha p Holm']:.3f}"; V[f"exh_{key}_m"] = xx["Model"]; V[f"exh_{key}_a"] = xx["Asset"].lower()
    V[f"exh_{key}_bp"] = n(xx["Alpha (daily)"] * 1e4, 1)
V["ex_rob_static"] = str(int(ex["Alpha Static Robust"].sum())); V["ex_rob_rcs"] = str(int(ex["Alpha RCS Robust"].sum()))
V["ex_rcs_close"] = str(int(ex["RCS > Static (close)"].sum())); V["ex_rcs_exec"] = str(int(ex["RCS > Static (exec)"].sum()))
V["ex_rcs_rob"] = str(int(ex["RCS Improvement Robust"].sum()))
# tabella miglioramento RCS per coppia
rows = []
for m in ORDER:
    cells = []
    for a in A:
        x = ex.loc[(a, m)]
        dc = x["Sharpe RCS (close)"] - x["Sharpe Static (close)"]; de = x["Sharpe RCS (exec)"] - x["Sharpe Static (exec)"]
        mark = "\\checkmark" if x["RCS Improvement Robust"] else ""
        cells += [n(dc), n(de), mark]
    rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
T["rcs_exec"] = "\n".join(rows)
rob_by_asset = ex.groupby(level=0)["RCS Improvement Robust"].sum()
for a in A: V[f"rcsrob_{a}"] = str(int(rob_by_asset[a]))
rob_by_model = ex.groupby(level=1)["RCS Improvement Robust"].sum().reindex(ORDER)
V["rcsrob_all4"] = ", ".join(rob_by_model[rob_by_model == 4].index) or "none"
V["rcsrob_none"] = ", ".join(rob_by_model[rob_by_model == 0].index) or "none"

# flat bars (snapshot OHLC)
for a in A:
    d = pd.read_csv(f"{D}/{TK[a]}_ohlc.csv", index_col=0, parse_dates=True); d.columns = [c.lower() for c in d.columns]
    o = d.loc["2025-07-02":"2026-04-30"]
    V[f"flat_{a}"] = str(int(((o.open == o.high) & (o.high == o.low) & (o.low == o.close)).sum()))
    V[f"flat_{a}_w"] = "none" if V[f"flat_{a}"] == "0" else V[f"flat_{a}"]

# ---------------------------------------------------------------- roll
rs = fix(pd.read_csv(f"{R}/roll_sensitivity.csv")); rd = pd.read_csv(f"{R}/roll_discontinuities.csv")
rows = []
for a in A:
    d = rd[rd.Asset == a]; s = rs[(rs.Asset == a) & (rs.Strategy == "static")]
    dS = s["Sharpe excl. roll window"] - s["Sharpe"]
    ratio = d["Mean |r| post-roll"].iloc[0] / d["Mean |r| other days"].iloc[0]
    V[f"roll_ratio_{a}"] = f"{ratio:.1f}"; V[f"roll_dsr_{a}"] = n(dS.mean())
    V[f"roll_post_{a}"] = f"{d['Mean |r| post-roll'].iloc[0] * 100:.2f}"; V[f"roll_other_{a}"] = f"{d['Mean |r| other days'].iloc[0] * 100:.2f}"
    rows.append(f"{a} & {int(d['N post-roll days'].iloc[0])} & {V[f'roll_post_{a}']} & {V[f'roll_other_{a}']} & \\riv{{{ratio:.1f}}} & {int(d['In roll window'].sum())} & {int(s['N Excluded'].iloc[0])} & {n(dS.mean())} & {n(dS.min())}/{n(dS.max())} & {int((s['Alpha p-value'] < .05).sum())}/{int((s['Alpha p-value excl. roll window'] < .05).sum())} \\\\")
T["roll"] = "\n".join(rows)
V["roll_top_max"] = WORDS[int(rd.groupby("Asset")["In roll window"].sum().max())]
gtop = rd[(rd.Asset == "Gold") & (rd["Rank |r|"] == 1)].iloc[0]
V["roll_gold_top_r"] = n(gtop.r * 100, 1); V["roll_gold_top_date"] = pd.Timestamp(gtop.Date).strftime("%-d %B %Y")
V["roll_p_before"] = str(int((rs["Alpha p-value"] < .05).sum())); V["roll_p_after"] = str(int((rs["Alpha p-value excl. roll window"] < .05).sum()))
V["roll_n"] = str(len(rs))
ga = rs[(rs.Asset == "Gold") & (rs.Model == "ARMA") & (rs.Strategy == "static")].iloc[0]
V["roll_arma_sr0"], V["roll_arma_sr1"] = n(ga["Sharpe"]), n(ga["Sharpe excl. roll window"])
V["roll_arma_p0"], V["roll_arma_p1"] = f"{ga['Alpha p-value']:.3f}", f"{ga['Alpha p-value excl. roll window']:.3f}"
big = rs.assign(dsr=(rs["Sharpe excl. roll window"] - rs["Sharpe"]).abs()).sort_values("dsr", ascending=False).iloc[0]
V["roll_bigsr_is_arma_gold"] = str(big.Asset == "Gold" and big.Model == "ARMA" and big.Strategy == "static")
for strat in ["static", "rcs"]:
    V[f"roll_holm_{strat}"] = f"{holm(rs[rs.Strategy == strat]['Alpha p-value excl. roll window'].values).min():.3f}"

# ---------------------------------------------------------------- RCS
for half, assets in [("a", ["Gold", "Silver"]), ("b", ["Copper", "Platinum"])]:
    rows = []
    for m in ORDER:
        cells = []
        for a in assets:
            x = btr[a].loc[m]; b_ = bmm.loc[(a, m)]
            ident = bool(b_["Identical to Long Filtered"])
            cells += [n(x["Sharpe Ratio"]) + ("$^{=}$" if ident else ""), f"({b_['p_emp vs Random Filtered']:.3f})",
                      "\\riv{" + n(x["Alpha vs Long Filtered (daily)"] * 1e4, 1) + "}", "\\riv{" + ("(--)" if ident else f"({x['Alpha vs Long Filtered p-value']:.3f})") + "}"]
        rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    for lab_, col in [("Passive long", "Sharpe Long Passive"), ("Filtered long", "Sharpe Long Filtered"),
                      ("Random signal, mean", "Sharpe Random Filtered Mean"), ("Random signal, 95th pct.", "Sharpe Random Filtered P95")]:
        rows.append(f"{lab_} & " + " & ".join(f"{n(bm.loc[a, col])} & & &" for a in assets) + " \\\\")
    T[f"rcs_{half}"] = "\n".join(rows)
sr_rcs = np.array([btr[a].loc[m, "Sharpe Ratio"] for a in A for m in ORDER]); sr_st = np.array([bt[a].loc[m, "Sharpe Ratio"] for a in A for m in ORDER])
V["rcs_gt"] = str(int((sr_rcs > sr_st).sum())); V["rcs_avg"] = n(sr_rcs.mean()); V["st_avg"] = n(sr_st.mean())
bmr = bmm.reset_index(); bmr = bmr[bmr.Model.isin(ORDER)]
pe = bmr[bmr["p_emp vs Random Filtered"] < 0.05]
V["pemp_n"] = str(len(pe)); V["pemp_list"] = ", ".join(f"{r.Model} on {r.Asset.lower()}" for _, r in pe.iterrows())
V["pemp_ident"] = WORDS[int(pe["Identical to Long Filtered"].sum())]
beat = bmr[bmr["RCS minus Long Filtered (Sharpe)"] > 0]
V["beatfl_n"] = str(len(beat)); V["beatfl_list"] = ", ".join(f"{r.Model} on {r.Asset.lower()}" for _, r in beat.iterrows())
V["ident_n"] = str(int(bmr["Identical to Long Filtered"].sum()))
V["ident_list"] = ", ".join(f"{r.Model} on {r.Asset.lower()}" for _, r in bmr[bmr["Identical to Long Filtered"]].iterrows())
top = bmr.sort_values("RCS minus Long Filtered (Sharpe)", ascending=False).iloc[0]
V["beatfl_top"], V["beatfl_top_a"], V["beatfl_top_v"] = top.Model, top.Asset.lower(), n(top["RCS minus Long Filtered (Sharpe)"])
rr = pd.concat([btr[a].assign(Asset=a) for a in A]).loc[ORDER].reset_index()
rr = rr.merge(bmr[["Asset", "Model", "Identical to Long Filtered"]], on=["Asset", "Model"])
ni = rr[~rr["Identical to Long Filtered"]]
V["fl_top_p"] = f"{rr[(rr.Model == top.Model) & (rr.Asset == top.Asset)]['Alpha vs Long Filtered p-value'].iloc[0]:.3f}"
b3 = bmr[(bmr.Asset != "Platinum")].sort_values("RCS minus Long Filtered (Sharpe)", ascending=False).iloc[0]
V["beatfl3_m"], V["beatfl3_a"], V["beatfl3_v"] = b3.Model, b3.Asset.lower(), n(b3["RCS minus Long Filtered (Sharpe)"])
V["beatfl3_p"] = f"{rr[(rr.Model == b3.Model) & (rr.Asset == b3.Asset)]['Alpha vs Long Filtered p-value'].iloc[0]:.3f}"
for a in A: V[f"flong_{a}"] = n(bm.loc[a, "Sharpe Long Filtered"])
mi = ni.sort_values("Alpha vs Long Filtered p-value").iloc[0]
V["fl_minp"], V["fl_minp_m"], V["fl_minp_a"] = f"{mi['Alpha vs Long Filtered p-value']:.3f}", mi.Model, mi.Asset.lower()
V["fl_minp_bp"] = n(mi["Alpha vs Long Filtered (daily)"] * 1e4, 1)
V["fl_npos"] = str(int((ni["Alpha vs Long Filtered (daily)"] > 0).sum())); V["fl_nni"] = str(len(ni))
V["fl_holm"] = f"{holm(ni['Alpha vs Long Filtered p-value'].values).min():.3f}"

# ---------------------------------------------------------------- per regime
pr = fix(pd.read_csv(f"{R}/performance_by_regime.csv")); pr = pr[pr.Model.isin(ORDER)]
st = pr[pr.Strategy == "static"]; rc = pr[pr.Strategy == "rcs"]
rows = []
for a in A:
    for g_ in REG:
        x = st[(st.Asset == a) & (st.Regime == g_)]; y = rc[(rc.Asset == a) & (rc.Regime == g_)]
        nd = int(x["N Days"].iloc[0])
        srs = n(x.Sharpe.mean()) if nd >= 5 else "--"
        srr = (n(y.Sharpe.mean()) if nd >= 5 else "--") if g_ != "volatile" else "--"
        rows.append(f"{a} & {g_} & {nd} & {srs} & {n(x['Mean Daily Return (%)'].mean() * 100, 1)} & {x['Directional Acc.'].mean():.1f} & {int((x['Directional Acc.'] > 50).sum())} & \\riv{{{srr}}} & \\riv{{{n(y['Mean Daily Return (%)'].mean() * 100, 1)}}} \\\\")
T["byreg"] = "\n".join(rows)
mrg = st.merge(rc, on=["Asset", "Model", "Regime"], suffixes=("_s", "_r"))
calm = mrg[mrg.Regime != "volatile"]
V["byreg_da_maxdiff"] = f"{(calm['Directional Acc._s'] - calm['Directional Acc._r']).abs().max():.2f}"
ok = calm[calm["N Days_s"] >= 5]
V["byreg_sr_maxdiff"] = f"{(ok['Sharpe_s'] - ok['Sharpe_r']).abs().max():.2f}"
V["byreg_vol_active"] = str(int(rc[rc.Regime == "volatile"]["N Active"].max()))
vd = st[st.Regime == "volatile"].groupby("Asset")["Directional Acc."].mean()
V["byreg_vol_da_lo"], V["byreg_vol_da_hi"] = f"{vd.min():.1f}", f"{vd.max():.1f}"
# long passivo per regime (lordo)
for a in A:
    p = pd.read_csv(f"{R}/predictions_{a}.csv", parse_dates=["Date"]); r_ = np.expm1(p[p.Model == "Random Walk"].set_index("Date").y_true)
    for g_ in REG:
        sel = r_[L[a].reindex(r_.index) == g_]
        V[f"plong_{a}_{g_}"] = n(sharpe(sel.values)) if len(sel) >= 5 else "--"
ms = st.groupby(["Asset", "Regime"]).Sharpe.mean()
V["byreg_below_long"] = ", ".join(f"{a}/{g_}" for a in three for g_ in REG if not (ms[(a, g_)] < float(V[f"plong_{a}_{g_}"].replace("$-$", "-"))))
# appendice: per modello e regime
for a in A:
    rows = []
    for strat, lab in [("static", "Static"), ("rcs", "RCS")]:
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{lab}}}}} \\\\")
        for m in ORDER:
            cells = []
            for g_ in REG:
                x = pr[(pr.Asset == a) & (pr.Model == m) & (pr.Strategy == strat) & (pr.Regime == g_)].iloc[0]
                exposed = x["N Active"] > 0
                srv = n(x.Sharpe) if (x["N Days"] >= 5 and exposed) else "--"
                dav = f"{x['Directional Acc.']:.1f}" if exposed else "--"
                cells += [srv, n(x["Mean Daily Return (%)"] * 100, 1), dav]
            rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
        if strat == "static": rows.append("\\midrule")
    T[f"app_byreg_{a}"] = "\n".join(rows)
    V[f"app_days_{a}"] = " / ".join(V[f"{k}_{a}"] for k in ["st", "no", "vo"])
# appendice: intervalli MDD, Calmar, DA
for a in A:
    rows = []
    for m in ORDER:
        cells = []
        for b in (bt, btr):
            x = b[a].loc[m]
            cells += [f"[{n(x['Sharpe CI Lower (95%)'])}, {n(x['Sharpe CI Upper (95%)'])}]",
                      f"[{n(x['Max Drawdown CI Lower (95%)'], 1)}, {n(x['Max Drawdown CI Upper (95%)'], 1)}]",
                      f"[{n(x['Calmar CI Lower (95%)'])}, {n(x['Calmar CI Upper (95%)'])}]",
                      (f"[{x['DA CI Lower (95%)']:.1f}, {x['DA CI Upper (95%)']:.1f}]" if not np.isnan(x['DA CI Lower (95%)']) else "--")]
        rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
    T[f"app_ci_{a}"] = "\n".join(rows)
# nota della tabella del platino (rilettura finale, N8)
_xp = btr["Platinum"].loc[ORDER]
assert ((_xp["Sharpe CI Lower (95%)"] < 0) & (_xp["Sharpe CI Upper (95%)"] > 0)).all()
assert (_xp["Max Drawdown CI Upper (95%)"].abs() < 0.05).all()
V["pt_cal_hi_n"] = str(int((_xp["Calmar CI Upper (95%)"] > 1000).sum()))
V["pt_cal_hi_max"] = n(_xp["Calmar CI Upper (95%)"].max())
V["pt_da_full_n"] = str(int(((_xp["DA CI Lower (95%)"] == 0) & (_xp["DA CI Upper (95%)"] == 100)).sum()))

# ---------------------------------------------------------------- D
# R14: controllo per asset con gli stessi asset in ogni replica, da tools/recompute_post_run.py;
# la riga "portfolio" coincide con quella dell'esecuzione (verificato dallo script).
Dres = pd.read_csv(f"{RC}/architecture_vs_regime_D.csv").set_index("Aggregation")
Drun = pd.read_csv(f"{R}/architecture_vs_regime_D.csv").set_index("Aggregation")
assert np.isclose(Dres.loc["portfolio", "D CI Upper (95%)"], Drun.loc["portfolio", "D CI Upper (95%)"])
S = fix(pd.read_csv(f"{R}/sharpe_by_model_regime.csv").rename(columns={"Unnamed: 0": "Model"})).set_index("Model").loc[ORDER]
rows = [f"{m} & " + " & ".join(n(S.loc[m, g_]) for g_ in REG) + f" & {S.loc[m].var(ddof=1):.2f} \\\\" for m in ORDER]
rows += ["\\midrule", "Variance across models & " + " & ".join(f"{S[g_].var(ddof=1):.2f}" for g_ in REG) + " & \\\\",
         "Dates in the cell & " + " & ".join(V[f"dcell_{g_}"] for g_ in REG) + " & \\\\"]
T["dmat"] = "\n".join(rows)
vr, vm = S.var(axis=1, ddof=1).mean(), S.var(axis=0, ddof=1).mean()
V["d_vr"], V["d_vm"] = f"{vr:.2f}", f"{vm:.2f}"
x = Dres.loc["portfolio"]; V["d"], V["d_lo"], V["d_hi"] = n(x.D), n(x["D CI Lower (95%)"]), n(x["D CI Upper (95%)"])
V["d_used"], V["d_disc"] = str(int(x["N Bootstrap Used"])), f"{100 * x['Frac. Discarded']:.1f}"
y = Dres.loc["per_asset"]; V["dpa"], V["dpa_lo"], V["dpa_hi"] = n(y.D), n(y["D CI Lower (95%)"]), n(y["D CI Upper (95%)"])
V["dpa_used"] = str(int(y["N Bootstrap Used"]))
V["dpa_ndisc"] = str(int(y["N Bootstrap Requested"] - y["N Bootstrap Used"]))
V["dpa_disc"] = f"{100 * (1 - y['N Bootstrap Used'] / y['N Bootstrap Requested']):.0f}"
_fail = pd.read_csv(f"{RC}/D_per_asset_discards.csv", index_col=0).iloc[:, 0]
assert _fail.max() <= int(V["dpa_ndisc"]) <= _fail.sum()
for a in _fail.index: V[f"dpa_fail_{a}"] = str(int(_fail[a]))
yr = Drun.loc["per_asset"]
V["dpa_lo_run"], V["dpa_hi_run"] = n(yr["D CI Lower (95%)"]), n(yr["D CI Upper (95%)"])
V["dpa_disc_run"] = f"{100 * yr['Frac. Discarded']:.1f}"
for k in ("dpa_lo_run", "dpa_hi_run", "dpa_disc_run", "dpa_lo", "dpa_hi"):
    V[k + "_it"] = V[k].replace(".", "{,}")
dpa = pd.read_csv(f"{R}/D_per_asset.csv", index_col=0).iloc[:, 0]
for a in ["Gold", "Silver", "Copper"]: V[f"di_{a}"] = n(dpa[a])
V["d_varmod_st"], V["d_varmod_vo"] = f"{S['stable'].var(ddof=1):.2f}", f"{S['volatile'].var(ddof=1):.2f}"
V["d_first"] = and_join(f"the {S[g_].idxmax()} in the {g_} regime" for g_ in REG)
tw = pd.read_csv(f"{R}/two_way_decomposition.csv").set_index("Component")
for k, c in [("m", "Share Model"), ("g", "Share Regime"), ("i", "Share Interaction"), ("e", "Share Residual")]:
    V[f"tw_{k}"] = f"{100 * tw.loc[c, 'Share']:.2f}"
    V[f"tw_{k}_lo"], V[f"tw_{k}_hi"] = f"{100 * tw.loc[c, 'CI Lower (95%)']:.2f}", f"{100 * tw.loc[c, 'CI Upper (95%)']:.2f}"
# D sui soli modelli non neurali e sole reti (descrittivo)
for name, grp in [("nn", NN), ("ml", [m for m in ORDER if m not in NN])]:
    Sg = S.loc[grp]; V[f"d_{name}"] = n(Sg.var(axis=1, ddof=1).mean() - Sg.var(axis=0, ddof=1).mean())

# ---------------------------------------------------------------- ranking
rows = []
mm_ = met[met.Model != "Random Walk"]
for a in A + ["Avg"]:
    if a == "Avg":
        rmv = mm_.groupby("Model").RMSE.mean(); b = pd.concat([bt[x].loc[ORDER] for x in A])
        srv = b.groupby(level=0)["Sharpe Ratio"].mean(); dav = b.groupby(level=0)["Directional Acc."].mean(); name = "Average over metals"
    else:
        rmv = mm_[mm_.Asset == a].set_index("Model").RMSE; srv = bt[a].loc[ORDER, "Sharpe Ratio"]; dav = bt[a].loc[ORDER, "Directional Acc."]; name = a
    idx = ORDER
    r1, r2, r3 = spearmanr(-rmv[idx], srv[idx]).statistic, spearmanr(-rmv[idx], dav[idx]).statistic, spearmanr(dav[idx], srv[idx]).statistic
    rows.append(f"{name} & {n(r1)} & {n(r2)} & {n(r3)} \\\\"); V[f"rk_{a}"] = (r1, r2, r3)
T["rank"] = "\n".join(rows)
V["rk_avg_rs"], V["rk_avg_rd"], V["rk_avg_ds"] = (n(v) for v in V.pop("rk_Avg"))
rk = {a: V.pop(f"rk_{a}") for a in A}
lo = min(rk, key=lambda a: rk[a][0]); hi = max(rk, key=lambda a: rk[a][0])
V["rk_lo"], V["rk_lo_a"], V["rk_hi"], V["rk_hi_a"] = n(rk[lo][0]), lo.lower(), n(rk[hi][0]), hi.lower()
# valore critico esatto di Spearman per n = 12 (test bilaterale 5%), da tabella (Zar 1972)
V["rk_crit"] = "0.587"
V["rk_sig"] = ", ".join(a.lower() for a in A if abs(rk[a][0]) > 0.587) or "none"

# ---------------------------------------------------------------- seed e TCN
sm = fix(pd.read_csv(f"{R}/dl_seed_metrics.csv")); disp = fix(pd.read_csv(f"{R}/dl_seed_dispersion.csv"))
rows, cur = [], None
for a in A:
    for i, m in enumerate(NN):
        x = disp[(disp.Asset == a) & (disp.Model == m)].iloc[0]
        rows.append(f"{a if i == 0 else ''} & {m} & {x['RMSE mean'] * 100:.3f} & {x['RMSE std'] * 100:.3f} & {n(x['Sharpe mean'])} & {x['Sharpe std']:.2f} & [{n(x['Sharpe min'])}, {n(x['Sharpe max'])}] & {x['Directional Acc. mean']:.1f} & [{x['Directional Acc. min']:.1f}, {x['Directional Acc. max']:.1f}] \\\\")
    if a != A[-1]: rows.append("\\addlinespace")
T["seeds"] = "\n".join(rows)
rwv = met[met.Model == "Random Walk"].set_index("Asset").RMSE
V["sd_rmse_max"] = f"{(disp['RMSE std'] / disp['RMSE mean']).max() * 100:.1f}"
V["sd_below_rw"] = str(int((sm.RMSE < sm.Asset.map(rwv)).sum())); V["sd_n"] = str(len(sm))
V["sd_sr_lo"], V["sd_sr_hi"], V["sd_sr_med"] = f"{disp['Sharpe std'].min():.2f}", f"{disp['Sharpe std'].max():.2f}", f"{disp['Sharpe std'].median():.2f}"
s42 = sm[sm.Seed == 42]; V["sd_between"] = f"{s42.groupby('Asset').Sharpe.std().mean():.2f}"
V["sd_signchg"] = str(int(sm.groupby(['Asset', 'Model']).Sharpe.apply(lambda x: (x > 0).any() and (x < 0).any()).sum()))
V["sd_pairs"] = str(sm.groupby(['Asset', 'Model']).ngroups)
avgs = sm.groupby(["Seed", "Model"]).Sharpe.mean().unstack()
V["sd_best"] = ", ".join(f"{avgs.loc[s].idxmax()} (seed {s})" for s in [1, 2, 3, 4, 42])
cs = [spearmanr(avgs.loc[i], avgs.loc[j]).statistic for k, i in enumerate(avgs.index) for j in avgs.index[k + 1:]]
V["sd_sp"] = n(np.mean(cs)); nnavg = avgs.mean(axis=1)
V["sd_nn_lo"], V["sd_nn_hi"] = n(nnavg.min()), n(nnavg.max())
t31 = pd.read_csv(f"{R}/tcn31_vs_tcn.csv").set_index("Asset")
T["tcn"] = "\n".join(f"{a} & {t31.loc[a, 'TCN-15 RMSE'] * 100:.3f} & {t31.loc[a, 'TCN-31 RMSE'] * 100:.3f} & {n(t31.loc[a, 'TCN-15 Sharpe'])} & {n(t31.loc[a, 'TCN-31 Sharpe'])} & {t31.loc[a, 'TCN-15 Directional Acc.']:.1f} & {t31.loc[a, 'TCN-31 Directional Acc.']:.1f} & {n(t31.loc[a, 'DM (TCN-31 vs TCN-15)'])} & {t31.loc[a, 'DM p-value']:.3f} \\\\" for a in A)
V["tcn_all_higher"] = str(bool((t31["TCN-31 RMSE"] > t31["TCN-15 RMSE"]).all()))
V["tcn_minp"] = f"{t31['DM p-value'].min():.3f}"
d_sr = t31["TCN-31 Sharpe"] - t31["TCN-15 Sharpe"]; V["tcn_sr_up"], V["tcn_sr_down"] = str(int((d_sr > 0).sum())), str(int((d_sr < 0).sum()))


# ---------------------------------------------------------------- Capitolo 5
fam = []
for strat, bdict in [("static", (bt, bto, btl)), ("rcs", (btr, btro, btrl))]:
    for ex_name, b in zip(["close", "open", "lag1"], bdict):
        x = pd.concat([b[a].assign(Asset=a) for a in A]).loc[ORDER].reset_index()
        r0 = x.sort_values("Alpha p Holm").iloc[0]
        fam.append((r0["Alpha p Holm"], strat, ex_name, r0["Model"], r0["Asset"], r0["Alpha (daily)"]))
f0 = min(fam, key=lambda z: z[0])
V["holm6_min"] = f"{f0[0]:.3f}"; V["holm6_sign"] = "negative" if f0[5] < 0 else "positive"
V["tw_sum"] = f"{100 - float(V['tw_e']):.2f}"
V["dm_holm_asset_l"] = V["dm_holm_asset"].lower()


# ---------------------------------------------------------------- revisione indipendente (20 nov.)
# Finestre di 210 giorni più volatili (R04)
for a in A:
    px = pd.read_csv(f"{D}/{TK[a]}.csv", index_col=0, parse_dates=True)
    r_ = np.log(px.Close / px.Close.shift(1)).dropna()
    sd = r_.rolling(210).std()
    oos = r_.loc["2025-07-02":"2026-04-30"]; assert len(oos) == 210
    end = sd.idxmax(); i_end = r_.index.get_loc(end)
    win = set(r_.index[i_end - 209:i_end + 1])
    V[f"ovl_{a}"] = str(len(win & set(oos.index)))
    V[f"vmaxend_{a}"] = end.strftime("%-d %B %Y")
assert V["ovl_Silver"] == "210", "il periodo fuori campione non è più la finestra più volatile dell'argento"

# Conteggi di RQ2 senza il platino (R03)
three_mask = np.array([a != "Platinum" for a in A for m in ORDER])
V["rcs_gt3"] = str(int(((sr_rcs > sr_st) & three_mask).sum()))
V["ex_rcs_rob3"] = str(int(ex.drop(index="Platinum", level=0)["RCS Improvement Robust"].sum()))
V["fl_notbeat"] = str(48 - int(V["beatfl_n"]))

# Benchmark della RCS con la variante eseguibile di riferimento (K13), da tools/recompute_post_run.py
xref = pd.read_csv(f"{RC}/executable_reference.csv", index_col=0).iloc[:, 0].to_dict()
bmx_all = fix(pd.read_csv(f"{RC}/rcs_benchmarks_by_model.csv"))
bmx = bmx_all[bmx_all.apply(lambda r: r.Execution == xref[r.Asset], axis=1)]
bmx = bmx[bmx.Model.isin(ORDER)]; assert len(bmx) == 48
bsx = pd.read_csv(f"{RC}/rcs_benchmarks.csv")
bsx = bsx[bsx.apply(lambda r: r.Execution == xref[r.Asset], axis=1)].set_index("Asset")
# controllo: i benchmark close del ricalcolo sono quelli dell'esecuzione
bmc = bmx_all[bmx_all.Execution == "close"].set_index(["Asset", "Model"])
assert np.allclose(bmc.loc[bmm.index, "Sharpe RCS"], bmm["Sharpe RCS"], atol=1e-4)
beatx = bmx[bmx["RCS minus Long Filtered (Sharpe)"] > 0]
assert all(bsx.loc[a, "Sharpe Long Filtered"] > bsx.loc[a, "Sharpe Long Passive"] for a in A)
V["bfx_beat_n"] = str(len(beatx)); V["bfx_beat3_n"] = str(int((beatx.Asset != "Platinum").sum()))
V["bfx_notbeat_n"] = str(48 - len(beatx))  # coppie in cui il long filtrato fa almeno altrettanto bene (abstract)
V["bfx_beat_list"] = and_join([f"{r.Model} on {r.Asset.lower()}" for _, r in beatx.iterrows()])
nix = bmx[~bmx["Identical to Long Filtered"]]
mx = nix.sort_values("Alpha vs Long Filtered p-value").iloc[0]
V["bfx_minp"], V["bfx_minp_m"], V["bfx_minp_a"] = f"{mx['Alpha vs Long Filtered p-value']:.3f}", mx.Model, mx.Asset.lower()
V["bfx_minp_bp"] = n(mx["Alpha vs Long Filtered (daily)"] * 1e4, 1)
V["bfx_minp_sign"] = "negative" if mx["Alpha vs Long Filtered (daily)"] < 0 else "positive"
V["bfx_holm"] = f"{holm(nix['Alpha vs Long Filtered p-value'].values).min():.3f}"
V["bfx_npos_sig"] = str(int(((nix["Alpha vs Long Filtered p-value"] < 0.05) & (nix["Alpha vs Long Filtered (daily)"] > 0)).sum()))
V["bfx_pemp_n"] = str(int((bmx["p_emp vs Random Filtered"] < 0.05).sum()))
V["bfx_pemp_n_w"] = WORDS[int(V["bfx_pemp_n"])].capitalize()
V["bfx_pemp_ident"] = str(int(((bmx["p_emp vs Random Filtered"] < 0.05) & bmx["Identical to Long Filtered"]).sum()))
V["bfx_ident_n"] = str(int(bmx["Identical to Long Filtered"].sum()))
V["fl_nni_x"] = str(len(nix))
for a in A:
    V[f"flongx_{a}"] = n(bsx.loc[a, "Sharpe Long Filtered"]); V[f"plongx_{a}"] = n(bsx.loc[a, "Sharpe Long Passive"])
    V[f"xref_{a}"] = {"open": "next open", "lag1": "one-day delay"}[xref[a]]
rows = []
for a in A:
    x = bmx[bmx.Asset == a]
    rows.append(f"{a} & {V[f'xref_{a}']} & {V[f'plongx_{a}']} & {V[f'flongx_{a}']} & {n(bsx.loc[a, 'Sharpe Random Filtered P95'])} & "
                f"{int((x['RCS minus Long Filtered (Sharpe)'] > 0).sum())} & {int((x['p_emp vs Random Filtered'] < 0.05).sum())} & "
                f"{int(x['Identical to Long Filtered'].sum())} & {p3(x[~x['Identical to Long Filtered']]['Alpha vs Long Filtered p-value'].min())} \\\\")
T["rcs_xbench"] = "\n".join(rows)

# Statistica D con la variante eseguibile (K13)
Dx = pd.read_csv(f"{RC}/architecture_vs_regime_D_exec.csv").set_index("Aggregation")
x = Dx.loc["portfolio"]; V["dx"], V["dx_lo"], V["dx_hi"] = n(x.D), n(x["D CI Lower (95%)"]), n(x["D CI Upper (95%)"])
V["dx_ndisc"] = str(int(x["N Bootstrap Requested"] - x["N Bootstrap Used"]))
y = Dx.loc["per_asset"]; V["dpax"], V["dpax_lo"], V["dpax_hi"] = n(y.D), n(y["D CI Lower (95%)"]), n(y["D CI Upper (95%)"])
V["dpax_ndisc"] = str(int(y["N Bootstrap Requested"] - y["N Bootstrap Used"]))
twx = pd.read_csv(f"{RC}/two_way_decomposition_exec.csv", index_col=0)
V["twx_mgi"] = f"{100 * (1 - twx.loc['Share Residual', 'Share']):.2f}"
V["twx_e"] = f"{100 * twx.loc['Share Residual', 'Share']:.1f}"
V["d_ndisc"] = str(int(Dres.loc["portfolio", "N Bootstrap Requested"] - Dres.loc["portfolio", "N Bootstrap Used"]))

# Holm della DA nelle varianti eseguibili (R06)
hf = pd.read_csv(f"{R}/holm_families.csv").set_index("Family")
for key, b, fam in [("open", bto, "open DA static"), ("lag1", btl, "lag1 DA static"),
                    ("ropen", btro, "open DA rcs"), ("rlag1", btrl, "lag1 DA rcs")]:
    xx = pd.concat([b[a] for a in A])["DA p Holm"].dropna()
    V[f"daexh_{key}"] = f"{xx.min():.3f}"; V[f"daexn_{key}"] = str(int(hf.loc[fam, "N Tests"]))
    assert int(hf.loc[fam, "N Sig. Holm (5%)"]) == 0

# RCS nella variante eseguibile non di riferimento (R06c)
nonref = {a: ("lag1" if xref[a] == "open" else "open") for a in A}
V["nref_rcs_n"] = str(sum(int((({"open": btro, "lag1": btrl}[nonref[a]][a].loc[ORDER, "Sharpe Ratio"]
                                > {"open": bto, "lag1": btl}[nonref[a]][a].loc[ORDER, "Sharpe Ratio"])).sum()) for a in A))
# migliora sia close-to-close sia con la variante non di riferimento (rilettura finale, N1)
_nref_both = {a: ((btr[a].loc[ORDER, "Sharpe Ratio"] > bt[a].loc[ORDER, "Sharpe Ratio"])
                  & ({"open": btro, "lag1": btrl}[nonref[a]][a].loc[ORDER, "Sharpe Ratio"]
                     > {"open": bto, "lag1": btl}[nonref[a]][a].loc[ORDER, "Sharpe Ratio"])) for a in A}
V["nref_rcs_both"] = str(sum(int(v.sum()) for v in _nref_both.values()))
V["nref_rcs_both3"] = str(sum(int(v.sum()) for a, v in _nref_both.items() if a != "Platinum"))
assert V["ex_rcs_exec"] == str(int(sum(int((btro[a].loc[ORDER, "Sharpe Ratio"] > bto[a].loc[ORDER, "Sharpe Ratio"]).sum()) if xref[a] == "open" else int((btrl[a].loc[ORDER, "Sharpe Ratio"] > btl[a].loc[ORDER, "Sharpe Ratio"]).sum()) for a in A)))

# Intervalli dello Sharpe della RCS (R05)
ci_rcs = pd.concat([btr[a].loc[ORDER] for a in A])
V["rcs_sr_ci0"] = str(int(((ci_rcs["Sharpe CI Lower (95%)"] < 0) & (ci_rcs["Sharpe CI Upper (95%)"] > 0)).sum()))
V["rcs_sr_ci_pos"] = str(int((ci_rcs["Sharpe CI Lower (95%)"] > 0).sum()))
V["rcs_sr_ci_neg"] = str(int((ci_rcs["Sharpe CI Upper (95%)"] < 0).sum()))
assert int(V["rcs_sr_ci0"]) + int(V["rcs_sr_ci_pos"]) + int(V["rcs_sr_ci_neg"]) == 48

# Composizione delle celle di D (revisione Cap. 1–2, R-04)
cnt = {g_: (L == g_).sum(axis=1) for g_ in REG}
for g_ in REG:
    c_ = cnt[g_][cnt[g_] > 0]; V[f"dcomp_{g_}"] = f"{c_.mean():.2f}"
V["dcomp_stable1"] = str(int((cnt["stable"] == 1).sum()))
assert V["dcomp_stable1"] != "0" and int(V["dcell_stable"]) == int((cnt["stable"] > 0).sum())

# Variazione massima di Holm del DM (R12)
_mx = _cmp.assign(d=(_cmp["p Holm (raw)"] - _cmp["p Holm (run)"]).abs()).sort_values("d").iloc[-1]
V["dm_holm_maxdiff_m"], V["dm_holm_maxdiff_a"] = _mx.Model.replace("CNN LSTM", "CNN-LSTM"), {"Gold": "l'oro", "Silver": "l'argento", "Copper": "il rame", "Platinum": "il platino"}[_mx.Asset]
if __name__ == "__main__":
    json.dump({"V": V, "T": T}, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
