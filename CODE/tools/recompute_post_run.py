"""
tools/recompute_post_run.py
---------------------------
Ricalcola, senza rifare il training, i risultati toccati dalle correzioni
successive a un'esecuzione completa di main.py, partendo dai suoi CSV:

  R14  statistica D: nel controllo per asset ogni replica bootstrap usa
       gli stessi asset della stima puntuale. La versione di portafoglio
       (principale) e la decomposizione a due fattori vengono ricalcolate
       come controllo e devono coincidere con quelle salvate.
  R11  test DM: p-value non arrotondati e correzione di Holm applicata a
       quelli (prima era applicata ai p-value arrotondati a 4 decimali).
  R5   GARCH: statistiche di un asset ripreso con --resume (nell'esecuzione
       del commit 2586f06 l'oro), ricostruite da garch_forecasts_<asset>.csv
       con la stessa funzione usata da main.py. Gli asset presenti in
       garch_volatility.csv vengono ricostruiti allo stesso modo e
       confrontati con i valori salvati, come controllo della procedura.
  K13  benchmark della regola di esposizione della RCS (long filtrato,
       intercetta sul long filtrato, segnale casuale) e statistica D con
       i due controlli, calcolati anche con la variante eseguibile di
       riferimento di ogni asset. Nell'esecuzione del commit 2586f06 erano
       calcolati solo in close-to-close. I benchmark close-to-close sono
       ricalcolati come controllo e devono coincidere con quelli salvati.

Le funzioni usate sono quelle di main.py e di src/ (nessuna copia della
logica): _reconstruct_asset_from_disk, _holm_correct, _garch_from_disk,
_d_statistics, _asset_data_executable. I parametri del bootstrap vengono
dal manifest dell'esecuzione ("config"), con i valori della pipeline come
ripiego.

Uso, da CODE/:
    python tools/recompute_post_run.py --results <cartella risultati> \
                                       --out <cartella di output>

La cartella dei risultati non viene modificata: i file intermedi scritti
dalla ricostruzione finiscono in una cartella temporanea.
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import main as M                                              # noqa: E402
from src.preprocessing import METALS, load_and_preprocess     # noqa: E402

GARCH_COLS = ["alpha", "beta", "persistence", "nu", "loss_RMSE",
              "loss_RMSE_scaled_proxy", "QLIKE"]


def _input_files(results_dir: str, assets) -> list:
    names = []
    for a in assets:
        names += [f"predictions_{a}.csv", f"garch_forecasts_{a}.csv"]
    missing = [n for n in names if not os.path.exists(os.path.join(results_dir, n))]
    if missing:
        raise FileNotFoundError(f"File mancanti in {results_dir}: {missing}")
    return names


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _close(a, b, atol=1e-6) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                            atol=atol, equal_nan=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True,
                    help="cartella con i CSV dell'esecuzione completa")
    ap.add_argument("--out", required=True, help="cartella di output")
    args = ap.parse_args()
    results_dir = os.path.abspath(args.results)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(results_dir, "run_manifest.json")) as f:
        run_manifest = json.load(f)
    cfg = run_manifest.get("config", {})
    boot = cfg.get("bootstrap", {})
    d_par = {"block_length": boot.get("block_length", 10),
             "n_boot": boot.get("n_boot", 2000),
             "seed": boot.get("seed", 42),
             "min_obs_per_cell": cfg.get("D_min_obs_per_cell", 5)}
    assets = list(METALS)
    inputs = _input_files(results_dir, assets)

    work = tempfile.mkdtemp(prefix="recompute_")
    for n in inputs:
        os.symlink(os.path.join(results_dir, n), os.path.join(work, n))
    M.RESULTS_DIR = work

    all_dfs = {a: load_and_preprocess(a, t, start=M.START, end=M.END)
               for a, t in METALS.items()}

    # ── Ricostruzione dalle previsioni salvate (stesse funzioni di --resume).
    # Il controllo del commit è disattivato: lo script gira per definizione
    # da un commit successivo a quello dell'esecuzione, che registra nel
    # proprio manifest.
    recon, bt_all = {}, {}
    for a in assets:
        print(f"Ricostruzione {a} da predictions_{a}.csv ...")
        recon[a] = M._reconstruct_asset_from_disk(a, all_dfs, check_commit=False)
        for execution, (bt_res, rcs_res) in recon[a]["bt_out"]["by_exec"].items():
            bt_all.setdefault(execution, {"static": {}, "rcs": {}})
            bt_all[execution]["static"][a] = bt_res
            bt_all[execution]["rcs"][a] = rcs_res

    checks = {}

    # ── R14 e K13: statistica D, close e variante eseguibile ──────────────
    asset_data = {a: recon[a]["asset_data"] for a in assets}
    print("\nStatistica D, close-to-close:")
    d_close = M._d_statistics(asset_data, **d_par)
    asset_data_ex = M._asset_data_executable(bt_all, asset_data)
    print("\nStatistica D, variante eseguibile di riferimento:")
    d_exec = M._d_statistics(asset_data_ex, **d_par)
    for suffix, d_out in (("", d_close), ("_exec", d_exec)):
        d_out["D"].to_csv(os.path.join(out_dir, f"architecture_vs_regime_D{suffix}.csv"),
                          index=False)
        d_out["per_asset"].to_csv(os.path.join(out_dir, f"D_per_asset{suffix}.csv"))
        d_out["per_asset_discards"].to_csv(
            os.path.join(out_dir, f"D_per_asset_discards{suffix}.csv"))
        d_out["regime_counts"].to_csv(os.path.join(out_dir, f"D_regime_counts{suffix}.csv"))
        d_out["two_way"].to_csv(os.path.join(out_dir, f"two_way_decomposition{suffix}.csv"))
        d_out["sharpe_by_model_regime"].to_csv(
            os.path.join(out_dir, f"sharpe_by_model_regime{suffix}.csv"))
    pd.Series({a: d["execution"] for a, d in asset_data_ex.items()},
              name="Execution").to_csv(os.path.join(out_dir, "executable_reference.csv"))

    d_old = pd.read_csv(os.path.join(results_dir, "architecture_vs_regime_D.csv"))
    d_new = d_close["D"]
    port_new = d_new[d_new.Aggregation == "portfolio"].iloc[0]
    port_old = d_old[d_old.Aggregation == "portfolio"].iloc[0]
    checks["portfolio_D_equals_run"] = all(
        _close(port_new[k], port_old[k]) for k in
        ("D", "D CI Lower (95%)", "D CI Upper (95%)", "Frac. Discarded"))
    tw_old = pd.read_csv(os.path.join(results_dir, "two_way_decomposition.csv"), index_col=0)
    checks["two_way_equals_run"] = _close(d_close["two_way"].values, tw_old.values)

    # ── K13: benchmark della RCS per esecuzione ──────────────────────────
    bench_rows, bench_model = [], []
    for a in assets:
        for execution, (summary, per_model) in recon[a]["bt_out"]["benchmarks"].items():
            bench_rows.append({"Asset": a, **summary})
            pm = per_model.assign(Asset=a, Execution=execution).reset_index()
            rcs = bt_all[execution]["rcs"][a]
            pm["Alpha vs Long Filtered (daily)"] = [
                rcs[m]["Alpha vs Long Filtered (daily)"] for m in pm["Model"]]
            pm["Alpha vs Long Filtered p-value"] = [
                rcs[m]["Alpha vs Long Filtered p-value"] for m in pm["Model"]]
            pm["Sharpe CI Lower (95%)"] = [rcs[m]["Sharpe CI Lower (95%)"] for m in pm["Model"]]
            pm["Sharpe CI Upper (95%)"] = [rcs[m]["Sharpe CI Upper (95%)"] for m in pm["Model"]]
            bench_model.append(pm)
    bench = pd.DataFrame(bench_rows)
    bench_m = pd.concat(bench_model, ignore_index=True)
    bench.to_csv(os.path.join(out_dir, "rcs_benchmarks.csv"), index=False)
    bench_m.to_csv(os.path.join(out_dir, "rcs_benchmarks_by_model.csv"), index=False)
    b_old = pd.read_csv(os.path.join(results_dir, "rcs_benchmarks.csv")).set_index("Asset")
    b_new = bench[bench.Execution == "close"].set_index("Asset")
    cols_b = [c for c in b_old.columns if c in b_new.columns]
    checks["close_benchmarks_equal_run"] = _close(
        b_new.loc[b_old.index, cols_b].values, b_old[cols_b].values, atol=1e-4)
    bm_old = pd.read_csv(os.path.join(results_dir, "rcs_benchmarks_by_model.csv"))
    bm_new = bench_m[bench_m.Execution == "close"].merge(
        bm_old, on=["Asset", "Model"], suffixes=("", " (run)"))
    checks["close_benchmarks_by_model_equal_run"] = (
        len(bm_new) == len(bm_old)
        and _close(bm_new["Sharpe RCS"], bm_new["Sharpe RCS (run)"], atol=1e-4)
        and _close(bm_new["p_emp vs Random Filtered"],
                   bm_new["p_emp vs Random Filtered (run)"], atol=1e-4)
        and bool((bm_new["Identical to Long Filtered"]
                  == bm_new["Identical to Long Filtered (run)"]).all()))

    # ── R11: DM con p-value non arrotondati e Holm ───────────────────────
    dm_tables = {a: recon[a]["dm_table"].copy() for a in assets}
    holm, n_dm = M._holm_correct(
        {a: dm_tables[a]["p-value (raw)"].to_dict() for a in assets})
    comp_rows = []
    for a in assets:
        t = dm_tables[a]
        t["p Holm"] = [holm[a][m][0] for m in t.index]
        t["Sig. Holm (5%)"] = ["Yes" if holm[a][m][1] else "No" for m in t.index]
        t.to_csv(os.path.join(out_dir, f"DM_test_{a}.csv"))
        old = pd.read_csv(os.path.join(results_dir, f"DM_test_{a}.csv"), index_col=0)
        for m in t.index:
            comp_rows.append({
                "Asset": a, "Model": m,
                "DM Stat": t.loc[m, "DM Stat"],
                "DM Stat (run)": old.loc[m, "DM Stat"],
                "p (raw)": t.loc[m, "p-value (raw)"],
                "p (run, 4 dec.)": old.loc[m, "p-value"],
                "p Holm (raw)": t.loc[m, "p Holm"],
                "p Holm (run)": old.loc[m, "p Holm"],
                "Sig. Holm (raw)": t.loc[m, "Sig. Holm (5%)"],
                "Sig. Holm (run)": old.loc[m, "Sig. Holm (5%)"],
            })
    dm_comp = pd.DataFrame(comp_rows)
    dm_comp.to_csv(os.path.join(out_dir, "dm_holm_comparison.csv"), index=False)
    checks["dm_stat_equals_run"] = _close(dm_comp["DM Stat"], dm_comp["DM Stat (run)"],
                                          atol=1e-4)

    # ── R5: GARCH dalle previsioni salvate ───────────────────────────────
    g_rows, g_dm = [], []
    for a in assets:
        rows, dm_row = M._garch_from_disk(a, all_dfs[a], M.N_FOLDS)
        g_rows += rows
        if dm_row is not None:
            g_dm.append(dm_row)
    g_new = pd.DataFrame(g_rows)
    g_dm_new = pd.DataFrame(g_dm)
    g_new.to_csv(os.path.join(out_dir, "garch_volatility.csv"), index=False)
    g_dm_new.to_csv(os.path.join(out_dir, "garch_dm_qlike.csv"), index=False)

    g_old = pd.read_csv(os.path.join(results_dir, "garch_volatility.csv"))
    g_dm_old = pd.read_csv(os.path.join(results_dir, "garch_dm_qlike.csv"))
    val_rows, rebuilt_only = [], []
    for a in assets:
        o = g_old[g_old.Asset == a]
        if o.empty:
            rebuilt_only.append(a)
            continue
        mg = g_new[g_new.Asset == a].merge(o, on=["Asset", "Dist", "Fold"],
                                           suffixes=("", " (run)"))
        row = {"Asset": a, "Rows": len(mg)}
        for c in GARCH_COLS:
            row[f"max |diff| {c}"] = float(np.nanmax(np.abs(
                mg[c].astype(float) - mg[f"{c} (run)"].astype(float))))
        od = g_dm_old[g_dm_old.Asset == a]
        nd = g_dm_new[g_dm_new.Asset == a] if len(g_dm_new) else od.iloc[0:0]
        row["|diff| DM Stat"] = (abs(float(nd["DM Stat"].iloc[0]) - float(od["DM Stat"].iloc[0]))
                                 if len(od) and len(nd) else float("nan"))
        val_rows.append(row)
    g_val = pd.DataFrame(val_rows)
    g_val.to_csv(os.path.join(out_dir, "garch_validation.csv"), index=False)

    shutil.rmtree(work, ignore_errors=True)

    # ── Manifest e resoconto ─────────────────────────────────────────────
    manifest = {
        "script": "tools/recompute_post_run.py",
        "commit": M._git_commit(),
        "commit_dirty": M._git_dirty(),
        "input_results": results_dir,
        "input_run_commit": run_manifest.get("commit"),
        "input_run_packages": run_manifest.get("packages"),
        "python_version": sys.version,
        "packages": M._package_versions(),
        "bootstrap_parameters": d_par,
        "date": datetime.datetime.now().isoformat(),
        "checks": {k: bool(v) for k, v in checks.items()},
    }
    with open(os.path.join(out_dir, "recompute_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    sig_changes = dm_comp[dm_comp["Sig. Holm (raw)"] != dm_comp["Sig. Holm (run)"]]
    gold = g_new[g_new.Asset.isin(rebuilt_only)]
    g_val_fmt = g_val.copy()
    for c in g_val_fmt.columns:
        if c.startswith("max |diff|") or c.startswith("|diff|"):
            g_val_fmt[c] = g_val_fmt[c].map(lambda v: f"{v:.1e}")
    dm_fmt = dm_comp[dm_comp["p (raw)"] < 0.10].copy()
    for c in ("p (raw)", "p Holm (raw)"):
        dm_fmt[c] = dm_fmt[c].map(lambda v: f"{v:.4g}")
    pa = d_close["D"][d_close["D"].Aggregation == "per_asset"].iloc[0]
    disc = d_close["per_asset_discards"]
    bench_fmt = bench[["Asset", "Execution", "Sharpe Long Passive", "Sharpe Long Filtered",
                       "Sharpe Random Filtered P95", "Frac. Days Exposed"]]
    report = [
        "# Ricalcolo post-esecuzione (R5, R11, R14, K13)",
        "",
        f"- Esecuzione di partenza: commit `{run_manifest.get('commit')}`",
        f"- Codice del ricalcolo: commit `{manifest['commit']}`"
        + (" con modifiche non registrate" if manifest["commit_dirty"] else ""),
        f"- Parametri del bootstrap: {d_par}",
        f"- Data: {manifest['date']}",
        "",
        "## Controlli di coerenza con l'esecuzione",
        "",
        "\n".join(f"- {k}: **{v}**" for k, v in manifest["checks"].items()),
        "",
        "## R14 — statistica D (close-to-close)",
        "",
        _md_table(d_new.round(6)),
        "",
        "Valori dell'esecuzione di partenza:",
        "",
        _md_table(d_old.round(6)),
        "",
        f"Controllo per asset, repliche scartate: "
        f"{int(pa['N Bootstrap Requested'] - pa['N Bootstrap Used'])}; per asset con una "
        f"cella sotto la soglia (una replica può contarne più d'uno): "
        + ", ".join(f"{a} {int(v)}" for a, v in disc.items()) + ".",
        "",
        "## K13 — statistica D con la variante eseguibile di riferimento",
        "",
        "Varianti: " + ", ".join(f"{a} {d['execution']}" for a, d in asset_data_ex.items()) + ".",
        "",
        _md_table(d_exec["D"].round(6)),
        "",
        "Decomposizione a due fattori:",
        "",
        _md_table(d_exec["two_way"].round(4).reset_index()),
        "",
        "## K13 — benchmark della regola di esposizione per esecuzione",
        "",
        _md_table(bench_fmt),
        "",
        "## R11 — DM e Holm sui p-value non arrotondati",
        "",
        f"Test nella famiglia: {n_dm}. "
        f"Significativi dopo Holm — prima: "
        f"{int((dm_comp['Sig. Holm (run)'] == 'Yes').sum())}, "
        f"ora: {int((dm_comp['Sig. Holm (raw)'] == 'Yes').sum())}. "
        f"Cambi di esito: {len(sig_changes)}.",
        f"Massima differenza tra p Holm: "
        f"{float(np.max(np.abs(dm_comp['p Holm (raw)'] - dm_comp['p Holm (run)']))):.4f}.",
        "",
        "Test con p grezzo sotto 0.10:",
        "",
        _md_table(dm_fmt),
        "",
        "## R5 — GARCH dalle previsioni salvate",
        "",
        "Controllo sugli asset presenti nell'esecuzione (differenza massima "
        "tra valori ricostruiti e salvati):",
        "",
        _md_table(g_val_fmt),
        "",
        "Asset assenti da garch_volatility.csv dell'esecuzione (ripresi con "
        f"--resume) e ricostruiti solo qui: {', '.join(rebuilt_only) or 'nessuno'}.",
        "",
        "DM sulla QLIKE (gaussiano contro Student-t), tutti gli asset:",
        "",
        _md_table(g_dm_new.round(6)),
        "",
        "Asset ricostruiti solo da qui, per fold:",
        "",
        _md_table(gold.round(6)) if not gold.empty else "(nessuno)",
        "",
    ]
    with open(os.path.join(out_dir, "recompute_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report))
    if not all(manifest["checks"].values()):
        print("\nATTENZIONE: almeno un controllo di coerenza con l'esecuzione è fallito.")


if __name__ == "__main__":
    main()
