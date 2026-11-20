"""
tools/summarize_robustness.py
-----------------------------
Riassume le esecuzioni di robustezza (commenti K08 e K09 del relatore) a
partire dalle previsioni salvate, con le stesse funzioni della pipeline:

  - dl_seed_metrics.csv     metriche per modello, asset e seed
                            (seed 42 = esecuzione principale)
  - dl_seed_dispersion.csv  media, deviazione standard, minimo e massimo
                            tra seed di RMSE, MAE, Sharpe statico e DA
  - tcn31_vs_tcn.csv        TCN (campo ricettivo 15) contro TCN-31, con DM
                            sulla loss quadratica tra le due previsioni

Controlla che tutte le esecuzioni provengano dallo stesso commit git
dell'esecuzione principale, con codice committato (run_manifest*.json e,
se c'è, la colonna Commit delle previsioni), e che ogni previsione di
robustezza abbia le stesse date e gli stessi rendimenti realizzati della
previsione principale dello stesso modello; altrimenti si ferma.

    cd CODE && RESULTS_DIR=<cartella risultati> python tools/summarize_robustness.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.preprocessing import METALS
from src.utils import compute_metrics
from src.walk_forward import aggregate_fold_results
from src.backtesting import run_backtest
from src.statistical_tests import diebold_mariano_test

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
DL = ["LSTM", "GRU", "BiLSTM", "Transformer", "TCN", "CNN LSTM"]


def _manifest(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def _commit(name):
    return _manifest(name).get("commit")


def _check_commit_column(p: pd.DataFrame, main_commit: str, path: str) -> None:
    """Colonna Commit delle previsioni (R19): stesso commit, codice pulito."""
    if "Commit" not in p.columns:
        print(f"  {os.path.basename(path)}: nessuna colonna Commit (file anteriore a R19)")
        return
    found = set(p["Commit"].astype(str))
    if found != {str(main_commit)}:
        sys.exit(f"{path}: previsioni dal codice {sorted(found)}, esecuzione principale "
                 f"dal commit {main_commit} (un suffisso -dirty indica codice non "
                 f"committato): non confrontabili.")


def _check_aligned(ref: pd.DataFrame, other: pd.DataFrame, what: str) -> None:
    """R4: stesse date e stessi rendimenti realizzati, nello stesso ordine."""
    a = ref.sort_values(["Fold", "Date"])
    b = other.sort_values(["Fold", "Date"])
    if len(a) != len(b) or not (a["Date"].values == b["Date"].values).all() or \
            not np.allclose(a["y_true"].values, b["y_true"].values, rtol=0.0, atol=1e-12):
        raise ValueError(f"{what}: date o rendimenti diversi da quelli "
                         f"dell'esecuzione principale")


def _metrics(g: pd.DataFrame, asset: str) -> dict:
    g = g.sort_values(["Fold", "Date"])
    agg = aggregate_fold_results([compute_metrics(x["y_true"].values, x["y_pred"].values)
                                  for _, x in g.groupby("Fold")])
    bt = run_backtest(g["y_true"].values, g["y_pred"].values,
                      dates=pd.DatetimeIndex(g["Date"]), asset_name=asset)
    e = g["y_true"].values - g["y_pred"].values
    return {"RMSE": agg["RMSE"], "RMSE pooled": float(np.sqrt(np.mean(e ** 2))),
            "MAE": agg["MAE"], "Sharpe": bt["Sharpe Ratio"],
            "Directional Acc.": bt["Directional Acc."]}


def main():
    main_commit = _commit("run_manifest.json")
    tags = sorted({re.match(r"run_manifest_(.+)\.json", os.path.basename(p)).group(1)
                   for p in glob.glob(os.path.join(RESULTS_DIR, "run_manifest_*.json"))})
    for name in ["run_manifest.json"] + [f"run_manifest_{t}.json" for t in tags]:
        if _manifest(name).get("commit_dirty"):
            sys.exit(f"{name}: esecuzione da codice non committato (commit_dirty).")
    for tag in tags:
        c = _commit(f"run_manifest_{tag}.json")
        if c != main_commit:
            sys.exit(f"Esecuzione '{tag}' dal commit {c}, principale dal commit "
                     f"{main_commit}: non confrontabili. Rilanciala dallo stesso commit.")

    rows = []
    for asset in METALS:
        main_path = os.path.join(RESULTS_DIR, f"predictions_{asset}.csv")
        main_pred = pd.read_csv(main_path, parse_dates=["Date"])
        _check_commit_column(main_pred, main_commit, main_path)
        for model in DL:
            g = main_pred[main_pred["Model"] == model]
            if len(g):
                rows.append({"Asset": asset, "Model": model, "Seed": 42, **_metrics(g, asset)})
        for tag in [t for t in tags if t.startswith("dl_seed")]:
            path = os.path.join(RESULTS_DIR, f"predictions_{tag}_{asset}.csv")
            if not os.path.exists(path):
                continue
            p = pd.read_csv(path, parse_dates=["Date"])
            _check_commit_column(p, main_commit, path)
            for model, g in p.groupby("Model"):
                _check_aligned(main_pred[main_pred["Model"] == model], g,
                               f"{asset}, {model}, {tag}")
                rows.append({"Asset": asset, "Model": model,
                             "Seed": int(g["Seed"].iloc[0]), **_metrics(g, asset)})
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(RESULTS_DIR, "dl_seed_metrics.csv"), index=False)
    metrics = ["RMSE", "RMSE pooled", "MAE", "Sharpe", "Directional Acc."]
    disp = long.groupby(["Asset", "Model"])[metrics].agg(["mean", "std", "min", "max"])
    disp.columns = [f"{m} {s}" for m, s in disp.columns]
    disp["N Seeds"] = long.groupby(["Asset", "Model"])["Seed"].nunique()
    disp.to_csv(os.path.join(RESULTS_DIR, "dl_seed_dispersion.csv"))
    print(disp[["RMSE mean", "RMSE std", "Sharpe mean", "Sharpe std", "N Seeds"]].round(4).to_string())

    if "tcn31" in tags:
        out = []
        for asset in METALS:
            main_pred = pd.read_csv(os.path.join(RESULTS_DIR, f"predictions_{asset}.csv"),
                                    parse_dates=["Date"])
            p31_path = os.path.join(RESULTS_DIR, f"predictions_tcn31_{asset}.csv")
            p31 = pd.read_csv(p31_path, parse_dates=["Date"])
            _check_commit_column(p31, main_commit, p31_path)
            g15 = main_pred[main_pred["Model"] == "TCN"].sort_values(["Fold", "Date"])
            g31 = p31.sort_values(["Fold", "Date"])
            # R4: confronto per data, non per posizione
            _check_aligned(g15, g31, f"{asset}, TCN-31")
            dm = diebold_mariano_test(g15["y_true"].values, g31["y_pred"].values,
                                      g15["y_pred"].values)
            m15, m31 = _metrics(g15, asset), _metrics(g31, asset)
            out.append({"Asset": asset,
                        **{f"TCN-15 {k}": v for k, v in m15.items()},
                        **{f"TCN-31 {k}": v for k, v in m31.items()},
                        "DM (TCN-31 vs TCN-15)": dm["dm_statistic"],
                        "DM p-value": dm["p_value"]})
        pd.DataFrame(out).to_csv(os.path.join(RESULTS_DIR, "tcn31_vs_tcn.csv"), index=False)
        print(pd.DataFrame(out).round(4).to_string())


if __name__ == "__main__":
    main()
