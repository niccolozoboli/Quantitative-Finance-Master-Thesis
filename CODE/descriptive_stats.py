"""
descriptive_stats.py
---------------------
Statistiche descrittive sui log-return giornalieri e matrice di
correlazione di Pearson tra i 4 asset, per la sezione di analisi
esplorativa (Capitolo 3) della tesi.

Usa load_and_preprocess() con START/END identici a main.py, quindi
stesso periodo e stesso preprocessing della pipeline principale.
Non fa training, non importa nulla da src/models — solo download
(via yfinance, serve connessione) e statistiche descrittive dirette.

Uso: da dentro CODE/
    python3 descriptive_stats.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.preprocessing import METALS, load_and_preprocess

START = "2010-01-01"
END   = "2026-05-01"   # identico a main.py


def compute_descriptive_stats(returns: pd.Series) -> dict:
    r = returns.dropna().values
    return {
        "Mean (%)":         100 * np.mean(r),
        "Std Dev (%)":       100 * np.std(r, ddof=1),
        "Skewness":          stats.skew(r, bias=False),
        "Excess Kurtosis":   stats.kurtosis(r, bias=False, fisher=True),
        "Min (%)":           100 * np.min(r),
        "Max (%)":           100 * np.max(r),
    }


def main():
    print("Caricamento dati (load_and_preprocess, identico a main.py: "
          f"start={START}, end={END})...")
    all_dfs = {}
    for name, ticker in METALS.items():
        all_dfs[name] = load_and_preprocess(name, ticker, start=START, end=END)
        print(f"  {name}: {len(all_dfs[name])} osservazioni")

    # ── Tabella statistiche descrittive ───────────────────────────────────
    rows = {name: compute_descriptive_stats(df["log_return"])
            for name, df in all_dfs.items()}
    stats_df = pd.DataFrame(rows).T
    stats_df = stats_df[["Mean (%)", "Std Dev (%)", "Skewness",
                          "Excess Kurtosis", "Min (%)", "Max (%)"]]

    print("\n" + "=" * 72)
    print("  STATISTICHE DESCRITTIVE — Log-Return Giornalieri")
    print("=" * 72)
    print(stats_df.round(4).to_string())

    print("\n--- LaTeX ---")
    print(stats_df.round(4).to_latex(
        float_format="%.4f",
        caption="Statistiche descrittive dei log-return giornalieri.",
        label="tab:descriptive_stats"))

    # ── Matrice di correlazione (Pearson) ─────────────────────────────────
    # Allineamento sulle date comuni ai 4 asset (dropna sull'intersezione)
    returns_df = pd.DataFrame(
        {name: df["log_return"] for name, df in all_dfs.items()}
    ).dropna()
    corr = returns_df.corr(method="pearson")

    print("\n" + "=" * 72)
    print(f"  MATRICE DI CORRELAZIONE (Pearson) — Log-Return "
          f"[{len(returns_df)} giorni comuni]")
    print("=" * 72)
    print(corr.round(4).to_string())

    print("\n--- LaTeX ---")
    print(corr.round(4).to_latex(
        float_format="%.4f",
        caption="Matrice di correlazione di Pearson tra i log-return giornalieri.",
        label="tab:correlation_matrix"))


if __name__ == "__main__":
    main()
