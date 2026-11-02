"""
tools/build_rolled_series.py
----------------------------
Serie realmente rollata per il periodo fuori campione (commento K03 del
relatore: "Se riesci a recuperare i prezzi dei singoli contratti almeno
per il periodo out-of-sample, ricostruisci una serie realmente rollata e
confrontala con il backtest corrente").

Per ogni data OOS tau si individua il contratto detenuto dalla chiusura
del giorno precedente (futures_calendar.held_contracts, stesso calendario
di roll del backtest) e si calcola il rendimento close-to-close di QUEL
contratto, R_rolled[tau] = P^c_tau / P^c_{tau-1} - 1: nessun salto tra
contratti diversi entra nel rendimento.

Da eseguire su Colab (questo ambiente non raggiunge Yahoo Finance):

    cd CODE && python tools/build_rolled_series.py

Scrive CODE/data/<ticker>_rolled.csv (Date, Contract, R_rolled,
R_continuous) solo se TUTTE le date OOS hanno un rendimento; altrimenti
non scrive la serie e main.py usa solo la sensitivity sui giorni di roll.
In ogni caso scrive CODE/data/rolled_series_report.csv, con l'esito del
download di ciascun contratto: è la traccia, da versionare, del tentativo
richiesto dal relatore (la tesi la cita per motivare la strada scelta). I ticker Yahoo dei singoli contratti (es. GCZ25.CMX)
seguono la convenzione <radice><codice mese><anno a 2 cifre>.<borsa>:
la loro disponibilità va verificata al momento dell'esecuzione.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf

from src.preprocessing import METALS, load_and_preprocess, _DATA_DIR
from src.futures_calendar import held_contracts

N_OOS = 210
MONTH_CODES = "FGHJKMNQUVXZ"
ROOT = {"Gold": ("GC", "CMX"), "Silver": ("SI", "CMX"),
        "Copper": ("HG", "CMX"), "Platinum": ("PL", "NYM")}


def contract_ticker(asset: str, year: int, month: int) -> str:
    root, exch = ROOT[asset]
    return f"{root}{MONTH_CODES[month - 1]}{year % 100:02d}.{exch}"


def download_close(ticker: str, start, end) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index)
    return s


def contract_report(asset: str, oos: pd.DatetimeIndex, prev: pd.DatetimeIndex,
                    held: list, prices: dict) -> list:
    """
    Una riga per contratto detenuto nel periodo OOS: ticker, numero di
    prezzi scaricati, primo e ultimo giorno disponibile, date OOS in cui il
    contratto è detenuto e quante di queste hanno un rendimento calcolabile
    (prezzo del contratto sia nella data sia nel giorno di borsa precedente).
    """
    rows = []
    for c in sorted(set(held)):
        p = prices[c]
        days = [(d, dp) for d, dp, h in zip(oos, prev, held) if h == c]
        ok = sum(d in p.index and dp in p.index for d, dp in days)
        rows.append({
            "Asset": asset, "Contract": contract_ticker(asset, *c),
            "N Prices Downloaded": int(len(p)),
            "First Price Date": p.index.min().date() if len(p) else None,
            "Last Price Date": p.index.max().date() if len(p) else None,
            "N OOS Days Held": len(days),
            "N OOS Days With Return": int(ok),
            "Complete": ok == len(days),
        })
    return rows


if __name__ == "__main__":
    report_rows = []
    for asset, ticker in METALS.items():
        df = load_and_preprocess(asset, ticker)
        oos = df.index[-N_OOS:]
        prev = df.index[-N_OOS - 1:-1]          # giorno di borsa precedente
        held = held_contracts(asset, oos)
        start = prev[0] - pd.Timedelta(days=10)
        end = oos[-1] + pd.Timedelta(days=5)
        prices = {c: download_close(contract_ticker(asset, *c), start, end)
                  for c in sorted(set(held))}
        report_rows += contract_report(asset, oos, prev, held, prices)
        rows = []
        for d, d_prev, c in zip(oos, prev, held):
            p = prices[c]
            r = (p.get(d, np.nan) / p.get(d_prev, np.nan) - 1
                 if d in p.index and d_prev in p.index else np.nan)
            rows.append({"Date": d, "Contract": contract_ticker(asset, *c),
                         "R_rolled": r,
                         "R_continuous": float(np.expm1(df.loc[d, "log_return"]))})
        out = pd.DataFrame(rows).set_index("Date")
        missing = int(out["R_rolled"].isna().sum())
        print(f"{asset}: contratti {sorted(out['Contract'].unique())}, "
              f"date OOS senza rendimento del contratto: {missing}/{N_OOS}")
        if missing:
            print(f"  {asset}: serie NON scritta — main.py userà solo la "
                  f"sensitivity sui giorni di roll.")
            continue
        path = os.path.join(_DATA_DIR, f"{ticker}_rolled.csv")
        out.to_csv(path)
        corr = out["R_rolled"].corr(out["R_continuous"])
        print(f"  scritto {path}; correlazione con la serie continua = {corr:.4f}")

    report = pd.DataFrame(report_rows)
    report_path = os.path.join(_DATA_DIR, "rolled_series_report.csv")
    report.to_csv(report_path, index=False)
    print(f"\nEsito del download dei contratti scritto in {report_path}")
    print(report.to_string(index=False))
