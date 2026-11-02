"""
preprocessing.py
----------------
Unified preprocessing pipeline for all models.

Design principle:
- Target variable = log_return (original scale, never Z-scored)
- Features       = Z-scored (only on training data, never on test)
- Prices         = kept for final inverse-transform and visualization

This ensures all models are evaluated on the same scale,
and predictions can be converted back to real price levels.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yfinance as yf


METALS = {
    "Gold":     "GC=F",
    "Silver":   "SI=F",
    "Copper":   "HG=F",
    "Platinum": "PL=F",
}

# CODE/data/ — snapshot locale dei prezzi grezzi, per riproducibilità:
# yfinance non garantisce di restituire la stessa serie storica a ogni
# chiamata (revisioni, dati mancanti aggiornati retroattivamente).
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _data_path(ticker: str) -> str:
    return os.path.join(_DATA_DIR, f"{ticker}.csv")


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance restituisce colonne (field, ticker) anche per un singolo
        # ticker: in memoria df["Close"] funziona comunque (selezione
        # parziale sul livello 0), ma un MultiIndex a una sola colonna
        # serializzato con to_csv produce righe di header multiple che
        # read_csv (usato dallo snapshot locale) non ricostruisce
        # correttamente — si appiattisce subito al nome di campo.
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Close"]].copy()
    df.dropna(inplace=True)
    return df


def load_and_preprocess(name: str, ticker: str,
                         start: str = "2010-01-01",
                         end: str   = "2025-01-01",
                         refresh: bool = False) -> pd.DataFrame:
    """
    Full pipeline:
      1. Load Close prices — da snapshot locale CODE/data/<ticker>.csv se
         presente e refresh=False, altrimenti scarica da yfinance e lo
         salva lì per le run successive (refresh=True forza un nuovo
         download anche se lo snapshot esiste già).
      2. Compute log-returns
      3. Keep both Close and log_return

    Returns DataFrame with columns: Close, log_return
    The 'Close' column is used for price-level visualization.
    The 'log_return' column is the forecasting target.
    Z-scoring is applied INSIDE the walk-forward folds, not here.

    Nota: se lo snapshot esiste, start/end NON vengono riapplicati — la
    serie caricata è esattamente quella salvata la prima volta. Per
    cambiare il periodo coperto serve refresh=True (o cancellare il file).
    """
    path = _data_path(ticker)
    if not refresh and os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = download_data(ticker, start=start, end=end)
        os.makedirs(_DATA_DIR, exist_ok=True)
        df.to_csv(path)

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot OHLC — esecuzione all'apertura successiva (commento K13)
# ─────────────────────────────────────────────────────────────────────────────
#
# Lo snapshot dei Close (CODE/data/<ticker>.csv) NON viene toccato: target,
# feature e backtest principale restano identici. Per la variante di
# esecuzione all'apertura si usa uno snapshot SEPARATO
# CODE/data/<ticker>_ohlc.csv, scaricato una sola volta con
# tools/fetch_ohlc.py (su Colab) e versionato. L'esecuzione principale non
# scarica mai nulla: se il file manca, o se non supera i controlli di
# validate_ohlc, la variante "open" non viene calcolata e il manifest lo
# registra.

OHLC_END = "2026-05-08"   # oltre END: serve l'apertura del giorno dopo l'ultimo OOS


def _ohlc_path(ticker: str) -> str:
    return os.path.join(_DATA_DIR, f"{ticker}_ohlc.csv")


def download_ohlc(ticker: str, start: str, end: str = OHLC_END) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df[["Open", "High", "Low", "Close"]].copy()


def load_ohlc_snapshot(ticker: str):
    """Snapshot OHLC se presente, altrimenti None (nessun download qui)."""
    path = _ohlc_path(ticker)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def validate_ohlc(ohlc: pd.DataFrame, close_df: pd.DataFrame,
                  oos_dates: pd.DatetimeIndex, rel_tol: float = 1e-6) -> dict:
    """
    Controlli prima di usare l'apertura come prezzo di esecuzione:
      1. ogni data OOS e il giorno di borsa successivo all'ultima data OOS
         sono presenti nello snapshot OHLC, con Open non mancante e > 0;
      2. Low <= Open <= High in quelle date;
      3. il Close OHLC coincide con lo snapshot dei Close nelle date OOS
         (stessa serie del provider, nessuna revisione nel frattempo).
    Ritorna {"ok": bool, "reason": str, ...}.
    """
    oos_dates = pd.DatetimeIndex(oos_dates)
    if ohlc is None:
        return {"ok": False, "reason": "snapshot OHLC assente"}
    after = ohlc.index[ohlc.index > oos_dates.max()]
    if len(after) == 0:
        return {"ok": False, "reason": "manca il giorno di borsa successivo all'ultima data OOS"}
    need = oos_dates.append(pd.DatetimeIndex([after.min()]))
    missing = need.difference(ohlc.index)
    if len(missing):
        return {"ok": False, "reason": f"{len(missing)} date OOS assenti nello snapshot OHLC"}
    sub = ohlc.loc[need]
    if sub["Open"].isna().any() or (sub["Open"] <= 0).any():
        return {"ok": False, "reason": "Open mancante o non positivo"}
    bad_range = ((sub["Open"] < sub["Low"] - 1e-9) | (sub["Open"] > sub["High"] + 1e-9)).sum()
    if bad_range:
        return {"ok": False, "reason": f"Open fuori da [Low, High] in {int(bad_range)} date"}
    c_old = close_df["Close"].reindex(oos_dates)
    c_new = ohlc["Close"].reindex(oos_dates)
    rel = (c_new / c_old - 1).abs()
    if rel.isna().any() or (rel > rel_tol).any():
        return {"ok": False, "reason": f"Close diverso dallo snapshot in {int((rel > rel_tol).sum())} date OOS"}
    return {"ok": True, "reason": "ok", "next_day_after_oos": str(after.min().date())}


def open_to_open_returns(ohlc: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    R^oo per ciascuna data di `dates`: O_{giorno successivo} / O_{data} - 1,
    cioè il rendimento di una posizione aperta all'apertura della data e
    chiusa all'apertura del giorno di borsa successivo. NaN se non disponibile.
    """
    o = ohlc["Open"].astype(float)
    nxt = o.shift(-1)
    r = (nxt / o - 1).reindex(pd.DatetimeIndex(dates))
    return r.values.astype(float)


def reconstruct_prices(last_known_price: float,
                        predicted_log_returns: np.ndarray) -> np.ndarray:
    """
    Converts predicted log-returns back to price levels.

    Formula: P_{t+k} = P_t * exp( sum_{i=1}^{k} r_{t+i} )

    This is the key function that allows model forecasts
    (in log-return space) to be compared against real prices.
    """
    cumulative = np.cumsum(predicted_log_returns)
    return last_known_price * np.exp(cumulative)
