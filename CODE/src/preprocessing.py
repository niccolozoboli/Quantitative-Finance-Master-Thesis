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
