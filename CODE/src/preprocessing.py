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


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.index = pd.to_datetime(df.index)
    df = df[["Close"]].copy()
    df.dropna(inplace=True)
    return df


def load_and_preprocess(name: str, ticker: str,
                         start: str = "2010-01-01",
                         end: str   = "2025-01-01") -> pd.DataFrame:
    """
    Full pipeline:
      1. Download Close prices
      2. Compute log-returns
      3. Keep both Close and log_return

    Returns DataFrame with columns: Close, log_return
    The 'Close' column is used for price-level visualization.
    The 'log_return' column is the forecasting target.
    Z-scoring is applied INSIDE the walk-forward folds, not here.
    """
    df = download_data(ticker, start=start, end=end)
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
