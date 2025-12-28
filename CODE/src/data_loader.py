 # per caricare dati da CSV, API, ecc.

import yfinance as yf
import pandas as pd
def download_commodity(symbol:str, start="2010-01-01", end="2024-12-31"):
    df = yf.download(symbol, start=start, end=end)
    df = df[['Close']].dropna()
    df.index.name = "Date"
    return df