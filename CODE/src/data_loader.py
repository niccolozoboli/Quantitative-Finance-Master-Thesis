import yfinance as yf
import pandas as pd
import os

def download_commodity_data(ticker, start_date, end_date, save_path):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if not os.path.exists("data"):
        os.makedirs("data")
    data.to_csv(save_path)
    return data

def load_commodity_data(filename):
    path = os.path.join("data", filename)
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df["Close"]  # or "Adj Close" if preferred