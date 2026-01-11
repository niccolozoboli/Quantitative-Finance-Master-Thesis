# === IMPORTS ===
from src.preprocessing import apply_log_return, scale_features
from src.data_loader import download_commodity_data
from src.models.arima import train_arima_model
from src.models.svr import train_svr_model
from src.models.random_forest import train_random_forest_model
from src.visualisation import plot_forecast_vs_actual
from src.models.transformer import train_transformer_model
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from src.models.xgboost import train_xgboost_model
from src.models.lstm import train_lstm_model
from src.models.gru import train_gru_model

# === CONFIG ===
metals = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Copper": "HG=F"
}

start_date = "2025-01-01"
end_date = "2025-12-31"

# === FUNCTIONS ===
def load_and_preprocess(name, ticker):
    file_path = f"data/{name}_2025.csv"
    df = download_commodity_data(ticker, start_date, end_date, save_path=file_path)
    df = apply_log_return(df, column="Close")
    df = scale_features(df, column="log_return")
    return df

def plot_forecast_vs_actual(y_true, y_pred, title="Forecast vs Real"):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true.index, y_true.values, label="Real", color="blue")
    plt.plot(y_true.index, y_pred, label="Forecast", color="orange", linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === ARIMA ===
for name, ticker in metals.items():
    print(f"\n📈 ARIMA Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    y_true, y_pred = train_arima_model(df)
    plot_forecast_vs_actual(y_true, y_pred, title=f"{name} (2025) - ARIMA Forecast vs Real")

# === SVR ===
for name, ticker in metals.items():
    print(f"\n🤖 SVR Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_svr_model(df)
    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - SVR Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")

    # === Random Forest ===
for name, ticker in metals.items():
    print(f"\n🌲 Random Forest Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_random_forest_model(df)
    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - Random Forest Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")

 

# === TRANSFORMER ===
for name, ticker in metals.items():
    print(f"\n🧠 Transformer Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_transformer_model(df)

    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - Transformer Forecast vs Real Performance"
    )

    print(f"Best params for {name}: {best_params}")

    # === XGBOOST ===
for name, ticker in metals.items():
    print(f"\n⚡ XGBoost Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_xgboost_model(df)

    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - XGBoost Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")

# === LSTM ===

for name, ticker in metals.items():
    print(f"\n🧬 LSTM Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_lstm_model(df)

    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - LSTM Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")

# === GRU ===

for name, ticker in metals.items():
    print(f"\n🔁 GRU Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_gru_model(df)

    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - GRU Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")