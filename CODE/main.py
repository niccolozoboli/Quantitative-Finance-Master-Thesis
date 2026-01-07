from src.preprocessing import apply_log_return, scale_features
from src.data_loader import download_commodity_data
import matplotlib.pyplot as plt
import datetime

# Lista dei metalli e ticker Yahoo Finance
metals = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Copper": "HG=F"
}

# Date
start_date = "2025-01-01"
end_date = "2025-12-31"

# Loop su ogni metallo
for name, ticker in metals.items():
    print(f"\n▶ Processing {name} ({ticker})")

    # 1. Scarica i dati
    save_path = f"data/{name}_2025.csv"
    df = download_commodity_data(ticker, start_date, end_date, save_path)

    # 2. Preprocessing: log return + scaling
    df = apply_log_return(df, column="Close")
    df = scale_features(df, column="log_return")

    # 3. Visualizza graficamente i dati preprocessati
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["scaled"], label=f"{name} Scaled Return", color="blue")
    plt.title(f"{name} - Scaled Log Returns (2025)")
    plt.xlabel("Date")
    plt.ylabel("Scaled Log Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ## ARIMA MODEL 

    from src.models.arima import train_arima_model
from src.visualisation import plot_forecast_vs_actual

# Forecasting con ARIMA e confronto con i dati reali
for name, ticker in metals.items():
    print(f"\n📈 ARIMA Forecasting for {name} ({ticker})")

    # 1. Ricarica i dati reali del 2025 dal file salvato
    file_path = f"data/{name}_2025.csv"
    df = download_commodity_data(ticker, start_date, end_date, save_path=file_path)
    
    # 2. Preprocessing identico (log return + scaling)
    df = apply_log_return(df, column="Close")
    df = scale_features(df, column="log_return")

    # 3. Applica il modello ARIMA
    y_true, y_pred = train_arima_model(df)

    # 4. Visualizza confronto tra reale e previsto
    plot_forecast_vs_actual(
        y_true, y_pred,
        title=f"{name} (2025) - ARIMA Forecast vs Real Performance"
    )

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