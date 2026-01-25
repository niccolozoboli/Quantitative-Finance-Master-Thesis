# === IMPORTS ===
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.preprocessing import apply_log_return, scale_features
from src.data_loader import download_commodity_data
from src.visualisation import plot_forecast_vs_actual

# MODELLI
from src.models.arima import train_arima_model
from src.models.svr import train_svr_model
from src.models.random_forest import train_random_forest_model
from src.models.xgboost import train_xgboost_model
from src.models.transformer import train_transformer_model
from src.models.lstm import train_lstm_model
from src.models.gru import train_gru_model
from src.models.decision_tree import train_decision_tree_model

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

def compute_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = (abs((y_true - y_pred) / y_true).mean()) * 100 if all(y_true != 0) else None
    return rmse, mae, mape

# === COLLECT ALL RESULTS ===
results_list = []

# === ARIMA ===
from src.models.arima import (
    train_arima_model,
    forecast_arima,
    evaluate_arima,
    plot_arima_results
)

for name, ticker in metals.items():
    print(f"\nARIMA Forecasting for {name} ({ticker})")

    df = load_and_preprocess(name, ticker)

    series = df["scaled"]
    train_series = series[:-21]
    test_series = series[-21:]

    model_fit = train_arima_model(train_series, order=(1, 1, 1))
    y_pred = forecast_arima(model_fit, steps=21)
    metrics = evaluate_arima(test_series, y_pred)

    plot_arima_results(
        df.index,
        train_series,
        test_series,
        y_pred,
        title=f"{name} (2025) - ARIMA Forecast vs Real"
    )

    results_list.append({
        "Asset": name,
        "Model": "ARIMA",
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "MAPE": None
    })


# === SVR ===
for name, ticker in metals.items():
    print(f"\nSVR Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_svr_model(df)
    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - SVR Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")
    rmse, mae, mape = compute_metrics(y_true, y_pred)
    results_list.append({"Asset": name, "Model": "SVR", "RMSE": rmse, "MAE": mae, "MAPE": mape})


# === RANDOM FOREST ===
for name, ticker in metals.items():
    print(f"\nRandom Forest Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_random_forest_model(df)
    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - Random Forest Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")
    rmse, mae, mape = compute_metrics(y_true, y_pred)
    results_list.append({"Asset": name, "Model": "Random Forest", "RMSE": rmse, "MAE": mae, "MAPE": mape})


# === XGBOOST ===
for name, ticker in metals.items():
    print(f"\nXGBoost Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_xgboost_model(df)
    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - XGBoost Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")
    rmse, mae, mape = compute_metrics(y_true, y_pred)
    results_list.append({"Asset": name, "Model": "XGBoost", "RMSE": rmse, "MAE": mae, "MAPE": mape})


# === TRANSFORMER ===
for name, ticker in metals.items():
    print(f"\nTransformer Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_transformer_model(df)
    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - Transformer Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")
    rmse, mae, mape = compute_metrics(y_true, y_pred)
    results_list.append({"Asset": name, "Model": "Transformer", "RMSE": rmse, "MAE": mae, "MAPE": mape})


# === LSTM ===
for name, ticker in metals.items():
    print(f"\nLSTM Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_lstm_model(df)
    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - LSTM Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")
    rmse, mae, mape = compute_metrics(y_true, y_pred)
    results_list.append({"Asset": name, "Model": "LSTM", "RMSE": rmse, "MAE": mae, "MAPE": mape})


# === GRU ===
for name, ticker in metals.items():
    print(f"\nGRU Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_gru_model(df)
    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - GRU Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")
    rmse, mae, mape = compute_metrics(y_true, y_pred)
    results_list.append({"Asset": name, "Model": "GRU", "RMSE": rmse, "MAE": mae, "MAPE": mape})

# === DECISION TREE ===
for name, ticker in metals.items():
    print(f"\nDecision Tree Forecasting for {name} ({ticker})")
    df = load_and_preprocess(name, ticker)
    dates, y_true, y_pred, best_params = train_decision_tree_model(df)
    plot_forecast_vs_actual(
        pd.Series(y_true, index=dates),
        pd.Series(y_pred, index=dates),
        title=f"{name} (2025) - Decision Tree Forecast vs Real Performance"
    )
    print(f"Best params for {name}: {best_params}")
    rmse, mae, mape = compute_metrics(y_true, y_pred)
    results_list.append({"Asset": name, "Model": "Decision Tree", "RMSE": rmse, "MAE": mae, "MAPE": mape})


# === SALVA METRICHE ===
results_df = pd.DataFrame(results_list)
results_df.to_csv("results/forecast_metrics.csv", index=False)
print("\nSaved all metrics to results/forecast_metrics.csv")
