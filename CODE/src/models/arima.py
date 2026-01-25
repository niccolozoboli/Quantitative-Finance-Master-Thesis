import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def train_arima_model(train_series, order=(1, 1, 1)):
    if not isinstance(train_series, pd.Series):
        raise ValueError("ARIMA requires a univariate pandas Series")

    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps):
    return model_fit.forecast(steps=steps)

def evaluate_arima(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae}

def plot_arima_results(dates, train_series, test_series, predictions, title):
    plt.figure(figsize=(12, 6))
    plt.plot(dates[:len(train_series)], train_series, label="Train")
    plt.plot(dates[len(train_series):], test_series, label="Test")
    plt.plot(dates[len(train_series):], predictions, label="Forecast")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Scaled values")
    plt.legend()
    plt.tight_layout()
    plt.show()