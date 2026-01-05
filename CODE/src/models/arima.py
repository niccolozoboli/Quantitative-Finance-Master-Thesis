import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def train_arima_model(train_series, order=(1, 1, 1)):
    """
    Trains an ARIMA model on the given training series.
    Args:
        train_series (pd.Series): The time series data (e.g., log returns).
        order (tuple): The (p,d,q) order of the ARIMA model.
    Returns:
        model_fit: The fitted ARIMA model.
    """
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps):
    """
    Generates forecast from ARIMA model.
    Args:
        model_fit: Fitted ARIMA model.
        steps (int): Number of steps to forecast.
    Returns:
        forecast (np.ndarray): Forecasted values.
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast

def evaluate_arima(true_values, predicted_values):
    """
    Evaluates the ARIMA forecast using standard metrics.
    Args:
        true_values (array-like): Actual values.
        predicted_values (array-like): Forecasted values.
    Returns:
        dict: MAE and RMSE values.
    """
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return {"MAE": mae, "RMSE": rmse}

def plot_arima_results(dates, train_series, test_series, predictions, title='ARIMA Forecast'):
    """
    Plots ARIMA predictions vs. actual.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates[:len(train_series)], train_series, label='Training Data', color='blue')
    plt.plot(dates[len(train_series):], test_series, label='Actual Prices', color='green')
    plt.plot(dates[len(train_series):], predictions, label='ARIMA Forecast', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()