from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_arima_model(series, order=(1, 1, 1)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def create_lagged_features(df, lag=5, column="log_return"):
    for i in range(1, lag + 1):
        df[f"lag_{i}"] = df[column].shift(i)
    df = df.dropna()
    return df