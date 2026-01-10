# src/models/random_forest.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def train_random_forest_model(df):
    X = df[["scaled"]].shift(1).dropna()
    y = df["scaled"].loc[X.index]

    # Split temporale: train (80%), test (20%)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = y_test.index

    # Grid Search su pochi iperparametri
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7]
    }

    model = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"📊 RMSE Random Forest: {rmse:.4f}")

    return dates_test, y_test, y_pred, grid.best_params_