import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def train_xgboost_model(df):
    # Crea feature lag (1 giorno indietro)
    df["lag_1"] = df["scaled"].shift(1)
    df.dropna(inplace=True)

    # Feature e target
    X = df[["lag_1"]]
    y = df["scaled"]

    # Split temporale (train: fino a novembre, test: dicembre)
    X_train, X_test = X[:-21], X[-21:]
    y_train, y_test = y[:-21], y[-21:]
    dates = df.index[-21:]

    # Definisci e allena il modello
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Previsioni
    y_pred = model.predict(X_test)

    # RMSE per controllo interno (opzionale)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Parametri usati
    best_params = model.get_params()

    return dates, y_test, y_pred, best_params