from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np

def train_svr_model(df):
    # Usa gli ultimi N valori per predire i successivi
    window = 5
    df["target"] = df["scaled"].shift(-1)
    df = df.dropna()

    X = np.array([df["scaled"].values[i-window:i] for i in range(window, len(df))])
    y = df["target"].values[window:]

    # Split temporale (80/10/10)
    n = len(X)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # Grid search su SVR
    param_grid = {
        "C": [1, 10],
        "epsilon": [0.01, 0.1],
        "kernel": ["rbf"]
    }

    model = GridSearchCV(SVR(), param_grid, cv=3)
    model.fit(X_train, y_train)

    best_model = model.best_estimator_
    y_pred = best_model.predict(X_test)

    return df.index[-len(y_test):], y_test, y_pred, model.best_params_