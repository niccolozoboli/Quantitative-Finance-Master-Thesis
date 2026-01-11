import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models

# === COSTRUZIONE DEL MODELLO GRU ===
def build_gru_model(input_shape):
    model = models.Sequential()
    model.add(layers.GRU(64, input_shape=input_shape, return_sequences=False))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

# === TRAINING DEL MODELLO GRU ===
def train_gru_model(df):
    sequence_length = 10
    data = df["scaled"].values
    X, y, dates = [], [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
        dates.append(df.index[i + sequence_length])

    X = np.array(X)
    y = np.array(y)
    dates = pd.to_datetime(dates)

    # Aggiungi dimensione per compatibilità GRU (samples, timesteps, features)
    X = X[..., np.newaxis]

    # Split temporale train/test
    X_train, X_test = X[:-21], X[-21:]
    y_train, y_test = y[:-21], y[-21:]
    test_dates = dates[-21:]

    # Costruzione e training del modello
    model = build_gru_model(input_shape=(X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # Previsioni
    y_pred = model.predict(X_test).flatten()

    # Parametri principali
    best_params = {
        "sequence_length": sequence_length,
        "units": 64,
        "dropout": 0.2,
        "batch_size": 16,
        "epochs": 20
    }

    return test_dates, y_test, y_pred, best_params