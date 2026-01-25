import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def train_bidirectional_model(df, sequence_length=10):
    data = df["scaled"].values
    dates = df.index

    X, y, out_dates = [], [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
        out_dates.append(dates[i + sequence_length])

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    out_dates = pd.to_datetime(out_dates)

    # Train-test split
    X_train, X_test = X[:-21], X[-21:]
    y_train, y_test = y[:-21], y[-21:]
    test_dates = out_dates[-21:]

    model = Sequential()
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    y_pred = model.predict(X_test).flatten()

    best_params = {
        "sequence_length": sequence_length,
        "units": 64,
        "dropout": 0.2,
        "epochs": 20,
        "batch_size": 16
    }

    return test_dates, y_test, y_pred, best_params