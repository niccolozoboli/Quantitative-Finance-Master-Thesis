import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def train_gru_model(data, window_size=30, epochs=50, batch_size=16):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = create_sequences(scaled_data, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(GRU(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = model.predict(X)
    predictions_inverse = scaler.inverse_transform(predictions)
    y_inverse = scaler.inverse_transform(y.reshape(-1, 1))

    mse = mean_squared_error(y_inverse, predictions_inverse)
    
    return model, predictions_inverse.flatten(), y_inverse.flatten(), mse