import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models

# === TRANSFORMER BLOCK ===
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# === COSTRUZIONE DEL MODELLO TRANSFORMER ===
def build_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = TransformerBlock(embed_dim=input_shape[-1], num_heads=2, ff_dim=64)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

# === TRAINING DEL MODELLO TRANSFORMER ===
def train_transformer_model(df):
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

    # Aggiungi dimensione per compatibilità (features)
    X = X[..., np.newaxis]

    # Split temporale train/test
    X_train, X_test = X[:-21], X[-21:]
    y_train, y_test = y[:-21], y[-21:]
    test_dates = dates[-21:]

    # Costruisci e allena il modello
    model = build_transformer_model(input_shape=(X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # Previsioni
    y_pred = model.predict(X_test).flatten()

    # Parametri ottimali salvati
    best_params = {
        "sequence_length": sequence_length,
        "num_heads": 2,
        "ff_dim": 64,
        "dropout": 0.2,
        "batch_size": 16,
        "epochs": 20
    }

    return test_dates, y_test, y_pred, best_params