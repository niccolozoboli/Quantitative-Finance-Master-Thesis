"""
dl_models.py
------------
All deep learning models: LSTM, GRU, BiLSTM, Transformer.
All return (test_dates, y_true, y_pred, params) in log_return original scale.

Architecture is FIXED across all assets and folds for comparability,
as described in the thesis (Section 3.4).
"""

import os
import random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
#### Prova 

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from keras.models import Sequential, Model
from keras.layers import (LSTM, GRU, Bidirectional, Dense,
                           Dropout, Input, GlobalAveragePooling1D,
                           LayerNormalization, MultiHeadAttention)
from keras.callbacks import EarlyStopping

from src.feature_engine import prepare_dl_fold

SEQ_LEN   = 10
UNITS     = 64
DROPOUT   = 0.2
EPOCHS    = 100
PATIENCE  = 10
BATCH     = 32


def _early_stop():
    return EarlyStopping(monitor="val_loss", patience=PATIENCE,
                          restore_best_weights=True, verbose=0)


def _val_split(X, y, ratio=0.1):
    n   = int(len(X) * (1 - ratio))
    return X[:n], X[n:], y[:n], y[n:]


# ── LSTM ──────────────────────────────────────────────────────────────────────

def run_lstm_fold(df, train_idx, test_idx):
    X_tr, X_te, y_tr, y_te, dates = prepare_dl_fold(
        df, train_idx, test_idx, seq_len=SEQ_LEN)

    X_tr, X_val, y_tr, y_val = _val_split(X_tr, y_tr)

    model = Sequential([
        LSTM(UNITS, input_shape=(SEQ_LEN, 1), return_sequences=True),
        Dropout(DROPOUT),
        LSTM(32, return_sequences=False),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
               epochs=EPOCHS, batch_size=BATCH,
               callbacks=[_early_stop()], verbose=0)

    params = {"architecture": "LSTM(64)->LSTM(32)->Dense(1)",
               "seq_len": SEQ_LEN, "dropout": DROPOUT}
    return dates, y_te, model.predict(X_te, verbose=0).flatten(), params


# ── GRU ───────────────────────────────────────────────────────────────────────

def run_gru_fold(df, train_idx, test_idx):
    X_tr, X_te, y_tr, y_te, dates = prepare_dl_fold(
        df, train_idx, test_idx, seq_len=SEQ_LEN)

    X_tr, X_val, y_tr, y_val = _val_split(X_tr, y_tr)

    model = Sequential([
        GRU(UNITS, input_shape=(SEQ_LEN, 1), return_sequences=True),
        Dropout(DROPOUT),
        GRU(32, return_sequences=False),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
               epochs=EPOCHS, batch_size=BATCH,
               callbacks=[_early_stop()], verbose=0)

    params = {"architecture": "GRU(64)->GRU(32)->Dense(1)",
               "seq_len": SEQ_LEN, "dropout": DROPOUT}
    return dates, y_te, model.predict(X_te, verbose=0).flatten(), params


# ── Bidirectional LSTM ────────────────────────────────────────────────────────

def run_bilstm_fold(df, train_idx, test_idx):
    X_tr, X_te, y_tr, y_te, dates = prepare_dl_fold(
        df, train_idx, test_idx, seq_len=SEQ_LEN)

    X_tr, X_val, y_tr, y_val = _val_split(X_tr, y_tr)

    model = Sequential([
        Bidirectional(LSTM(UNITS, return_sequences=True),
                      input_shape=(SEQ_LEN, 1)),
        Dropout(DROPOUT),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
               epochs=EPOCHS, batch_size=BATCH,
               callbacks=[_early_stop()], verbose=0)

    params = {"architecture": "BiLSTM(64)->BiLSTM(32)->Dense(1)",
               "seq_len": SEQ_LEN, "dropout": DROPOUT}
    return dates, y_te, model.predict(X_te, verbose=0).flatten(), params


# ── Transformer ───────────────────────────────────────────────────────────────

def run_transformer_fold(df, train_idx, test_idx):
    X_tr, X_te, y_tr, y_te, dates = prepare_dl_fold(
        df, train_idx, test_idx, seq_len=SEQ_LEN)

    X_tr, X_val, y_tr, y_val = _val_split(X_tr, y_tr)

    inp = Input(shape=(SEQ_LEN, 1))
    x   = MultiHeadAttention(num_heads=2, key_dim=1)(inp, inp)
    x   = Dropout(DROPOUT)(x)
    x   = LayerNormalization(epsilon=1e-6)(inp + x)
    ff  = Dense(64, activation="relu")(x)
    ff  = Dense(1)(ff)
    x   = LayerNormalization(epsilon=1e-6)(x + ff)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(32, activation="relu")(x)
    x   = Dropout(DROPOUT)(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
               epochs=EPOCHS, batch_size=BATCH,
               callbacks=[_early_stop()], verbose=0)

    params = {"architecture": "Transformer(heads=2,ff=64)->Dense(1)",
               "seq_len": SEQ_LEN, "dropout": DROPOUT}
    return dates, y_te, model.predict(X_te, verbose=0).flatten(), params
