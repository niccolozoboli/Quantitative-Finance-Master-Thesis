"""
dl_models.py
------------
Modelli deep learning: LSTM, GRU, BiLSTM, Transformer, TCN, CNN-LSTM.
VERSIONE 2.0 — Input MULTIVARIATO (seq_len, N_features).

Cambiamenti rispetto alla v1:
- Input shape: (seq_len, 1) → (seq_len, N_features)
- n_features viene letto dinamicamente da prepare_dl_fold
- Aggiunto TCN (Temporal Convolutional Network)
- Aggiunto CNN-LSTM ibrido
- SEQ_LEN aumentato da 10 a 20
- Architettura Transformer migliorata (più heads, FFN più grande)

Tutti i modelli restituiscono:
    (test_dates, y_true, y_pred, params)
in scala log_return originale — compatibile con il resto del pipeline.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from keras.models import Sequential, Model
from keras.layers import (
    LSTM, GRU, Bidirectional, Dense, Dropout,
    Input, GlobalAveragePooling1D, LayerNormalization,
    MultiHeadAttention, Conv1D, MaxPooling1D, Flatten,
    BatchNormalization, Add
)
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from src.feature_engine import prepare_dl_fold

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN  = 20    # aumentato da 10 — più contesto per attention e conv layers
DROPOUT  = 0.2
EPOCHS   = 100
PATIENCE = 10
BATCH    = 32
LR       = 0.001


def _early_stop():
    return EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0
    )


def _val_split(X, y, ratio=0.1):
    """Split finale del training set per validation (early stopping)."""
    n = int(len(X) * (1 - ratio))
    return X[:n], X[n:], y[:n], y[n:]


def _get_fold_data(df, train_idx, test_idx, cross_asset_dfs=None):
    """
    Helper condiviso: chiama prepare_dl_fold e fa il val split.
    Restituisce tutto quello che serve per il training.
    """
    X_tr, X_te, y_tr, y_te, dates, n_feat = prepare_dl_fold(
        df, train_idx, test_idx,
        cross_asset_dfs=cross_asset_dfs,
        seq_len=SEQ_LEN
    )
    X_tr, X_val, y_tr, y_val = _val_split(X_tr, y_tr)
    return X_tr, X_val, X_te, y_tr, y_val, y_te, dates, n_feat


# ─────────────────────────────────────────────────────────────────────────────
# MODELLO 1 — LSTM
# ─────────────────────────────────────────────────────────────────────────────

def run_lstm_fold(df, train_idx, test_idx, cross_asset_dfs=None):
    """
    LSTM a due layer con input multivariato.
    Layer 1: LSTM(64, return_sequences=True)
    Layer 2: LSTM(32, return_sequences=False)
    """
    X_tr, X_val, X_te, y_tr, y_val, y_te, dates, n_feat = \
        _get_fold_data(df, train_idx, test_idx, cross_asset_dfs)

    model = Sequential([
        LSTM(64, input_shape=(SEQ_LEN, n_feat), return_sequences=True),
        Dropout(DROPOUT),
        LSTM(32, return_sequences=False),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer=Adam(LR), loss="mse")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[_early_stop()], verbose=0
    )

    params = {
        "architecture": f"LSTM(64)->LSTM(32)->Dense(1)",
        "seq_len": SEQ_LEN, "n_features": n_feat, "dropout": DROPOUT
    }
    return dates, y_te, model.predict(X_te, verbose=0).flatten(), params


# ─────────────────────────────────────────────────────────────────────────────
# MODELLO 2 — GRU
# ─────────────────────────────────────────────────────────────────────────────

def run_gru_fold(df, train_idx, test_idx, cross_asset_dfs=None):
    """
    GRU a due layer. Più veloce di LSTM, parametri ridotti.
    Preferibile quando il dataset è più piccolo o il tempo di training
    è un vincolo (es. molti fold).
    """
    X_tr, X_val, X_te, y_tr, y_val, y_te, dates, n_feat = \
        _get_fold_data(df, train_idx, test_idx, cross_asset_dfs)

    model = Sequential([
        GRU(64, input_shape=(SEQ_LEN, n_feat), return_sequences=True),
        Dropout(DROPOUT),
        GRU(32, return_sequences=False),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer=Adam(LR), loss="mse")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[_early_stop()], verbose=0
    )

    params = {
        "architecture": f"GRU(64)->GRU(32)->Dense(1)",
        "seq_len": SEQ_LEN, "n_features": n_feat, "dropout": DROPOUT
    }
    return dates, y_te, model.predict(X_te, verbose=0).flatten(), params


# ─────────────────────────────────────────────────────────────────────────────
# MODELLO 3 — BiLSTM
# ─────────────────────────────────────────────────────────────────────────────

def run_bilstm_fold(df, train_idx, test_idx, cross_asset_dfs=None):
    """
    Bidirectional LSTM. Processa la sequenza in entrambe le direzioni.
    All'interno di una finestra fissa (SEQ_LEN giorni già osservati),
    non accede a informazione futura — tutti i dati sono disponibili
    al momento della previsione.
    """
    X_tr, X_val, X_te, y_tr, y_val, y_te, dates, n_feat = \
        _get_fold_data(df, train_idx, test_idx, cross_asset_dfs)

    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True),
            input_shape=(SEQ_LEN, n_feat)
        ),
        Dropout(DROPOUT),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(DROPOUT),
        Dense(1)
    ])
    model.compile(optimizer=Adam(LR), loss="mse")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[_early_stop()], verbose=0
    )

    params = {
        "architecture": f"BiLSTM(64)->BiLSTM(32)->Dense(1)",
        "seq_len": SEQ_LEN, "n_features": n_feat, "dropout": DROPOUT
    }
    return dates, y_te, model.predict(X_te, verbose=0).flatten(), params


# ─────────────────────────────────────────────────────────────────────────────
# MODELLO 4 — TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────

def run_transformer_fold(df, train_idx, test_idx, cross_asset_dfs=None):
    """
    Transformer con multi-head self-attention.
    Versione migliorata rispetto alla v1:
    - 4 heads invece di 2 (più capacità espressiva)
    - key_dim=16 (adeguato al nuovo n_features)
    - FFN più grande (128 invece di 64)
    - Due blocchi di attention in sequenza

    L'attention mechanism permette al modello di pesare
    selettivamente i diversi step temporali nella finestra,
    catturando dipendenze non locali che LSTM fatica a gestire.
    """
    X_tr, X_val, X_te, y_tr, y_val, y_te, dates, n_feat = \
        _get_fold_data(df, train_idx, test_idx, cross_asset_dfs)

    inp = Input(shape=(SEQ_LEN, n_feat))

    # Blocco 1: Multi-head attention + residual
    attn1 = MultiHeadAttention(num_heads=4, key_dim=16)(inp, inp)
    attn1 = Dropout(DROPOUT)(attn1)
    x1    = LayerNormalization(epsilon=1e-6)(inp + attn1)

    # Feed-forward 1
    ff1 = Dense(128, activation="relu")(x1)
    ff1 = Dense(n_feat)(ff1)
    x1  = LayerNormalization(epsilon=1e-6)(x1 + ff1)

    # Blocco 2: secondo strato di attention
    attn2 = MultiHeadAttention(num_heads=4, key_dim=16)(x1, x1)
    attn2 = Dropout(DROPOUT)(attn2)
    x2    = LayerNormalization(epsilon=1e-6)(x1 + attn2)

    # Feed-forward 2
    ff2 = Dense(128, activation="relu")(x2)
    ff2 = Dense(n_feat)(ff2)
    x2  = LayerNormalization(epsilon=1e-6)(x2 + ff2)

    # Pooling e output
    x2  = GlobalAveragePooling1D()(x2)
    x2  = Dense(64, activation="relu")(x2)
    x2  = Dropout(DROPOUT)(x2)
    out = Dense(1)(x2)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(LR), loss="mse")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[_early_stop()], verbose=0
    )

    params = {
        "architecture": "Transformer(heads=4,ff=128,blocks=2)->Dense(1)",
        "seq_len": SEQ_LEN, "n_features": n_feat, "dropout": DROPOUT
    }
    return dates, y_te, model.predict(X_te, verbose=0).flatten(), params


# ─────────────────────────────────────────────────────────────────────────────
# MODELLO 5 — TCN (Temporal Convolutional Network)  ← NUOVO
# ─────────────────────────────────────────────────────────────────────────────

def run_tcn_fold(df, train_idx, test_idx, cross_asset_dfs=None):
    """
    Temporal Convolutional Network con dilated causal convolutions.

    RAZIONALE:
    Le TCN sono state proposte come alternativa agli RNN per sequenze
    temporali (Bai et al., 2018: "An Empirical Evaluation of Generic
    Convolutional and Recurrent Networks for Sequence Modeling").
    Vantaggi rispetto a LSTM/GRU:
    - Parallelizzabile (training più veloce)
    - Gradiente stabile (no vanishing gradient)
    - Campo ricettivo controllabile con dilation

    ARCHITETTURA:
    Conv1D con dilation_rate crescente (1, 2, 4) — ogni layer
    vede un campo temporale doppio del precedente.
    'causal' padding garantisce che il passo t non veda t+1, t+2...
    (zero look-ahead anche all'interno del layer convoluzionale).

    Campo ricettivo totale con kernel=3, dilations=[1,2,4]:
    (3-1)*1 + (3-1)*2 + (3-1)*4 + 1 = 15 timestep
    Con SEQ_LEN=20, copriamo quasi l'intera finestra.
    """
    X_tr, X_val, X_te, y_tr, y_val, y_te, dates, n_feat = \
        _get_fold_data(df, train_idx, test_idx, cross_asset_dfs)

    inp = Input(shape=(SEQ_LEN, n_feat))

    # Layer 1: dilation=1 (vede 3 step contigui)
    x = Conv1D(64, kernel_size=3, dilation_rate=1,
               padding="causal", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT)(x)

    # Layer 2: dilation=2 (vede 5 step effettivi)
    x = Conv1D(64, kernel_size=3, dilation_rate=2,
               padding="causal", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT)(x)

    # Layer 3: dilation=4 (vede 9 step effettivi)
    x = Conv1D(32, kernel_size=3, dilation_rate=4,
               padding="causal", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT)(x)

    # Aggregazione e output
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(32, activation="relu")(x)
    x   = Dropout(DROPOUT)(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(LR), loss="mse")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[_early_stop()], verbose=0
    )

    params = {
        "architecture": "TCN(dilations=[1,2,4],kernel=3)->Dense(1)",
        "seq_len": SEQ_LEN, "n_features": n_feat,
        "receptive_field": 15, "dropout": DROPOUT
    }
    return dates, y_te, model.predict(X_te, verbose=0).flatten(), params


# ─────────────────────────────────────────────────────────────────────────────
# MODELLO 6 — CNN-LSTM ibrido  ← NUOVO
# ─────────────────────────────────────────────────────────────────────────────

def run_cnn_lstm_fold(df, train_idx, test_idx, cross_asset_dfs=None):
    """
    CNN-LSTM ibrido: convoluzione locale + memoria temporale.

    RAZIONALE:
    Molto citato nella letteratura su commodity price forecasting
    (es. Livieris et al. 2020 — già nella tua bibliografia).
    La CNN cattura pattern locali nella sequenza (es. inversioni a V,
    picchi di volatilità), mentre l'LSTM cattura le dipendenze
    temporali a più lungo raggio.

    ARCHITETTURA:
    1. Conv1D(32, kernel=3) — estrae pattern locali su 3 giorni
    2. MaxPooling1D(2) — riduce la dimensione temporale (20→10)
    3. LSTM(64) — impara dipendenze su sequenza compressa
    4. Dense(1) — output

    Questo design è deliberatamente asimmetrico:
    la CNN comprime la rappresentazione locale,
    l'LSTM lavora su una sequenza più corta ma più ricca.
    """
    X_tr, X_val, X_te, y_tr, y_val, y_te, dates, n_feat = \
        _get_fold_data(df, train_idx, test_idx, cross_asset_dfs)

    model = Sequential([
        # Strato convoluzionale — pattern locali
        Conv1D(
            filters=32,
            kernel_size=3,
            padding="causal",
            activation="relu",
            input_shape=(SEQ_LEN, n_feat)
        ),
        BatchNormalization(),
        Dropout(DROPOUT),

        # Pooling — compressione temporale
        MaxPooling1D(pool_size=2),

        # LSTM — dipendenze temporali sulla sequenza compressa
        LSTM(64, return_sequences=False),
        Dropout(DROPOUT),

        # Output
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer=Adam(LR), loss="mse")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[_early_stop()], verbose=0
    )

    params = {
        "architecture": "Conv1D(32,k=3)->MaxPool->LSTM(64)->Dense(1)",
        "seq_len": SEQ_LEN, "n_features": n_feat, "dropout": DROPOUT
    }
    return dates, y_te, model.predict(X_te, verbose=0).flatten(), params