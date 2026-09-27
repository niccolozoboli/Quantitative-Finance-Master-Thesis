"""
tests/test_report_relatore_b4.py
----------------------------------
B4 (report del relatore, 7/9/2026): il refit riallena sull'INTERO T_k per
e* epoche (l'epoca di minimo val_loss stimata in fase 1, sul 90% con
early stopping sul restante 10%), non su un sottoinsieme.

In un file separato da test_report_relatore.py perché richiede
TensorFlow/training (più lento, ma sempre CPU/dati sintetici, < 1 minuto).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.models.dl_models import _train_with_refit, BATCH


def _build_tiny_model(d: int):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam

    def build():
        m = Sequential([Dense(4, activation="relu", input_shape=(d,)), Dense(1)])
        m.compile(optimizer=Adam(0.01), loss="mse")
        return m
    return build


def test_b4_refit_uses_e_star_epochs_on_full_train_set():
    rng = np.random.default_rng(0)
    n, d = 137, 3   # n non multiplo di BATCH, deliberatamente
    X = rng.normal(size=(n, d)).astype("float32")
    y = rng.normal(size=(n,)).astype("float32")

    build = _build_tiny_model(d)
    model, e_star = _train_with_refit(build, X, y, seed=42)

    assert e_star >= 1

    # optimizer.iterations conta gli step di gradiente totali: se il refit
    # ha usato TUTTO X (non solo il 90% di fase 1) per e_star epoche,
    # deve valere iterations == e_star * ceil(n / BATCH) — questo fallirebbe
    # se il refit avesse usato solo il sottoinsieme di training di fase 1
    # (90% di n, un numero diverso di batch per epoca).
    steps_per_epoch_full = int(np.ceil(n / BATCH))
    expected_iterations = e_star * steps_per_epoch_full
    actual_iterations = int(model.optimizer.iterations.numpy())

    assert actual_iterations == expected_iterations, (
        f"attese {expected_iterations} iterazioni (refit su tutto T_k, "
        f"n={n}) ma il modello ne ha registrate {actual_iterations} — "
        f"il refit non sta usando l'intero training set"
    )


def test_b4_e_star_matches_argmin_val_loss():
    """e* deve coincidere con l'epoca di minimo val_loss della fase 1
    (non con l'epoca di stop di EarlyStopping, che è successiva per via
    della patience)."""
    rng = np.random.default_rng(1)
    n, d = 150, 3
    X = rng.normal(size=(n, d)).astype("float32")
    y = (X[:, 0] * 0.5 + rng.normal(0, 0.01, n)).astype("float32")

    from keras.utils import set_random_seed
    from src.models.dl_models import _val_split, _early_stop, EPOCHS

    build = _build_tiny_model(d)

    set_random_seed(42)
    model_phase1 = build()
    X_tr, X_val, y_tr, y_val = _val_split(X, y, ratio=0.1)
    history = model_phase1.fit(
        X_tr, y_tr, validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[_early_stop()], verbose=0
    )
    expected_e_star = int(np.argmin(history.history["val_loss"])) + 1

    set_random_seed(42)
    _, e_star = _train_with_refit(build, X, y, seed=42)

    assert e_star == expected_e_star
