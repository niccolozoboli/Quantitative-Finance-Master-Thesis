"""
feature_engine.py
-----------------
Feature construction for ML and DL models.

Key design choices:
- Features are Z-scored (fit on train only → no leakage)
- Target is log_return in ORIGINAL scale (never Z-scored)
- DL models use raw Z-scored log_return as sequence input
- ML models use lag + rolling features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── ML Feature Engineering ────────────────────────────────────────────────────

def build_ml_features(df: pd.DataFrame,
                       lags: list = [1, 2, 3, 5, 10],
                       rolling_windows: list = [5, 10]) -> pd.DataFrame:
    """
    Builds feature matrix for ML models (RF, XGBoost, GBM, SVR, DT).

    Features:
    - Lagged log-returns: lag_1, lag_2, lag_3, lag_5, lag_10
    - Rolling mean:       roll_mean_5, roll_mean_10
    - Rolling std:        roll_std_5,  roll_std_10

    Target: next-day log_return (original scale — NOT Z-scored)

    All features use shift(1) or more to avoid look-ahead.
    """
    df = df.copy()
    r = df["log_return"]

    for lag in lags:
        df[f"lag_{lag}"] = r.shift(lag)

    for w in rolling_windows:
        df[f"roll_mean_{w}"] = r.shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = r.shift(1).rolling(w).std()

    # Target = next-day log_return (original scale)
    df["target"] = r.shift(-1)
    df.dropna(inplace=True)
    return df


def get_feature_cols(lags: list = [1, 2, 3, 5, 10],
                      rolling_windows: list = [5, 10]) -> list:
    cols  = [f"lag_{l}" for l in lags]
    cols += [f"roll_mean_{w}" for w in rolling_windows]
    cols += [f"roll_std_{w}"  for w in rolling_windows]
    return cols


def prepare_ml_fold(df: pd.DataFrame,
                     train_idx, test_idx,
                     lags: list = [1, 2, 3, 5, 10],
                     rolling_windows: list = [5, 10]):
    """
    Prepares train/test split for a single walk-forward fold.
    - Features are Z-scored (fit on train only)
    - Target is log_return in original scale
    """
    df_feat = build_ml_features(df, lags=lags, rolling_windows=rolling_windows)
    feat_cols = get_feature_cols(lags=lags, rolling_windows=rolling_windows)

    # Align indices
    valid_train = df_feat.index.isin(df.index[train_idx])
    valid_test  = df_feat.index.isin(df.index[test_idx])

    X_train = df_feat.loc[valid_train, feat_cols].values
    y_train = df_feat.loc[valid_train, "target"].values
    X_test  = df_feat.loc[valid_test,  feat_cols].values
    y_test  = df_feat.loc[valid_test,  "target"].values
    dates   = df_feat.loc[valid_test].index

    # Z-score features — fit on train only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, dates


# ── DL Sequence Engineering ───────────────────────────────────────────────────

def prepare_dl_fold(df: pd.DataFrame,
                     train_idx, test_idx,
                     seq_len: int = 10):
    """
    Prepares sliding-window sequences for DL models (LSTM, GRU, BiLSTM, Transformer).

    Input:  sequence of seq_len Z-scored log-returns
    Target: next-day log_return in ORIGINAL scale

    Z-scaler fit on training data only → no leakage.
    """
    train_df = df.iloc[train_idx]
    test_df  = df.iloc[test_idx]

    # Fit Z-scaler on training log-returns only
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[["log_return"]]).flatten()
    test_scaled  = scaler.transform(test_df[["log_return"]]).flatten()

    # Build sequences from full series (train + test context)
    full_scaled = np.concatenate([train_scaled, test_scaled])
    full_returns = df["log_return"].values[
        list(train_idx) + list(test_idx)
    ]

    X_train, y_train = [], []
    X_test,  y_test  = [], []

    n_train = len(train_scaled)

    for i in range(seq_len, len(full_scaled)):
        seq = full_scaled[i - seq_len : i].reshape(seq_len, 1)
        tgt = full_returns[i]                        # original scale target
        if i < n_train:
            X_train.append(seq)
            y_train.append(tgt)
        else:
            X_test.append(seq)
            y_test.append(tgt)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test  = np.array(X_test)
    y_test  = np.array(y_test)

    # Dates for test predictions
    test_start = train_idx[-1] + seq_len + 1
    dates = df.index[test_idx[len(test_idx) - len(y_test):]]

    return X_train, X_test, y_train, y_test, dates
