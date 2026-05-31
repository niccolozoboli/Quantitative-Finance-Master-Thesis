"""
ml_models.py
------------
Modelli tree-based e kernel ML.
VERSIONE 2.0 — accetta cross_asset_dfs.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb

from src.feature_engine import prepare_ml_fold


def _grid_search(estimator, param_grid, X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=3)
    gs   = GridSearchCV(
        estimator, param_grid, cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def run_rf_fold(df, train_idx, test_idx, cross_asset_dfs=None):
    X_tr, X_te, y_tr, y_te, dates, feat_cols = prepare_ml_fold(
        df, train_idx, test_idx, cross_asset_dfs=cross_asset_dfs)
    param_grid = {
        "n_estimators":      [100, 200],
        "max_depth":         [3, 5, 7, None],
        "min_samples_split": [5, 10]
    }
    model, params = _grid_search(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid, X_tr, y_tr)
    params["n_features"] = len(feat_cols)
    return dates, y_te, model.predict(X_te), params


def run_xgboost_fold(df, train_idx, test_idx, cross_asset_dfs=None):
    X_tr, X_te, y_tr, y_te, dates, feat_cols = prepare_ml_fold(
        df, train_idx, test_idx, cross_asset_dfs=cross_asset_dfs)
    param_grid = {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [3, 5, 7],
        "subsample":     [0.8, 1.0]
    }
    model, params = _grid_search(
        xgb.XGBRegressor(objective="reg:squarederror",
                         random_state=42, verbosity=0),
        param_grid, X_tr, y_tr)
    params["n_features"] = len(feat_cols)
    return dates, y_te, model.predict(X_te), params


def run_gbm_fold(df, train_idx, test_idx, cross_asset_dfs=None):
    X_tr, X_te, y_tr, y_te, dates, feat_cols = prepare_ml_fold(
        df, train_idx, test_idx, cross_asset_dfs=cross_asset_dfs)
    param_grid = {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [3, 5]
    }
    model, params = _grid_search(
        GradientBoostingRegressor(random_state=42),
        param_grid, X_tr, y_tr)
    params["n_features"] = len(feat_cols)
    return dates, y_te, model.predict(X_te), params


def run_svr_fold(df, train_idx, test_idx, cross_asset_dfs=None):
    X_tr, X_te, y_tr, y_te, dates, feat_cols = prepare_ml_fold(
        df, train_idx, test_idx, cross_asset_dfs=cross_asset_dfs)
    param_grid = {
        "C":       [0.1, 1, 10],
        "epsilon": [0.001, 0.01, 0.1],
        "gamma":   ["scale", "auto"]
    }
    model, params = _grid_search(
        SVR(kernel="rbf"),
        param_grid, X_tr, y_tr)
    params["n_features"] = len(feat_cols)
    return dates, y_te, model.predict(X_te), params


def run_dt_fold(df, train_idx, test_idx, cross_asset_dfs=None):
    X_tr, X_te, y_tr, y_te, dates, feat_cols = prepare_ml_fold(
        df, train_idx, test_idx, cross_asset_dfs=cross_asset_dfs)
    param_grid = {
        "max_depth":         [3, 5, 7, None],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf":  [3, 5, 10]
    }
    model, params = _grid_search(
        DecisionTreeRegressor(random_state=42),
        param_grid, X_tr, y_tr)
    params["n_features"] = len(feat_cols)
    return dates, y_te, model.predict(X_te), params