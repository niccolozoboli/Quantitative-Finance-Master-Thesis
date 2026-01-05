from src.data_loader import download_commodity
from src.preprocessing import apply_log_return, scale_features
from src.feature_engine import create_lagged_features, add_rolling_features
from src.split import train_val_test_split
from src.evaluate import compute_metrics
from src.plot import plot_predictions, plot_feature_importance

# import models
from src.models.arima import train_arima_model
from src.models.svr import train_svr
from src.models.random_forest import train_random_forest
from src.models.decision_tree import train_decision_tree
from src.models.gradient_boosting import train_gradient_boosting
from src.models.xgboost_model import train_xgboost
from src.models.lstm import train_lstm
from src.models.gru import train_gru
#from src.models.transformer import train_transformer  # opzionale

import pandas as pd

def run_experiments(ticker="GC=F"):

    df = download_commodity(ticker)
    df = apply_log_return(df)
    df = add_rolling_features(df)
    df = scale_features(df)

    df_features = create_lagged_features(df)
    X = df_features.drop(columns=["log_return"])
    y = df_features["log_return"]

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # dictionary to store all results
    results = {}

    # ARIMA
    arima_model = train_arima_model(df["log_return"])
    arima_pred = arima_model.predict(start=len(df)-len(y_test), end=len(df)-1)
    results["ARIMA"] = compute_metrics(y_test, arima_pred)
    plot_predictions(y_test, arima_pred, "ARIMA Prediction")

    # SVR
    svr_model = train_svr(X_train, y_train)
    svr_pred = svr_model.predict(X_test)
    results["SVR"] = compute_metrics(y_test, svr_pred)
    plot_predictions(y_test, svr_pred, "SVR Prediction")

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results["RandomForest"] = compute_metrics(y_test, rf_pred)
    plot_feature_importance(rf_model, X.columns)

    # Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    results["DecisionTree"] = compute_metrics(y_test, dt_pred)

    # Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    results["GradientBoosting"] = compute_metrics(y_test, gb_pred)

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results["XGBoost"] = compute_metrics(y_test, xgb_pred)

    # LSTM
    lstm_model = train_lstm(X_train, y_train, X_val, y_val)
    lstm_pred = lstm_model.predict(X_test)
    results["LSTM"] = compute_metrics(y_test, lstm_pred)
    plot_predictions(y_test, lstm_pred, "LSTM Predictions")

    # GRU
    gru_model = train_gru(X_train, y_train, X_val, y_val)
    gru_pred = gru_model.predict(X_test)
    results["GRU"] = compute_metrics(y_test, gru_pred)

    print("=== RESULTS ===")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")

if __name__ == "__main__":
    run_experiments("GC=F")