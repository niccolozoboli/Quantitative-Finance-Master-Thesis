# src/models/xgboost_model.py

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror',
                              n_estimators=100,
                              learning_rate=0.1,
                              max_depth=5,
                              random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_xgboost_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return predictions, rmse