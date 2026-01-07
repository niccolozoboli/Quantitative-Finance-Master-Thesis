# src/models/gradient_boosting.py

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_gradient_boosting_model(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=200,
                                       learning_rate=0.05,
                                       max_depth=4,
                                       random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_gradient_boosting_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return predictions, rmse