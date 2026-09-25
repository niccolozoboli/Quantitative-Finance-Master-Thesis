import numpy as np
import pandas as pd
import warnings
import itertools
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def select_order(series, p_range=range(0, 4), d=0, q_range=range(0, 4)):
    best_aic, best_order = np.inf, (1, d, 1)
    for p, q in itertools.product(p_range, q_range):
        if p == 0 and q == 0:
            continue
        try:
            res = ARIMA(series, order=(p, d, q)).fit()
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, d, q)
        except Exception:
            continue
    return best_order, round(best_aic, 4)


def run_arima_fold(df: pd.DataFrame, train_idx, test_idx):
    """
    Fits ARIMA on log_return (train). Order selection and parameter
    estimation use only the training data. Test predictions are then
    ONE-STEP-AHEAD, non-dynamic forecasts r_hat_{t+1|t}: the fitted
    parameters are applied (not re-estimated, refit=False) to the series
    extended through the end of the test window, and each test-day
    prediction conditions on the actually realized log-returns up to the
    previous day (dynamic=False) — never on the model's own prior-day
    forecasts. Returns y_true and y_pred in original log_return scale.
    """
    series     = df["log_return"].values
    train_s    = pd.Series(series[train_idx])
    test_s     = series[test_idx]
    test_dates = df.index[test_idx]
    n_train    = len(train_idx)
    n_test     = len(test_idx)

    order, aic = select_order(train_s)
    model_fit  = ARIMA(train_s, order=order).fit()

    full_idx = np.concatenate([train_idx, test_idx])
    full_s   = pd.Series(series[full_idx])
    applied  = model_fit.apply(full_s, refit=False)
    y_pred   = np.asarray(applied.get_prediction(
        start=n_train, end=n_train + n_test - 1, dynamic=False
    ).predicted_mean)

    params = {"order": order, "aic": aic}
    return test_dates, test_s, y_pred, params
