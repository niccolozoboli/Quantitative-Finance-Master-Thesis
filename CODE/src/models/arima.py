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
    Fits ARIMA on log_return (train), forecasts log_return (test).
    Returns y_true and y_pred in original log_return scale.
    """
    series     = df["log_return"].values
    train_s    = pd.Series(series[train_idx])
    test_s     = series[test_idx]
    test_dates = df.index[test_idx]
    n_test     = len(test_idx)

    order, aic = select_order(train_s)
    model_fit  = ARIMA(train_s, order=order).fit()
    y_pred     = model_fit.forecast(steps=n_test).values

    params = {"order": order, "aic": aic}
    return test_dates, test_s, y_pred, params
