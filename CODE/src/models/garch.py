"""
garch.py
--------
GARCH(1,1) volatility model.

IMPORTANT: GARCH forecasts conditional VARIANCE (volatility),
not price direction. It is evaluated against realized volatility
(|log_return|), NOT against log_return directly.

This is why GARCH appears in a separate section in the thesis
and is not ranked alongside directional forecasting models
on RMSE/MAE metrics.
"""

import numpy as np
import pandas as pd
import warnings
from arch import arch_model

warnings.filterwarnings("ignore")


def run_garch_fold(df: pd.DataFrame, train_idx, test_idx):
    """
    Fits GARCH(1,1) on train log_returns.
    Produces one-step-ahead conditional volatility forecasts.

    Returns:
      test_dates      : DatetimeIndex
      realized_vol    : |log_return| — proxy for realized volatility
      forecast_vol    : sqrt(conditional variance) from GARCH
      params          : dict with alpha, beta, persistence
    """
    series     = df["log_return"].values * 100   # scale for numerical stability
    train_s    = pd.Series(series[train_idx])
    test_dates = df.index[test_idx]
    n_test     = len(test_idx)

    model_fit = arch_model(train_s, vol="Garch", p=1, q=1,
                            mean="Constant", dist="normal").fit(disp="off")

    # Rolling one-step-ahead forecast
    history      = list(train_s)
    forecast_vol = []
    for t in range(n_test):
        mod = arch_model(pd.Series(history), vol="Garch", p=1, q=1,
                          mean="Constant", dist="normal")
        res = mod.fit(disp="off", show_warning=False)
        fc  = res.forecast(horizon=1)
        forecast_vol.append(np.sqrt(fc.variance.values[-1, 0]) / 100)
        history.append(series[test_idx[t]])

    realized_vol = np.abs(df["log_return"].values[test_idx])

    params = {
        "alpha":       round(float(model_fit.params.get("alpha[1]", 0)), 6),
        "beta":        round(float(model_fit.params.get("beta[1]",  0)), 6),
        "persistence": round(
            float(model_fit.params.get("alpha[1]", 0)) +
            float(model_fit.params.get("beta[1]",  0)), 6
        )
    }
    return test_dates, realized_vol, np.array(forecast_vol), params
