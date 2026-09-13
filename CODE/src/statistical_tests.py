"""
statistical_tests.py
--------------------
Statistical validation tools for forecasting comparisons.

0. ADF Stationarity Test (Augmented Dickey-Fuller)
1. Random Walk (Naive) Baseline
2. Diebold-Mariano Test (Diebold & Mariano, 1995)
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller


def _align(y_true, y_pred):
    """Trim both arrays to the same length. Always safe."""
    n = min(len(y_true), len(y_pred))
    return np.array(y_true)[-n:], np.array(y_pred)[-n:]


# ── 0. ADF Stationarity Test ──────────────────────────────────────────────────

def adf_test(series: np.ndarray, regression: str = "c") -> dict:
    """
    Augmented Dickey-Fuller test for a unit root.
    H0: series has a unit root (non-stationary).
    Rejecting H0 (p < 0.05) => series is stationary.
    Reporting-only: does not feed back into ARIMA's fixed d=0.
    """
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    stat, pvalue, usedlag, nobs, crit_values, _ = adfuller(
        series, regression=regression, autolag="AIC")
    return {
        "ADF Stat":        round(float(stat), 4),
        "p-value":         round(float(pvalue), 4),
        "Lags Used":       int(usedlag),
        "N Obs":           int(nobs),
        "Crit. 5%":        round(float(crit_values["5%"]), 4),
        "Stationary (5%)": "Yes" if pvalue < 0.05 else "No",
    }


def build_adf_table(price_series: dict, return_series: dict) -> pd.DataFrame:
    """
    ADF test table per Capitolo 3: livello prezzo vs log-return, per asset.
    price_series / return_series: dict {asset_name: array-like 1D}
    """
    rows = []
    for asset in price_series:
        rows.append({"Asset": asset, "Series": "Price",
                     **adf_test(price_series[asset])})
        rows.append({"Asset": asset, "Series": "Log-Return",
                     **adf_test(return_series[asset])})
    return pd.DataFrame(rows).set_index(["Asset", "Series"])


# ── 1. Random Walk Baseline ───────────────────────────────────────────────────

def random_walk_forecast(y_true: np.ndarray) -> np.ndarray:
    """
    Random Walk (Naive) forecast: predicts zero log-return at every step.
    Under the EMH, the best short-horizon forecast of log-returns is zero.
    This is the canonical baseline in financial forecasting.
    """
    return np.zeros_like(y_true)


# ── 2. Diebold-Mariano Test ───────────────────────────────────────────────────

def _newey_west_lag(n: int) -> int:
    """
    Automatic HAC bandwidth (Newey & West, 1994 plug-in rule):
        L = floor(4 * (T/100)^(2/9))
    Data-driven lag for the long-run variance estimator, independent of the
    forecast horizon h — it targets residual serial correlation in the loss
    differential d_t (e.g. from volatility clustering), not MA(h-1) structure.
    """
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def diebold_mariano_test(y_true: np.ndarray,
                          y_pred1: np.ndarray,
                          y_pred2: np.ndarray,
                          h: int = 1,
                          loss: str = "mse",
                          hac_lag: int = None) -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy, with a
    Newey-West HAC long-run variance estimator (Bartlett kernel).

    h        : forecast horizon (documentation only here — this pipeline's
               forecasts are 1-step-ahead, so h=1 is correct and not used to
               drive the variance calculation).
    hac_lag  : number of lags L for the Newey-West correction. If None
               (default), L is chosen automatically via _newey_west_lag(n)
               from the sample size of this specific comparison — exposed
               in the returned dict as "hac_lag" so it is never buried
               inside the function and can be reported/audited externally.
    H0: equal predictive accuracy between model1 and model2.
    """
    y_true  = np.array(y_true).flatten()
    y_pred1 = np.array(y_pred1).flatten()
    y_pred2 = np.array(y_pred2).flatten()

    # Align all three to shortest length
    n = min(len(y_true), len(y_pred1), len(y_pred2))
    y_true  = y_true[-n:]
    y_pred1 = y_pred1[-n:]
    y_pred2 = y_pred2[-n:]

    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    if loss == "mse":
        d = e1**2 - e2**2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    n      = len(d)
    d_mean = np.mean(d)

    L = hac_lag if hac_lag is not None else _newey_west_lag(n)
    gamma0  = np.var(d, ddof=1)
    hac_var = gamma0
    for j in range(1, L + 1):
        gamma_j  = np.cov(d[j:], d[:-j])[0, 1]
        w_j      = 1 - j / (L + 1)   # Bartlett kernel taper
        hac_var += 2 * w_j * gamma_j
    hac_var = max(hac_var, 1e-12)

    dm_stat = d_mean / np.sqrt(hac_var / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    if p_value < 0.05:
        better = "model1" if dm_stat < 0 else "model2"
    else:
        better = "equal"

    return {
        "dm_statistic": round(float(dm_stat), 4),
        "p_value":      round(float(p_value), 4),
        "hac_lag":      int(L),
        "significant":  p_value < 0.05,
        "better":       better,
        "interpretation": (
            "Model 1 significantly better" if better == "model1" else
            "Model 2 significantly better" if better == "model2" else
            "No significant difference (p={:.3f})".format(p_value)
        )
    }


def dm_vs_random_walk(model_name: str,
                       y_true: np.ndarray,
                       y_pred_model: np.ndarray,
                       hac_lag: int = None) -> dict:
    """Tests model against the random walk baseline."""
    y_true, y_pred_model = _align(y_true, y_pred_model)
    y_rw = random_walk_forecast(y_true)
    result = diebold_mariano_test(y_true, y_pred_model, y_rw,
                                   loss="mse", hac_lag=hac_lag)
    result["model"]     = model_name
    result["benchmark"] = "Random Walk"
    return result


# ── 3. Summary table ──────────────────────────────────────────────────────────

def build_dm_table(y_true: np.ndarray,
                    predictions: dict,
                    hac_lag: int = None) -> pd.DataFrame:
    """
    DM test summary table for all models vs random walk.
    Automatically aligns lengths — safe against ARIMA off-by-one.

    hac_lag: forwarded to dm_vs_random_walk / diebold_mariano_test. None
             (default) means each row's L is auto-computed from that row's
             own sample size. The L actually used is always reported in the
             "HAC Lag (L)" column — never hidden — so it can be verified and
             cited (e.g. "L calcolato automaticamente via regola di
             Newey-West, L=4 per T=210") directly from the saved CSV.
    """
    rows = []
    for model_name, y_pred in predictions.items():
        result = dm_vs_random_walk(model_name, y_true, y_pred, hac_lag=hac_lag)
        rows.append({
            "Model":       model_name,
            "DM Stat":     result["dm_statistic"],
            "p-value":     result["p_value"],
            "HAC Lag (L)": result["hac_lag"],
            "Sig. (5%)":   "Yes" if result["significant"] else "No",
            "Result":      result["interpretation"]
        })
    return pd.DataFrame(rows).set_index("Model")