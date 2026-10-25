"""
statistical_tests.py
--------------------
Statistical validation tools for forecasting comparisons.

0. ADF unit-root diagnostic (Augmented Dickey-Fuller)
1. Random Walk (Naive) Baseline
2. Diebold-Mariano Test (Diebold & Mariano, 1995), on any pair of losses
3. Directional-accuracy tests under H0: E[h_t] = 1/2 (commento K04)
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def _align(y_true, y_pred):
    """Trim both arrays to the same length. Always safe."""
    n = min(len(y_true), len(y_pred))
    return np.array(y_true)[-n:], np.array(y_pred)[-n:]


# ── 0. ADF unit-root diagnostic ───────────────────────────────────────────────

def adf_test(series: np.ndarray, regression: str = "c") -> dict:
    """
    Augmented Dickey-Fuller test, H0: unit root. Costante nella regressione
    di test, ritardi scelti con AIC.

    Diagnostica descrittiva sul log-rendimento: il rifiuto di H0 è evidenza
    contro quella particolare forma di non stazionarietà, non una prova che
    il processo sia stazionario. Non viene usato per scegliere d: d = 0 è
    fissato ex ante dalla definizione del target (blu §1.3, §3.2).
    """
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    stat, pvalue, usedlag, nobs, crit_values, _ = adfuller(
        series, regression=regression, autolag="AIC")
    return {
        "ADF Stat":              round(float(stat), 4),
        "p-value":               float(pvalue),
        "Lags Used":             int(usedlag),
        "N Obs":                 int(nobs),
        "Crit. 5%":              round(float(crit_values["5%"]), 4),
        "Reject unit root (5%)": "Yes" if pvalue < 0.05 else "No",
    }


def build_adf_table(return_series: dict) -> pd.DataFrame:
    """
    Tabella ADF (Tab. 3.3) sulla sola variabile che entra nella stima, il
    log-rendimento (report del relatore, 7 settembre 2026, §3).
    return_series: dict {asset_name: array-like 1D}
    """
    rows = [{"Asset": asset, "Series": "Log-Return", **adf_test(r)}
            for asset, r in return_series.items()]
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
    Troncamento HAC con la regola pratica
        B = floor(4 * (T/100)^(2/9))     (B = 4 per T = 210)
    associata alla letteratura sulla banda di Newey e West (1994). È una
    scelta di implementazione fissa, non la procedura di selezione ottimale
    dipendente dai dati sviluppata in quell'articolo (blu §1.3, eq. 1.10).
    """
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_lrv(x: np.ndarray, L: int) -> float:
    """
    Varianza di lungo periodo di Newey-West (kernel di Bartlett), eq. 1.10:
        Omega = gamma_0 + 2 sum_{j=1}^{L} (1 - j/(L+1)) gamma_j,
    con gamma_j = (1/T) sum_{t=j+1}^{T} (x_t - xbar)(x_{t-j} - xbar)
    (autocovarianza campionaria standard: media globale, divisore T).
    """
    x = np.asarray(x, dtype=float).flatten()
    T = len(x)
    xc = x - x.mean()
    omega = float(xc @ xc) / T
    for j in range(1, L + 1):
        omega += 2 * (1 - j / (L + 1)) * float(xc[j:] @ xc[:-j]) / T
    return omega


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
    hac_var = max(newey_west_lrv(d, L), 1e-12)

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


def dm_test_loss(loss1: np.ndarray, loss2: np.ndarray,
                 hac_lag: int = None) -> dict:
    """
    Diebold-Mariano su due serie di loss qualsiasi (es. QLIKE di due
    previsioni di volatilità): d_t = loss1_t - loss2_t, H0: E[d_t] = 0,
    stessa varianza HAC di diebold_mariano_test. DM < 0 => loss1 minore.
    """
    d = np.asarray(loss1, dtype=float).flatten() - np.asarray(loss2, dtype=float).flatten()
    d = d[~np.isnan(d)]
    n = len(d)
    L = hac_lag if hac_lag is not None else _newey_west_lag(n)
    dm = float(d.mean() / np.sqrt(max(newey_west_lrv(d, L), 1e-12) / n))
    return {"DM Stat": dm, "p-value": float(2 * (1 - stats.norm.cdf(abs(dm)))),
            "HAC Lag (L)": int(L), "N": n}


# ── 3. Directional accuracy under H0: E[h_t] = 1/2 ─────────────────────────────
#
# Commento K04 del relatore: il bootstrap non centrato di DA^* è centrato
# sulla DA campionaria, quindi la sua area di coda a 1/2 non è un p-value
# valido. Il nullo va imposto: test HAC sulla media di h_t - 1/2 (test
# principale, coerente con DM e alpha) oppure bootstrap a blocchi della
# sequenza centrata h_t - h̄ (controllo). h_t è l'indicatore di hit sui soli
# giorni con posizione non nulla, nell'ordine temporale; per la RCS i giorni
# attivi non sono contigui e la dipendenza è trattata sulla sequenza dei
# giorni attivi. Sotto MIN_ACTIVE_DA giorni attivi il test non è calcolato.

MIN_ACTIVE_DA = 30


def _active_hits(hit, active) -> np.ndarray:
    hit = np.asarray(hit, dtype=float).flatten()
    active = np.asarray(active, dtype=float).flatten()
    return hit[active == 1]


def da_test_hac(hit: np.ndarray, active: np.ndarray,
                min_active: int = MIN_ACTIVE_DA) -> dict:
    """t_DA = (h̄ - 1/2) / sqrt(Omega_h / n_a), Omega_h Newey-West con B da n_a."""
    h = _active_hits(hit, active)
    n = len(h)
    out = {"N Active": n, "DA t (HAC)": float("nan"),
           "DA p-value (HAC)": float("nan"), "DA HAC Lag": None}
    if n < min_active:
        return out
    L = _newey_west_lag(n)
    omega = newey_west_lrv(h, L)
    if omega <= 1e-12:
        return out
    t = float((h.mean() - 0.5) / np.sqrt(omega / n))
    out.update({"DA t (HAC)": t,
                "DA p-value (HAC)": float(2 * (1 - stats.norm.cdf(abs(t)))),
                "DA HAC Lag": int(L)})
    return out


def da_test_centred_bootstrap(hit: np.ndarray, active: np.ndarray,
                              block_length: int = 10, n_boot: int = 2000,
                              seed: int = 42,
                              min_active: int = MIN_ACTIVE_DA) -> dict:
    """
    Moving block bootstrap della sequenza centrata hc_t = h_t - h̄ (impone H0
    sulla distribuzione simulata): p = Pr*(|mean(hc*)| >= |h̄ - 1/2|).
    """
    h = _active_hits(hit, active)
    n = len(h)
    if n < min_active or n < block_length:
        return {"DA p-value (Boot H0)": float("nan")}
    hc = h - h.mean()
    stat_obs = abs(h.mean() - 0.5)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length
    stats_b = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
        stats_b[b] = abs(hc[idx].mean())
    return {"DA p-value (Boot H0)": float(np.mean(stats_b >= stat_obs))}
