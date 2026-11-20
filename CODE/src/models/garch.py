"""
garch.py
--------
GARCH(1,1) con media costante — benchmark di VOLATILITÀ, non di rendimento
medio: non entra nella classifica RMSE né nei test DM sui rendimenti.

Valutazione (commenti K06 e K01 del relatore):
  - descrittiva: SE tra sigma_hat_{t+1} e |r_{t+1}|, proxy rumoroso e
    distorto di scala (E|r| ≈ sigma*sqrt(2/pi) sotto normalità e media
    trascurabile), e SE rispetto al proxy riscalato |r_{t+1}|*sqrt(pi/2);
  - forecast evaluation: QLIKE = log(sigma_hat^2) + r^2 / sigma_hat^2,
    robusta al rumore del proxy r^2 e definita anche per r = 0;
  - robustezza: stessa specificazione con innovazioni Student-t, unico
    competitore, confrontato con un DM sulla QLIKE.
"""

import numpy as np
import pandas as pd
import warnings
from arch import arch_model

warnings.filterwarnings("ignore")


GARCH_DISTS = ("normal", "t")


def qlike_loss(sigma_hat: np.ndarray, r: np.ndarray) -> np.ndarray:
    """QLIKE sulla varianza: log(sigma_hat^2) + r^2 / sigma_hat^2."""
    v = np.asarray(sigma_hat, dtype=float) ** 2
    return np.log(v) + np.asarray(r, dtype=float) ** 2 / v


def fit_garch_params(df: pd.DataFrame, train_idx, dist: str = "normal") -> dict:
    """
    Stima su T_k del GARCH(1,1) a media costante (rendimenti x 100) e
    restituisce alpha, beta, persistenza e nu. È la stessa stima usata da
    run_garch_fold per i parametri riportati in tesi; la ripresa
    (--resume) e tools/recompute_post_run.py la richiamano per ricostruire
    i parametri senza rifare le previsioni giornaliere (rilievo R5).
    """
    if dist not in GARCH_DISTS:
        raise ValueError(f"dist deve essere uno di {GARCH_DISTS}")
    train_s = pd.Series(df["log_return"].values[train_idx] * 100)
    model_fit = arch_model(train_s, vol="Garch", p=1, q=1,
                           mean="Constant", dist=dist).fit(disp="off")
    return {
        "alpha":       round(float(model_fit.params.get("alpha[1]", 0)), 6),
        "beta":        round(float(model_fit.params.get("beta[1]",  0)), 6),
        "persistence": round(
            float(model_fit.params.get("alpha[1]", 0)) +
            float(model_fit.params.get("beta[1]",  0)), 6
        ),
        "nu":          (round(float(model_fit.params.get("nu")), 4)
                        if "nu" in model_fit.params else None),
    }


def garch_fold_losses(sigma_hat: np.ndarray, r: np.ndarray) -> dict:
    """
    Loss di un fold: RMSE rispetto a |r| (proxy descrittivo, arrotondato a
    8 decimali come compute_metrics), RMSE rispetto al proxy riscalato
    |r|*sqrt(pi/2) e QLIKE media. Unica definizione usata dal ciclo
    principale di main.py, dalla ripresa e dallo script di ricalcolo.
    """
    sigma_hat = np.asarray(sigma_hat, dtype=float)
    r = np.asarray(r, dtype=float)
    abs_r = np.abs(r)
    return {
        "loss_RMSE": round(float(np.sqrt(np.mean((abs_r - sigma_hat) ** 2))), 8),
        "loss_RMSE_scaled_proxy": float(np.sqrt(np.mean(
            (sigma_hat - abs_r * np.sqrt(np.pi / 2)) ** 2))),
        "QLIKE": float(np.mean(qlike_loss(sigma_hat, r))),
    }


def run_garch_fold(df: pd.DataFrame, train_idx, test_idx, dist: str = "normal"):
    """
    GARCH(1,1) a media costante, quasi-massima verosimiglianza con
    innovazioni `dist` ("normal" o "t"), stimato su T_k; previsioni a un
    passo di sigma_{t+1} ristimando il modello ogni giorno sui rendimenti
    osservati fino a t.

    Returns:
      test_dates      : DatetimeIndex
      realized_vol    : |log_return| (proxy descrittivo)
      forecast_vol    : sigma_hat_{t+1}
      params          : alpha, beta, persistenza (e nu per la t) stimati su T_k
    """
    if dist not in GARCH_DISTS:
        raise ValueError(f"dist deve essere uno di {GARCH_DISTS}")
    series     = df["log_return"].values * 100   # scale for numerical stability
    train_s    = pd.Series(series[train_idx])
    test_dates = df.index[test_idx]
    n_test     = len(test_idx)

    # Rolling one-step-ahead forecast
    history      = list(train_s)
    forecast_vol = []
    for t in range(n_test):
        mod = arch_model(pd.Series(history), vol="Garch", p=1, q=1,
                          mean="Constant", dist=dist)
        res = mod.fit(disp="off", show_warning=False)
        fc  = res.forecast(horizon=1)
        forecast_vol.append(np.sqrt(fc.variance.values[-1, 0]) / 100)
        history.append(series[test_idx[t]])

    realized_vol = np.abs(df["log_return"].values[test_idx])

    params = fit_garch_params(df, train_idx, dist=dist)
    return test_dates, realized_vol, np.array(forecast_vol), params
