"""
metrics_econ.py
----------------
Modulo unico per le metriche economiche di performance — usato da
run_backtest, block_bootstrap_ci e architecture_vs_regime_D (statistica D),
così le tre sezioni condividono esattamente le stesse formule (D4 del
report del relatore, 7 settembre 2026).

Definizioni (identiche alla Notazione della tesi):
    SR     = sqrt(252) * mean(y) / std(y, ddof=1)
    W_t    = prod_{j<=t} (1 + y_j)                      (ricchezza composta)
    MDD    = max_t { 1 - W_t / max_{u<=t} W_u }          (frazione positiva)
    Calmar = 252 * mean(y) / MDD

y è la serie giornaliera di rendimento NETTO della strategia (già al netto
di costi di transazione e di roll) — mai il rendimento lordo.
"""

import numpy as np


def sharpe_ratio(y: np.ndarray) -> float:
    """SR annualizzato, ddof=1. Ritorna 0.0 se la serie ha varianza ~0."""
    y = np.asarray(y, dtype=float).flatten()
    std = np.std(y, ddof=1) if len(y) > 1 else 0.0
    if std <= 1e-10:
        return 0.0
    return float(np.sqrt(252) * np.mean(y) / std)


def wealth_path(y: np.ndarray) -> np.ndarray:
    """W_t = prod_{j<=t} (1 + y_j)."""
    y = np.asarray(y, dtype=float).flatten()
    return np.cumprod(1.0 + y)


def max_drawdown(y: np.ndarray) -> float:
    """
    MDD = max_t { 1 - W_t / max_{u<=t} W_u } — frazione POSITIVA della
    ricchezza (0 = nessun drawdown, 0.5 = dimezzamento della ricchezza dal
    picco). Chi consuma questo valore per la tabella "Max Drawdown (%)"
    (segno negativo, storicamente atteso dai plot/CSV esistenti) applica il
    segno a valle, non qui.
    """
    y = np.asarray(y, dtype=float).flatten()
    if len(y) == 0:
        return 0.0
    W = wealth_path(y)
    running_max = np.maximum.accumulate(W)
    drawdown = 1.0 - W / running_max
    return float(drawdown.max())


def calmar_ratio(y: np.ndarray, mdd: float = None) -> float:
    """Calmar = 252 * mean(y) / MDD. Ritorna 0.0 se MDD ~ 0."""
    y = np.asarray(y, dtype=float).flatten()
    if mdd is None:
        mdd = max_drawdown(y)
    if mdd <= 1e-10:
        return 0.0
    return float(252 * np.mean(y) / mdd)
