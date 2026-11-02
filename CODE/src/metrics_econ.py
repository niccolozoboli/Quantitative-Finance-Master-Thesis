"""
metrics_econ.py
----------------
Modulo unico per le metriche economiche di performance — usato da
run_backtest, block_bootstrap_ci e architecture_vs_regime_D (statistica D),
così le tre sezioni condividono esattamente le stesse formule.

Definizioni (identiche all'eq. 1.15 della tesi, versione rivista dal
relatore, ottobre 2026 — commento K05):
    SR_0   = sqrt(252) * mean(y) / std(y, ddof=1)
             (Sharpe annualizzato a benchmark zero: nessun risk-free
             sottratto; esposizione a notional unitario)
    W_0    = 1,  W_t = prod_{j<=t} (1 + y_j)              (ricchezza composta)
    MDD    = max_{1<=t<=T} { 1 - W_t / max_{0<=u<=t} W_u } (frazione positiva)
    r_ann  = W_T^(252/T) - 1                               (rendimento annuo composto)
    Calmar = r_ann / MDD

Il picco parte da W_0 = 1, quindi una perdita il primo giorno conta come
drawdown. Il Calmar usa il rendimento composto annualizzato: la versione
precedente 252*mean(y)/MDD non è il Calmar ratio standard.

Convenzioni per i casi degeneri (da dichiarare in tesi): SR = 0 se la
serie ha varianza ~0 (es. strategia sempre flat); Calmar = 0 se MDD ~0.

y è la serie giornaliera di rendimento NETTO della strategia (già al netto
di costi di transazione e di roll) — mai il rendimento lordo.
"""

import numpy as np


def sharpe_ratio(y: np.ndarray) -> float:
    """SR_0 annualizzato, ddof=1. Ritorna 0.0 se la serie ha varianza ~0."""
    y = np.asarray(y, dtype=float).flatten()
    std = np.std(y, ddof=1) if len(y) > 1 else 0.0
    if std <= 1e-10:
        return 0.0
    return float(np.sqrt(252) * np.mean(y) / std)


def wealth_path(y: np.ndarray) -> np.ndarray:
    """[W_0, W_1, ..., W_T] con W_0 = 1 e W_t = prod_{j<=t} (1 + y_j)."""
    y = np.asarray(y, dtype=float).flatten()
    return np.concatenate([[1.0], np.cumprod(1.0 + y)])


def max_drawdown(y: np.ndarray) -> float:
    """
    MDD = max_{1<=t<=T} { 1 - W_t / max_{0<=u<=t} W_u } — frazione POSITIVA
    della ricchezza (0 = nessun drawdown, 0.5 = dimezzamento dal picco).
    Il picco include W_0 = 1. Chi consuma questo valore per la tabella
    "Max Drawdown (%)" (segno negativo) applica il segno a valle.
    """
    y = np.asarray(y, dtype=float).flatten()
    if len(y) == 0:
        return 0.0
    W = wealth_path(y)
    running_max = np.maximum.accumulate(W)
    drawdown = 1.0 - W / running_max
    return float(drawdown[1:].max())


def annualised_return(y: np.ndarray) -> float:
    """r_ann = W_T^(252/T) - 1. NaN se T = 0 o se la ricchezza si azzera."""
    y = np.asarray(y, dtype=float).flatten()
    T = len(y)
    if T == 0:
        return float("nan")
    W_T = float(np.prod(1.0 + y))
    if W_T <= 0:
        return float("nan")
    return float(W_T ** (252.0 / T) - 1.0)


def calmar_ratio(y: np.ndarray, mdd: float = None) -> float:
    """Calmar = r_ann / MDD (rendimento composto annualizzato). 0.0 se MDD ~ 0."""
    y = np.asarray(y, dtype=float).flatten()
    if mdd is None:
        mdd = max_drawdown(y)
    if mdd <= 1e-10:
        return 0.0
    return float(annualised_return(y) / mdd)
