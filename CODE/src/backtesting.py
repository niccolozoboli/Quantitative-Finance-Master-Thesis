"""
backtesting.py
--------------
Backtesting layer — cuore del contributo Quantitative Finance.
MODULO NUOVO.

Trasforma ogni previsione in un segnale di trading e misura
il valore economico generato.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def run_backtest(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 transaction_cost: float = 0.0001,
                 threshold: float = 0.0) -> dict:
    """
    Strategia long/short basata sul segno della previsione.

    y_pred > threshold  → LONG  (+1)
    y_pred < -threshold → SHORT (-1)
    altrimenti          → FLAT  (0)

    transaction_cost = 1 bps (realistico per commodity futures).
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    signal = np.where(y_pred >  threshold,  1.0,
             np.where(y_pred < -threshold, -1.0, 0.0))

    gross_pnl       = signal * y_true
    position_change = np.abs(np.diff(signal, prepend=0.0))
    costs           = transaction_cost * position_change
    net_pnl         = gross_pnl - costs

    cumulative_pnl = np.cumsum(net_pnl)
    annual_return  = net_pnl.mean() * 252
    annual_vol     = net_pnl.std()  * np.sqrt(252)
    sharpe         = annual_return / annual_vol if annual_vol > 1e-10 else 0.0

    running_max  = np.maximum.accumulate(cumulative_pnl)
    drawdown     = cumulative_pnl - running_max
    max_drawdown = float(drawdown.min())

    calmar = (annual_return / abs(max_drawdown)
              if abs(max_drawdown) > 1e-10 else 0.0)

    active_mask = signal != 0
    if active_mask.sum() > 0:
        correct = (np.sign(y_pred[active_mask]) ==
                   np.sign(y_true[active_mask]))
        da      = float(correct.mean() * 100)
        btest   = binomtest(int(correct.sum()), int(active_mask.sum()), p=0.5)
        da_pval = float(btest.pvalue)
    else:
        da      = 50.0
        da_pval = 1.0

    turnover = float(position_change.mean())
    win_rate = float((net_pnl > 0).mean() * 100)

    return {
        "Annual Return (%)": round(annual_return * 100, 3),
        "Annual Vol (%)":    round(annual_vol    * 100, 3),
        "Sharpe Ratio":      round(sharpe,              4),
        "Max Drawdown (%)":  round(max_drawdown  * 100, 3),
        "Calmar Ratio":      round(calmar,              4),
        "Directional Acc.":  round(da,                  2),
        "DA p-value":        round(da_pval,             4),
        "Win Rate (%)":      round(win_rate,             2),
        "Turnover":          round(turnover,             4),
        "N Days":            len(net_pnl),
        "Cum. PnL":          cumulative_pnl,
        "Net PnL":           net_pnl
    }


def regime_conditional_strategy(y_pred: np.ndarray,
                                  dates: pd.DatetimeIndex,
                                  regimes: pd.Series,
                                  stable_scale:   float = 1.0,
                                  normal_scale:   float = 0.5,
                                  volatile_scale: float = 0.0) -> np.ndarray:
    """
    Modula la posizione in base al regime di mercato.

    STABILE:  posizione piena  (il modello ha edge documentato)
    NORMALE:  metà posizione   (edge incerto)
    VOLATILE: flat             (nessun edge, rischio alto)

    Questa è la regime-conditional strategy — contributo originale.
    """
    y_pred         = np.array(y_pred).flatten()
    regime_aligned = regimes.reindex(dates).fillna("normal").values
    scale = np.where(regime_aligned == "stable",   stable_scale,
            np.where(regime_aligned == "normal",   normal_scale,
                                                   volatile_scale))
    return y_pred * scale


def buy_and_hold_backtest(y_true: np.ndarray,
                           transaction_cost: float = 0.0001) -> dict:
    """Benchmark passivo: sempre long."""
    y_true   = np.array(y_true).flatten()
    signal   = np.ones_like(y_true)
    costs    = np.zeros_like(y_true)
    costs[0] = transaction_cost
    net_pnl  = signal * y_true - costs

    cumulative_pnl = np.cumsum(net_pnl)
    annual_return  = net_pnl.mean() * 252
    annual_vol     = net_pnl.std()  * np.sqrt(252)
    sharpe         = annual_return / annual_vol if annual_vol > 1e-10 else 0.0
    running_max    = np.maximum.accumulate(cumulative_pnl)
    max_drawdown   = float((cumulative_pnl - running_max).min())

    return {
        "Annual Return (%)": round(annual_return * 100, 3),
        "Annual Vol (%)":    round(annual_vol    * 100, 3),
        "Sharpe Ratio":      round(sharpe,              4),
        "Max Drawdown (%)":  round(max_drawdown  * 100, 3),
        "Cum. PnL":          cumulative_pnl,
        "Net PnL":           net_pnl
    }


def backtest_summary_table(bt_results: dict) -> pd.DataFrame:
    """Tabella riassuntiva per la tesi (Sezione 4.4)."""
    scalar_keys = [
        "Annual Return (%)", "Annual Vol (%)", "Sharpe Ratio",
        "Max Drawdown (%)", "Calmar Ratio",
        "Directional Acc.", "DA p-value", "Win Rate (%)", "Turnover"
    ]
    rows = []
    for model_name, bt in bt_results.items():
        row = {"Model": model_name}
        for k in scalar_keys:
            row[k] = bt.get(k, None)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("Sharpe Ratio", ascending=False)