"""
backtesting.py
--------------
Traduce ogni previsione in un segnale di trading long/short e misura
il valore economico generato (Sharpe, drawdown, directional accuracy),
con costi di transazione.
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


def block_bootstrap_ci(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        transaction_cost: float = 0.0001,
                        threshold: float = 0.0,
                        block_length: int = 10,
                        n_boot: int = 2000,
                        ci: float = 0.95,
                        seed: int = 42) -> dict:
    """
    Moving Block Bootstrap (Kunsch, 1989) — CI al 95% per Sharpe Ratio, Max
    Drawdown e Calmar Ratio, più un test su Directional Accuracy che non
    assume indipendenza degli hit giornalieri (a differenza del binomiale
    già riportato da run_backtest, che resta invariato — questo è un
    supplemento, non una sostituzione).

    block_length=10 (~2 settimane di trading): sopra la regola asintotica
    standard b ~ T^(1/3) (Hall, Horowitz & Jing, 1995; con T=210 -> ~5.9),
    verificata empiricamente contro l'ACF di P&L/hit reali (dipendenza
    residua entro la banda di rumore bianco oltre lag ~2) — compromesso tra
    dipendenza catturata e diversità campionaria (210/10 = 21 blocchi).
    n_boot=2000: standard per CI percentile-based (Efron & Tibshirani,
    1993). Entrambi parametri espliciti, non sepolti nel corpo funzione.

    Ogni replica richiama run_backtest() invariata — le CI sono quindi
    coerenti al 100% con le definizioni usate per i valori puntuali.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)
    assert n >= block_length, (
        f"n={n} osservazioni insufficienti per block_length={block_length}"
    )

    rng        = np.random.default_rng(seed)
    n_blocks   = int(np.ceil(n / block_length))
    max_start  = n - block_length

    sharpe_boot, mdd_boot, calmar_boot, da_boot = [], [], [], []
    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx    = np.concatenate(
            [np.arange(s, s + block_length) for s in starts])[:n]
        bt = run_backtest(y_true[idx], y_pred[idx],
                          transaction_cost=transaction_cost,
                          threshold=threshold)
        sharpe_boot.append(bt["Sharpe Ratio"])
        mdd_boot.append(bt["Max Drawdown (%)"])
        calmar_boot.append(bt["Calmar Ratio"])
        da_boot.append(bt["Directional Acc."])

    sharpe_boot = np.array(sharpe_boot)
    mdd_boot    = np.array(mdd_boot)
    calmar_boot = np.array(calmar_boot)
    da_boot     = np.array(da_boot)

    alpha            = 1 - ci
    lo_pct, hi_pct   = 100 * alpha / 2, 100 * (1 - alpha / 2)

    p_below        = float(np.mean(da_boot <= 50.0))
    p_above        = float(np.mean(da_boot >= 50.0))
    da_boot_pvalue = float(min(1.0, 2 * min(p_below, p_above)))

    return {
        "Sharpe CI Lower (95%)":
            round(float(np.percentile(sharpe_boot, lo_pct)), 4),
        "Sharpe CI Upper (95%)":
            round(float(np.percentile(sharpe_boot, hi_pct)), 4),
        "Max Drawdown CI Lower (95%)":
            round(float(np.percentile(mdd_boot, lo_pct)), 3),
        "Max Drawdown CI Upper (95%)":
            round(float(np.percentile(mdd_boot, hi_pct)), 3),
        "Calmar CI Lower (95%)":
            round(float(np.percentile(calmar_boot, lo_pct)), 4),
        "Calmar CI Upper (95%)":
            round(float(np.percentile(calmar_boot, hi_pct)), 4),
        "DA CI Lower (95%)":
            round(float(np.percentile(da_boot, lo_pct)), 2),
        "DA CI Upper (95%)":
            round(float(np.percentile(da_boot, hi_pct)), 2),
        "DA Bootstrap p-value":   round(da_boot_pvalue, 4),
        "DA Bootstrap Sig. (5%)": da_boot_pvalue < 0.05,
        "Block Length":           block_length,
        "N Bootstrap":            n_boot,
    }


_D_REGIMES = ["stable", "normal", "volatile"]


def _simple_sharpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Sharpe Ratio "semplice", senza costi di transazione — stessa formula di
    compute_metrics() in utils.py. Usata (invece dello Sharpe di
    run_backtest) perché indipendente dall'ordine temporale: i giorni di un
    singolo regime sono un sottoinsieme non contiguo della serie, e uno
    Sharpe basato su path/turnover tratterebbe giorni non adiacenti come se
    lo fossero, introducendo un artefatto nei costi di transazione.
    """
    if len(y_true) == 0:
        return 0.0
    position = np.sign(y_pred)
    pnl      = position * y_true
    std      = np.std(pnl)
    return float(np.mean(pnl) / std * np.sqrt(252)) if std > 1e-10 else 0.0


def architecture_vs_regime_D(asset_data: dict,
                              block_length: int = 10,
                              n_boot: int = 2000,
                              min_obs_per_cell: int = 5,
                              ci: float = 0.95,
                              seed: int = 42) -> dict:
    """
    Statistica D per RQ3 — prevalenza dell'effetto architettura vs effetto
    regime:

        D = (1/M) * sum_m Var_g(S_m,g) - (1/G) * sum_g Var_m(S_m,g)

    S_m,g = Sharpe "semplice" (_simple_sharpe, no costi) del modello m nel
    regime g. D > 0 -> prevale la variabilità tra regimi a modello fissato;
    D < 0 -> prevale la variabilità tra modelli a regime fissato.

    Design (confermato esplicitamente, non lasciato a default arbitrari):

    - Random Walk esclusa da M: predice sempre rendimento zero, Sharpe=0 in
      ogni regime per costruzione — stesso precedente di build_dm_table,
      che la esclude dai confronti DM.
    - D è calcolato aggregando (pooling) le osservazioni di TUTTI gli asset
      in asset_data per ciascun regime, non un D separato per asset. Scelta
      esplicita: alcuni asset hanno celle regime quasi vuote nella finestra
      OOS (es. Platinum: 1 solo giorno "stable" su 210) che renderebbero un
      D per-asset inaffidabile; il pooling prende in prestito osservazioni
      dagli altri asset. Nota: ogni coppia (asset, giorno) pesa allo stesso
      modo — nessuna normalizzazione per la volatilità propria dell'asset.
    - Bootstrap: Moving Block Bootstrap (Künsch, 1989). Per ogni replica, un
      set di blocchi viene estratto UNA VOLTA per ciascun asset e applicato
      identicamente a tutti i modelli di quell'asset — preserva la
      dipendenza trasversale tra strategie valutate sugli stessi rendimenti
      realizzati (asset diversi restano indipendenti tra loro).
      block_length=10, n_boot=2000: stessa configurazione e stessa
      giustificazione di block_bootstrap_ci() — riuso deliberato per
      coerenza metodologica nella tesi (b ~ T^(1/3) di Hall, Horowitz &
      Jing 1995 con margine, verificata su ACF reale; B=2000 per CI
      percentile-based, Efron & Tibshirani 1993).
    - min_obs_per_cell=5: repliche in cui un qualunque regime scende sotto
      questa soglia (es. i pochi giorni "stable" di Platinum non vengono
      estratti in quella replica) sono scartate, non conteggiate nella
      distribuzione bootstrap di D. "N Bootstrap Used" riporta quante
      repliche su n_boot sono state effettivamente utilizzabili.

    asset_data: {asset_name: {"y_true": array (T,), "dates": DatetimeIndex,
                               "regimes": pd.Series,
                               "y_pred": {model_name: array (T,)}}}
                (eventuale "Random_Walk" dentro y_pred viene ignorata)
    """
    def compute_D(y_true_all, regime_all, y_pred_all, model_names):
        M = len(model_names)
        S = np.zeros((M, len(_D_REGIMES)))
        counts = {}
        for gi, g in enumerate(_D_REGIMES):
            mask = regime_all == g
            counts[g] = int(mask.sum())
            for mi, m in enumerate(model_names):
                S[mi, gi] = _simple_sharpe(y_true_all[mask], y_pred_all[m][mask])
        return S, counts

    # ── Allinea ciascun asset a una lunghezza comune tra i suoi modelli ────────
    prepared = {}
    for asset_name, data in asset_data.items():
        model_names_a = [m for m in data["y_pred"] if m != "Random_Walk"]
        n_common = min(len(data["y_true"]),
                       *(len(data["y_pred"][m]) for m in model_names_a))
        y_true = np.asarray(data["y_true"])[-n_common:]
        dates  = data["dates"][-n_common:]
        y_pred = {m: np.asarray(data["y_pred"][m])[-n_common:]
                  for m in model_names_a}
        regime_arr = data["regimes"].reindex(dates).fillna("normal").values
        prepared[asset_name] = {
            "y_true": y_true, "y_pred": y_pred, "regime": regime_arr, "n": n_common,
        }

    model_names = sorted(set.intersection(
        *(set(p["y_pred"].keys()) for p in prepared.values())))

    # ── Stima puntuale (nessun bootstrap) ──────────────────────────────────────
    y_true_all = np.concatenate([p["y_true"] for p in prepared.values()])
    regime_all = np.concatenate([p["regime"] for p in prepared.values()])
    y_pred_all = {m: np.concatenate([prepared[a]["y_pred"][m] for a in prepared])
                  for m in model_names}

    S_point, counts_point = compute_D(y_true_all, regime_all, y_pred_all, model_names)
    var_g_point = S_point.var(axis=1, ddof=1)   # per modello, sui G regimi
    var_m_point = S_point.var(axis=0, ddof=1)   # per regime, sugli M modelli
    D_point = float(var_g_point.mean() - var_m_point.mean())

    # ── Block bootstrap congiunto (per asset, condiviso tra i modelli) ────────
    rng    = np.random.default_rng(seed)
    D_boot = []
    for _ in range(n_boot):
        yt_parts, reg_parts = [], []
        yp_parts = {m: [] for m in model_names}
        for p in prepared.values():
            n         = p["n"]
            n_blocks  = int(np.ceil(n / block_length))
            max_start = n - block_length
            starts    = rng.integers(0, max_start + 1, size=n_blocks)
            idx       = np.concatenate(
                [np.arange(s, s + block_length) for s in starts])[:n]
            yt_parts.append(p["y_true"][idx])
            reg_parts.append(p["regime"][idx])
            for m in model_names:
                yp_parts[m].append(p["y_pred"][m][idx])

        yt_b  = np.concatenate(yt_parts)
        reg_b = np.concatenate(reg_parts)
        yp_b  = {m: np.concatenate(yp_parts[m]) for m in model_names}

        counts_b = {g: int((reg_b == g).sum()) for g in _D_REGIMES}
        if min(counts_b.values()) < min_obs_per_cell:
            continue   # replica scartata: cella regime degenere

        S_b, _  = compute_D(yt_b, reg_b, yp_b, model_names)
        var_g_b = S_b.var(axis=1, ddof=1)
        var_m_b = S_b.var(axis=0, ddof=1)
        D_boot.append(float(var_g_b.mean() - var_m_b.mean()))

    D_boot = np.array(D_boot)
    n_used = len(D_boot)
    alpha  = 1 - ci
    if n_used > 0:
        lo = round(float(np.percentile(D_boot, 100 * alpha / 2)), 6)
        hi = round(float(np.percentile(D_boot, 100 * (1 - alpha / 2))), 6)
    else:
        lo, hi = None, None

    return {
        "D":                    round(D_point, 6),
        "D CI Lower (95%)":     lo,
        "D CI Upper (95%)":     hi,
        "Sharpe by Model-Regime": pd.DataFrame(
            S_point, index=model_names, columns=_D_REGIMES),
        "N Obs per Regime (point estimate)": counts_point,
        "Block Length":         block_length,
        "N Bootstrap Requested": n_boot,
        "N Bootstrap Used":     n_used,
        "Min Obs per Cell":     min_obs_per_cell,
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


def backtest_summary_table(bt_results: dict) -> pd.DataFrame:
    """Tabella riassuntiva per la tesi (Sezione 4.4)."""
    scalar_keys = [
        "Annual Return (%)", "Annual Vol (%)", "Sharpe Ratio",
        "Max Drawdown (%)", "Calmar Ratio",
        "Directional Acc.", "DA p-value", "Win Rate (%)", "Turnover",
        "Sharpe CI Lower (95%)", "Sharpe CI Upper (95%)",
        "Max Drawdown CI Lower (95%)", "Max Drawdown CI Upper (95%)",
        "Calmar CI Lower (95%)", "Calmar CI Upper (95%)",
        "DA CI Lower (95%)", "DA CI Upper (95%)",
        "DA Bootstrap p-value", "DA Bootstrap Sig. (5%)",
        "Block Length", "N Bootstrap",
    ]
    rows = []
    for model_name, bt in bt_results.items():
        row = {"Model": model_name}
        for k in scalar_keys:
            row[k] = bt.get(k, None)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("Sharpe Ratio", ascending=False)