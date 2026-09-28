"""
backtesting.py
--------------
Traduce ogni previsione in un segnale di trading long/short e misura
il valore economico generato (Sharpe, drawdown, directional accuracy),
con costi di transazione e di roll.

Notazione (identica alla tesi):
    r_{t+1}   = log-return (target dei modelli)
    R_{t+1}   = P_{t+1}/P_t - 1 = exp(r_{t+1}) - 1   (rendimento semplice,
                usato SOLO qui nel backtest)
    s_{m,t}   = sign(r_hat_{m,t+1|t})
    a_t       in {1, 1/2, 0}                          (scala di regime)
    w_{m,t}   = a_t * s_{m,t}                          (posizione)
    y_{m,t+1} = w_{m,t} R_{t+1}
                - c|w_{m,t} - w_{m,t-1}|
                - 2c|w_{m,t}| * 1{t in roll dates}      (D1-D2)
    con c = cost_per_side (1e-4 per lato) e w = 0 prima del primo giorno OOS.

SR, MDD, Calmar sono importati da src/metrics_econ.py (D4) — stessa
definizione ovunque nel progetto (backtest, bootstrap, statistica D).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistical_tests import _newey_west_lag
from src.futures_calendar import roll_mask
from src import metrics_econ

_D_REGIMES = ["stable", "normal", "volatile"]


def run_backtest(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  dates: pd.DatetimeIndex = None,
                  asset_name: str = None,
                  cost_per_side: float = 0.0001,
                  threshold: float = 0.0,
                  position: np.ndarray = None) -> dict:
    """
    Strategia long/short basata sul segno della previsione.

    y_pred > threshold  → LONG  (+1, scalato da `position` se fornito)
    y_pred < -threshold → SHORT (-1)
    altrimenti           → FLAT  (0)

    Se `position` è fornito, viene usato direttamente come segnale w_t
    (es. per la regime-conditional strategy).

    `dates` + `asset_name` sono necessari per il costo di roll (2c|w_t| sui
    giorni del calendario di roll — vedi src/futures_calendar.py, A4-A5).
    Se omessi (es. test unitari su dati sintetici), il costo di roll è
    zero ovunque — comportamento esplicito, non un default silenzioso.

    cost_per_side = 1 bps per lato (realistico per commodity futures).
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    n = len(y_true)

    R = np.expm1(y_true)   # rendimento semplice — D1: usato nel P&L, mai r

    if position is not None:
        signal = np.asarray(position, dtype=float).flatten()
    else:
        signal = np.where(y_pred >  threshold,  1.0,
                 np.where(y_pred < -threshold, -1.0, 0.0))

    gross_pnl       = signal * R
    position_change = np.abs(np.diff(signal, prepend=0.0))
    turnover_cost    = cost_per_side * position_change

    if dates is not None and asset_name is not None:
        # roll_mask(dates)[i] è True quando dates[i] STESSA è un giorno di
        # roll. Il costo di roll fatto alla chiusura del giorno t si paga
        # però sul rendimento del giorno dopo (y_{t+1}, D1-D2/notazione
        # tesi: -2c|w_t|*1{t in roll dates} dentro y_{t+1}) — quindi va
        # applicato al rendimento in posizione i se dates[i-1] (non
        # dates[i]) è un giorno di roll: si sposta la maschera di una
        # posizione in avanti. Nessun costo sul primo giorno della serie
        # (non c'è un dates[i-1] osservabile in questo array).
        rmask_on_own_date = roll_mask(asset_name, dates)
        rmask = np.zeros(n, dtype=bool)
        rmask[1:] = rmask_on_own_date[:-1]
    else:
        rmask = np.zeros(n, dtype=bool)
    roll_cost = 2 * cost_per_side * np.abs(signal) * rmask

    net_pnl = gross_pnl - turnover_cost - roll_cost

    sharpe = metrics_econ.sharpe_ratio(net_pnl)
    mdd    = metrics_econ.max_drawdown(net_pnl)
    calmar = metrics_econ.calmar_ratio(net_pnl, mdd)

    annual_return = float(net_pnl.mean() * 252)
    annual_vol    = float(net_pnl.std(ddof=1) * np.sqrt(252)) if n > 1 else 0.0

    active_mask = signal != 0
    hit = (np.sign(y_pred) == np.sign(y_true)).astype(float)
    # B10: DA = frazione di giorni con w != 0 in cui sign(r_hat) == sign(r).
    # Nessuna inferenza binomiale qui — l'unica inferenza è il moving block
    # bootstrap (block_bootstrap_ci), coerente con la dipendenza seriale dei
    # rendimenti (H0: DA = 50%).
    if active_mask.sum() > 0:
        da = float(hit[active_mask].mean() * 100)
    else:
        da = 50.0

    turnover = float(position_change.mean())
    win_rate = float((net_pnl > 0).mean() * 100)

    return {
        "Annual Return (%)": round(annual_return * 100, 3),
        "Annual Vol (%)":    round(annual_vol    * 100, 3),
        "Sharpe Ratio":      round(sharpe,              4),
        "Max Drawdown (%)":  round(-mdd * 100, 3),
        "Calmar Ratio":      round(calmar,              4),
        "Directional Acc.":  round(da,                  2),
        "Win Rate (%)":      round(win_rate,             2),
        "Turnover":          round(turnover,             4),
        "N Roll Days":       int(rmask.sum()),
        "N Days":            n,
        "Cum. PnL":          np.cumsum(net_pnl),
        "Net PnL":           net_pnl,
        "Position":          signal,
        "Hit":               hit,
        "Active":            active_mask.astype(float),
    }


def block_bootstrap_ci(net_pnl: np.ndarray,
                        hit: np.ndarray,
                        active: np.ndarray,
                        block_length: int = 10,
                        n_boot: int = 2000,
                        ci: float = 0.95,
                        seed: int = 42) -> dict:
    """
    Moving Block Bootstrap (Kunsch, 1989) — CI al 95% per Sharpe Ratio, Max
    Drawdown e Calmar Ratio, più un test bootstrap su Directional Accuracy
    (H0: DA = 50%).

    D5: il bootstrap ricampiona a blocchi la serie NETTA `net_pnl` già
    calcolata sul percorso OOS completo (run_backtest chiamata una sola
    volta a monte) — non ri-esegue la strategia sui blocchi ricampionati.
    Ricalcolare turnover/costi su indici ricampionati (non contigui nella
    serie originale) introdurrebbe costi di transazione spuri alle giunture
    tra blocchi che non sono mai stati effettivamente pagati. Per la DA si
    ricampionano a blocchi le coppie (hit, active) — stesso principio.

    block_length=10, n_boot=2000, seed=42, intervallo percentile — stessi
    parametri in tutto il progetto (bootstrap, statistica D).
    """
    net_pnl = np.asarray(net_pnl, dtype=float).flatten()
    hit     = np.asarray(hit,     dtype=float).flatten()
    active  = np.asarray(active,  dtype=float).flatten()
    n = len(net_pnl)
    assert n >= block_length, (
        f"n={n} osservazioni insufficienti per block_length={block_length}"
    )

    rng       = np.random.default_rng(seed)
    n_blocks  = int(np.ceil(n / block_length))
    max_start = n - block_length

    sharpe_boot, mdd_boot, calmar_boot, da_boot = [], [], [], []
    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx    = np.concatenate(
            [np.arange(s, s + block_length) for s in starts])[:n]

        y_b   = net_pnl[idx]
        mdd_b = metrics_econ.max_drawdown(y_b)
        sharpe_boot.append(metrics_econ.sharpe_ratio(y_b))
        mdd_boot.append(-mdd_b * 100)
        calmar_boot.append(metrics_econ.calmar_ratio(y_b, mdd_b))

        act_b = active[idx]
        hit_b = hit[idx]
        da_b  = float(hit_b[act_b == 1].mean() * 100) if act_b.sum() > 0 else 50.0
        da_boot.append(da_b)

    sharpe_boot = np.array(sharpe_boot)
    mdd_boot    = np.array(mdd_boot)
    calmar_boot = np.array(calmar_boot)
    da_boot     = np.array(da_boot)

    alpha          = 1 - ci
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)

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


def estimate_alpha(y: np.ndarray, b: np.ndarray, hac_lag: int = None) -> dict:
    """
    D6 (RQ1) — regressione y_{m,t} = alpha_m + beta_m b_t + u_{m,t}, al
    posto del t-test su E[y]. b_t = R_t (long passivo, w=1 sempre, stesso
    percorso finanziario di A4 — nessun costo, è un fattore di benchmark).

    Errori HAC (Newey-West, kernel di Bartlett) via statsmodels
    (cov_type="HAC"), con L = _newey_west_lag(n) — stessa regola di banda
    usata dal test Diebold-Mariano, per coerenza metodologica.

    H0: alpha_m = 0.
    """
    y = np.asarray(y, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    n = len(y)

    L = hac_lag if hac_lag is not None else _newey_west_lag(n)
    X = sm.add_constant(b)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(L, 0)})

    return {
        "Alpha (daily)":        float(model.params[0]),
        "Beta":                 float(model.params[1]),
        "Alpha t-stat (HAC)":   float(model.tvalues[0]),
        "Alpha p-value":        float(model.pvalues[0]),
        "HAC Lag (L)":          int(L),
    }


def _sharpe_by_regime(y: np.ndarray, regime_arr: np.ndarray,
                       min_obs: int = 0) -> np.ndarray:
    """SR (metrics_econ) del vettore `y` ristretto a ciascun regime di _D_REGIMES."""
    out = np.zeros(len(_D_REGIMES))
    for gi, g in enumerate(_D_REGIMES):
        mask = regime_arr == g
        out[gi] = metrics_econ.sharpe_ratio(y[mask]) if mask.sum() >= min_obs else 0.0
    return out


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

    D7: S_m,g = SR (metrics_econ.sharpe_ratio, D4) della serie NETTA y della
    strategia STATICA del modello m (già calcolata da run_backtest
    sull'intero percorso OOS, D1-D2), ristretta ai giorni con regime g,
    aggregando (pooling) i 4 asset. M = 12 (Random Walk esclusa dalle serie
    y passate in asset_data), G = 3.

    Bootstrap: Moving Block Bootstrap sulle DATE COMUNI ai 4 asset,
    applicato alle serie y già calcolate (nessuna ri-esecuzione della
    strategia) — stessa logica di block_bootstrap_ci.

    asset_data: {asset_name: {"y": {model_name: net_pnl array (T,)},
                               "dates": DatetimeIndex (T,),
                               "regimes": pd.Series}}
                (un'eventuale "Random_Walk" dentro "y" viene ignorata)
    """
    def compute_D(regime_all, y_all, model_names):
        M = len(model_names)
        S = np.zeros((M, len(_D_REGIMES)))
        counts = {}
        for gi, g in enumerate(_D_REGIMES):
            mask = regime_all == g
            counts[g] = int(mask.sum())
            for mi, m in enumerate(model_names):
                S[mi, gi] = metrics_econ.sharpe_ratio(y_all[m][mask])
        return S, counts

    # ── Allinea ciascun asset a una lunghezza comune tra i suoi modelli ────────
    prepared = {}
    for asset_name, data in asset_data.items():
        model_names_a = [m for m in data["y"] if m != "Random_Walk"]
        n_common = min(len(data["dates"]),
                       *(len(data["y"][m]) for m in model_names_a))
        dates  = data["dates"][-n_common:]
        y_dict = {m: np.asarray(data["y"][m])[-n_common:] for m in model_names_a}
        regime_arr = data["regimes"].reindex(dates).fillna("normal").values
        prepared[asset_name] = {
            "y": y_dict, "regime": regime_arr, "dates": dates, "n": n_common,
        }

    model_names = sorted(set.intersection(
        *(set(p["y"].keys()) for p in prepared.values())))

    # ── Stima puntuale (nessun bootstrap) ──────────────────────────────────────
    regime_all = np.concatenate([p["regime"] for p in prepared.values()])
    y_all = {m: np.concatenate([prepared[a]["y"][m] for a in prepared])
             for m in model_names}

    S_point, counts_point = compute_D(regime_all, y_all, model_names)
    var_g_point = S_point.var(axis=1, ddof=1)   # per modello, sui G regimi
    var_m_point = S_point.var(axis=0, ddof=1)   # per regime, sugli M modelli
    D_point = float(var_g_point.mean() - var_m_point.mean())

    # ── Allinea i 4 asset sulle date OOS comuni (per il bootstrap congiunto) ──
    common_dates = None
    for p in prepared.values():
        common_dates = (p["dates"] if common_dates is None
                         else common_dates.intersection(p["dates"]))
    common_dates = common_dates.sort_values()
    n_common     = len(common_dates)

    aligned = {}
    for asset_name, p in prepared.items():
        pos = pd.Series(np.arange(p["n"]), index=p["dates"])
        loc = pos.reindex(common_dates).values.astype(int)
        aligned[asset_name] = {
            "regime": p["regime"][loc],
            "y": {m: p["y"][m][loc] for m in model_names},
        }

    # ── Block bootstrap congiunto sulle date comuni (un solo set di blocchi
    # di posizioni-data per replica, applicato a tutti gli asset/modelli) ─────
    rng       = np.random.default_rng(seed)
    n_blocks  = int(np.ceil(n_common / block_length))
    max_start = n_common - block_length
    D_boot    = []
    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx    = np.concatenate(
            [np.arange(s, s + block_length) for s in starts])[:n_common]

        reg_parts = []
        y_parts   = {m: [] for m in model_names}
        for a in aligned.values():
            reg_parts.append(a["regime"][idx])
            for m in model_names:
                y_parts[m].append(a["y"][m][idx])

        reg_b = np.concatenate(reg_parts)
        y_b   = {m: np.concatenate(y_parts[m]) for m in model_names}

        counts_b = {g: int((reg_b == g).sum()) for g in _D_REGIMES}
        if min(counts_b.values()) < min_obs_per_cell:
            continue   # replica scartata: cella regime degenere

        S_b, _  = compute_D(reg_b, y_b, model_names)
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
        "N Common Dates":       n_common,
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
    signal = np.where(y_pred > 0, 1.0, np.where(y_pred < 0, -1.0, 0.0))
    return signal * scale


def backtest_summary_table(bt_results: dict) -> pd.DataFrame:
    """Tabella riassuntiva per la tesi (Sezione 4.4)."""
    scalar_keys = [
        "Annual Return (%)", "Annual Vol (%)", "Sharpe Ratio",
        "Max Drawdown (%)", "Calmar Ratio",
        "Directional Acc.", "Win Rate (%)", "Turnover", "N Roll Days",
        "Sharpe CI Lower (95%)", "Sharpe CI Upper (95%)",
        "Max Drawdown CI Lower (95%)", "Max Drawdown CI Upper (95%)",
        "Calmar CI Lower (95%)", "Calmar CI Upper (95%)",
        "DA CI Lower (95%)", "DA CI Upper (95%)",
        "DA Bootstrap p-value", "DA Bootstrap Sig. (5%)",
        "DA Bootstrap p Holm", "DA Bootstrap Sig. Holm (5%)",
        "Block Length", "N Bootstrap",
        "Alpha (daily)", "Beta", "Alpha t-stat (HAC)", "Alpha p-value",
        "Alpha p Holm", "Alpha Sig. Holm (5%)",
    ]
    rows = []
    for model_name, bt in bt_results.items():
        row = {"Model": model_name}
        for k in scalar_keys:
            row[k] = bt.get(k, None)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("Sharpe Ratio", ascending=False)
