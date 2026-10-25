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


EXECUTIONS = ("close", "lag1", "open")


def lag_positions(w: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Posizione eseguita con `lag` giorni di ritardo: w_exec[i] = w[i-lag],
    con w_exec = 0 nei primi `lag` giorni (nessuna posizione prima del
    primo giorno fuori campione). Sensitivity del timing (commento K13).
    """
    w = np.asarray(w, dtype=float).flatten()
    if lag <= 0:
        return w.copy()
    if lag >= len(w):
        return np.zeros_like(w)
    return np.concatenate([np.zeros(lag), w[:-lag]])


def run_backtest(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  dates: pd.DatetimeIndex = None,
                  asset_name: str = None,
                  cost_per_side: float = 0.0001,
                  threshold: float = 0.0,
                  position: np.ndarray = None,
                  execution: str = "close",
                  open_returns: np.ndarray = None) -> dict:
    """
    Strategia long/short basata sul segno della previsione.

    y_pred > threshold  → LONG  (+1, scalato da `position` se fornito)
    y_pred < -threshold → SHORT (-1)
    altrimenti           → FLAT  (0)

    Se `position` è fornito, viene usato direttamente come segnale w_t
    (es. per la regime-conditional strategy).

    `execution` (commento K13 del relatore):
      "close" — specificazione principale: la posizione decisa con
                l'informazione fino alla chiusura di t è applicata al
                rendimento close-to-close da t a t+1. È un'idealizzazione
                (P_t è insieme informazione ed esecuzione), dichiarata in tesi.
      "lag1"  — sensitivity: la stessa posizione è eseguita un giorno dopo,
                w_exec[i] = w[i-1] (lag_positions), con prima posizione 0.
      "open"  — esecuzione all'apertura successiva: la posizione decisa alla
                chiusura di t è eseguita all'apertura di t+1 e tenuta fino
                all'apertura di t+2. `open_returns[i]` deve essere il
                rendimento open-to-open O_{i+1}/O_i - 1, con O_i l'apertura
                della data dates[i]; NaN dove non disponibile (quei giorni
                vengono esclusi dalla serie).
    Costi di transazione e di roll sono sempre calcolati sulla posizione
    effettivamente eseguita. La DA confronta il segno della previsione
    eseguita con il segno del rendimento effettivamente realizzato.

    `dates` + `asset_name` sono necessari per il costo di roll (2c|w_t| sui
    giorni del calendario di roll — vedi src/futures_calendar.py).
    Se omessi (es. test unitari su dati sintetici), il costo di roll è
    zero ovunque — comportamento esplicito, non un default silenzioso.

    cost_per_side = 1 bps per lato.
    """
    if execution not in EXECUTIONS:
        raise ValueError(f"execution deve essere uno di {EXECUTIONS}")

    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    if position is not None:
        signal = np.asarray(position, dtype=float).flatten()
    else:
        signal = np.where(y_pred >  threshold,  1.0,
                 np.where(y_pred < -threshold, -1.0, 0.0))
    pred_exec = y_pred

    if execution == "close":
        R = np.expm1(y_true)   # rendimento semplice — usato nel P&L, mai r
    elif execution == "lag1":
        R = np.expm1(y_true)
        signal = lag_positions(signal, 1)
        pred_exec = lag_positions(y_pred, 1)
    else:  # "open"
        if open_returns is None:
            raise ValueError("execution='open' richiede open_returns")
        R = np.asarray(open_returns, dtype=float).flatten()
        if len(R) != len(y_true):
            raise ValueError("open_returns deve avere la stessa lunghezza di y_true")

    keep = ~np.isnan(R)
    n = int(keep.sum())

    gross_pnl       = signal * np.where(keep, R, 0.0)
    position_change = np.abs(np.diff(signal, prepend=0.0))
    turnover_cost   = cost_per_side * position_change

    if dates is not None and asset_name is not None:
        # roll_mask(dates)[i] è True quando dates[i] STESSA è un giorno di
        # roll. Il costo di roll fatto alla chiusura del giorno t si paga
        # sul rendimento del giorno dopo (y_{t+1}: -2c|w_t|*1{t in roll
        # dates} dentro y_{t+1}) — quindi va applicato al rendimento in
        # posizione i se dates[i-1] (non dates[i]) è un giorno di roll.
        # Nessun costo sul primo giorno della serie.
        rmask_on_own_date = roll_mask(asset_name, dates)
        rmask = np.zeros(len(y_true), dtype=bool)
        rmask[1:] = rmask_on_own_date[:-1]
    else:
        rmask = np.zeros(len(y_true), dtype=bool)
    roll_cost = 2 * cost_per_side * np.abs(signal) * rmask

    net_full = gross_pnl - turnover_cost - roll_cost
    net_pnl  = net_full[keep]

    sharpe = metrics_econ.sharpe_ratio(net_pnl)
    mdd    = metrics_econ.max_drawdown(net_pnl)
    calmar = metrics_econ.calmar_ratio(net_pnl, mdd)

    annual_return = metrics_econ.annualised_return(net_pnl)
    mean_x252     = float(net_pnl.mean() * 252) if n > 0 else float("nan")
    annual_vol    = float(net_pnl.std(ddof=1) * np.sqrt(252)) if n > 1 else 0.0

    realised = np.where(keep, R, 0.0)
    active_mask = (signal != 0)[keep]
    hit = (np.sign(pred_exec) == np.sign(realised)).astype(float)[keep]
    # DA = frazione di giorni con w != 0 in cui il segno della previsione
    # eseguita coincide con il segno del rendimento realizzato. Nessuna
    # inferenza qui: i test sotto H0 sono in statistical_tests
    # (da_test_hac, da_test_centred_bootstrap), gli intervalli in
    # block_bootstrap_ci.
    if active_mask.sum() > 0:
        da = float(hit[active_mask].mean() * 100)
    else:
        da = float("nan")

    turnover = float(position_change[keep].mean()) if n > 0 else 0.0
    win_rate = float((net_pnl > 0).mean() * 100) if n > 0 else float("nan")

    return {
        "Execution":                  execution,
        "Annual Return (%)":          round(annual_return * 100, 3),
        "Mean Daily Return x252 (%)": round(mean_x252 * 100, 3),
        "Annual Vol (%)":             round(annual_vol    * 100, 3),
        "Sharpe Ratio":               round(sharpe,              4),
        "Max Drawdown (%)":           round(-mdd * 100, 3),
        "Calmar Ratio":               round(calmar,              4),
        "Directional Acc.":           round(da,                  2),
        "Win Rate (%)":               round(win_rate,             2),
        "Turnover":                   round(turnover,             4),
        "N Roll Days":                int(rmask[keep].sum()),
        "N Days":                     n,
        "Cum. PnL":                   np.cumsum(net_pnl),
        "Net PnL":                    net_pnl,
        "Position":                   signal[keep],
        "Hit":                        hit,
        "Active":                     active_mask.astype(float),
        "Kept":                       keep,
    }


def block_bootstrap_ci(net_pnl: np.ndarray,
                        hit: np.ndarray,
                        active: np.ndarray,
                        block_length: int = 10,
                        n_boot: int = 2000,
                        ci: float = 0.95,
                        seed: int = 42) -> dict:
    """
    Moving Block Bootstrap (Kunsch, 1989) — intervalli al 95% per Sharpe
    Ratio, Max Drawdown, Calmar Ratio e Directional Accuracy.

    Il bootstrap ricampiona a blocchi la serie NETTA `net_pnl` già
    calcolata sul percorso OOS completo (run_backtest chiamata una sola
    volta a monte) — non ri-esegue la strategia sui blocchi ricampionati,
    così alle giunture tra blocchi non nascono costi di transazione spuri.
    Gli intervalli sono quindi condizionati alle previsioni stimate, al
    percorso delle posizioni e al modello dei costi; non includono
    l'incertezza della ristima della pipeline.

    Per la DA si ricampionano a blocchi le coppie (hit, active) e si
    riporta SOLO l'intervallo: il bootstrap non centrato è centrato sulla
    DA campionaria, non sul valore nullo 1/2, quindi la sua area di coda a
    1/2 non è un p-value valido (commento K04 del relatore). I test sotto
    H0 sono in statistical_tests.da_test_hac / da_test_centred_bootstrap.

    block_length=10, n_boot=2000, seed=42, intervallo percentile — stessi
    parametri in tutto il progetto.
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
        da_b  = (float(hit_b[act_b == 1].mean() * 100)
                 if act_b.sum() > 0 else float("nan"))
        da_boot.append(da_b)

    sharpe_boot = np.array(sharpe_boot)
    mdd_boot    = np.array(mdd_boot)
    calmar_boot = np.array(calmar_boot, dtype=float)
    da_boot     = np.array(da_boot, dtype=float)

    alpha          = 1 - ci
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def _pct(x, q):
        x = x[~np.isnan(x)]
        return float(np.percentile(x, q)) if len(x) else float("nan")

    return {
        "Sharpe CI Lower (95%)":       round(_pct(sharpe_boot, lo_pct), 4),
        "Sharpe CI Upper (95%)":       round(_pct(sharpe_boot, hi_pct), 4),
        "Max Drawdown CI Lower (95%)": round(_pct(mdd_boot, lo_pct), 3),
        "Max Drawdown CI Upper (95%)": round(_pct(mdd_boot, hi_pct), 3),
        "Calmar CI Lower (95%)":       round(_pct(calmar_boot, lo_pct), 4),
        "Calmar CI Upper (95%)":       round(_pct(calmar_boot, hi_pct), 4),
        "DA CI Lower (95%)":           round(_pct(da_boot, lo_pct), 2),
        "DA CI Upper (95%)":           round(_pct(da_boot, hi_pct), 2),
        "Block Length":                block_length,
        "N Bootstrap":                 n_boot,
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


# ─────────────────────────────────────────────────────────────────────────────
# Statistica D (RQ3) — commenti K02 e K15 del relatore
# ─────────────────────────────────────────────────────────────────────────────

D_AGGREGATIONS = ("portfolio", "per_asset")


def _D_from_S(S: np.ndarray) -> float:
    """D = (1/M) sum_m Var_g(S_m,g) - (1/G) sum_g Var_m(S_m,g), varianze ddof=1 (eq. 1.2)."""
    return float(np.var(S, axis=1, ddof=1).mean() - np.var(S, axis=0, ddof=1).mean())


def _prepare_D_inputs(asset_data: dict):
    """
    Allinea gli asset sulle date OOS comuni. Ritorna (model_names, assets,
    common_dates, Y, RG) con Y[m] array (A, n) dei rendimenti netti statici
    e RG array (A, n) delle etichette di regime osservabili al momento della
    decisione. Stima puntuale e bootstrap usano le stesse date comuni.
    """
    prepared = {}
    for asset_name, data in asset_data.items():
        model_names_a = [m for m in data["y"] if m not in ("Random_Walk", "Random Walk")]
        n_a = min(len(data["dates"]), *(len(data["y"][m]) for m in model_names_a))
        dates = pd.DatetimeIndex(data["dates"][-n_a:])
        prepared[asset_name] = {
            "dates": dates,
            "y": {m: np.asarray(data["y"][m], dtype=float)[-n_a:] for m in model_names_a},
            "regime": data["regimes"].reindex(dates).fillna("normal").values,
        }
    model_names = sorted(set.intersection(*(set(p["y"]) for p in prepared.values())))
    assets = list(prepared)
    common = None
    for p_ in prepared.values():
        common = p_["dates"] if common is None else common.intersection(p_["dates"])
    common = common.sort_values()
    loc = {a: pd.Series(np.arange(len(prepared[a]["dates"])),
                        index=prepared[a]["dates"]).reindex(common).values.astype(int)
           for a in assets}
    Y = {m: np.stack([prepared[a]["y"][m][loc[a]] for a in assets]) for m in model_names}
    RG = np.stack([prepared[a]["regime"][loc[a]] for a in assets])
    return model_names, assets, common, Y, RG


def _D_operator(Y: dict, RG: np.ndarray, model_names: list, assets: list,
                idx: np.ndarray, aggregation: str, min_obs: int) -> dict:
    """
    Unico operatore usato sia per la stima puntuale (idx = arange(n)) sia
    per ogni replica bootstrap (idx = date ricampionate): il pooling è
    identico per costruzione (commento K15).

    "portfolio" (principale): per modello m, regime g e data tau,
        P_{m,g,tau} = media dei y_{m,i,tau} sugli asset i con g_{i,tau} = g,
    definita nelle date in cui almeno un asset è nel regime g. Se gli asset
    sono in regimi diversi nella stessa data, ciascuno entra nella cella del
    proprio regime: la data contribuisce a più celle, con sottoinsiemi
    diversi di asset. S_{m,g} = SR_0 della serie P_{m,g,.} (un rendimento
    per data, quindi l'annualizzazione con sqrt(252) è coerente).
    Cella valida se ha almeno `min_obs` date.

    "per_asset" (controllo): D_i calcolata sul singolo asset; D = media delle
    D_i sugli asset le cui tre celle hanno almeno `min_obs` giorni.
    """
    rg = RG[:, idx]
    G, M = len(_D_REGIMES), len(model_names)
    if aggregation == "portfolio":
        S = np.zeros((M, G))
        counts = {}
        for gi, g in enumerate(_D_REGIMES):
            mask = rg == g
            k = mask.sum(axis=0)
            has = k > 0
            counts[g] = int(has.sum())
            for mi, m in enumerate(model_names):
                port = np.where(mask, Y[m][:, idx], 0.0).sum(axis=0)[has] / k[has]
                S[mi, gi] = metrics_econ.sharpe_ratio(port)
        valid = min(counts.values()) >= min_obs
        return {"D": _D_from_S(S) if valid else float("nan"), "S": S,
                "counts": counts, "valid": valid, "assets_used": list(assets)}

    if aggregation == "per_asset":
        D_i, used, S_all, counts_all = {}, [], {}, {}
        for ai, a in enumerate(assets):
            S = np.zeros((M, G))
            counts = {}
            for gi, g in enumerate(_D_REGIMES):
                mask = rg[ai] == g
                counts[g] = int(mask.sum())
                for mi, m in enumerate(model_names):
                    S[mi, gi] = metrics_econ.sharpe_ratio(Y[m][ai, idx][mask])
            S_all[a], counts_all[a] = S, counts
            if min(counts.values()) >= min_obs:
                D_i[a] = _D_from_S(S)
                used.append(a)
            else:
                D_i[a] = float("nan")
        valid = len(used) > 0
        D = float(np.mean([D_i[a] for a in used])) if valid else float("nan")
        return {"D": D, "D_i": D_i, "S_by_asset": S_all, "counts": counts_all,
                "valid": valid, "assets_used": used}

    raise ValueError(f"aggregation deve essere uno di {D_AGGREGATIONS}")


def _block_indices(rng, n: int, block_length: int) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]


def architecture_vs_regime_D(asset_data: dict,
                              aggregation: str = "portfolio",
                              block_length: int = 10,
                              n_boot: int = 2000,
                              min_obs_per_cell: int = 5,
                              ci: float = 0.95,
                              seed: int = 42) -> dict:
    """
    Statistica D per RQ3 (eq. 1.2): dispersione media tra regimi a modello
    fissato meno dispersione media tra modelli a regime fissato. È una
    diagnostica descrittiva di dispersione, non un'attribuzione causale.

    S_m,g è calcolato sulla serie NETTA y della strategia STATICA del
    modello m (già calcolata da run_backtest sull'intero percorso OOS; la
    restrizione ai giorni del regime non ricalcola costi). M = 12 (Random
    Walk esclusa), G = 3. L'aggregazione tra asset è quella di _D_operator.

    Bootstrap: moving block bootstrap sulle date comuni ai 4 asset, con gli
    stessi blocchi di date per tutti i modelli e gli asset; le etichette di
    regime viaggiano con le date. La regola di scarto (replica scartata se
    una cella ha meno di `min_obs_per_cell` date) fa parte dello stimatore:
    la frazione di repliche scartate è restituita e va riportata.

    asset_data: {asset_name: {"y": {model_name: net_pnl (T,)},
                               "dates": DatetimeIndex (T,),
                               "regimes": pd.Series}}
    """
    model_names, assets, common, Y, RG = _prepare_D_inputs(asset_data)
    n = len(common)

    point = _D_operator(Y, RG, model_names, assets, np.arange(n),
                        aggregation, min_obs_per_cell)

    rng = np.random.default_rng(seed)
    D_boot = []
    for _ in range(n_boot):
        idx = _block_indices(rng, n, block_length)
        res = _D_operator(Y, RG, model_names, assets, idx,
                          aggregation, min_obs_per_cell)
        if res["valid"]:
            D_boot.append(res["D"])
    D_boot = np.array(D_boot)
    n_used = len(D_boot)
    a_ = 1 - ci
    lo = round(float(np.percentile(D_boot, 100 * a_ / 2)), 6) if n_used else None
    hi = round(float(np.percentile(D_boot, 100 * (1 - a_ / 2))), 6) if n_used else None

    out = {
        "Aggregation":            aggregation,
        "D":                      round(point["D"], 6),
        "D CI Lower (95%)":       lo,
        "D CI Upper (95%)":       hi,
        "N Common Dates":         n,
        "Block Length":           block_length,
        "N Bootstrap Requested":  n_boot,
        "N Bootstrap Used":       n_used,
        "Frac. Discarded":        round(1 - n_used / n_boot, 4),
        "Min Obs per Cell":       min_obs_per_cell,
        "Assets Used":            ",".join(point["assets_used"]),
        "N Obs per Regime (point estimate)": point["counts"],
    }
    if aggregation == "portfolio":
        out["Sharpe by Model-Regime"] = pd.DataFrame(
            point["S"], index=model_names, columns=_D_REGIMES)
    else:
        out["D by Asset"] = point["D_i"]
        out["Sharpe by Model-Regime"] = pd.concat(
            {a: pd.DataFrame(point["S_by_asset"][a], index=model_names,
                             columns=_D_REGIMES) for a in assets},
            names=["Asset", "Model"])
    return out


def _two_way_shares(z: np.ndarray, model_id: np.ndarray, regime_id: np.ndarray,
                    M: int, G: int) -> dict:
    """
    Quote di varianza (somme dei quadrati di tipo II) di un modello a due
    fattori modello x regime con interazione, su osservazioni non
    bilanciate. Calcolo diretto con minimi quadrati (numpy).
    """
    zc = z - z.mean()
    sst = float(zc @ zc)

    def rss_groups(gid, K):
        s = np.bincount(gid, weights=z, minlength=K)
        c = np.bincount(gid, minlength=K)
        means = np.divide(s, c, out=np.zeros(K), where=c > 0)
        r = z - means[gid]
        return float(r @ r)

    rss_model = rss_groups(model_id, M)
    rss_regime = rss_groups(regime_id, G)
    rss_full = rss_groups(model_id * G + regime_id, M * G)
    X = np.zeros((len(z), M + G - 1))
    X[np.arange(len(z)), model_id] = 1.0
    nz = regime_id > 0
    X[np.where(nz)[0], M + regime_id[nz] - 1] = 1.0
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    r = z - X @ beta
    rss_add = float(r @ r)
    return {"Share Model": (rss_regime - rss_add) / sst,
            "Share Regime": (rss_model - rss_add) / sst,
            "Share Interaction": (rss_add - rss_full) / sst,
            "Share Residual": rss_full / sst}


def two_way_decomposition(asset_data: dict, block_length: int = 10,
                          n_boot: int = 2000, ci: float = 0.95,
                          seed: int = 42) -> pd.DataFrame:
    """
    Controllo per RQ3 suggerito dal relatore (commento K02): decomposizione
    a due fattori modello-regime sui rendimenti giornalieri.

    Osservazioni: z_{m,i,tau} = y_{m,i,tau} / s_{m,i}, rendimento netto
    statico standardizzato con la deviazione standard del modello m
    sull'asset i (così gli asset con volatilità diversa pesano allo stesso
    modo; scelta di questa implementazione). Ogni osservazione ha il regime
    del proprio asset in quella data. Si riportano le quote di varianza
    attribuite a modello, regime e interazione con un intervallo bootstrap
    sulle date (stessi blocchi per tutti i modelli e gli asset).

    Nessun p-value F: le osservazioni non sono indipendenti tra modelli
    valutati sulle stesse date.
    """
    model_names, assets, common, Y, RG = _prepare_D_inputs(asset_data)
    n, M, G = len(common), len(model_names), len(_D_REGIMES)
    g_index = {g: k for k, g in enumerate(_D_REGIMES)}

    def shares(idx):
        z_parts, m_parts, g_parts = [], [], []
        rg = np.vectorize(g_index.get)(RG[:, idx])
        for mi, m in enumerate(model_names):
            y = Y[m][:, idx]
            sd = y.std(axis=1, ddof=1, keepdims=True)
            sd[sd <= 1e-12] = 1.0
            z_parts.append((y / sd).ravel())
            m_parts.append(np.full(y.size, mi))
            g_parts.append(rg.ravel())
        return _two_way_shares(np.concatenate(z_parts), np.concatenate(m_parts),
                               np.concatenate(g_parts), M, G)

    point = shares(np.arange(n))
    rng = np.random.default_rng(seed)
    boot = [shares(_block_indices(rng, n, block_length)) for _ in range(n_boot)]
    a_ = 1 - ci
    rows = []
    for k, v in point.items():
        b = np.array([x[k] for x in boot])
        rows.append({"Component": k, "Share": round(v, 6),
                     "CI Lower (95%)": round(float(np.percentile(b, 100 * a_ / 2)), 6),
                     "CI Upper (95%)": round(float(np.percentile(b, 100 * (1 - a_ / 2))), 6)})
    return pd.DataFrame(rows).set_index("Component")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark della RCS — commento K14 del relatore
# ─────────────────────────────────────────────────────────────────────────────

def exposure_benchmarks(y_true: np.ndarray, dates: pd.DatetimeIndex,
                        regimes: pd.Series, asset_name: str,
                        rcs_results: dict, n_rand: int = 1000,
                        seed: int = 42, cost_per_side: float = 0.0001) -> tuple:
    """
    Separa il valore del forecast dal valore del filtro di volatilità:
    la stessa regola di esposizione a_t della RCS è applicata a
      (i)  una posizione long semplice:  w_t = a_t;
      (ii) un segnale non informativo:   w_t = a_t * e_t, e_t = ±1 casuale
           (n_rand repliche, seed fisso).
    Per ogni modello si riporta p_emp = (1 + #{SR_rand >= SR_RCS}) /
    (n_rand + 1) e se la sua RCS coincide con il long filtrato (posizione
    mai negativa quando esposta).

    rcs_results: {model_label: dict di run_backtest della RCS}
    Ritorna (summary_dict, per_model_DataFrame, a_t).
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    n = len(y_true)
    ones = np.ones(n)
    a = regime_conditional_strategy(ones, dates, regimes)

    long_passive = run_backtest(y_true, ones, dates=dates, asset_name=asset_name,
                                cost_per_side=cost_per_side, position=ones)
    long_filt = run_backtest(y_true, ones, dates=dates, asset_name=asset_name,
                             cost_per_side=cost_per_side, position=a)
    rng = np.random.default_rng(seed)
    sr_rand = np.array([
        run_backtest(y_true, e, dates=dates, asset_name=asset_name,
                     cost_per_side=cost_per_side, position=e * a)["Sharpe Ratio"]
        for e in rng.choice([-1.0, 1.0], size=(n_rand, n))])

    summary = {
        "Sharpe Long Passive":           long_passive["Sharpe Ratio"],
        "Sharpe Long Filtered":          long_filt["Sharpe Ratio"],
        "Calmar Long Filtered":          long_filt["Calmar Ratio"],
        "Sharpe Random Filtered Mean":   round(float(sr_rand.mean()), 4),
        "Sharpe Random Filtered P95":    round(float(np.percentile(sr_rand, 95)), 4),
        "Frac. Days Exposed":            round(float((a > 0).mean()), 4),
        "N Random":                      n_rand,
    }
    rows = []
    for label, bt in rcs_results.items():
        pos = np.asarray(bt["Position"], dtype=float)
        sr = bt["Sharpe Ratio"]
        rows.append({
            "Model": label,
            "Sharpe RCS": sr,
            "p_emp vs Random Filtered": round(float((1 + np.sum(sr_rand >= sr)) / (n_rand + 1)), 4),
            "RCS minus Long Filtered (Sharpe)": round(sr - long_filt["Sharpe Ratio"], 4),
            "Identical to Long Filtered": bool(np.all(pos[a > 0] > 0) and np.all(pos[a == 0] == 0)),
        })
    return summary, pd.DataFrame(rows).set_index("Model"), a


def performance_by_regime(results: dict, dates: pd.DatetimeIndex,
                          regimes: pd.Series, strategy: str) -> pd.DataFrame:
    """
    Per ogni modello e regime (etichetta osservabile al momento della
    decisione): numero di giorni, giorni attivi, Sharpe della serie netta
    ristretta ai giorni del regime, rendimento medio giornaliero e DA sui
    giorni attivi (commento K14: performance e directional accuracy
    condizionate al regime).
    """
    reg = regimes.reindex(pd.DatetimeIndex(dates)).fillna("normal").values
    rows = []
    for label, bt in results.items():
        y = np.asarray(bt["Net PnL"]); hit = np.asarray(bt["Hit"])
        act = np.asarray(bt["Active"]) == 1
        r_ = reg[-len(y):]
        for g in _D_REGIMES:
            m = r_ == g
            ma = m & act
            rows.append({"Model": label, "Strategy": strategy, "Regime": g,
                         "N Days": int(m.sum()), "N Active": int(ma.sum()),
                         "Sharpe": round(metrics_econ.sharpe_ratio(y[m]), 4) if m.sum() > 1 else float("nan"),
                         "Mean Daily Return (%)": round(float(y[m].mean() * 100), 5) if m.sum() else float("nan"),
                         "Directional Acc.": round(float(hit[ma].mean() * 100), 2) if ma.sum() else float("nan")})
    return pd.DataFrame(rows)


def regime_conditional_strategy(y_pred: np.ndarray,
                                  dates: pd.DatetimeIndex,
                                  regimes: pd.Series,
                                  stable_scale:   float = 1.0,
                                  normal_scale:   float = 0.5,
                                  volatile_scale: float = 0.0) -> np.ndarray:
    """
    Regime-Conditional Strategy: w_t = a_t * sign(r_hat), con la regola di
    esposizione
        stable   -> a_t = 1
        normal   -> a_t = 1/2
        volatile -> a_t = 0
    e `regimes` già traslato al giorno precedente (etichetta osservabile al
    momento della decisione). La regola non presuppone che il segnale sia
    migliore nei regimi stabili: il contributo del filtro rispetto a quello
    del forecast è misurato da exposure_benchmarks (commento K14).
    """
    y_pred         = np.array(y_pred).flatten()
    regime_aligned = regimes.reindex(dates).fillna("normal").values
    scale = np.where(regime_aligned == "stable",   stable_scale,
            np.where(regime_aligned == "normal",   normal_scale,
                                                   volatile_scale))
    signal = np.where(y_pred > 0, 1.0, np.where(y_pred < 0, -1.0, 0.0))
    return signal * scale


def backtest_summary_table(bt_results: dict) -> pd.DataFrame:
    """Tabella riassuntiva per la tesi (Cap. 4)."""
    scalar_keys = [
        "Execution",
        "Annual Return (%)", "Mean Daily Return x252 (%)", "Annual Vol (%)",
        "Sharpe Ratio", "Max Drawdown (%)", "Calmar Ratio",
        "Directional Acc.", "Win Rate (%)", "Turnover", "N Roll Days", "N Days",
        "Sharpe CI Lower (95%)", "Sharpe CI Upper (95%)",
        "Max Drawdown CI Lower (95%)", "Max Drawdown CI Upper (95%)",
        "Calmar CI Lower (95%)", "Calmar CI Upper (95%)",
        "DA CI Lower (95%)", "DA CI Upper (95%)",
        "N Active", "DA t (HAC)", "DA p-value (HAC)", "DA HAC Lag",
        "DA p-value (Boot H0)", "DA p Holm", "DA Sig. Holm (5%)",
        "Block Length", "N Bootstrap",
        "Alpha (daily)", "Beta", "Alpha t-stat (HAC)", "Alpha p-value",
        "Alpha p Holm", "Alpha Sig. Holm (5%)",
        "Alpha vs Long Filtered (daily)", "Alpha vs Long Filtered p-value",
    ]
    rows = []
    for model_name, bt in bt_results.items():
        row = {"Model": model_name}
        for k in scalar_keys:
            row[k] = bt.get(k, None)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("Sharpe Ratio", ascending=False)
