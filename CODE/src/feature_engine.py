"""
feature_engine.py
-----------------
Feature construction for ML and DL models.
VERSIONE 2.0 — Quantitative Finance oriented.

Cambiamenti rispetto alla versione 1.0:
- Feature set esteso con segnali momentum, mean reversion, volatility regime
- Cross-asset features (Gold-Silver spread, Gold-Copper ratio)
- Input multivariato per i modelli DL: shape (seq_len, N_FEATURES) invece di (seq_len, 1)
- Tutti i segnali costruiti con shift(1) o più → zero look-ahead garantito

Design principles invariati:
- Features Z-scored fit on train only → no leakage
- Target = log_return in ORIGINAL scale (mai Z-scored)
- Compatibile con il walk-forward framework esistente

Riferimenti:
- Momentum: Jegadeesh & Titman (1993), Asness et al. (2013)
- Mean reversion: De Bondt & Thaler (1985)
- Vol ratio come regime proxy: Ang & Timmermann (2012)
- Cross-asset: già documentato nella tesi (correlazioni Gold-Silver 0.79)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE FEATURE
# ─────────────────────────────────────────────────────────────────────────────

# Lag standard (invariati dalla v1)
DEFAULT_LAGS = [1, 2, 3, 5, 10]

# Rolling windows (invariate dalla v1)
DEFAULT_ROLLING = [5, 10, 21]   # aggiunto 21 (1 mese trading)

# Sequenza per modelli DL
SEQ_LEN = 20   # aumentato da 10 a 20 — più contesto per attention/LSTM


# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE 1 — FEATURE ENGINEERING ML
# ─────────────────────────────────────────────────────────────────────────────

def build_ml_features(df: pd.DataFrame,
                       cross_asset_dfs: dict = None,
                       lags: list = DEFAULT_LAGS,
                       rolling_windows: list = DEFAULT_ROLLING) -> pd.DataFrame:
    """
    Costruisce la feature matrix per i modelli ML (RF, XGBoost, GBM, SVR, DT).

    FEATURE GROUPS:
    ───────────────
    Gruppo 1 — Lag features (invariato dalla v1)
        lag_1, lag_2, lag_3, lag_5, lag_10
        Catturano autocorrelazione a breve termine e momentum a brevissimo.

    Gruppo 2 — Rolling statistics (esteso dalla v1)
        roll_mean_5, roll_mean_10, roll_mean_21
        roll_std_5,  roll_std_10,  roll_std_21
        Media e volatilità rolling su finestre multiple.

    Gruppo 3 — Momentum signals (NUOVO)
        mom_5:  rendimento cumulativo ultimi 5gg  (momentum settimanale)
        mom_21: rendimento cumulativo ultimi 21gg (momentum mensile)
        mom_63: rendimento cumulativo ultimi 63gg (momentum trimestrale)
        Fonte: letteratura momentum trading (Jegadeesh & Titman 1993).
        In commodity, momentum a 1-3 mesi è ampiamente documentato.

    Gruppo 4 — Mean reversion signal (NUOVO)
        z_score_21: z-score del rendimento corrente rispetto alla media 21gg
        Valori estremi (>2 o <-2) segnalano potenziale mean reversion.
        Utile nei regimi stabili dove il prezzo tende a tornare alla media.

    Gruppo 5 — Volatility regime proxy (NUOVO)
        vol_ratio: rolling_std_5 / rolling_std_21
        >1 → volatilità in espansione (regime volatile in arrivo)
        <1 → volatilità in contrazione (regime stabile)
        Permette ai modelli ML di condizionarsi sul regime SENZA usare
        la classificazione regime (che usa dati futuri per i percentili).

    Gruppo 6 — Cross-asset features (NUOVO, opzionale)
        gs_spread:    Gold - Silver log-return (divergenza metalli preziosi)
        gc_spread:    Gold - Copper log-return (precious vs industrial)
        Richiedono i DataFrame degli altri asset passati come cross_asset_dfs.
        Se non disponibili, queste feature vengono saltate silenziosamente.

    Gruppo 7 — Calendar features (NUOVO)
        dow_sin, dow_cos: giorno della settimana codificato con sin/cos
        Cattura effetti calendario (Monday effect, weekend effect)
        documentati nelle commodity futures.

    Parameters:
        df              : DataFrame con colonne 'log_return' (e 'Close')
        cross_asset_dfs : dict {'Gold': df_gold, 'Silver': df_silver, ...}
                          Gli altri asset per i cross-asset spread.
                          Passare None per saltare questa feature group.
        lags            : lista di lag per le lag features
        rolling_windows : lista di finestre per rolling statistics

    Returns:
        DataFrame con tutte le feature e colonna 'target' = next-day log_return
        Righe con NaN droppate.
    """
    df = df.copy()
    r = df["log_return"]

    # ── Gruppo 1: Lag features ────────────────────────────────────────────────
    for lag in lags:
        df[f"lag_{lag}"] = r.shift(lag)

    # ── Gruppo 2: Rolling statistics ──────────────────────────────────────────
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = r.shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = r.shift(1).rolling(w).std()

    # ── Gruppo 3: Momentum signals ────────────────────────────────────────────
    # Somma dei rendimenti nelle ultime N giornate (esclusa quella corrente)
    for horizon in [5, 21, 63]:
        df[f"mom_{horizon}"] = r.shift(1).rolling(horizon).sum()

    # ── Gruppo 4: Mean reversion (z-score 21gg) ───────────────────────────────
    roll_mean_21 = r.shift(1).rolling(21).mean()
    roll_std_21  = r.shift(1).rolling(21).std()
    df["z_score_21"] = (r.shift(1) - roll_mean_21) / (roll_std_21 + 1e-8)

    # ── Gruppo 5: Volatility ratio (regime proxy real-time) ───────────────────
    vol_short = r.shift(1).rolling(5).std()
    vol_long  = r.shift(1).rolling(21).std()
    df["vol_ratio"] = vol_short / (vol_long + 1e-8)

    # ── Gruppo 6: Cross-asset features (se disponibili) ───────────────────────
    if cross_asset_dfs is not None:
        # Gold-Silver spread: divergenza tra i due metalli preziosi
        # Quando il Gold sale e Silver scende (o viceversa), segnala
        # disaccoppiamento che spesso precede un riallineamento
        if "Gold" in cross_asset_dfs and "Silver" in cross_asset_dfs:
            gold_r   = cross_asset_dfs["Gold"]["log_return"].reindex(df.index)
            silver_r = cross_asset_dfs["Silver"]["log_return"].reindex(df.index)
            df["gs_spread"] = gold_r.shift(1) - silver_r.shift(1)

        # Gold-Copper spread: precious vs industrial
        # Proxy per risk-on/risk-off: quando Copper sale più di Gold,
        # il mercato è in modalità risk-on (crescita economica attesa)
        if "Gold" in cross_asset_dfs and "Copper" in cross_asset_dfs:
            gold_r   = cross_asset_dfs["Gold"]["log_return"].reindex(df.index)
            copper_r = cross_asset_dfs["Copper"]["log_return"].reindex(df.index)
            df["gc_spread"] = gold_r.shift(1) - copper_r.shift(1)

    # ── Gruppo 7: Calendar features ───────────────────────────────────────────
    # Encoding ciclico: evita il salto discreto tra venerdì (4) e lunedì (0)
    dow = df.index.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # ── Target: next-day log_return ───────────────────────────────────────────
    df["target"] = r.shift(-1)
    df.dropna(inplace=True)

    return df


def get_feature_cols(cross_asset_dfs: dict = None,
                      lags: list = DEFAULT_LAGS,
                      rolling_windows: list = DEFAULT_ROLLING) -> list:
    """
    Restituisce la lista ordinata delle colonne feature.
    Deve essere chiamata con gli stessi argomenti di build_ml_features.
    """
    cols = []

    # Gruppo 1: lag
    cols += [f"lag_{l}" for l in lags]

    # Gruppo 2: rolling
    cols += [f"roll_mean_{w}" for w in rolling_windows]
    cols += [f"roll_std_{w}"  for w in rolling_windows]

    # Gruppo 3: momentum
    cols += ["mom_5", "mom_21", "mom_63"]

    # Gruppo 4: mean reversion
    cols += ["z_score_21"]

    # Gruppo 5: vol ratio
    cols += ["vol_ratio"]

    # Gruppo 6: cross-asset (solo se passati)
    if cross_asset_dfs is not None:
        if "Gold" in cross_asset_dfs and "Silver" in cross_asset_dfs:
            cols += ["gs_spread"]
        if "Gold" in cross_asset_dfs and "Copper" in cross_asset_dfs:
            cols += ["gc_spread"]

    # Gruppo 7: calendar
    cols += ["dow_sin", "dow_cos"]

    return cols


def prepare_ml_fold(df: pd.DataFrame,
                     train_idx,
                     test_idx,
                     cross_asset_dfs: dict = None,
                     lags: list = DEFAULT_LAGS,
                     rolling_windows: list = DEFAULT_ROLLING):
    """
    Prepara train/test split per un singolo fold walk-forward.

    - Features Z-scored: scaler fit SOLO su train → no leakage
    - Target in scala originale (log_return non scalato)
    - cross_asset_dfs: dict con DataFrame degli altri asset (opzionale)

    Returns:
        X_train, X_test, y_train, y_test, dates, feature_names
    """
    df_feat   = build_ml_features(df, cross_asset_dfs=cross_asset_dfs,
                                   lags=lags, rolling_windows=rolling_windows)
    feat_cols = get_feature_cols(cross_asset_dfs=cross_asset_dfs,
                                  lags=lags, rolling_windows=rolling_windows)

    # Filtra solo le colonne effettivamente presenti nel DataFrame
    # (sicurezza se alcuni cross-asset non sono disponibili)
    feat_cols = [c for c in feat_cols if c in df_feat.columns]

    # Allinea indici tra df originale e df con feature
    valid_train = df_feat.index.isin(df.index[train_idx])
    valid_test  = df_feat.index.isin(df.index[test_idx])

    X_train = df_feat.loc[valid_train, feat_cols].values
    y_train = df_feat.loc[valid_train, "target"].values
    X_test  = df_feat.loc[valid_test,  feat_cols].values
    y_test  = df_feat.loc[valid_test,  "target"].values
    dates   = df_feat.loc[valid_test].index

    # Z-score: fit SOLO su train, transform su entrambi
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, dates, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE 2 — FEATURE ENGINEERING DL (INPUT MULTIVARIATO)
# ─────────────────────────────────────────────────────────────────────────────

def build_dl_features(df: pd.DataFrame,
                       cross_asset_dfs: dict = None) -> pd.DataFrame:
    """
    Costruisce il DataFrame di feature per i modelli DL.

    Subset delle ML features ottimizzato per sequenze temporali:
    - Non include lag/rolling stats espliciti (Gruppi 1-2): il modello DL
      riceve già la sequenza grezza di SEQ_LEN=20 giorni di log_return,
      quindi può in linea di principio apprendere da solo pattern
      equivalenti (medie mobili, autocorrelazione a breve termine)
      ENTRO quella finestra — scelta di design, non un'omissione.
      NB: questo argomento non copre segnali con lookback > SEQ_LEN
      (es. mom_63), che restano fuori dalla finestra e vanno quindi
      inclusi esplicitamente come feature (vedi sotto).
    - Include momentum (incluso mom_63, esplicito perché fuori dalla
      finestra di sequenza), z-score, vol_ratio, cross-asset, calendar
    - Shape finale: (T, N_DL_FEATURES)

    La colonna 'log_return' è sempre inclusa come prima feature
    (è il segnale primario della sequenza).
    """
    df = df.copy()
    r  = df["log_return"]

    features = pd.DataFrame(index=df.index)

    # Feature 1: log_return (segnale primario)
    features["log_return"] = r

    # Feature 2-4: Momentum multi-scala
    # mom_63 esplicito: con SEQ_LEN=20 il modello non può "vedere" da solo
    # un segnale a 63 giorni, va quindi fornito come feature pre-calcolata
    # (a differenza di mom_5/mom_21 che ricadono nella finestra di sequenza).
    features["mom_5"]  = r.shift(1).rolling(5).sum()
    features["mom_21"] = r.shift(1).rolling(21).sum()
    features["mom_63"] = r.shift(1).rolling(63).sum()

    # Feature 5: Mean reversion z-score
    mu  = r.shift(1).rolling(21).mean()
    sig = r.shift(1).rolling(21).std()
    features["z_score_21"] = (r.shift(1) - mu) / (sig + 1e-8)

    # Feature 6: Volatility ratio (regime proxy)
    vol_s = r.shift(1).rolling(5).std()
    vol_l = r.shift(1).rolling(21).std()
    features["vol_ratio"] = vol_s / (vol_l + 1e-8)

    # Feature 7-8: Cross-asset (se disponibili)
    if cross_asset_dfs is not None:
        if "Gold" in cross_asset_dfs and "Silver" in cross_asset_dfs:
            gold_r   = cross_asset_dfs["Gold"]["log_return"].reindex(df.index)
            silver_r = cross_asset_dfs["Silver"]["log_return"].reindex(df.index)
            features["gs_spread"] = gold_r.shift(1) - silver_r.shift(1)

        if "Gold" in cross_asset_dfs and "Copper" in cross_asset_dfs:
            gold_r   = cross_asset_dfs["Gold"]["log_return"].reindex(df.index)
            copper_r = cross_asset_dfs["Copper"]["log_return"].reindex(df.index)
            features["gc_spread"] = gold_r.shift(1) - copper_r.shift(1)

    # Feature 9-10: Calendar encoding
    dow = df.index.dayofweek.astype(float)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    features.dropna(inplace=True)
    return features


def prepare_dl_fold(df: pd.DataFrame,
                     train_idx,
                     test_idx,
                     cross_asset_dfs: dict = None,
                     seq_len: int = SEQ_LEN):
    """
    Prepara sequenze sliding-window MULTIVARIATE per i modelli DL.

    DIFFERENZA CHIAVE rispetto alla v1:
    - v1: input shape (seq_len, 1)  — solo log_return
    - v2: input shape (seq_len, N)  — log_return + momentum + z_score + ...

    Questo permette ai modelli LSTM/GRU/Transformer di condizionarsi
    su informazioni finanziariamente rilevanti ad ogni step temporale,
    non solo sul valore grezzo del rendimento.

    Z-scaler: fit su train ONLY, colonna per colonna → no leakage.
    Target: log_return next-day in scala originale (mai scalato).

    Returns:
        X_train : (n_train_samples, seq_len, n_features)
        X_test  : (n_test_samples,  seq_len, n_features)
        y_train : (n_train_samples,)
        y_test  : (n_test_samples,)
        dates   : DatetimeIndex dei giorni di test
        n_features : int — numero di feature per step (utile per definire i modelli)
    """
    # Costruisci feature matrix completa
    feat_df = build_dl_features(df, cross_asset_dfs=cross_asset_dfs)

    # Allinea con gli indici originali del fold
    train_dates = df.index[train_idx]
    test_dates  = df.index[test_idx]

    train_feat = feat_df[feat_df.index.isin(train_dates)]
    test_feat  = feat_df[feat_df.index.isin(test_dates)]

    feat_cols = train_feat.columns.tolist()
    n_features = len(feat_cols)

    # Z-score: fit su train, colonna per colonna
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat[feat_cols].values)
    test_scaled  = scaler.transform(test_feat[feat_cols].values)

    # Target: log_return in scala originale (prima colonna = log_return)
    # Recuperato dal df originale per sicurezza
    train_targets = df["log_return"].reindex(train_feat.index).values
    test_targets  = df["log_return"].reindex(test_feat.index).values

    # Sliding window sulle feature scalate
    # Concatena train + test per costruire sequenze di contesto
    full_scaled  = np.vstack([train_scaled, test_scaled])
    full_targets = np.concatenate([train_targets, test_targets])
    n_train = len(train_scaled)

    X_train, y_train = [], []
    X_test,  y_test  = [], []

    for i in range(seq_len, len(full_scaled)):
        seq = full_scaled[i - seq_len : i, :]   # shape: (seq_len, n_features)
        tgt = full_targets[i]                    # target: next-day return

        if i < n_train:
            X_train.append(seq)
            y_train.append(tgt)
        else:
            X_test.append(seq)
            y_test.append(tgt)

    X_train = np.array(X_train)   # (n_train, seq_len, n_features)
    y_train = np.array(y_train)
    X_test  = np.array(X_test)    # (n_test,  seq_len, n_features)
    y_test  = np.array(y_test)

    # Date per le previsioni di test
    dates = test_feat.index[seq_len - len(y_test) + len(y_test) - len(y_test):]
    # Gestione robusta: allinea le date all'effettivo numero di campioni di test
    dates = test_feat.index[len(test_feat) - len(y_test):]

    return X_train, X_test, y_train, y_test, dates, n_features


# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE 3 — UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def get_n_features(df: pd.DataFrame,
                    cross_asset_dfs: dict = None) -> int:
    """
    Calcola il numero di feature DL senza costruire il dataset completo.
    Utile per definire l'input shape dei modelli prima del training.
    """
    feat_df = build_dl_features(df, cross_asset_dfs=cross_asset_dfs)
    return len(feat_df.columns)


def feature_summary(df: pd.DataFrame,
                     cross_asset_dfs: dict = None) -> pd.DataFrame:
    """
    Stampa un summary delle feature costruite con statistiche descrittive.
    Utile per debugging e per la sezione 3.2 della tesi.
    """
    feat_df = build_dl_features(df, cross_asset_dfs=cross_asset_dfs)
    summary = feat_df.describe().T
    summary["missing"] = feat_df.isnull().sum()
    return summary