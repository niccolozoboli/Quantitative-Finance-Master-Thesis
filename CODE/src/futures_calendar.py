"""
futures_calendar.py
--------------------
Calendario di roll per le serie futures continue Yahoo Finance
(GC=F, SI=F, HG=F, PL=F) — A4/A5 del report del relatore (7 settembre
2026).

CONTESTO E LIMITE DICHIARATO
-----------------------------
GC=F/SI=F/HG=F/PL=F su Yahoo Finance sono serie continue front-month NON
back-adjusted (auto_adjust=True non fa nulla di rilevante per i futures:
niente dividendi/split). Yahoo non pubblica la propria metodologia di
switch del contratto sottostante. La ricostruzione da singoli contratti
(Opzione 1, valutata e scartata — vedi discussione A4-A5) non è
praticabile con i dati liberamente disponibili per l'intero periodo
2010-2026: Yahoo non ha storico dei contratti scaduti prima di ~2020
(verificato: GCZ25.CMX disponibile dal 2020, GCZ12.CMX/GCZ10.CMX assenti).

Si adotta quindi l'Opzione 2: la serie GC=F/SI=F/HG=F/PL=F viene usata
così com'è, con un CALENDARIO DI ROLL sovrapposto, costruito ex ante dalle
specifiche ufficiali CME/COMEX (non dalle date di splice reali di Yahoo,
che restano sconosciute). Il costo di roll (2c|w_t|) viene applicato sulle
date di questo calendario come approssimazione dichiarata del costo reale
di rollover di una posizione fisica sul front contract — non una
ricostruzione esatta di quale contratto Yahoo abbia quotato giorno per
giorno.

FONTI E REGOLA
---------------
Mesi di consegna (fonte: specifiche ufficiali CME Group, "Listed
Contracts"):
    Gold (GC):     Feb, Apr, Jun, Aug, Oct, Dec
    Silver (SI):   Jan, Mar, May, Jul, Sep, Dec
    Copper (HG):   Mar, May, Jul, Sep, Dec
    Platinum (PL): Jan, Apr, Jul, Oct

Silver — testo CME (fact card): "Trading is conducted for delivery during
the current calendar month; the next two calendar months; any January,
March, May, and September falling within a 23-month period; and any July
and December falling within a 60-month period."
    https://www.cmegroup.com/trading/metals/files/fact-card-silver-futures-options.pdf

Copper — testo CME (rulebook COMEX Ch.111, §111102): "The current
calendar month, the next 23 calendar months, and any March, May, July,
September, and December falling within a 60-month period."
    https://www.cmegroup.com/rulebook/COMEX/1a/111.pdf

Gold — mesi di consegna attivi Feb/Apr/Jun/Aug/Oct/Dec, Last Trading Day =
terzultimo giorno lavorativo del mese di consegna, periodo di consegna dal
primo giorno lavorativo del mese di consegna:
    https://www.cmegroup.com/trading/metals/files/fact-card-gold-futures-options.pdf

Platinum — mesi di consegna attivi Gen/Apr/Lug/Ott (+ mese corrente e
successivi 2):
    https://www.cmegroup.com/markets/metals/precious/platinum.contractSpecs.html

REGOLA DI ROLL (indicata esplicitamente dall'utente, sessione 2026-09-27):
il periodo di consegna inizia il primo giorno lavorativo del mese di
consegna (regola COMEX §111102, generalizzata alle 4 serie per coerenza —
stessa convenzione di consegna fisica COMEX/NYMEX). Il First Notice Day
(FND) è quindi l'ultimo giorno lavorativo del mese PRIMA del mese di
consegna. Il roll avviene alla chiusura del giorno lavorativo che precede
il FND, cioè:

    roll_date = (ultimo giorno lavorativo del mese prima della consegna) - 1 giorno lavorativo

Nota: i giorni lavorativi sono calcolati Lun-Ven (numpy busday, nessun
calendario festività CME) — ulteriore approssimazione dichiarata, che può
spostare una roll_date di 1-2 giorni rispetto al calendario reale CME.
"""

import numpy as np
import pandas as pd

DELIVERY_MONTHS = {
    "Gold":     [2, 4, 6, 8, 10, 12],
    "Silver":   [1, 3, 5, 7, 9, 12],
    "Copper":   [3, 5, 7, 9, 12],
    "Platinum": [1, 4, 7, 10],
}


def _last_business_day_of_month(year: int, month: int) -> pd.Timestamp:
    period_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return (period_end - pd.offsets.BDay(0)) if period_end.weekday() < 5 \
        else period_end - pd.offsets.BDay(1)


def _first_notice_day(delivery_year: int, delivery_month: int) -> pd.Timestamp:
    """FND = ultimo giorno lavorativo del mese prima del mese di consegna."""
    prev_month = delivery_month - 1
    prev_year = delivery_year
    if prev_month == 0:
        prev_month = 12
        prev_year -= 1
    return _last_business_day_of_month(prev_year, prev_month)


def get_roll_dates(asset_name: str, start: str, end: str) -> pd.DatetimeIndex:
    """
    Calendario di roll ex ante per `asset_name` nell'intervallo [start, end].
    roll_date = FND - 1 giorno lavorativo, per ogni mese di consegna del
    ciclo dell'asset che cade in [start, end] (con un anno di margine su
    entrambi i lati per catturare consegne a cavallo di fine periodo).
    """
    if asset_name not in DELIVERY_MONTHS:
        raise ValueError(f"Asset sconosciuto: {asset_name}")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    years = range(start_ts.year - 1, end_ts.year + 2)

    roll_dates = []
    for year in years:
        for month in DELIVERY_MONTHS[asset_name]:
            fnd = _first_notice_day(year, month)
            roll_date = fnd - pd.offsets.BDay(1)
            if start_ts <= roll_date <= end_ts:
                roll_dates.append(roll_date)

    return pd.DatetimeIndex(sorted(set(roll_dates)))


def roll_mask(asset_name: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Maschera booleana 1{t in roll dates}, allineata a `dates` (le date
    effettivamente presenti nella serie di trading, non il calendario
    lavorativo puro): ogni roll_date viene mappata alla data di trading più
    vicina (all'indietro, cioè l'ultimo giorno di mercato aperto a quella
    data o prima) presente in `dates`, coerente con "roll alla chiusura del
    giorno lavorativo che precede il FND".
    """
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return np.zeros(0, dtype=bool)

    roll_dates = get_roll_dates(asset_name, dates.min(), dates.max())
    mask = np.zeros(len(dates), dtype=bool)
    if len(roll_dates) == 0:
        return mask

    sorted_dates = dates.sort_values()
    pos = sorted_dates.searchsorted(roll_dates, side="right") - 1
    valid = pos >= 0
    matched_dates = sorted_dates[pos[valid]]
    mask |= dates.isin(matched_dates)
    return mask
