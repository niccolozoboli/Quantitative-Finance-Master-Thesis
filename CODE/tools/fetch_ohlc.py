"""
tools/fetch_ohlc.py
-------------------
Scarica UNA volta gli snapshot Open/High/Low/Close dei quattro futures in
CODE/data/<ticker>_ohlc.csv (commento K13 del relatore: esecuzione al primo
prezzo disponibile dopo la formazione del segnale).

Da eseguire su Colab (questo ambiente non raggiunge Yahoo Finance), poi
caricare i file nel branch PRIMA dell'esecuzione completa, così dati,
codice e risultati stanno nello stesso commit. Lo snapshot dei Close
(CODE/data/<ticker>.csv) non viene toccato.

    cd CODE && python tools/fetch_ohlc.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (METALS, download_ohlc, _ohlc_path,
                               load_and_preprocess, validate_ohlc, OHLC_END)

START = "2010-01-01"

if __name__ == "__main__":
    for name, ticker in METALS.items():
        path = _ohlc_path(ticker)
        if os.path.exists(path):
            print(f"{ticker}: {path} esiste già — non riscarico")
            continue
        df = download_ohlc(ticker, start=START, end=OHLC_END)
        df.to_csv(path)
        close_df = load_and_preprocess(name, ticker)
        oos = close_df.index[-210:]
        print(f"{ticker}: {len(df)} righe salvate in {path} — controlli:",
              validate_ohlc(df, close_df, oos))
