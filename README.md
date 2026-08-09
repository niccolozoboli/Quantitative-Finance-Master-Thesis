# Metal Commodity Return Forecasting

Tesi magistrale in Data Science — confronto sistematico tra modelli
statistici classici, Machine Learning e Deep Learning per la previsione
dei log-return giornalieri di quattro commodity metalliche (Gold, Silver,
Copper, Platinum), con valutazione sia statistica sia di valore economico
(backtesting).

## Cosa fa il progetto

Per ciascun asset, la pipeline:

1. scarica e pre-processa i prezzi (log-return, test di stazionarietà ADF)
2. costruisce feature specifiche per ML/DL (lag, rolling stats, momentum,
   mean-reversion, regime di volatilità, spread cross-asset, encoding
   calendariale ciclico)
3. allena e valuta 13 modelli — Random Walk, ARIMA, GARCH, Decision Tree,
   Random Forest, Gradient Boosting, XGBoost, SVR, LSTM, GRU, BiLSTM,
   Transformer, TCN, CNN-LSTM — su 10 fold di walk-forward validation
   cronologica (21 giorni di test per fold, 210 giorni out-of-sample
   totali per asset)
4. confronta i modelli su RMSE/MAE, test di Diebold-Mariano contro il
   Random Walk, e performance per regime di mercato (stabile/normale/
   volatile)
5. traduce le previsioni in una strategia di trading long/short con
   costi di transazione, misurando Sharpe Ratio, Max Drawdown, Calmar
   Ratio e Directional Accuracy — inclusa una Regime-Conditional Strategy
   che modula l'esposizione in base al regime di volatilità

## Installazione

```
pip install -r requirements.txt
```

## Esecuzione

```
cd CODE
python3 main.py
```

Richiede una connessione internet (i dati vengono scaricati da Yahoo
Finance via `yfinance`). Non richiede GPU: i modelli Deep Learning
girano anche su CPU, con tempi di training più lunghi.

## Struttura

```
CODE/
├── main.py                    punto di ingresso, orchestrazione pipeline
├── descriptive_stats.py       statistiche descrittive e correlazioni (Cap. 3)
├── src/
│   ├── preprocessing.py       download dati, calcolo log-return
│   ├── feature_engine.py      feature engineering per ML e DL
│   ├── walk_forward.py        split walk-forward cronologico
│   ├── regime.py              classificazione regime di volatilità
│   ├── statistical_tests.py   test ADF, Diebold-Mariano
│   ├── backtesting.py         strategia di trading e metriche economiche
│   ├── visualization.py       tutti i grafici della tesi
│   ├── utils.py                metriche di errore, seed, salvataggio risultati
│   └── models/                 implementazione dei 13 modelli
└── results/                    CSV e grafici generati dalla pipeline
    ├── metrics.csv
    ├── backtesting_all_assets.csv
    └── plots/
```

## Metodologia (sintesi)

Target: log-return giornalieri (non prezzi grezzi), verificati stazionari
via test ADF. Feature normalizzate con z-score stimato solo sul training
di ciascun fold (nessun leakage). Split walk-forward cronologico a 10
fold, finestra di training espandibile. Seed fissato a 42 (Python,
NumPy, TensorFlow) per riproducibilità. Costo di transazione di 1 basis
point per trade nel backtesting. Dettagli completi nel Capitolo 3 della
tesi.
