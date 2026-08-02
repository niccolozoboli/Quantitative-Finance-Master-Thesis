# Progetto: Tesi Magistrale — Metal Commodity Return Forecasting

## Regola più importante
Questo codice ha già prodotto i risultati riportati nella tesi di laurea, 
già scritta, con numeri specifici citati in Abstract, Capitolo 4 e 5. 
Qualsiasi modifica che alteri la logica di calcolo di feature, split, 
training o metriche rischia di invalidare quei numeri.

Prima di modificare qualunque file:
1. Spiega la causa del problema riscontrato
2. Specifica se la correzione può alterare i risultati numerici già calcolati
3. Se sì, FERMATI e chiedi conferma esplicita prima di procedere
4. Se è un bug puramente di naming/robustezza (es. una chiave sbagliata in 
   una fase di stampa/reporting) senza impatto sui numeri già calcolati, 
   puoi procedere ma documenta cosa hai cambiato e perché

## Specifica metodologica (ground truth dalla tesi)

### Dataset
Gold (GC=F), Silver (SI=F), Copper (HG=F), Platinum (PL=F). Prezzi di 
chiusura giornalieri aggiustati, gennaio 2010 – aprile 2026, via yfinance. 
~4.100 osservazioni per asset.

### Preprocessing
Log-return (non prezzi grezzi). Stazionarietà verificata con ADF. 
Z-score normalizzato per fold, stimato solo su training (mai leakage). 
7 gruppi di feature: lag (1,2,3,5,10gg), rolling stats (5,10,21gg), 
momentum (5,21,63gg), mean-reversion z-score 21gg, volatility regime 
proxy, cross-asset spread (Gold-Silver, Gold-Copper), calendar encoding 
ciclico (sin/cos giorno settimana).

### Split
Walk-forward chronologico, 10 fold, finestra di test di 21 giorni trading 
per fold. 210 giorni out-of-sample totali per asset, luglio 2025 – 
aprile 2026.

### Modelli (13 totali)
- Classici: ARIMA (ordine via AIC, grid p∈{0,1,2,3}, q∈{0,1,2,3}, d=0), 
  GARCH(1,1)
- ML: SVR, Decision Tree, Random Forest, Gradient Boosting, XGBoost 
  (grid search per fold, TimeSeriesSplit a 3 split interni)
- DL: LSTM, GRU, BiLSTM (64→32 unità, dropout 0.2), Transformer (4 head, 
  2 blocchi attention, FFN 128), TCN (dilation 1,2,4, kernel 3, filtri 
  64,64,32, receptive field 15), CNN-LSTM (Conv1D 32 filtri k=3 → 
  MaxPool 2 → LSTM 64)
- Seed fissato a 42 (Python random, NumPy, TensorFlow)
- Random Walk come baseline

### Metriche
RMSE, MAE, MAPE (non usato per confronto). Diebold-Mariano contro Random 
Walk. Sharpe ratio, Max Drawdown, Calmar ratio, Directional Accuracy 
(test binomiale). Transaction cost 1 basis point/trade. Regime-Conditional 
Strategy: 100% stabile, 50% normale, 0% volatile.

## Risultati già riportati in tesi — NON RICALCOLARE, sono già scritti
- RMSE medio: ARIMA 0.027337 (1°), CNN-LSTM 0.027339, LSTM 0.027345, 
  Random Walk 0.027346
- Sharpe medio (base strategy): BiLSTM 0.973 (1°), ARIMA 0.609, TCN 0.608
- RCS su Gold: ARIMA 3.812, TCN 3.787
- GARCH persistence: Gold 0.9843, Silver 0.9919, Copper 0.9933, 
  Platinum 0.9967
- DM test: 14/48 rifiuti significativi; solo BiLSTM su Silver favorevole 
  (DM -2.0387, p=0.0415)

Se una modifica cambierebbe uno di questi numeri, è un problema serio da 
segnalare esplicitamente, mai da "sistemare" in silenzio.