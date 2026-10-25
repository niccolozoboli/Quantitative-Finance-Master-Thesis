# Piano delle correzioni al codice — commenti del relatore (ottobre 2026)

Fonte delle richieste: tesi annotata dal relatore (`main-3.tex`, compilata il
9 ottobre 2026: 15 commenti in verde, passaggi riscritti in blu) e report del
7 settembre 2026. Base di partenza: commit `26f3e22` (risultati del commit
`4ea8510`, esecuzione Colab del 27 settembre 2026).

Ogni voce è spuntata solo quando il test relativo passa. `Kxx` rimanda
all'inventario dei commenti (K01–K15 = i 15 commenti in verde).

## Scelte fissate prima dell'esecuzione (non vanno cambiate dopo aver visto i risultati)

| Punto | Scelta | Strada del relatore |
|---|---|---|
| Timing (K13) | Esecuzione principale `close` (idealizzazione dichiarata); sensitivity `lag1` sempre; `open` solo se lo snapshot OHLC supera i controlli | «spostare l'esecuzione al primo prezzo disponibile» / «sensitivity con un ritardo di un giorno» |
| Test DA (K04) | Test HAC su h_t − 1/2 (principale) + bootstrap a blocchi centrato (controllo); non calcolato se i giorni attivi sono < 30 | «Implementa il test sotto il nullo» |
| Holm (K18) | 5 famiglie principali: DM, alpha statica, alpha RCS, DA statica, DA RCS. Famiglie di sensitivity separate | Blu §1.3 / §3.5 |
| Statistica D (K15, K02) | Principale: portafoglio equal-weight per data, con gli asset nel regime g; controlli: D per asset aggregata e decomposizione a due fattori | «Scegli una delle due costruzioni» + controlli suggeriti |
| RCS (K14) | Benchmark: long passivo filtrato e segnale casuale filtrato (1.000 repliche); performance e DA per regime | «Aggiungi almeno un benchmark…» |
| GARCH (K06, K01) | QLIKE + SE su \|r\|√(π/2); GARCH-t come unico competitore; DM sulla QLIKE | «La QLIKE è preferibile» + Student-t |
| Futures (K03) | Contratti singoli per il periodo fuori campione se disponibili (script separato); in ogni caso sensitivity escludendo le finestre di roll | Entrambe le strade |
| TCN (K08) | Campo ricettivo 15 resta nel modello principale; TCN con dilatazioni [1,2,4,8] (campo 31) come controllo di robustezza | Entrambe le strade |
| Calendario (K11) | Uniformato: nelle reti la riga u codifica il weekday del giorno di borsa successivo, come i modelli ML | «uniforma la costruzione» |
| Seed (K09) | Seed 42 = specificazione principale; seed aggiuntivi 1, 2, 3, 4 per le 6 reti, in esecuzioni separate dallo stesso commit | «ripeti… con più seed» |

## Modifiche

### Metriche (K05, K16)
- [x] C1–C5 `src/metrics_econ.py`: W_0 = 1, MDD da u = 0, `annualised_return`, Calmar composto.
- [x] C7 `backtesting.run_backtest`: `Annual Return (%)` composto; `Mean Daily Return x252 (%)` aritmetico.
- [x] C50–C51 test aggiornati e nuovi.
- Nota: C6 (Sharpe NaN a varianza nulla, non proposta dal relatore) **non applicata**: i NaN si propagherebbero a intervalli bootstrap e D. Resta la convenzione 0, da dichiarare in tesi.

### Test DA e Holm (K04, K17, K18)
- [x] C10 `block_bootstrap_ci`: rimosso il p-value non centrato; restano gli intervalli.
- [x] C22–C23 `statistical_tests`: `da_test_hac`, `da_test_centred_bootstrap`, `MIN_ACTIVE_DA = 30`.
- [x] C16 colonne di `backtest_summary_table`.
- [x] C43 `main.py`: 5 famiglie principali + famiglie di sensitivity.

### Timing (K13, K24)
- [x] C8–C9 `run_backtest(execution=…)`, `lag_positions`.
- [x] C27 `preprocessing.load_ohlc_snapshot` + controlli; snapshot dei Close invariato.
- [x] C41–C42 backtest per ogni esecuzione, colonna `Execution`.

### RCS (K14)
- [x] C13 `exposure_benchmarks`; C14 `performance_by_regime`; C15 alpha rispetto al long filtrato; C17 docstring.

### Statistica D (K02, K15, K27, K28, K38)
- [x] C11 `architecture_vs_regime_D` riscritta con un unico operatore per stima e bootstrap; C12; C45 output.

### GARCH (K01, K06, K25)
- [x] C31–C32 `models/garch.py`; C24 `dm_test_loss`; C40 sezione GARCH di `main.py`.

### Roll (K03, K23)
- [x] C25 `roll_window_mask`; C44 `roll_sensitivity.csv`; C47 `tools/build_rolled_series.py`; C26 docstring.

### Reti (K08, K09, K11, K29)
- [x] C28 calendario uniforme (C29, il conteggio dei giorni con calendario diverso, non serve più: dopo l'uniformazione la differenza non esiste); C33 argomento `seed`; C34 `run_tcn31_fold`; C35 varianze del positional encoding; C37 flag `--dl-seed`, `--robustness tcn31`; C48 `tools/summarize_robustness.py`.

### Pulizia
- [x] C18–C19, C38 ADF solo sui log-rendimenti, colonna «Reject unit root (5%)».
- [x] C20–C21 HAC: docstring e autocovarianza standard.
- [x] C30, C46 commenti e terminologia; C36 manifest con le scelte; C39 `RMSE pooled`.

### Colab ed esecuzione
- [x] C56 notebook: cella dati, branch, cartelle, celle per seed e TCN-31.
- [x] C57 risultati del 27/9 conservati in `results_run_20260927/`.

## Verifica prima dell'esecuzione su Colab (fatta il 25/10/2026, commit di questo branch)
- [x] Tutti i test verdi.
- [x] `--smoke` completo su CPU.
- [x] Ripasso delle previsioni del 27/9 nel nuovo post-processing: Sharpe, turnover, giorni di roll, DA, alpha e intervalli bootstrap identici; MDD diverso solo nei 2 casi con perdita prima del primo picco; Calmar cambiato come atteso; DM con differenza massima 0,031 (autocovarianza standard), stessi 2 test significativi dopo Holm.

## Da fare su Colab (vedi colab_run.ipynb)
1. Fase A: `tools/fetch_ohlc.py` e `tools/build_rolled_series.py`; i file in `data/` vanno aggiunti al branch con un commit.
2. Fase B: `python main.py` dal commit che contiene i dati.
3. Fase C: `--robustness tcn31` e `--dl-seed 1…4` dallo stesso commit.
4. Fase D: `tools/summarize_robustness.py`; i risultati vanno in `CODE/results/`.
