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
| Timing (K13) | Esecuzione principale `close` (idealizzazione dichiarata); sensitivity `lag1` sempre; `open` solo se lo snapshot OHLC supera i controlli. Regola di lettura: un risultato economico conta solo se presente anche nella variante eseguibile (`open` per oro, argento e rame; `lag1` per il platino, le cui aperture sono in gran parte barre piatte; `lag1` anche se `open` manca) | «spostare l'esecuzione al primo prezzo disponibile» / «sensitivity con un ritardo di un giorno» |
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

- [x] C58 `backtesting.executable_check` + `EXECUTABLE_REFERENCE`; `results/executable_check.csv`; regola nel manifest (revisione del testo del 2/11/2026, §1.2 e §1.3).

### RCS (K14)
- [x] C13 `exposure_benchmarks`; C14 `performance_by_regime`; C15 alpha rispetto al long filtrato; C17 docstring.

### Statistica D (K02, K15, K27, K28, K38)
- [x] C11 `architecture_vs_regime_D` riscritta con un unico operatore per stima e bootstrap; C12; C45 output.

### GARCH (K01, K06, K25)
- [x] C31–C32 `models/garch.py`; C24 `dm_test_loss`; C40 sezione GARCH di `main.py`.

### Roll (K03, K23)
- [x] C25 `roll_window_mask`; C44 `roll_sensitivity.csv`; C47 `tools/build_rolled_series.py`; C26 docstring.
- [x] C59 `build_rolled_series.contract_report`: `data/rolled_series_report.csv` con l'esito del download di ogni contratto, scritto sempre (traccia del tentativo citata in tesi). **Da rifare su Colab** e versionare il report.

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
1. Fase A: `tools/fetch_ohlc.py` (fatto, commit `3dfb188`) e `tools/build_rolled_series.py` (da rilanciare per ottenere `data/rolled_series_report.csv`); i file in `data/` vanno aggiunti al branch con un commit.
2. Fase B: `python main.py` dal commit che contiene i dati.
3. Fase C: `--robustness tcn31` e `--dl-seed 1…4` dallo stesso commit.
4. Fase D: `tools/summarize_robustness.py`; i risultati vanno in `CODE/results/`.

## Rilievi della revisione del codice (novembre 2026, branch `fix/rilievi-codice` da `2586f06`)
Nessun rilievo richiede di rieseguire `main.py`: i valori toccati (e i benchmark e la $D$ con la variante eseguibile) si ricalcolano dai file della run con `tools/recompute_post_run.py` (cella D2 del notebook).
- [x] R4 `main._align_fold_output`: le previsioni di un fold sono accettate solo se date e rendimenti coincidono con quelli del fold (prima: riallineamento per posizione). Verificato sui 40 fold dei 4 asset: nella run `2586f06` le date coincidevano già.
- [x] R5 `garch.fit_garch_params`, `garch.garch_fold_losses`, `main._garch_from_disk`: con `--resume` le statistiche GARCH di un asset completato si ricostruiscono da `garch_forecasts_<asset>.csv`. Sugli altri tre metalli la ricostruzione riproduce i valori salvati (persistenza entro 2e-6, DM esatta).
- [x] R7, R8 `results/model_params.csv` (e `model_params_<asset>.csv`, scritto a fine asset): parametri scelti per fold (ordine ARMA e AIC, combinazione della griglia, e* delle reti). Con `--resume` vengono riletti, e da lì si ricostruiscono anche le epoche e* (`main._model_params_from_disk`). Nella run `2586f06` non esistono: dichiarato in tesi.
- [x] R11 `statistical_tests`: p-value DM con `norm.sf` e colonna `p-value (raw)`; Holm sui p non arrotondati (`main._holm_correct`, ora a livello di modulo).
- [x] R14 `backtesting._D_operator(required_assets=...)`: nel controllo per asset ogni replica usa gli asset della stima puntuale.
- [x] R17 `requirements-lock.txt` con le versioni del manifest; manifest con scipy, matplotlib e `commit_dirty`.
- [x] R18 un errore in un modello o nel GARCH ferma l'esecuzione (prima il fold veniva saltato).
- [x] R19 colonna `Commit` in `predictions_<asset>.csv`, verificata da `--resume`.
- [x] R20 docstring (`descriptive_stats.py`, `get_n_features`, `run_cnn_lstm_fold`, stringhe `architecture`), commento di `END`, `.gitignore`, `results_pre_revision/LEGGIMI.md`.
- [x] K13 (revisione indipendente): i benchmark della RCS (long filtrato, segnale casuale, p-value empirico) e la statistica $D$ si calcolano anche con la variante eseguibile di riferimento (open per oro, argento e rame; lag1 per il platino). `main._run_backtests_for_asset` restituisce i benchmark per close, lag1 e open (colonna `Execution`); `main._asset_data_executable` costruisce i rendimenti eseguibili e, per lag1, sposta di un giorno anche l'etichetta del regime; `main._d_statistics` e `main._write_d_outputs` scrivono i file di $D$ con suffisso `""` (close) e `"_exec"`. `backtesting.exposure_benchmarks(..., execution=, alt_returns=)` esegue il long filtrato e il segnale casuale con la stessa tempistica della strategia.
- [x] Versione del codice: `main._code_version()` restituisce l'hash con il suffisso `-dirty` se `main._git_dirty()` trova file tracciati modificati o file di codice non tracciati (`.py`, `.ipynb`, `.txt`, `.toml`, `.cfg`); `--resume` lo controlla e avvisa se il codice non è committato.
- [x] Ordine dei file: `predictions_<asset>.csv` e `model_params_<asset>.csv` sono scritti prima del GARCH; se il GARCH si interrompe, `--resume` lo ricalcola (`main._garch_live`) senza rieseguire i modelli.
- [x] `tools/recompute_post_run.py` riscritto: legge i parametri dal manifest della run, ricalcola $D$ (close ed eseguibile), i benchmark della RCS per ogni esecuzione (`rcs_benchmarks.csv`, `rcs_benchmarks_by_model.csv`, `executable_reference.csv`), il DM con Holm sui p non arrotondati e il GARCH dell'oro dai file salvati (`garch_validation.csv`); scrive i controlli e le versioni dei pacchetti nel manifest del ricalcolo.
- [x] `tools/thesis_values_cap4.py`: genera tutti i numeri e i corpi delle tabelle dei Capitoli 4–5 e dell'Appendice A dai file della run, dai prezzi e dall'output del ricalcolo (prima era uno script fuori dal repository).
- [x] `tools/summarize_robustness.py`: confronto TCN-31 / TCN solo dopo aver verificato che date e `y_true` coincidano.
- [x] Test: `tests/test_rilievi_novembre.py` (unitari e di integrazione su una pipeline ridotta: commit e parametri, Holm sui p non arrotondati, benchmark e $D$ eseguibili, ripresa e rifiuto di un commit diverso, arresto su errore di un modello, previsioni disallineate, operatore del bootstrap); `tests/test_report_relatore_resume.py` aggiornato.
- [x] Rilettura finale (seconda revisione indipendente, 20 novembre 2026):
  - Test di mutazione: con la suite precedente sopravvivevano 12 mutazioni non equivalenti su 21 (tra cui intercetta della RCS sul long passivo invece che sul filtrato, segnale casuale e intercetta sempre close-to-close, etichetta di regime non spostata con lag1, $D$ eseguibile calcolata sulle serie close, p empirico senza il +1). Aggiunti i test che le intercettano (`test_alpha_vs_filtered_long_uses_same_execution`, `test_random_signal_and_empirical_p_follow_execution`, `test_d_executable_uses_reference_series_and_decision_label`, `test_code_version_marks_uncommitted_code`, `test_predictions_saved_before_garch_with_code_version`, `test_run_partial_rejects_misaligned_forecasts`, `test_summarize_robustness_checks`, caso `tiny_relative` dell'allineamento). Riesecuzione: le 21 mutazioni non equivalenti della rilettura più 3 nuove (ripiego per asset, controlli di `summarize_robustness`) sono tutte intercettate, 24 su 24; la ventiduesima della rilettura (`1 - cdf` al posto di `norm.sf`) è equivalente. Suite completa: 93 test superati.
  - Ripiego della variante eseguibile per asset: `backtesting.executions_for_asset`; prima, se l'apertura mancava per un solo asset, quell'asset riceveva comunque "open" (KeyError in `executable_check` e `_asset_data_executable`). Nessun effetto sulla run, dove l'apertura c'è per tutti i metalli. Test `test_executable_fallback_is_per_asset`.
  - `tools/summarize_robustness.py`: si ferma se un manifest ha `commit_dirty`, se la colonna `Commit` delle previsioni non è il commit dell'esecuzione principale, o se le previsioni dei seed o di TCN-31 non hanno le stesse date e gli stessi `y_true` di quelle principali (`_check_aligned`). Sulla run `2586f06` dà gli stessi `dl_seed_dispersion.csv` e `tcn31_vs_tcn.csv`.
  - `colab_run.ipynb`: installa `requirements-lock.txt` per default (`USE_LOCK = True`), necessario perché il ricalcolo del GARCH (cella D2) dipende dalla versione di arch. Il lock non è stato provato qui per intero: nell'ambiente di verifica sono installate le versioni fissate di numpy, pandas, arch, scikit-learn e statsmodels, non quelle di tensorflow, keras e yfinance.
  - `main._git_dirty`: docstring allineata al comportamento (un file tracciato modificato, anche di dati, rende il codice "dirty").
  - `tools/thesis_values_cap4.py`: tolti codice morto e blocchi `__main__` ripetuti.
  - Limiti noti, dichiarati e non modificati: con l'esecuzione all'apertura il costo di roll segue la tempistica close-to-close (stesso trattamento per strategia e benchmark, i confronti non cambiano); la performance per regime e la sensitivity sui giorni di roll restano close-to-close (il Capitolo 1 promette la variante eseguibile solo per alpha, confronto RCS-statica, benchmark e $D$); la famiglia di Holm sulle intercette rispetto al long filtrato è calcolata nel Capitolo 4 come controllo aggiuntivo, fuori dalle famiglie del §1.3.
- [x] `tools/figure_cap4.py`: lo script che genera le figure 4.1–4.3 della tesi (Sharpe statico con intervalli, RCS contro statica con i benchmark, dispersione dei seed), aggiunto così come era stato consegnato il 14 novembre (R16). Rieseguito sui risultati della run `2586f06` dà le stesse figure: stesso contenuto e stesse dimensioni, differenze solo di rasterizzazione (anti-aliasing, versione di matplotlib). `garch_gold.py` e `analisi_cap4.py` non sono stati aggiunti: i valori che producevano (GARCH dell'oro, numeri del Capitolo 4) vengono ora da `tools/recompute_post_run.py` e `tools/thesis_values_cap4.py`.
- [x] File nuovi aggiunti e `codice_tesi_fixed.zip`, `json/settings.local.json` tolti dal repository (restano in `.gitignore`). `tools/recompute_post_run.py`, `tools/thesis_values_cap4.py` e `tools/figure_cap4.py` si rieseguono da questo commit, che è quello citato in tesi (§2.4).
