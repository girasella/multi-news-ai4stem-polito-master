# Contenuto di `scripts/`

## `run_benchmark_test.py`

Driver non presidiato per la sessione di benchmark con `SCOPE='test'`: esegue i notebook 10
(First-k), 17 (LDA), 15 (LSA), 16 (clustering SBERT), 11 (Centroid+MMR), 03 (BART), 04 (PEGASUS),
07 (Qwen), 09 (Mistral), 08 (Gemma) e 06 (PRIMERA) — in quest'ordine, dal più veloce al più lento
— sull'intera split test pulita (5.610 righe), uno dopo l'altro, senza dover riaprire e rilanciare
a mano ogni notebook. I notebook 10, 11, 15 e 16 generano ciascuno due varianti di metodo
(`firstk_psr`/`firstk_nltk`, `centroid_mmr`/`centroid_mmr_bert`, `lsa`/`lsa_steinberger` e
`sbert_kmeans`/`sbert_agglom`).

### Uso

```
python scripts/run_benchmark_test.py             # corsa completa, tutte le righe (~3,5-5 giorni su GPU CUDA)
python scripts/run_benchmark_test.py --limit 2   # smoke test: 2 righe per metodo, da capo a fondo
python scripts/run_benchmark_test.py --only 10,11  # solo i notebook indicati (il 05 viene comunque rieseguito)
```

`--only` accetta i prefissi numerici dei notebook separati da virgola; è utile quando gli altri
notebook hanno già completato la loro corsa `test` (rieseguirli ricaricherebbe i modelli e
ricalcolerebbe le metriche per nulla). I controlli di preflight si restringono a ciò che serve
alla selezione (ollama per 07-09, GPU per 03/04/06/11).

Si lancia da qualunque directory: i percorsi sono risolti rispetto alla posizione dello script
stesso. Servono le dipendenze dei notebook (`pip install -r requirements-notebooks.txt`) più
`jupyter`/`nbconvert`.

### Che cosa fa

1. **Preflight**: verifica che sia disponibile una GPU CUDA (se è selezionato uno fra
   03/04/06/11), che `ollama` risponda su `http://localhost:11434` con i tre tag di modello
   richiesti già scaricati (se è selezionato uno fra 07-09), che `data/tab/complete.tab` esista e
   che `jupyter nbconvert` sia importabile — fallisce subito, prima di impegnarsi in una corsa di
   giorni.
2. **Esegue ogni notebook selezionato** con `jupyter nbconvert --to notebook --execute
   --inplace`, impostando `SUMM_SCOPE=test` (e `SUMM_LIMIT=N` se è stato passato `--limit`)
   nell'ambiente del sottoprocesso. Ogni notebook dei metodi legge il proprio scope da
   `os.environ.get('SUMM_SCOPE', 'sample')`, quindi aprirli a mano in Jupyter senza questa
   variabile d'ambiente continua a eseguire lo scope `sample` di default, invariato. Un notebook
   fallito viene registrato nel log e **non** interrompe la corsa: grazie al ciclo di generazione
   riprendibile condiviso (`notebooks/summ_utils.py`), rilanciare più tardi questo script
   completa semplicemente le righe ancora mancanti.
3. **Deriva le metriche `test` di TextRank/LexRank** filtrando su `split == 'test'` il loro CSV
   per-esempio già committato con `SCOPE='full'`, invece di rieseguire i notebook 01/02 (che
   coprono già l'intero dataset, split test compresa). Il risultato è numericamente identico a una
   corsa dedicata sullo scope test, perché le metriche sono calcolate per esempio.
4. **Riesegue il notebook 05**, così che le viste di confronto riflettano i nuovi risultati.

### Output

- `results/summaries/{metodo}_test.tsv` e
  `results/metrics/{metodo}_test_{per_example.csv,aggregate.json}` per i quindici slug di metodo
  generati (sette notebook a metodo singolo più le due varianti di ciascuno dei notebook
  10/11/15/16), più i due derivati (`textrank`, `lexrank`).
- `run_benchmark_test.log` nella radice del repository (in `.gitignore`) — in sola aggiunta, con
  marca temporale, una riga per inizio/fine di ogni notebook più un riepilogo finale.

### Prima di una corsa lunga

Disattivare la sospensione di Windows (`powercfg /change standby-timeout-ac 0`), assicurarsi che
`ollama serve` sia in esecuzione con i tag richiesti (`ollama list`) e mettere in conto che la
macchina resterà occupata per diversi giorni: lo script non limita l'uso di GPU e CPU.

## `run_geval.py`

Driver non presidiato per il backfill **G-Eval (LLM-as-a-Judge)** — notebook 14. Fa giudicare ogni
riassunto generato sulla split test da `gpt-5.4-mini` su Azure, con punteggi 1–5 su coherence,
consistency, fluency e relevance. **100.621 giudizi** sui 18 metodi, ore di chiamate API a
pagamento; per la metodologia vedi la sezione *G-Eval* di `notebooks/README.md`. La corsa è
completa e l'intera cache è committata (€95,19, 94,2% dei giudizi riusciti — il resto è finito nel
content filter di Azure); un rilancio giudica solo ciò che manca, quindi non costa nulla a meno
che non si aggiungano metodi o righe.

### Uso

```
python scripts/run_geval.py --righe 1        # smoke: una chiamata per ogni metodo non ancora in
                                             # cache, dimostra che la prompt cache funziona
python scripts/run_geval.py --pilota 20      # pilota: 20 righe, misura il costo per giudizio
python scripts/run_geval.py --budget 120     # corsa completa, stop netto una volta spesi $120
python scripts/run_geval.py --righe 500 --thread 12
python scripts/run_geval.py --solo-metriche  # riscrive CSV/JSON dalla cache, ZERO chiamate API
python scripts/run_geval.py --costo          # report di costo dalla cache, ZERO chiamate API
python scripts/run_geval.py --riprova-errori # scarta i fallimenti in cache per ritentarli
python scripts/run_geval.py --no-05          # non rieseguire il notebook 05 alla fine
```

Vanno eseguiti in quest'ordine: `--righe 1`, poi `--pilota 20`, poi la corsa completa. È il pilota
a trasformare la stima di costo in un numero misurato.

### Che cosa fa

Esegue `notebooks/14_geval.ipynb` in-place tramite `nbconvert`, passando le variabili d'ambiente
`GEVAL_*` (`GEVAL_SCOPE`, `GEVAL_RIGHE`, `GEVAL_PILOTA`, `GEVAL_THREAD`, `GEVAL_BUDGET`,
`GEVAL_OGNI`, `GEVAL_RIPROVA_ERRORI`, `GEVAL_SOLO_METRICHE`) — il notebook resta l'unica fonte di
verità e l'artefatto documentale. Il preflight fallisce subito in caso di: `AZURE_OPENAI_ENDPOINT`
o `AZURE_OPENAI_API_KEY` mancanti, esito negativo del **ping da 1 token sul deployment del
giudice** (così un nome di deployment sbagliato costa secondi, non ore), `complete.tab` assente,
TSV dei riassunti mancanti per uno qualsiasi dei 18 metodi, `nbconvert` non disponibile. Il
notebook 05 viene rieseguito alla fine, a meno di `--pilota` o `--no-05`.

`run_benchmark_test.py` non viene toccato: il G-Eval non sta sul percorso di generazione.

### Output

- `results/metrics/geval_cache_test.jsonl` — una riga JSON per ogni coppia `(metodo, row_id)` con
  i quattro punteggi (o l'errore) **e i conteggi di token**. È l'artefatto pagato: viene
  committato, e i file di metrica si riderivano da lì a costo zero.
- `results/metrics/{metodo}_test_geval_{per_example.csv,aggregate.json}` per i 18 metodi —
  deliberatamente **separati** dai file di metrica standard (in `notebooks/README.md` è spiegato
  perché unirli romperebbe `valuta_e_salva`).
- `run_geval.log` nella radice del repository (in `.gitignore`) — in sola aggiunta, con marca
  temporale.

### Monitoraggio dei costi

**Azure non ha un'API di costo in tempo reale.** Cost Management ha 8–24 h di ritardo, quindi a
corsa in corso è inutile. La fonte di verità è l'oggetto `usage` di ogni risposta: i conteggi di
token sono esatti e immediati, e i prezzi unitari vengono dall'**Azure Retail Prices API**
pubblica e non autenticata (`su.prezzi_retail_azure()`, con `su.PREZZI_GEVAL` fissato come
fallback).

Durante una corsa il notebook stampa un blocco di costo ogni 1.500 giudizi (ritmo, TPM osservato,
ETA, quota di input in cache, quota di token di reasoning, costo diviso per voce e **proiezione
del totale**). Poiché i conteggi finiscono anche in cache, `--costo` ricalcola le stesse cifre
offline: si lancia **da un secondo terminale mentre la corsa lunga è in esecuzione**. `--budget` è
un tetto duro (**complessivo** fra le sessioni, non per lancio: somma quanto è già in cache prima
di confrontarlo con il tetto) e ferma la corsa in modo pulito, riprendibile al lancio successivo.

**`--valuta` deve corrispondere alla valuta di fatturazione della sottoscrizione: non è un default
cosmetico.** L'Azure Retail Prices API risponde in USD di default, ma i listini per valuta di
Azure **non** sono conversioni al cambio l'uno dell'altro: verificato su questo account
(fatturazione in EUR), il listino EUR è un **~0,8776×** fisso del numero USD su ogni meter (input,
cached e output allo stesso modo), non il tasso EUR/USD del giorno. Lanciare con la `--valuta`
sbagliata non altera la contabilità dei token — resta esatta — ma la cifra in dollari o euro
stampata e confrontata con `--budget` non corrisponderà a quanto viene effettivamente addebitato
alla sottoscrizione. Scoperto la prima volta quando una corsa tracciata a $36,00 si è rivelata,
24 h dopo e a saldo EUR reale, pari a €31,59 — un divario del 12% spiegato interamente da questo
sconto EUR fisso, non dal ritardo di Cost Management.

Per la riconciliazione del giorno dopo con la fatturazione reale:

```
az costmanagement query --type ActualCost --timeframe MonthToDate \
  --scope "/subscriptions/<sub-id>/resourceGroups/rg-antonio.girasella-0716"
```

Serve prima `az login --tenant <tenant-id>`: la CLI potrebbe essere autenticata su un tenant
diverso da quello che possiede la risorsa AI Foundry.

### Prima di una corsa lunga

Disattivare la sospensione di Windows (`powercfg /change standby-timeout-ac 0`), controllare la
**quota TPM** del deployment nel portale Azure (è quella, non `--thread`, il vero collo di
bottiglia) e ricordare che qui si spendono soldi veri: tenere d'occhio il contatore di costo
stampato e impostare `--budget`.

## `import_llm_results.py`

Importatore una tantum dei risultati del benchmark LLM locale di Federica (corse con LM Studio,
archiviate in [`notebooks/llm/`](../notebooks/llm/README.md)) nella struttura condivisa
`results/`.

> **Nota storica:** i file in `results/` committati per `qwen`/`gemma`/`mistral` sono stati poi
> rigenerati da zero con corse locali su ollama dei notebook 07–09 (qwen/gemma 2026-07-16,
> mistral 2026-07-17) e non corrispondono più a questa importazione. Lo script è conservato per
> documentare e riprodurre l'import originale da LM Studio; la sua protezione contro la
> sovrascrittura (sotto) gli impedisce di calpestare i risultati ollama.

### Uso

```
python scripts/import_llm_results.py
```

Si lancia da qualunque directory: i percorsi sono risolti rispetto alla posizione dello script
stesso. Servono le dipendenze dei notebook (`pip install -r requirements-notebooks.txt`), perché
riusa `notebooks/summ_utils.py` per la scrittura e la valutazione.

### Che cosa fa

Per ciascuno di `qwen`, `gemma`, `mistral`:

1. legge `notebooks/llm/{nome}_summary_evaluation_results.csv` e **verifica** che sia allineato
   1:1 e nell'ordine con `results/sample/sample_100_seed42.tsv` (ogni `reference_summary` deve
   coincidere con il `summary` del campione);
2. scrive `results/summaries/{nome}_sample.tsv` nel formato del repository (`row_id`,
   `generated_summary`), saltando le righe la cui generazione è fallita (contenuto vuoto o
   `Error:` — gemma ne ha 81, quindi ne restano 19);
3. ricalcola ROUGE/BLEU/METEOR con la normalizzazione condivisa del benchmark
   (`summ_utils.valuta_e_salva`) dentro `results/metrics/{nome}_sample_per_example.csv` e
   `..._aggregate.json`, la cui `config` registra la provenienza della corsa originale (LM Studio,
   checkpoint, prompt, parametri). I valori di metrica contenuti nei CSV di origine usano
   impostazioni di normalizzazione diverse e **non** vengono riportati; nemmeno la loro colonna
   BERTScore.

### Sicurezza

Lo script **si rifiuta di partire** se un file di destinazione
`results/summaries/{nome}_sample.tsv` esiste già: quel file potrebbe nel frattempo contenere righe
rigenerate via ollama (notebook 07–09), e reimportare mescolerebbe silenziosamente due backend.
Per reimportare, cancellare prima il file.

## `convert_to_tab.py`

Converte i file canonici del dataset in [`data/text/`](../data/README.md) in file `.tab` puliti di
[Orange Data Mining](https://orangedatamining.com), dentro `data/tab/`.

### Uso

```
python scripts/convert_to_tab.py
```

Si lancia da qualunque directory: i percorsi sono risolti rispetto alla posizione dello script
stesso. Nessuna dipendenza di terze parti (solo libreria standard di Python 3: `csv`, `hashlib`,
`os`). Mettere in conto qualche minuto di esecuzione: legge in streaming ~680 MB di testo sorgente
due volte e scrive ~1,3 GB di output. I file già presenti in `data/tab/` vengono sovrascritti.

### Input

I sei file canonici, che lo script non modifica mai:

| file | contenuto |
|------|-----------|
| `data/text/{train,val,test}.src.cleaned` | articoli sorgente, un esempio per riga, articoli uniti da `\|\|\|\|\|`, ritorni a capo codificati come `NEWLINE_CHAR` |
| `data/text/{train,val,test}.tgt` | un riassunto per riga, allineato riga per riga con il `.src.cleaned` corrispondente |

### Output

| file | contenuto |
|------|-----------|
| `data/tab/{train,val,test}.tab` | un `.tab` Orange pulito per split — colonne `document`, `summary` (entrambe `string`/`meta`) |
| `data/tab/complete.tab` | le tre split unite, nell'ordine train → val → test, con una terza colonna `split` (`discrete`/`meta`, valori `train`/`val`/`test`) che registra l'origine di ogni riga |
| `data/tab/excluded_rows.tsv` | elenco delle righe scartate — colonne `split`, `line` (indice a base 0 nei file di `data/text/`), `reason` |

In tutti gli output `.tab`, `NEWLINE_CHAR` viene riportato a veri ritorni a capo e il separatore
`|||||` resta dentro `document`. Le righe sono scritte con il modulo `csv` di Python (delimitate
da tabulazioni, con quoting), che è anche il modo in cui Orange stesso interpreta i file `.tab`:
ritorni a capo e tabulazioni incorporati sopravvivono quindi correttamente al giro di andata e
ritorno.

### Pulizia

L'output `.tab` è *pulito*: le righe la cui sorgente presenta un problema di qualità noto
(individuato dalla dashboard EDA, [`multi_news_dashboard.html`](../multi_news_dashboard.html))
vengono scartate. Con i dati attuali questo rimuove 115 dei 56.216 esempi (92 train, 11 val, 12
test), lasciandone 56.101. Una riga è esclusa quando la sua sorgente è:

1. **più corta di `MIN_SRC_WORDS` (50) parole** — comprese le sorgenti completamente vuote;
   probabili scraping falliti (55 righe);
2. **più lunga di `MAX_SRC_WORDS` (100.000) parole** — gli outlier estremi, il cui testo sorgente
   è semanticamente slegato dal riassunto (errori di scraping o di link a monte, non semplicemente
   testo lungo) (8 righe);
3. **un duplicato esatto di una sorgente precedente** — hash SHA-1 con spaziatura normalizzata,
   scandendo train → val → test; si tiene solo la prima occorrenza, il che elimina anche il
   *leakage* train/eval dei gruppi di duplicati a cavallo fra split (52 righe etichettate come
   duplicati; altre 25 righe ridondanti sono già intercettate dalla regola 1, dato che le sorgenti
   vuote o troncate si duplicano fra loro).

I conteggi di parole usano `str.split()`, indipendente dal tokenizer, con `NEWLINE_CHAR`
ripristinato e `|||||` escluso, in linea con la metodologia della dashboard. Le soglie sono le
costanti `MIN_SRC_WORDS` / `MAX_SRC_WORDS` in cima allo script. I riassunti non sono mai un
criterio di scarto.

**Conseguenza:** `data/tab/` *non* è allineata riga per riga con `data/text/` — per passare
dall'una all'altra si usa `excluded_rows.tsv`.

### Come funziona

Due passate:

1. `find_dirty_rows()` legge in streaming il file `.src.cleaned` di ogni split, calcola per ogni
   sorgente il numero di parole e l'hash, e restituisce l'insieme delle righe `(split, line)` da
   escludere con il relativo motivo.
2. `convert_split()` rilegge in streaming ogni coppia `.src.cleaned`/`.tgt`, accoppiandole riga
   per riga e scrivendo ogni riga non esclusa contemporaneamente nel `.tab` della sua split e in
   `complete.tab`. `write_manifest()` produce infine `excluded_rows.tsv`.

Tutto è letto in streaming riga per riga: nessun file viene caricato interamente in memoria.

### Quando rilanciarlo

I file di `data/tab/` sono derivati, mai modificati a mano. Rilanciare lo script ogni volta che
`data/text/` cambia o che si modificano i criteri di pulizia, e aggiornare conteggi di righe e
dimensioni in [`data/README.md`](../data/README.md) se cambiano.

## `analyze_dataset.py`

Analisi esplorativa sull'**intero** dataset Multi-News (train+val+test aggregate, senza dettaglio
per split): è lo script dietro le cifre incorporate in
[`multi_news_dashboard.html`](../multi_news_dashboard.html).

### Uso

```
python scripts/analyze_dataset.py
```

Si lancia dalla radice del repository. L'unica dipendenza di terze parti è `numpy` (percentili e
istogrammi); tutto il resto è libreria standard. Mettere in conto qualche minuto: legge in
streaming ~680 MB di testo sorgente una volta.

### Input

I sei file canonici in [`data/text/`](../data/README.md), letti riga per riga e mai modificati —
gli stessi file, e la stessa gestione di `NEWLINE_CHAR` / `|||||`, che usa
`multi_news.py::_generate_examples`. Nota che si tratta dei dati canonici **non puliti**, quindi
le righe sporche che `convert_to_tab.py` scarta qui sono ancora conteggiate (ed è proprio il
punto: la dashboard le riporta).

### Output

`scripts/dataset_stats.json` — un unico oggetto JSON aggregato (committato), con le chiavi di
primo livello `meta`, `structure`, `sources`, `lengths`, `hist`, `paper_reference`, `correlations`
e `quality`. Il letterale `const D = {...}` inline della dashboard è costruito da questo file:
modificare a mano i numeri della dashboard significa modificare quel letterale.

### Che cosa fa

Una sola passata in streaming per split, calcolando solo ciò che costa poco:

- **Struttura** — sorgenti e target vuoti, duplicati esatti (SHA-1 sul testo minuscolo con
  spaziatura normalizzata) e quasi-duplicati (impronta delle prime 15 parole normalizzate),
  numero di articoli per esempio.
- **Lunghezze** — conteggi di parole (`str.split()`, indipendente dal tokenizer, con
  `NEWLINE_CHAR` ripristinato e il separatore `|||||` escluso), conteggi euristici di frasi
  (`[.!?]+`), rapporti di compressione, percentili e istogrammi, più l'insieme del vocabolario
  globale (`[a-z0-9']+`).
- **Qualità** — conteggi rispetto alle soglie di anomalia dichiarate in cima allo script
  (`SUM_MIN=20`, `SUM_MAX=600`, `SRC_MIN=50` parole) e le righe outlier estreme, riferite come
  `split:riga` a base 0.

**Deliberatamente omesse** (perimetro concordato, dichiarato nel docstring del modulo) le metriche
pesanti: percentuali di n-grammi nuovi, copertura e densità dei frammenti estrattivi, e
riconoscimento della lingua. La dashboard le mostra solo come valori di riferimento statici presi
dal paper, mai come cifre ricalcolate. Su tutto ciò che dipende dal tokenizer sono attesi scarti
sistematici rispetto alla tabella del paper (per esempio vocabolario 494.577 qui contro 666.515 in
Fabbri et al.): tokenizzazione diversa, non un bug.

### Quando rilanciarlo

Ogni volta che `data/text/` cambia. Dopo il rilancio, riportare i nuovi valori nel letterale
`const D` della dashboard e aggiornare le cifre citate in
[`data/README.md`](../data/README.md) e in [`CLAUDE.md`](../CLAUDE.md) — non lo fa nulla in
automatico.
