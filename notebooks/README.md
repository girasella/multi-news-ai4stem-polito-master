# Notebook di benchmark della summarization

Questa cartella contiene i notebook (documentati in italiano) che applicano e valutano cinque
metodi di summarization sul dataset Multi-News pulito ([data/tab/complete.tab](../data/tab/complete.tab)):
due estrattivi (TextRank, LexRank) e tre abstractive (BART, PEGASUS, PRIMERA), usando la libreria
[pyAutoSummarizer](https://github.com/Valdecy/pyAutoSummarizer) (PRIMERA, non supportato dalla
libreria, usa direttamente `transformers`; le metriche sono comunque quelle di pyAutoSummarizer
per tutti i metodi).

## Installazione

```bash
pip install -r requirements-notebooks.txt     # dalla root del repository
```

Sulla macchina con GPU NVIDIA installare prima la build CUDA di PyTorch (vedi commento in
[requirements-notebooks.txt](../requirements-notebooks.txt)); tutti i notebook rilevano la GPU
automaticamente e la usano se disponibile.

## Notebook e ordine di esecuzione

| # | Notebook | Cosa fa |
|---|----------|---------|
| 00 | [00_prepara_campione.ipynb](00_prepara_campione.ipynb) | **Da eseguire per primo.** Estrae il campione casuale riproducibile condiviso (default: 100 esempi, seed 42) da `complete.tab` e lo salva in `results/sample/`. |
| 01 | [01_textrank.ipynb](01_textrank.ipynb) | TextRank (estrattivo, sentence embeddings + PageRank). Ambiti `sample` e `full`. |
| 02 | [02_lexrank.ipynb](02_lexrank.ipynb) | LexRank (estrattivo, TF-IDF + PageRank). Ambiti `sample` e `full`. |
| 03 | [03_bart.ipynb](03_bart.ipynb) | BART (`facebook/bart-large-cnn`, abstractive). Solo `sample`. |
| 04 | [04_pegasus.ipynb](04_pegasus.ipynb) | PEGASUS (`google/pegasus-multi_news`, abstractive). Solo `sample`. |
| 05 | [05_confronto.ipynb](05_confronto.ipynb) | Confronto: tabelle e grafici dalle metriche salvate. Eseguibile su qualunque sottoinsieme di risultati. |
| 06 | [06_primera.ipynb](06_primera.ipynb) | PRIMERA (`allenai/PRIMERA-multinews`, abstractive multi-documento, input 4096 token). Solo `sample`. |

I notebook dei metodi (01–04 e 06) sono indipendenti tra loro e condividono le routine di
[summ_utils.py](summ_utils.py) (caricamento dati, ciclo con ripresa, metriche).

## Parametri principali (cella di configurazione di ogni notebook)

- `N_SAMPLES`, `SEED` — identificano il file campione; devono combaciare con il notebook 00.
- `SCOPE` (solo 01/02) — `'sample'` = campione condiviso (confronto con gli abstractive);
  `'full'` = intero `complete.tab`, 56.101 esempi in streaming (confronto tra estrattivi).
- `LIMIT` — `None` per la corsa completa; un intero piccolo (es. `3`) per uno smoke test.
- `N_SENTENCES` (solo 01/02) — frasi estratte per riassunto (default 11, la mediana di frasi
  per riassunto del corpus; i riassunti estratti risultano comunque più lunghi dei riferimenti,
  perché le frasi di cronaca sono più lunghe di quelle dei digest).

## File prodotti

```
results/
  sample/sample_{N}_seed{S}.tsv        # campione condiviso (row_id, split, document, summary)
  summaries/{metodo}_{scope}.tsv       # riassunti generati (row_id, generated_summary)
  metrics/{metodo}_{scope}_per_example.csv   # ROUGE-1/2/L (F1,P,R), BLEU, METEOR per esempio
  metrics/{metodo}_{scope}_aggregate.json    # medie complessive e per split + configurazione usata
```

I riassunti sono la parte costosa: vengono scritti **incrementalmente** (una riga per esempio,
flush immediato) e un'esecuzione interrotta **riprende** da dove era arrivata, saltando i `row_id`
già presenti nel file. Le metriche invece si ricalcolano in pochi secondi **leggendo solo i file
salvati**: la sezione «Valutazione» di ogni notebook è rieseguibile senza rigenerare nulla.
I file `*_full.tsv` (decine di MB) sono esclusi da git ([.gitignore](../.gitignore)); campione e
metriche sono versionati.

## Tempi indicativi

| Corsa | CPU (questa macchina) | GPU NVIDIA |
|---|---|---|
| LexRank, campione 100 | ~1 min | ~1 min |
| TextRank, campione 100 | ~5 min | ~2 min |
| BART / PEGASUS, campione 100 | ~1–2,5 h ciascuno (~30–90 s/esempio) | ~5–10 min |
| PRIMERA, campione 100 | sconsigliata (minuti per esempio: input 4096, 5 beam) | ~30–60 min |
| LexRank, `full` (56.101) | ore | ore (non serve la GPU) |
| TextRank, `full` (56.101) | ~6–12 h | ~1 h — **consigliata la GPU** |

Al primo avvio vengono scaricati i modelli da Hugging Face (MiniLM ~90 MB; BART ~1,6 GB;
PEGASUS ~2,3 GB; PRIMERA ~1,8 GB).

## Esecuzione su Google Colab

1. Caricare su Colab: il notebook desiderato, `summ_utils.py` e il file campione, ricreando i
   percorsi relativi (`/content/results/sample/sample_100_seed42.tsv`). Per `SCOPE='full'` serve
   anche `data/tab/complete.tab` (~658 MB, per esempio via Drive in `/content/data/tab/`).
2. Attivare il runtime GPU (Runtime → Cambia tipo di runtime → GPU).
3. La prima cella installa `pyAutoSummarizer` automaticamente se manca; i percorsi puntano a
   `/content` quando il notebook rileva Colab.
4. A fine corsa scaricare `results/summaries/` e `results/metrics/` e copiarli nel repository.

## Avvertenze metodologiche

- **Leakage PEGASUS e PRIMERA**: il campione proviene da tutte e tre le split e sia
  `google/pegasus-multi_news` sia `allenai/PRIMERA-multinews` sono stati addestrati sulla split
  train di questo dataset → i loro punteggi su righe train sono ottimistici. Gli aggregati
  riportano anche le medie per split; il confronto pulito è sulla sola split `test` (vista
  dedicata nel notebook 05).
- **ROUGE della libreria**: pyAutoSummarizer calcola ROUGE-N su insiemi di n-grammi *unici*
  (non i conteggi "clipped" dello standard): i valori sono coerenti tra i metodi di questo
  benchmark ma non confrontabili in assoluto con la letteratura.
- **Troncamento a 1024 token**: BART e PEGASUS vedono solo l'inizio di ogni cluster di articoli
  (limite dei checkpoint); vale per entrambi, quindi il confronto tra i due resta equo. PRIMERA
  (notebook 06) arriva invece a 4096 token, con budget uguale per articolo e separatore
  `<doc-sep>`: il divario con BART/PEGASUS riflette quindi anche la diversa copertura
  dell'input, non solo il modello.
- **Righe saltate dagli estrattivi**: su rari testi (~1% del campione) il costruttore di
  `psr.summarization` solleva un `IndexError` (bug della libreria: dopo la pulizia le liste di
  frasi possono disallinearsi). Il ciclo registra l'errore e prosegue: la riga manca dal file dei
  riassunti di quel metodo. Il notebook 05 confronta i metodi sull'**intersezione** dei `row_id`
  valutati da tutti, quindi le medie restano eque.
