# Notebook di benchmark della summarization

Questa cartella contiene i notebook (documentati in italiano) che applicano e valutano nove
metodi di summarization sul dataset Multi-News pulito ([data/tab/complete.tab](../data/tab/complete.tab)):
due estrattivi (TextRank, LexRank), tre abstractive specializzati (BART, PEGASUS, PRIMERA), tre
LLM generalisti eseguiti in locale (Qwen2.5-7B, Gemma 4 E4B, Mistral-7B — notebook
07–09, via [ollama](https://ollama.com)) e un LLM cloud su **Azure AI Foundry**
(GPT-5-mini — notebook 10), usando la libreria
[pyAutoSummarizer](https://github.com/Valdecy/pyAutoSummarizer) (PRIMERA usa direttamente
`transformers`, gli LLM il client `openai`; le metriche sono comunque quelle di
pyAutoSummarizer per tutti i metodi). Due ulteriori notebook Azure (Claude Haiku 4.5 e
DeepSeek-V3.2, ex 11–12) sono stati rimossi perché il deployment dei modelli non è riuscito:
restano recuperabili dalla history git se si riproverà con un altro approccio.

La sottocartella [llm/](llm/README.md) è un **archivio**: i notebook e i risultati originali di
Federica (LM Studio), da cui erano stati inizialmente importati i risultati committati dei metodi
`qwen`/`gemma`/`mistral` — poi sostituiti dalle corse ollama dei notebook 07–09 (vedi sotto).

> ⚠️ **L'ambito `sample` non è più pubblicato.** Fino a luglio 2026 ogni metodo veniva anche
> valutato sul campione condiviso da 100 esempi e i risultati (`results/summaries/{metodo}_sample.tsv`
> e le metriche corrispondenti) erano versionati. Con la corsa `test` completa disponibile per
> tutti e nove i metodi (sezione dedicata più sotto), quel confronto più piccolo non aggiungeva
> informazione ed è stato rimosso da `results/` per tenere il repository più snello. `SCOPE='sample'`
> resta comunque il default di ogni notebook dei metodi, per uno smoke test locale rapido
> (tipicamente con `LIMIT` a un numero piccolo) prima di lanciare una corsa `test`/`full`; il file
> di input `results/sample/sample_100_seed42.tsv` resta versionato per questo scopo.

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
| 03 | [03_bart.ipynb](03_bart.ipynb) | BART (`facebook/bart-large-cnn`, abstractive). Ambiti `sample` e `test`. |
| 04 | [04_pegasus.ipynb](04_pegasus.ipynb) | PEGASUS (`google/pegasus-multi_news`, abstractive). Ambiti `sample` e `test`. |
| 05 | [05_confronto.ipynb](05_confronto.ipynb) | Confronto: tabelle e grafici dalle metriche salvate. Eseguibile su qualunque sottoinsieme di risultati. |
| 06 | [06_primera.ipynb](06_primera.ipynb) | PRIMERA (`allenai/PRIMERA-multinews`, abstractive multi-documento, input 4096 token). Ambiti `sample` e `test`. |
| 07 | [07_qwen.ipynb](07_qwen.ipynb) | Qwen2.5-7B-Instruct (LLM locale via ollama, prompt zero-shot). Ambiti `sample` e `test`. |
| 08 | [08_gemma.ipynb](08_gemma.ipynb) | Gemma 4 E4B (LLM locale via ollama). Ambiti `sample` e `test`. |
| 09 | [09_mistral.ipynb](09_mistral.ipynb) | Mistral-7B-Instruct-v0.3 (LLM locale via ollama). Ambiti `sample` e `test`. |
| 10 | [10_azure_gpt.ipynb](10_azure_gpt.ipynb) | GPT-5-mini (Azure OpenAI). Ambiti `sample`, `test` e `full` (56.101 righe, sequenziale). |

I notebook dei metodi (01–04 e 06–10) sono indipendenti tra loro e condividono le routine di
[summ_utils.py](summ_utils.py) (caricamento dati, ciclo con ripresa, metriche).

## LLM locali (notebook 07–09)

> Nota storica: le informazioni sotto descrivono la validazione originale sull'ambito `sample`
> (100 esempi). I risultati committati oggi sono quelli dell'ambito `test` (sezione dedicata più
> sotto); i file `*_sample.tsv` a cui si fa riferimento qui non sono più nel repository.

I primi risultati di `qwen`/`gemma`/`mistral`, poi superati dalla corsa `test` completa,
provenivano dalle **corse ollama di questi notebook** (qwen/gemma 2026-07-16, mistral
2026-07-17; 100/100 esempi ciascuna). Avevano sostituito i risultati della corsa originale di
Federica via **LM Studio** (Mac M4, 2026-07-16), a suo tempo importati con
[`scripts/import_llm_results.py`](../scripts/README.md) dai CSV archiviati in
[llm/](llm/README.md). Una prima corsa ollama di mistral (2026-07-16) aveva usato per errore
Mistral Small ~24B ed è stata scartata e rifatta con il modello corretto (vedi notebook 09).
Modelli usati:

```bash
ollama pull qwen2.5:7b-instruct              # 07
ollama pull gemma4                           # 08
ollama pull mistral:7b-instruct-v0.3-q4_K_M  # 09
ollama serve                                 # se il servizio non è già attivo
```

Il modello gira nel server ollama; i notebook usano solo il client `openai` puntato
all'endpoint OpenAI-compatibile (`http://localhost:11434/v1`). Avvertenze:

- **Ripresa = rischio di mescolare corse**: il ciclo condiviso salta i `row_id` già presenti
  nel TSV, quindi rieseguire la generazione sopra un file esistente aggiunge solo le righe
  mancanti — con un backend, un modello o una configurazione diversi si mescolerebbero due
  corse nello stesso file. Per una corsa pulita eliminare prima
  `results/summaries/{metodo}_sample.tsv`.
- **gemma usa `MAX_TOKENS=1500`** (gli altri 200/300): il reasoning di Gemma 4 consuma il
  budget di `max_tokens` prima della risposta visibile (`finish_reason=length`, `content`
  vuoto) — con i 300 della corsa originale LM Studio il modello restituiva una risposta vuota
  in 81 casi su 100 (solo 19 esempi valutati); con 1500 la corsa ollama copre tutti i 100.
- **qwen usa `max_tokens=200`** (come la corsa originale; mistral 300, gemma 1500): riassunti
  più corti, da tenere presente leggendo precisione/recall.
- Il prompt (in inglese) e `temperature=0.3` replicano la corsa originale; il BERTScore
  presente nei CSV di Federica non è stato portato in `results/` (la pipeline condivisa non
  lo calcola).

## Corsa completa sulla split test (notebook 03-04, 06-09)

`scripts/run_benchmark_test.py` esegue in **un'unica sessione non presidiata** i sei notebook
che finora coprivano solo il campione (03 BART, 04 PEGASUS, 06 PRIMERA, 07 Qwen, 08 Gemma, 09
Mistral) con `SCOPE='test'` — l'intera split test pulita, 5.610 righe — dal più veloce al più
lento, senza bisogno di riaprire i notebook uno per uno tra una corsa e l'altra:

```bash
python scripts/run_benchmark_test.py            # corsa completa (~3,5-5 giorni su questa macchina)
python scripts/run_benchmark_test.py --limit 2   # prova end-to-end economica, 2 righe per metodo
```

Meccanismo: ogni notebook legge l'ambito da `SCOPE = os.environ.get('SUMM_SCOPE', 'sample')` —
se aperto a mano in Jupyter senza questa variabile d'ambiente si comporta come sempre
(`SCOPE='sample'`); lo script imposta `SUMM_SCOPE=test` (e `SUMM_LIMIT` se `--limit` è passato)
ed esegue ciascun notebook via `jupyter nbconvert --execute --inplace`, salvando l'output
eseguito nel notebook stesso. TextRank e LexRank **non vengono rieseguiti**: le loro metriche
`test` sono derivate filtrando la corsa `full` già committata (stesso risultato numerico, perché
le metriche sono calcolate per esempio). Il notebook 05 viene poi rieseguito automaticamente per
aggiornare le viste di confronto.

Un notebook fallito viene **registrato e non blocca** i successivi (`run_benchmark_test.log`
nella root del repo); grazie alla ripresa condivisa, rilanciare lo script in seguito completa
solo le righe mancanti. Prima di una corsa lunga: disattivare la sospensione di Windows
(`powercfg /change standby-timeout-ac 0`), tenere `ollama serve` attivo e verificare i tag con
`ollama list` (`qwen2.5:7b-instruct`, `gemma4:latest`, `mistral:7b-instruct-v0.3-q4_K_M`).

## LLM su Azure AI Foundry (notebook 10)

Il notebook 10 replica il protocollo dei notebook 07–09 (stesso prompt zero-shot in inglese,
documento passato da `prepara_documento`; senza il prefisso `/no_think`, artefatto di qwen) su
**GPT-5-mini** servito da Azure AI Foundry. GPT-5-mini è un modello con *reasoning* e devia in
modo documentato (niente `temperature`, `max_completion_tokens=1500` con
`reasoning_effort='minimal'` — il caso gemma del notebook 08; la famiglia gpt-4o-mini è ritirata
da Azure e non è più deployabile). Tre ambiti:

- `sample` — il campione condiviso da 100 esempi (confronto con tutti gli altri metodi, costo di
  pochi centesimi);
- `test` — l'intera split **test** pulita di `complete.tab` (5.610 righe = 5.622 − 12 righe
  sporche): confronto senza le avvertenze di leakage, con numerosità ~56 volte maggiore;
- `full` — l'**intero dataset** (56.101 righe) in chiamate sequenziali sul deployment Standard
  (~2–4 giorni, interrompibile e riprendibile). ⚠️ La Batch API di Azure OpenAI (sconto 50%)
  **non offre gpt-5-mini in nessuna regione** (solo gpt-4.1*, gpt-4o*, gpt-5, gpt-5.1 e serie
  o), quindi la corsa completa va a prezzo pieno.

### Configurazione di Azure (una tantum, nel portale)

1. Creare una risorsa **Azure AI Foundry** + progetto (es. *Sweden Central*).
2. Deployment: `gpt-5-mini` di tipo **Global Standard**.
3. Variabili d'ambiente (mai chiavi nel codice o nei notebook): `AZURE_OPENAI_ENDPOINT` (solo
   la **radice** della risorsa, es. `https://<risorsa>.services.ai.azure.com`, senza path) e
   `AZURE_OPENAI_API_KEY`.

### Costi indicativi (luglio 2026, prezzi Azure pay-as-you-go)

Stime con ~2.900 token di input e ~300 di output per esempio (GPT-5-mini: 0,25/2,00 $/M):

| Corsa | Costo stimato |
|---|---|
| `sample` (100 esempi) | centesimi |
| `test` (5.610) | ~8 $ (corsa reale 2026-07-17: ~7 €) |
| `full` (56.101, sequenziale a prezzo pieno) | ~80 $ |

### Avvertenze

- **Ripresa = rischio di mescolare corse**, come per i notebook 07–09: ogni ambito scrive su un
  TSV separato e rieseguire sopra un file esistente aggiunge solo le righe mancanti; con un
  deployment o una configurazione diversi eliminare prima il TSV.
- **Smoke test**: prima di una corsa `test` o `full` lanciare con `LIMIT = 3` per verificare
  endpoint, chiavi e formato delle risposte.
- **Content filter di Azure**: alcuni cluster di cronaca (hate/violence a severità media)
  vengono respinti dal filtro con errore `content_filter` prima di raggiungere il modello: le
  righe restano assenti dal TSV e non sono ritentabili (nella corsa test 2026-07-17: 139 righe
  su 5.610). Il confronto nel notebook 05 resta equo (intersezione dei `row_id`); per coprirle
  serve un content filter personalizzato con soglie *high-only* associato al deployment.

## Parametri principali (cella di configurazione di ogni notebook)

- `N_SAMPLES`, `SEED` — identificano il file campione; devono combaciare con il notebook 00.
- `SCOPE` — `'sample'` = campione condiviso (tutti i metodi); `'full'` = intero `complete.tab`,
  56.101 esempi in streaming (01/02 e 10); `'test'` = intera split test, 5.610 esempi in
  streaming (03-04, 06-10; nei notebook 03-04 e 06-09 letto dalla variabile d'ambiente
  `SUMM_SCOPE`, impostata da `scripts/run_benchmark_test.py` — default `'sample'` se assente).
- `LIMIT` — `None` per la corsa completa; un intero piccolo (es. `3`) per uno smoke test. Nei
  notebook 03-04 e 06-09 letto anche dalla variabile d'ambiente `SUMM_LIMIT` (usata da
  `run_benchmark_test.py --limit N`).
- `N_SENTENCES` (solo 01/02) — frasi estratte per riassunto (default 11, la mediana di frasi
  per riassunto del corpus; i riassunti estratti risultano comunque più lunghi dei riferimenti,
  perché le frasi di cronaca sono più lunghe di quelle dei digest).
- `MODELLO`, `OLLAMA_URL`, `MAX_TOKENS`, `TEMPERATURE` (solo 07–09) — tag del modello ollama
  (verificare con `ollama list`), endpoint e parametri di generazione.
- `DEPLOYMENT`, endpoint da variabili d'ambiente (solo 10) — nome del deployment Azure e
  parametri del client (vedi la sezione Azure sopra); il notebook 10 usa la rotta **v1** di
  Azure OpenAI (`<endpoint>/openai/v1/`, senza api-version datata).

## File prodotti

```
results/
  sample/sample_{N}_seed{S}.tsv        # campione condiviso (row_id, split, document, summary) — solo input
  summaries/{metodo}_{scope}.tsv       # riassunti generati (row_id, generated_summary); scope = test o full
  metrics/{metodo}_{scope}_per_example.csv   # ROUGE-1/2/L (F1,P,R), BLEU, METEOR per esempio
  metrics/{metodo}_{scope}_aggregate.json    # medie complessive e per split + configurazione usata
```

I riassunti sono la parte costosa: vengono scritti **incrementalmente** (una riga per esempio,
flush immediato) e un'esecuzione interrotta **riprende** da dove era arrivata, saltando i `row_id`
già presenti nel file. Le metriche invece si ricalcolano in pochi secondi **leggendo solo i file
salvati**: la sezione «Valutazione» di ogni notebook è rieseguibile senza rigenerare nulla.
Il file campione e le corse `test`/`full` sono versionati (compresi i TSV `*_full.tsv` e
`*_test.tsv`, grandi ma rigenerabili a pagamento); i risultati dell'ambito `sample`, superato
dalla corsa `test` completa, non sono più committati — restano generabili localmente
(`SCOPE='sample'`, il default di ogni notebook) per uno smoke test rapido.

## Tempi indicativi

| Corsa | CPU (questa macchina) | GPU NVIDIA |
|---|---|---|
| LexRank, campione 100 | ~1 min | ~1 min |
| TextRank, campione 100 | ~5 min | ~2 min |
| BART / PEGASUS, campione 100 | ~1–2,5 h ciascuno (~30–90 s/esempio) | ~5–10 min |
| PRIMERA, campione 100 | sconsigliata (minuti per esempio: input 4096, 5 beam) | ~30–60 min |
| LexRank, `full` (56.101) | ore | ore (non serve la GPU) |
| TextRank, `full` (56.101) | ~6–12 h | ~1 h — **consigliata la GPU** |
| LLM via ollama, campione 100 | dipende da modello e hardware (su questa macchina: qwen ~9 min, mistral ~18 min, gemma ~24 min) | — |
| BART, `test` (5.610) | — | ~2 h |
| PEGASUS, `test` (5.610) | — | ~5-9 h |
| Qwen, `test` (5.610) | — | ~8 h |
| Mistral, `test` (5.610) | — | ~17 h |
| Gemma, `test` (5.610) | — | ~22 h |
| PRIMERA, `test` (5.610) | sconsigliata | ~28-56 h — **richiede la GPU** |
| GPT-5-mini (10), campione 100 | ~5–15 min (dipende dalla latenza dell'API) | — |
| GPT-5-mini (10), split test 5.610 | ~8 h sequenziali (corsa reale 2026-07-17: ~5 s/esempio) | — |
| GPT-5-mini (10), `full` intero dataset | ~2–4 giorni di chiamate sequenziali (riprendibile) | — |

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
- **Righe saltate dagli estrattivi**: su rari testi (22/5.610 nella split test, ~0,4%) il
  costruttore di `psr.summarization` solleva un `IndexError` (bug della libreria: dopo la pulizia
  le liste di frasi possono disallinearsi). Il ciclo registra l'errore e prosegue: la riga manca
  dal file dei riassunti di quel metodo. Il notebook 05 confronta i metodi sull'**intersezione**
  dei `row_id` valutati da tutti, quindi le medie restano eque.
- **METEOR non affidabile per output degeneri**: la formula `meteor()` di pyAutoSummarizer
  (`meteor = fmean * (1 - penalty**3)`) non è limitata a [0,1]. Nella corsa `test`, PEGASUS
  produce due riassunti patologici (un loop di ripetizione del beam search e un probabile
  mismatch sorgente/riferimento) con METEOR rispettivamente -1959.12 e -2.10, che trascinano la
  sua media riportata da ~0.42 a 0.079 — solo quella colonna va letta con questa avvertenza (vedi
  il dettaglio nel notebook 05). LexRank ha un caso molto più lieve, con effetto trascurabile.
