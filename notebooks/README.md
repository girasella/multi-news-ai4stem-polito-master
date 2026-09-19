# Notebook di benchmark della summarization

Questa cartella contiene i notebook (documentati in italiano) che applicano e valutano i
metodi di summarization sul dataset Multi-News pulito ([data/tab/complete.tab](../data/tab/complete.tab)):
la baseline posizionale First-k / Lead (notebook 10, in due varianti di segmentazione), tre
estrattivi (TextRank, LexRank e Centroid-based + MMR *custom* in due varianti di vettorizzazione,
TF-IDF e BERT — notebook 11), tre abstractive specializzati (BART, PEGASUS, PRIMERA), tre
LLM generalisti eseguiti in locale (Qwen2.5-7B, Gemma 4 E4B, Mistral-7B — notebook
07–09, via [ollama](https://ollama.com)), un LLM cloud su **Azure AI Foundry**
(GPT-5-mini — notebook 12) e cinque estrattivi **non supervisionati** basati su riduzione di
dimensionalità, clustering e topic modeling (LSA in due varianti di selezione — notebook 15;
clustering su sentence embeddings SBERT in due varianti — notebook 16; LDA — notebook 17),
usando la libreria
[pyAutoSummarizer](https://github.com/Valdecy/pyAutoSummarizer) (PRIMERA usa direttamente
`transformers`, gli LLM il client `openai`; le metriche sono comunque quelle di
pyAutoSummarizer per tutti i metodi). Due ulteriori notebook Azure (Claude Haiku 4.5 e
DeepSeek-V3.2) sono stati rimossi perché il deployment dei modelli non è riuscito:
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
| 10 | [10_firstk.ipynb](10_firstk.ipynb) | First-k / Lead (baseline posizionale, prime k frasi per articolo). Genera **due varianti** confrontabili — `firstk_psr` e `firstk_nltk` — che differiscono solo per il segmentatore di frasi. Ambiti `sample`, `test` e `full`. |
| 11 | [11_centroid_mmr.ipynb](11_centroid_mmr.ipynb) | Centroid-based (MEAD) + MMR (estrattivo nativo MDS, implementazione *custom* scikit-learn: centroide + selezione greedy MMR con parametro `λ`). Genera **due varianti** — `centroid_mmr` (vettorizzazione TF-IDF) e `centroid_mmr_bert` (embeddings BERT `all-MiniLM-L6-v2`) — che differiscono solo per il vettorizzatore. Ambiti `sample`, `test` e `full`. |
| 12 | [12_azure_gpt.ipynb](12_azure_gpt.ipynb) | GPT-5-mini (Azure OpenAI). Ambiti `sample`, `test` e `full` (56.101 righe, sequenziale). |
| 13 | [13_bertscore.ipynb](13_bertscore.ipynb) | BERTScore (`roberta-large`, backfill separato). Aggiunge `bertscore_f1/p/r` alle metriche `test` già calcolate dai 18 metodi, senza rieseguire i notebook 01–04/06–12 e 15–17. |
| 14 | [14_geval.ipynb](14_geval.ipynb) | G-Eval (LLM-as-a-Judge, giudice GPT-5.4-mini su Azure; backfill separato). Assegna a ogni riassunto della split `test` quattro punteggi 1–5 — coherence, consistency, fluency, relevance — scritti in **file dedicati** (`*_geval_*`), non uniti alle metriche standard. Guidato da [`scripts/run_geval.py`](../scripts/run_geval.py). |
| 15 | [15_lsa.ipynb](15_lsa.ipynb) | LSA / SVD (estrattivo non supervisionato: TF-IDF + `TruncatedSVD`). Genera **due varianti** — `lsa` (top-k per norma latente, come `sumy`) e `lsa_steinberger` (greedy con deflazione, anti-ridondanza MDS) — che differiscono solo per la regola di selezione. Ambiti `sample`, `test` e `full`. |
| 16 | [16_sbert_clustering.ipynb](16_sbert_clustering.ipynb) | Clustering su sentence embeddings SBERT (`all-MiniLM-L6-v2`) con selezione del **medoide** di ogni cluster. Genera **due varianti** — `sbert_kmeans` (KMeans, distanza euclidea su embedding L2-normalizzati) e `sbert_agglom` (Agglomerative, cosine + average linkage) — che differiscono solo per l'algoritmo di clustering. Ambiti `sample`, `test` e `full`. |
| 17 | [17_lda.ipynb](17_lda.ipynb) | Topic modeling con LDA (`CountVectorizer` + `LatentDirichletAllocation`): le frasi vengono allocate ai topic in proporzione al peso di ciascuno. Slug `lda`. Ambiti `sample`, `test` e `full`. |
| 18 | [18_analisi_lunghezze.ipynb](18_analisi_lunghezze.ipynb) | Analisi **per cluster** delle lunghezze dei riassunti generati dai 18 metodi (issue #15): non genera nulla, legge le metriche `test` già salvate e la mediana del riferimento per numero di articoli. |
| 19 | [19_confronto_giudici.ipynb](19_confronto_giudici.ipynb) | **Validazione del giudice** G-Eval: rigiudica un campione appaiato di riassunti gia' valutati con un secondo giudice di lignaggio diverso (DeepSeek-V3.2-Speciale) per verificare che il primato di `gpt5mini` non sia family bias. Non genera nulla e non spende nulla: legge le due cache di giudizi. |

I notebook dei metodi (01–04, 06–12 e 15–17) sono indipendenti tra loro e condividono le routine di
[summ_utils.py](summ_utils.py) (caricamento dati, ciclo con ripresa, metriche).

## Baseline First-k (notebook 10)

First-k / Lead è la baseline posizionale del paper Multi-News (con `k=3` = "First-3"): per ogni
articolo del cluster prende le prime `K_SENTENCES` frasi e le concatena. Il notebook produce
**due varianti** che si distinguono solo per come dividono ogni articolo in frasi, così da
misurare se una segmentazione più raffinata cambia le metriche MDS:

- **`firstk_psr`** — segmentazione di `psr.summarization` (per punteggiatura), la **stessa** di
  TextRank/LexRank: confronto equo con gli altri estrattivi.
- **`firstk_nltk`** — `nltk.sent_tokenize` (modello Punkt, gestisce le abbreviazioni):
  segmentazione più raffinata.

I due segmentatori sono alternative (una per variante); la valutazione resta quella condivisa di
pyAutoSummarizer. È una baseline senza modelli né ranking, quindi rapidissima anche su CPU.

## Centroid-based + MMR (notebook 11)

Primo metodo **nativo MDS** del benchmark, che gestisce esplicitamente la **ridondanza tra fonti**
(§3.4 del [documento-guida](../Tecniche_MDS_non_LLM_MultiNews.md); MMR è trattato a lezione con
la formula esplicita). È il principale
**contributo implementativo originale** del gruppo: nessuna libreria plug-and-play, la pipeline è
scritta a mano con scikit-learn. Passi:

1. **Segmentazione** in frasi via `psr.summarization` (la **stessa** di TextRank/LexRank/firstk_psr),
   così il pool di candidati è identico agli altri estrattivi.
2. **Vettorizzazione** di tutte le frasi del cluster → **centroide** (media dei vettori); la
   **rilevanza** di una frase è la sua cosine similarity col centroide (idea *MEAD*).
3. Selezione **greedy MMR**: a ogni passo si aggiunge la frase che massimizza
   `λ·rilevanza − (1−λ)·max(similarità con le già scelte)`, fino a `N_SENTENCES` frasi. Il parametro
   `LAMBDA` (default 0.7) bilancia salienza e diversità; le frasi scelte sono riportate in ordine di
   documento.

**Due varianti** che differiscono solo per il passo 2 (il vettorizzatore), per misurare se una
rappresentazione semantica migliora le metriche MDS:

- **`centroid_mmr`** — **TF-IDF** (`sklearn.TfidfVectorizer`): lessicale, sparso, rapidissimo su CPU.
- **`centroid_mmr_bert`** — **embeddings BERT** (`all-MiniLM-L6-v2`, lo stesso modello di TextRank):
  semantico, coglie ridondanze anche senza parole in comune. Più pesante (encoding): su
  `test`/`full` conviene la GPU (rilevata via `su.rileva_device()`).

Attesa: in letteratura, su Multi-News, MMR supera LexRank.

Il notebook include una sezione **"Come funziona, passo per passo"** che apre la pipeline su un
documento-esempio (`RIGA_DEMO`) e con un vettorizzatore a scelta (`VETTORIZZAZIONE_DEMO`), con tre
figure: scatter PCA delle frasi + centroide, heatmap della similarità frase-frase (ridondanza) ed
effetto di `λ` sulle frasi selezionate. Con `SALVA_FIGURE` le figure vengono salvate anche come PNG
in `results/figures/centroid_mmr/`. La sezione è puramente illustrativa: `analizza_mmr` (che espone
gli artefatti intermedi) è la stessa funzione usata in produzione da `make_genera`, quindi le figure
riflettono esattamente ciò che il metodo calcola.

## LSA / SVD (notebook 15)

Estrattivo non supervisionato classico (§3.5 del [documento-guida](../Tecniche_MDS_non_LLM_MultiNews.md)):
le frasi del cluster, vettorizzate TF-IDF, vengono proiettate in uno spazio **latente** da una
`TruncatedSVD`; i "concetti" latenti sostituiscono le parole, e una frase è saliente se ha una
forte componente sui concetti principali. Segmentazione via `psr.summarization`, la stessa degli
altri estrattivi, quindi il pool di candidati è identico.

Parametri della corsa committata: `N_SENTENCES = 11` (mediana del corpus, come 01/02/11) e
`K_LATENTE = 11`. Il default `K_LATENTE = N_SENTENCES` segue la convenzione di Gong & Liu (una
dimensione per frase estratta) ed è anche un **vincolo** della variante con deflazione: dopo
`K_LATENTE` sottrazioni lo spazio residuo si annulla. Viene comunque cappato al rango della
matrice del singolo cluster.

**Due varianti** che condividono la stessa SVD e differiscono solo per la regola di selezione:

- **`lsa`** — le `N_SENTENCES` frasi con vettore latente più lungo (‖z‖). È la regola di
  Gong & Liu senza update, la stessa di `sumy.LsaSummarizer`: nessuna memoria di ciò che è già
  stato scelto, quindi nessuna difesa dalla ridondanza tra fonti.
- **`lsa_steinberger`** — greedy con **deflazione**: a ogni passo si prende la frase col residuo
  più lungo e si sottrae la sua direzione da tutte le altre (proiezione ortogonale), così una
  frase che ripete la precedente vale ~0 e non viene riscelta. È l'anti-ridondanza della variante
  multi-documento di Steinberger, assente in `sumy`.

Sulla split test la variante con deflazione è la migliore dei cinque metodi non supervisionati
(ROUGE-1 F1 0,376 contro 0,351 di `lsa`): la conferma che, su MDS, gestire esplicitamente la
ridondanza tra fonti paga.

Come i notebook 11/16/17, include una sezione **"Come funziona, passo per passo"** che apre la
pipeline su un documento-esempio (`RIGA_DEMO`), con figure salvate in `results/figures/lsa/`
quando `SALVA_FIGURE` è attivo. Le figure usano `analizza_lsa`, la stessa funzione della
produzione, quindi riflettono esattamente ciò che il metodo calcola.

## Clustering su sentence embeddings SBERT (notebook 16)

§3.7 del [documento-guida](../Tecniche_MDS_non_LLM_MultiNews.md). Le frasi del cluster vengono
codificate con **`all-MiniLM-L6-v2`** (lo stesso encoder di TextRank
e di `centroid_mmr_bert`), con embedding **L2-normalizzati**, e poi raggruppate in
`N_CLUSTER = 11` gruppi; da ogni gruppo si estrae il **medoide** — la frase più vicina al
centroide, quindi sempre testo originale e mai un centroide sintetico. L'idea è che i cluster
corrispondano ai sottotemi ricorrenti fra le fonti, così il riassunto ne copre uno per gruppo
invece di ripetere il più frequente.

**Due varianti** che differiscono solo per l'algoritmo di clustering:

- **`sbert_kmeans`** — `sklearn.KMeans` (distanza euclidea; su embedding L2-normalizzati è
  monotona rispetto alla cosine).
- **`sbert_agglom`** — `sklearn.AgglomerativeClustering` (metrica cosine, average linkage):
  nessuna assunzione di cluster sferici.

Su questa corsa KMeans è nettamente avanti (ROUGE-1 F1 0,359 contro 0,329), e produce riassunti
più lunghi (264 parole contro 216): la figura sulle taglie dei cluster nella sezione esplicativa
mostra il comportamento dei due algoritmi sullo stesso documento.

L'encoding è l'unico passo pesante: il notebook rileva il device con `su.rileva_device()` e la
corsa committata è stata fatta **su CPU**, quindi il metodo non richiede GPU.
Figure della sezione esplicativa in `results/figures/sbert_clustering/`.

## Topic modeling con LDA (notebook 17)

§3.6 del [documento-guida](../Tecniche_MDS_non_LLM_MultiNews.md). Le frasi vengono vettorizzate
con `CountVectorizer(stop_words='english')` e date in pasto a una
`LatentDirichletAllocation` con `N_TOPICS = 5` (cappato al numero di frasi disponibili):
`theta` dà la distribuzione sui topic di ogni frase e la loro somma il **peso** di ogni topic nel
cluster. Il budget di `N_SENTENCES = 11` frasi viene poi **allocato in proporzione a quei pesi**
(arrotondamento con correzione del residuo, così la somma delle quote torna esatta), e dentro
ogni topic si prendono le frasi con appartenenza più alta. Se i duplicati impediscono di
raggiungere il budget, si completa con le frasi più salienti.

⚠️ **Attenzione alla lunghezza** (vedi anche «Avvertenze metodologiche»): a parità di budget
nominale, le frasi che questa allocazione seleziona sono molto più lunghe, e il riassunto medio
arriva a **477 parole** contro 216–264 degli altri quattro metodi non supervisionati. Va tenuto
presente leggendo le metriche sensibili alla lunghezza.

Figure della sezione esplicativa in `results/figures/lda/`.

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

## Corsa completa sulla split test (notebook 03-04, 06-11, 15-17)

`scripts/run_benchmark_test.py` esegue in **un'unica sessione non presidiata** gli undici notebook
dei metodi non coperti dalla derivazione da `full` (10 First-k, 17 LDA, 15 LSA, 16 SBERT
clustering, 11 Centroid+MMR, 03 BART, 04
PEGASUS, 07 Qwen, 09 Mistral, 08 Gemma, 06 PRIMERA) con `SCOPE='test'` — l'intera split test
pulita, 5.610 righe — dal più veloce al più lento, senza bisogno di riaprire i notebook uno per
uno tra una corsa e l'altra:

```bash
python scripts/run_benchmark_test.py             # corsa completa (~3,5-5 giorni su questa macchina)
python scripts/run_benchmark_test.py --limit 2   # prova end-to-end economica, 2 righe per metodo
python scripts/run_benchmark_test.py --only 10,11  # solo i notebook indicati (il 05 viene comunque rieseguito)
```

`--only` accetta i prefissi numerici dei notebook separati da virgola: utile quando gli altri
metodi hanno già completato la corsa (rieseguirli ricaricherebbe i modelli e ricalcolerebbe le
metriche per nulla). I controlli preliminari si restringono di conseguenza: ollama serve solo
per 07-09, la GPU per 03/04/06/11.

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

## LLM su Azure AI Foundry (notebook 12)

Il notebook 12 replica il protocollo dei notebook 07–09 (stesso prompt zero-shot in inglese,
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

## BERTScore (notebook 13, backfill)

Aggiunge il **BERTScore** (Zhang et al., 2020 — similarità semantica su embedding contestuali
BERT, `roberta-large`) alle metriche `test` di tutti e 18 i metodi, come **backfill**
sui riassunti già generati: non tocca i notebook 01–04/06–12 e 15–17 né il loro ciclo di
generazione/valutazione dal vivo.

Il notebook è **incrementale**: salta i metodi che hanno già le colonne `bertscore_*` nel CSV
per-esempio, perché `su.valuta_e_salva` riscrive CSV e JSON del metodo e ricalcolarli
rigenererebbe file già pubblicati a parità di valori. È così che i cinque metodi dei notebook
15–17 sono stati aggiunti senza toccare i tredici preesistenti (verificato: metriche lessicali e
`config` invariate). Per un ricalcolo integrale si imposta `BERTSCORE_FORZA=1`.

`pyAutoSummarizer` espone un wrapper di comodo `bert_score()`, ma questo ricarica
`roberta-large` da zero a **ogni chiamata** (nessuna cache tra chiamate nella libreria
`bert-score` che chiama) — inutilizzabile dentro il ciclo per-esempio condiviso, che
richiederebbe migliaia di ricaricamenti per metodo. Il notebook 13 usa invece
`bert_score.BERTScorer` **direttamente**: un solo caricamento del modello per metodo, poi
tutte le coppie candidato/riferimento valutate in un'unica chiamata batched
(`su.calcola_bertscore_batch`, `batch_size=64`).

Stima (misurata su questa macchina, GPU RTX PRO 2000 Blackwell Laptop, su un sottoinsieme di
300 righe di `firstk_psr`): caricamento del modello ~6 s (una tantum per metodo) e ~21 righe/s
di scoring puro — **circa 1 ora di calcolo GPU in totale** per tredici metodi sulla split test
(~5.600 righe ciascuno), e 24 minuti misurati per i cinque del backfill successivo. TextRank
e LexRank (che hanno solo una corsa `_full.tsv`, non `_test.tsv` dedicata) vedono anche
ROUGE/BLEU/METEOR ricalcolati direttamente sulle sole righe della split test invece di essere
derivati filtrando la corsa `full` — numericamente equivalente, ma più semplice da tenere in un
unico ciclo uniforme su tutti i metodi. La configurazione storica di ciascun metodo (`config` nel
JSON aggregato) viene preservata, non sovrascritta dal backfill.

## G-Eval — LLM-as-a-Judge (notebook 14, backfill)

Aggiunge una metrica **LLM-as-a-Judge** (G-Eval, Liu et al., 2023): un modello giudice assegna
a ogni riassunto quattro punteggi in scala **1–5** — *coherence*, *consistency* (fedeltà
fattuale), *fluency*, *relevance*. È l'unica metrica del benchmark che **non passa dal
riferimento umano**: ROUGE/BLEU/METEOR misurano sovrapposizione lessicale con il `summary` di
riferimento e il BERTScore similarità semantica con lo stesso, ma nessuna delle due dice se un
riassunto è coerente o se ha inventato dei fatti.

### Perché non `psr.g_eval()`

`pyAutoSummarizer` espone già un G-Eval, ma il suo codice (v1.2.0) lo rende inutilizzabile qui
per **quattro** motivi: (1) costruisce `openai.OpenAI(api_key=...)` **senza `base_url`**, quindi
parla solo con OpenAI e non con Azure; (2) ha `max_tokens=5` e `temperature=0.0` **cablati**,
entrambi fatali per un modello *reasoning*; (3) legge la sorgente da `self.full_txt`, cioè
servirebbe una `psr.summarization(sorgente)` nuova e pesante per **ogni** esempio — il pattern a
istanza fittizia condivisa di `su.crea_valutatore()` non regge; (4) fa **una chiamata API per
dimensione**, quadruplicando il costo.

La funzione è quindi reimplementata in `summ_utils.py`, ma le **rubriche restano verbatim**:
`su.RUBRICHE_GEVAL` è derivato *meccanicamente* da `su.PROMPT_GEVAL_ORIGINALI` (copia letterale
dei template della libreria) tagliando la coda `"Reply with a single digit only."`, qui
sostituita dalla richiesta di un unico oggetto JSON. La metrica resta così difendibile come *il
G-Eval di pyAutoSummarizer, reimplementato per un endpoint non-OpenAI*.

### Scelta del giudice: `gpt-5.4-mini`

Deployment **GlobalStandard** dedicato su Azure AI Foundry (swedencentral), distinto da quello
del notebook 12. Due proprietà lo motivano:

1. **Indipendente da tutti e 18 i metodi valutati**: nessun self-judging. Usare `gpt-5-mini` (il
   modello del notebook 12, presente nel benchmark come metodo `gpt5mini`) significherebbe
   fargli giudicare i propri output, con il noto bias di auto-preferenza.
2. **Più recente di ogni generatore del benchmark** (`gpt-5-mini` è 2025-08-07, `gpt-5.4-mini`
   è 2026-03-17). Conta soprattutto per **consistency**: individuare un'allucinazione sottile
   dentro un cluster multi-documento è un compito di ragionamento, e la correlazione
   giudice–umano scala con la capacità del giudice proprio su quella dimensione.

Verificato dal vivo sulla sottoscrizione: DeepSeek/Grok/Llama/Mistral **non sono deployabili**
su questo account AIServices (stessa ragione per cui i notebook Claude Haiku e DeepSeek furono
abbandonati), e **`gpt-5.1-mini` non esiste** — la linea 5.1 è `gpt-5.1` / `-chat` / `-codex*`.

Essendo un modello *reasoning* valgono le regole del notebook 12: **niente `temperature`**,
`max_completion_tokens=1500` (non 200: i token di ragionamento consumano il budget **prima**
dell'output visibile — stesso modo di fallire di gemma e gpt-5-mini) e
`reasoning_effort='minimal'`. Le corse **non sono riproducibili bit-a-bit**: l'artefatto di
riproducibilità è la cache JSONL committata.

### Protocollo

- **una chiamata per (metodo, riga)**, con tutte e quattro le dimensioni in un unico JSON
  (`response_format` con `json_schema` **strict**: il formato è garantito lato server; in strict
  non sono ammessi `minimum`/`maximum`, quindi la scala 1–5 è espressa con `enum`);
- sorgente troncata a **3.500 parole** (`su.MAX_PAROLE_SORGENTE_GEVAL`): lascia **intero il
  91,5%** dei cluster della split test (mediana 1.288 parole, p90 3.244, p95 4.499) mettendo
  comunque un tetto ai casi estremi, che arrivano a 35.362 parole. È molto più di quanto vedano
  BART/PEGASUS (1.024 token). Il costo marginale rispetto a un tetto di 2.100 parole (copertura
  76%) è di circa **$5 sull'intera corsa**, perché quasi tutte le parole in più finiscono nel
  prefisso condiviso, pagato a tariffa *cached*;
- ambito: split `test` intera, **100.621 giudizi** (meno di 18 × 5.610 perché la copertura è
  disomogenea: firstk_psr, le varianti centroid e i cinque metodi dei notebook 15–17 hanno 5.588
  righe, gpt5mini 5.471), tutti eseguiti: i 72.681 dei tredici metodi originali (2026-08-17) più
  i 27.940 dei cinque aggiunti col backfill dell'issue #12 (2026-08-31).

### Concorrenza e prompt cache

L'ordine dei messaggi è un **contratto, non uno stile**: il `system` (rubriche + formato JSON) è
identico su tutte le chiamate e il `user` comincia con la **sorgente troncata** — la stessa per
tutti i metodi di una riga — mettendo in fondo il **riassunto**, unica parte che cambia.

Di conseguenza `su.giudica_geval_concorrente` prende la **riga** come unità di lavoro ed esegue
i giudizi di quella riga **in sequenza dentro lo stesso thread**: il primo popola la prompt cache
di Azure e i successivi la riusano. Il parallelismo è **tra** righe diverse. Parallelizzare per
singolo giudizio farebbe partire insieme tutte le chiamate della stessa riga, mancando la cache
tutte quante. Nota che l'ammortamento peggiora quando i metodi da giudicare per riga sono pochi:
nel backfill dei cinque metodi dei notebook 15–17 la chiamata che paga il prefisso intero si
divide su 5 giudizi invece che su 13, e la quota di input in cache scende (56% cumulativo contro
il 59% della prima corsa).

### ⚠️ La concorrenza degrada la cache (misurato)

Il numero di thread **non** è un parametro innocuo di velocità: è un fattore di costo. Misurato
confrontando le **sole righe cacheabili** delle due corse (il confronto sui totali grezzi
sarebbe fuorviante, perché i due campioni hanno lunghezze di sorgente molto diverse):

| thread | chiamate con hit, nelle righe cacheabili | hit rate complessivo | proiezione corsa completa |
|---|---|---|---|
| 8 | 45,9% | 36,1% | **~$107** (~3 h) |
| 2 | **64,8%** | **51,1%** | **~$87** (~12 h) |

La causa è il routing: un deployment **GlobalStandard** manda ogni richiesta a una qualunque
istanza di backend e la prompt cache è **per istanza**. Con più righe in volo, le chiamate 2–13
di una riga atterrano più spesso su istanze che non hanno mai visto quel prefisso.

Quindi `--thread` basso costa meno e impiega di più: **~$19 di risparmio per ~9 ore in più**. Il
collo di bottiglia non è la quota TPM del deployment (nessun 429 osservato) ma questo
compromesso.

### Il tetto strutturale: la soglia dei 1.024 token

Azure non attiva la prompt cache sotto i **1.024 token di prefisso**, e i token in cache sono
quantizzati a blocchi di 256. Sulla split test **il 14,8% delle righe (830 su 5.610) ha un
prefisso troppo corto e non entra mai in cache**, indipendentemente da thread e troncamento: le
loro 13 chiamate si pagano tutte a tariffa piena.

Sommato al fatto che, anche nelle righe cacheabili, si arriva a ~65% di chiamate con hit e non
al massimo teorico di 12/13 (92%), il tetto realistico complessivo è intorno al **51%** — non al
75% che una singola riga isolata può far sembrare raggiungibile.

### Ripresa e gestione degli errori

Ogni giudizio è scritto e flushato subito su `results/metrics/geval_cache_{scope}.jsonl` (una
riga JSON per coppia `(metodo, row_id)`, con anche i conteggi di token) e un rilancio salta le
coppie già presenti: Ctrl-C non perde nulla di pagato. Gli errori **permanenti** (content filter
di Azure, 400/401/403) vengono scritti in cache con il campo `errore`, così una riesecuzione non
li ripaga; quelli **transitori** (429, 5xx, timeout) non vengono scritti affatto e passano per
`su.chiama_con_backoff` (backoff esponenziale + jitter, rispetta `Retry-After`).

Poiché la sorgente troncata è identica per tutti i metodi di una riga, un `content_filter` sul
primo giudizio viene **propagato** agli altri dodici senza pagarli. `--riprova-errori` li rimette
in gioco (utile solo dopo aver attaccato al deployment un filtro contenuti *high-only*).

### Perché file separati dalle metriche standard

I punteggi finiscono in `{metodo}_{scope}_geval_per_example.csv` e `..._geval_aggregate.json`,
**mai** uniti al CSV per-esempio standard. Il motivo è in `su.valuta_e_salva`, la cui media
interna somma **ogni** colonna su **ogni** riga: il giudice lascia scoperte alcune righe e la
media esploderebbe con un `KeyError`; e restringere le righe valutate riscriverebbe i CSV già
committati, cambiando medie e `n_esempi` di ROUGE/BLEU/METEOR/BERTScore — in modo drastico per i
cinque metodi dei notebook 15–17, giudicati solo al ~60%. Il notebook 05 aggancia
le colonne con un merge **LEFT** (che non cambia il numero di righe) e riporta la copertura
effettiva nella colonna `n_geval`.

Effetto collaterale utile: G-Eval è immune al fatto che
`scripts/run_benchmark_test.py::deriva_metriche_test` ricalcoli le medie di textrank/lexrank
usando solo `su.COLONNE_METRICHE` — un rilancio del driver **fa cadere le colonne BERTScore**
dagli aggregati di quei due metodi (problema preesistente, non introdotto qui; il notebook 13
lo aggira ricalcolandoli direttamente).

### Configurazione di Azure (una tantum, nel portale)

Creare un deployment **GlobalStandard** di `gpt-5.4-mini` (versione 2026-03-17) sulla risorsa
AI Foundry e annotarne la quota TPM. Le credenziali arrivano **solo** da variabili d'ambiente,
le stesse del notebook 12 (`AZURE_OPENAI_ENDPOINT`, la radice della risorsa senza path, e
`AZURE_OPENAI_API_KEY`); il nome del deployment si sovrascrive con `AZURE_GEVAL_DEPLOYMENT` se
diverso da `gpt-5.4-mini`. `scripts/run_geval.py` verifica il deployment con un ping da 1 token
**prima** di iniziare, così un nome sbagliato costa secondi e non ore.

### Costi e monitoraggio

Prezzi verificati sulla **Azure Retail Prices API** (GlobalStandard, $/1M token: input **0,75**,
input in cache **0,075**, output **4,50**); il notebook li rilegge a ogni avvio con
`su.prezzi_retail_azure()`, con `su.PREZZI_GEVAL` come fallback.

**Misurato sul pilota (269 giudizi):** con `reasoning_effort='minimal'` il giudice emette
**zero token di reasoning** — l'output è di ~30 token per giudizio, cioè **~$10 sull'intera
corsa**. Il termine di output, che a priori sembrava dominante e imprevedibile, di fatto non
conta.

La corsa è quindi **interamente vincolata dall'input**, e la variabile che sposta il totale è
l'**hit rate della prompt cache** (vedi la sezione sulla concorrenza): **~$87 a 2 thread, ~$107
a 8**. Non è il modello, non è il prompt di sistema (che da solo costa $2,23 in tutto) e quasi
non è nemmeno il troncamento: passare da 2.100 a 3.500 parole — cioè dal 76% al 91,5% di
sorgenti intere — costa solo **~$7** in più, perché quasi tutte le parole aggiunte finiscono nel
prefisso condiviso.

**Non esiste un'API Azure che riporti il costo in tempo reale** (Cost Management ha 8–24 h di
ritardo). La fonte di verità è l'oggetto `usage` di ogni risposta, che `su.ContatoreCosti` somma
e stampa ogni 1.500 giudizi con ritmo, TPM osservato, ETA, quota di input in cache, quota di
reasoning, costo per voce e **proiezione a fine corsa**. Gli stessi conteggi sono nella cache,
quindi la stima si rilegge **da un secondo terminale a corsa in corso** con
`python scripts/run_geval.py --costo`. `--budget` è un tetto **complessivo** (somma quanto è
già in cache, non riparte da zero a ogni rilancio): al superamento la corsa si ferma in modo
pulito e basta rilanciare con un tetto più alto.

⚠️ **La valuta del listino non è cosmetica.** La Azure Retail Prices API, senza parametro,
ritorna prezzi in **USD** — ma la sottoscrizione potrebbe fatturare in un'altra valuta, e il
listino di Azure per quella valuta **non è una conversione al cambio del momento**: è un
listino a sé, verificato qui a **~0,8776× il numero USD su ogni meter** (input, cache, output
identicamente). La prima corsa completa è stata tracciata come "$36,00" con il listino USD di
default; il credito Azure realmente consumato, verificato 24 h dopo (tempo sufficiente perché
il ritardo di Cost Management si esaurisca), corrispondeva a **€31,59** — un divario del 12%
dovuto **non** al ritardo di rendicontazione ma alla valuta sbagliata nel calcolo. Usare
`su.prezzi_retail_azure(valuta='EUR')` (o `--valuta EUR` da riga di comando) quando la
sottoscrizione fattura in euro; il numero di token contati resta comunque esatto in entrambi
i casi, cambia solo la cifra.

### Avvertenze

- **Copertura parziale**: alcune righe vengono respinte dal content filter di Azure o producono
  risposte non conformi — 1.071 giudizi su 100.621 (1,1%) dopo il ritentativo del 2026-09-19
  (erano 5.855, il 5,8%, nella corsa di agosto), per una copertura per metodo fra 96,8% e
  99,0%. Le medie vanno lette insieme a `n_geval`, che può essere minore di `n_esempi`.
  **Le righe respinte non sono le stesse in corse diverse** (vedi l'avvertenza dedicata più
  sotto): la copertura più alta dei cinque metodi dei notebook 15–17 dipende dal filtro, non dai
  metodi.
- **Bias del giudice**: un LLM giudice tende a premiare i testi **più lunghi** e quelli generati
  da altri LLM. Il confronto fra metodi estrattivi (che qui producono 216–450 parole) e
  astrattivi (55–210) su questa metrica va preso con cautela.
- **Non riproducibile bit-a-bit** (modello reasoning, `temperature` non impostabile): riprodurre
  i numeri significa ripartire dalla cache committata, non rilanciare le chiamate.
- Cambiare deployment, rubriche o troncamento **dopo** aver popolato la cache mescolerebbe corse
  diverse: in quel caso cancellare prima `geval_cache_{scope}.jsonl`.

## Analisi per cluster delle lunghezze (notebook 18)

Non genera né rigenera nulla: legge la colonna `parole_generate` già presente nei CSV
per-esempio committati e la incrocia, in un'unica scansione in streaming di
`data/tab/complete.tab`, con la lunghezza del riassunto di riferimento e il numero di articoli
di ogni cluster. Nasce dalla richiesta del relatore di verificare le lunghezze e, in caso di
disomogeneità significative, contenere la distribuzione dentro un intervallo comune — dettagli
e decisioni in [issue #15](https://github.com/girasella/multi-news-ai4stem-polito-master/issues/15)
(sostituisce la #14, limitata a `lda`/notebook 15–17).

Le medie aggregate per metodo (55 parole per `bart` → 477 per `lda`, viste nel notebook 05)
nascondono la dispersione al livello a cui il confronto avviene davvero: **dentro un singolo
cluster**, il riassunto più lungo dei 18 metodi è in mediana **8,5 volte** il più corto — una
distanza pari a circa il doppio del riassunto di riferimento di quel cluster. Nessun metodo si
adatta al cluster: la correlazione fra lunghezza generata e lunghezza del riferimento arriva al
massimo a 0,51 (`primera`) ed è circa zero per gli estrattivi a budget di frasi (`lsa`,
`lsa_steinberger`, `sbert_agglom`), che con "11 frasi" producono la stessa lunghezza qualunque
sia la dimensione del cluster.

Il riferimento inoltre non è costante: la sua mediana cresce con il numero di articoli, da ~164
parole (1 articolo) a ~327 (8+). Il candidato naturale a budget per-cluster è quindi
`T(n_articoli)`, la mediana del riferimento per bucket di articoli, **tarata sulla split train**
e scritta in `scripts/budget_lunghezza.json`. La **Vista 5** del notebook mette però alla prova
questo e altri candidati calcolabili dal solo input (decili di lunghezza della sorgente,
regressioni log-log, combinazioni), tarandoli su train e valutandoli su test, e l'esito è
netto: **nessuno prevede bene la lunghezza del riferimento**. Il migliore mette il riferimento
entro ±25% del budget nel 58% dei cluster contro il 54% di un budget **costante**, con
correlazione di rango mai sopra 0,46; la tendenza per numero di articoli è reale ma spiega
poco del singolo cluster (l'81% dei cluster di test ha 2–3 articoli, dove `T` ≈ la costante).
L'idea intuitiva «scala con la lunghezza della sorgente» va peggio di tutte (MAE ~138 parole),
perché il rapporto di compressione varia di 5× fra cluster.

Ne segue che il budget che davvero elimina la lunghezza come confondente è il **riferimento
stesso, cluster per cluster** (protocollo *length-matched* / *oracle-length*, es. Sun et al.
2019): usa la sola *lunghezza* gold, mai il contenuto, e va dichiarato come tale perché risponde
a «quanto è buona la selezione dei contenuti a parità di lunghezza», non a «cosa otterrebbe il
metodo in produzione». La sua applicazione ai 18 metodi è documentata in una issue dedicata.

Output: `results/metrics/analisi_lunghezze_{ambito}.json` (committato, stesso pattern di
`scripts/dataset_stats.json`; include la tabella del confronto fra predittori) e
`scripts/budget_lunghezza.json` (scritto solo se mancante).

## Ambito `test_budgetref` — confronto a lunghezza del riferimento (issue #16)

Il seguito operativo del notebook 18: un ambito in cui ogni metodo ha, per ogni cluster, **lo
stesso budget di parole — la lunghezza del riassunto di riferimento** (`su.budget_riferimento`).
È il protocollo *length-matched* / *oracle-length* (Sun et al., 2019): con
`|candidato| ≈ |riferimento|` richiamo e precisione ROUGE coincidono e l'obiezione «i riassunti
lunghi vincono sul recall» non può più nascere per costruzione. Usa un'informazione gold — la sola
*lunghezza*, mai il contenuto — e va dichiarato come tale: i numeri rispondono a «quanto è buona la
selezione dei contenuti a parità di lunghezza», **non** a «cosa otterrebbe il metodo in
produzione», e non sostituiscono i numeri di testa dell'ambito `test`.

Si attiva con `SUMM_SCOPE='test_budgetref'`: i notebook leggono le stesse righe dell'ambito `test`
(`su.split_base`) e scrivono file con suffisso `_test_budgetref`; l'ambito `test` committato non
viene toccato. Due interventi distinti, perché il solo troncamento non basta (porta il rapporto
max/min mediano per cluster da 8,5× a ~4×: `bart` resta sotto banda nel 100% dei cluster, `qwen`
nel 99%, e un riassunto corto non si tronca verso l'alto):

- **Tetto, tutti e 18 i metodi** — `scripts/applica_budget.py` tronca ogni riassunto a
  `1,25 × budget` e ricalcola le metriche (ROUGE/BLEU/METEOR + BERTScore).
- **Rigenerazione a budget, gli 11 estrattivi** (notebook 01, 02, 10, 11, 15, 16, 17; driver
  `scripts/run_benchmark_test.py --scope test_budgetref`) — rigenerati con il budget in **parole**
  al posto del budget in **frasi**. Il criterio di ordinamento di ciascun metodo è invariato;
  cambia solo *dove* ci si ferma (`su.seleziona_per_budget`: si accumulano frasi nell'ordine del
  metodo, la frase che sfora entra se ne entra più della metà). Adattamenti per notebook: in 01/02
  si usa l'ordine di rank di `show_summary`; in 10 l'ordine è round-robin fra gli articoli (1ª
  frase di ognuno, poi 2ª, …); in 11 e 15 il greedy produce un ordine lungo fino a
  `K_ORDINE_BUDGET = 50` frasi poi tagliato; in 15 e 16 il parametro **strutturale**
  (`K_LATENTE`, `N_CLUSTER`) resta 11 — vincolo segnalato nella issue #14 — e in 16 l'ordine è a
  giri (medoide di ogni cluster, poi la seconda frase più vicina al centroide, …); in 17
  l'allocazione proporzionale ai topic ripartisce parole invece di frasi.
- **Rigenerazione a budget, gli LLM** (notebook 07, 08, 09, 12) — rigenerati con il budget **nel prompt**: alla
  stessa richiesta zero-shot si aggiunge la sola frase «of approximately *N* words»
  (`PROMPT_USER_BUDGET`, derivato meccanicamente da `PROMPT_USER`), il limite di token sale a
  `MAX_TOKENS_BUDGET = 1500` (con 200/300 un bersaglio da 300 parole sarebbe irraggiungibile). Un
  LLM segue l'istruzione di lunghezza solo **approssimativamente**: la lunghezza obiettivo è raggiunta
  *circa*, il tetto si applica comunque dopo, e la quota in banda è un risultato misurato. Non
  sono nella lista del driver: si lanciano singolarmente, pilota su poche righe prima della corsa
  (ore per modello).
- **Solo tetto per `bart`, `pegasus`, `primera`** — decisione deliberata, non solo di costo
  (pur essendo 2 h + 5–9 h + 28–56 h di GPU). La rigenerazione a budget qui passerebbe da `min_length` in
  token, che costringe il beam search a generare oltre la lunghezza naturale del modello: e' il
  meccanismo dei loop di ripetizione gia' documentati per PEGASUS (riga 51178, METEOR −1959, alla
  lunghezza *naturale*). Su `bart`, che dovrebbe quadruplicare l'output, si misurerebbe il degrado
  da allungamento forzato invece della selezione dei contenuti. `pegasus` (mediana 0,83× il
  riferimento) e `primera` (0,97×) sono comunque gia' al bordo o dentro la banda da soli; `bart`
  (0,26×, 0% in banda) resta l'**eccezione dichiarata** del confronto e va letto come tale.

**G-Eval non è ricalcolato** (cache indicizzata su `(metodo, row_id)`, ri-giudizio ~€70–95; misura
la fedeltà alla fonte, non la sovrapposizione col riferimento). Il confronto prima/dopo è la
**Vista 3 del notebook 05**; la dispersione per cluster «dopo» si ottiene rieseguendo il notebook
18 con `SUMM_SCOPE='test_budgetref'`.

**Esito finale (corse 2026-09-13/17: 15 slug su 18 rigenerati, tetto su tutti e 18).** I 15
rigenerati cadono in banda nel **71–100%** dei cluster (gli 11 estrattivi 93–99%, `gpt5mini` 100%,
`gemma` 99%, `qwen` 79%, `mistral` 71%), con lunghezza mediana pari al riferimento (~215 parole).
Il rapporto max/min dentro il cluster passa da 8,5× a 4,7× sui 18 — ma quel 4,7 è quasi tutto
`bart`, che a 0,26× è il minimo in quasi ogni cluster mentre il tetto tiene il massimo a 1,25×:
**senza `bart` si passa da 3,8× a 1,6×, e sui soli 15 rigenerati da 3,7× a 1,4×**. Fra i metodi
effettivamente riportati a lunghezza pari la dispersione è quindi quasi annullata — il numero sui
18 sottostima il contenimento.

La graduatoria si riassesta: `lda` perde tutto il vantaggio (recall 0,499 → 0,348, ultimo fra gli
estrattivi in F1, 0,338); le baseline `firstk_*` salgono dal 9°–10° al **5°–6° posto** (F1 0,368),
sopra metodi di selezione molto più elaborati; i metodi corti **guadagnano**, perché la rigenerazione a budget
dà loro lo spazio che non usavano (`qwen` 0,344 → 0,351, `mistral` 0,354 → 0,365) — prova che il
confondente tagliava in entrambe le direzioni, non solo a favore dei lunghi; `primera` (0,446) e
`pegasus` (0,424), già a lunghezza di riferimento e non rigenerati, restano in testa: il loro
vantaggio non era di lunghezza. Il fattore del prompt è calibrato **per modello** (`qwen` 1,2,
`gemma` 0,9, `mistral` 1,0, `gpt5mini` 0,9): i modelli sbagliano in direzioni opposte, e `mistral`
resta molto disperso (p10 0,76×, p90 1,98×) contro il 99–100% in banda di `gemma` e `gpt5mini`.

⚠️ **Content filter di Azure e filtro custom.** La corsa `test` di `gpt5mini` (vecchio account,
policy di default) copriva 5.471 righe su 5.610; la prima corsa `test_budgetref` sul nuovo
account ne copriva solo **5.250** (360 rifiuti, tutti `content_filter`, 6,4%), perche' la
severita' del filtro e' configurata **per risorsa** e non solo instabile fra corse. Attaccando al
deployment un **filtro contenuti custom permissivo** e rilanciando le sole righe mancanti, 312
delle 360 sono state recuperate: copertura finale **5.562/5.610 = 99,1%**, la piu' alta di tutte
le corse Azure del progetto. Le 48 righe residue restano respinte anche dalla policy permissiva.
La provenienza resta pulita — stessa risorsa, stesso modello, stesso prompt, stessi parametri:
cambia solo la policy del filtro, che regola l'**accesso** e non altera la generazione. Effetto
sul confronto: l'intersezione comune ai 18 metodi sale a **5.527** cluster, sopra i 5.437 di
partenza. **Per riprodurre la corsa serve il filtro custom sul deployment**, altrimenti si
ricade a ~93,6% di copertura.

Avvertenza: sulla riga 50540 (sorgente degenere, un elenco di birrifici) il METEOR non limitato di
pyAutoSummarizer esplode (−77.557 per `lda`, −3.074 per `lexrank`) e trascina le medie a −13,5 e
−0,16 contro ~0,40 e ~0,39 reali — stesso fenomeno della riga 51178 di PEGASUS, stessa scelta:
nessun file modificato.

## Validazione del giudice G-Eval (notebook 19)

Il G-Eval committato ha **un solo giudice**, `gpt-5.4-mini`, e fra i 18 metodi giudicati ce n'è
uno della stessa famiglia: `gpt5mini`. `CLAUDE.md` ha sempre rivendicato l'indipendenza a
livello di *modello* («no self-judging»), mai a livello di **famiglia**. La domanda è legittima
perché i numeri la sollevano da soli: fra i quattro LLM l'ordine G-Eval è esattamente invertito
rispetto a BERTScore e ROUGE-1 F1, e `gpt5mini` è terzo su quattro sulle metriche ancorate al
riferimento ma primo in assoluto su G-Eval (4,89).

[19_confronto_giudici.ipynb](19_confronto_giudici.ipynb) risponde con un **confronto appaiato**:
gli stessi riassunti, già generati e già giudicati, vengono rigiudicati da
`DeepSeek-V3.2-Speciale` — lignaggio completamente diverso, stesso prompt, stesse rubriche. Il
campione è di 1.000 righe estratte con seme fisso su 7 metodi: i 4 LLM più **3 controlli
non-LLM** (`primera`, `firstk_psr`, `lexrank`), che sono la parte che fa funzionare il disegno,
perché separano un effetto generico «al giudice piace la prosa LLM» da un effetto specifico
della famiglia OpenAI. La corsa è
[`scripts/pilota_giudice_deepseek.py`](../scripts/pilota_giudice_deepseek.py); scrive su una
cache e un JSON **separati**, quindi non tocca nulla di committato.

Esito su 6.799 giudizi appaiati (983 cluster, 2026-09-17/18, €9,16):

- **ordinamento identico**, ρ di Spearman **1,000**, nessuna inversione su sette posizioni;
- il contrasto che misurerebbe il bias — `delta(gpt5mini) − media(delta degli altri 3 LLM)`,
  calcolato riga per riga — vale **+0,049** con IC95% `[+0,022, +0,076]`. Il family bias
  prevedeva un contrasto **negativo**: il segno è quello sbagliato per l'ipotesi, quindi
  correggere per questo effetto *allargherebbe* il primato di `gpt5mini` invece di ridurlo.

Due avvertenze da non perdere. **Lo zero è fuori dall'intervallo**: con ~970 cluster appaiati
l'errore standard è 0,014 e anche cinque centesimi di punto risultano statisticamente
distinguibili, quindi non si scriva «compatibile con zero» — ciò che rende conclusivo il
risultato è il *segno*, non la significatività. E i due giudici **non sono intercambiabili**:
DeepSeek è più severo sui metodi non-LLM (`−0,276` contro `+0,005` sugli LLM) e quindi allarga
di 0,28 punti il distacco fra prosa LLM e prosa estrattiva. L'effetto «al giudice piace la
prosa LLM» è reale ed è più forte nel giudice DeepSeek che in quello OpenAI: è una proprietà
del paradigma LLM-as-a-Judge, condivisa, non una distorsione di famiglia. Il disaccordo si
concentra su `coherence` (−0,32) e `relevance` (−0,30, quasi tutta sui controlli: −0,675 contro
−0,017 sugli LLM), mentre `fluency` è identica e `consistency` va in senso opposto (+0,19).

**Perché il secondo giudice resta un pilota.** Non per qualità ma per portata del deployment:
20 RPM nominali, **6,5 giudizi/min effettivi** (il resto è attesa di backoff sui 429), misurati
su 17,99 h di corsa. Estrapolato ai 100.621 giudizi di un ambito fa **10,8 giorni e €132** per
ambito, 21,6 giorni per entrambi. `gpt-5.4-mini` copre lo stesso ambito in 3–12 h **e costa
meno per giudizio** (€93 contro €132) nonostante un listino per token più alto, perché il suo
deployment GlobalStandard ha la **prompt cache**, che un deployment MaaS non offre: la sorgente
troncata condivisa fra i metodi di una riga — il motivo per cui l'ordine dei messaggi è un
contratto nel notebook 14 — con DeepSeek si ripaga a ogni chiamata. Alzare i thread non aiuta;
servirebbe un aumento di quota su `DeepSeek-V3.2-Speciale`.

Conseguenza operativa: il G-Eval dell'ambito `test_budgetref` si esegue con **`gpt-5.4-mini`**,
stesso giudice dei numeri pubblicati (quindi continuità del confronto prima/dopo), con questa
validazione appaiata come garanzia metodologica.

### G-Eval nell'ambito `test_budgetref`

Il notebook 14 accetta `GEVAL_SCOPE=test_budgetref` (driver: `scripts/run_geval.py --scope
test_budgetref`). Legge le righe della split base, prende i riassunti dalla corsa a budget quando
esiste e da `_test.tsv` altrimenti, e **riapplica il tetto a 1,25 × riferimento**
(`su.tetto_riferimento`, lo stesso di `applica_budget.py`) prima di giudicare, così il giudice
vede esattamente il testo su cui sono state calcolate le metriche dell'ambito. I giudizi dei testi
rimasti identici all'ambito `test` — bart 94 %, pegasus 84 %, primera 74 % — vengono copiati dalla
cache `test` invece di essere ripagati (voci con `riuso_da`, zero token). Il giudice è lo stesso
`gpt-5.4-mini` dei numeri pubblicati, validato nel notebook 19. L'eseguito va in
`results/notebook_runs/test_budgetref/`, non in-place.

**Esito (corsa 2026-09-18/19, 100.712 giudizi, 99.040 riusciti, €76,26 di spesa utile): la
lunghezza non era il confondente del G-Eval.** Dove il riallineamento ha ribaltato il recall
ROUGE e rimescolato la sua graduatoria, **le prime undici posizioni del G-Eval restano
identiche** e il massimo spostamento è 0,34 su 5. I due estrattivi più lunghi *guadagnano*
tagliando (`lexrank` +0,34, `textrank` +0,27: coerenza, consistenza e fluenza in su, pertinenza
in lieve calo); la **pertinenza è l'unica dimensione sensibile alla lunghezza**, controparte del
recall (`lda` −0,50, `centroid_mmr*` −0,28, `lsa_steinberger` −0,23); fra gli LLM allungati
`qwen` +0,15 supera `mistral` −0,13 (tutto sulla consistency: costretto a scrivere di più,
mistral aggiunge cose che nella fonte non ci sono), `gpt5mini` non si muove. Tabella e grafico
prima/dopo nella Vista 3 del notebook 05.

## Parametri principali (cella di configurazione di ogni notebook)

- `N_SAMPLES`, `SEED` — identificano il file campione; devono combaciare con il notebook 00.
- `SCOPE` — `'sample'` = campione condiviso (tutti i metodi); `'full'` = intero `complete.tab`,
  56.101 esempi in streaming (01/02, 10/11, 12 e 15-17); `'test'` = intera split test, 5.610
  esempi in streaming (03-04, 06-11, 12 e 15-17; nei notebook 03-04, 06-11 e 15-17 letto dalla
  variabile d'ambiente `SUMM_SCOPE`, impostata da `scripts/run_benchmark_test.py` — default
  `'sample'` se assente).
- `LIMIT` — `None` per la corsa completa; un intero piccolo (es. `3`) per uno smoke test. Nei
  notebook 03-04, 06-11 e 15-17 letto anche dalla variabile d'ambiente `SUMM_LIMIT` (usata da
  `run_benchmark_test.py --limit N`).
- `BUDGET` (01/02, 07-12, 15-17) — `su.budget_riferimento` quando `SCOPE` è un ambito `*_budgetref`,
  altrimenti `None`: il canale con cui `ciclo_summarization` passa a `genera()` la lunghezza
  obiettivo del cluster. Con esso `K_ORDINE_BUDGET` (11, 15), `MAX_TOKENS_BUDGET` e
  `PROMPT_USER_BUDGET` (07-09, 12) — vedi la sezione sull'ambito `test_budgetref`.
- `N_SENTENCES` (01/02, 11 e 15-17) — frasi estratte per riassunto (default 11, la mediana di
  frasi per riassunto del corpus; i riassunti estratti risultano comunque più lunghi dei
  riferimenti, perché le frasi di cronaca sono più lunghe di quelle dei digest). Attenzione: a
  parità di questo parametro la lunghezza in **parole** varia molto tra metodi — vedi
  l'avvertenza su LDA in fondo.
- `K_LATENTE` (15), `N_CLUSTER` (16), `N_TOPICS` (17) — dimensioni latenti della SVD, numero di
  cluster e numero di topic; i primi due valgono `N_SENTENCES` per costruzione, il terzo è
  indipendente (default 5, cappato al numero di frasi disponibili).
- `RIGA_DEMO`, `SALVA_FIGURE` (11 e 15-17) — riga usata dalla sezione esplicativa «Come funziona,
  passo per passo» e salvataggio delle sue figure in `results/figures/{metodo}/`.
- `MODELLO`, `OLLAMA_URL`, `MAX_TOKENS`, `TEMPERATURE` (solo 07–09) — tag del modello ollama
  (verificare con `ollama list`), endpoint e parametri di generazione.
- `DEPLOYMENT`, endpoint da variabili d'ambiente (solo 12) — nome del deployment Azure e
  parametri del client (vedi la sezione Azure sopra); il notebook 12 usa la rotta **v1** di
  Azure OpenAI (`<endpoint>/openai/v1/`, senza api-version datata).

## File prodotti

```
results/
  sample/sample_{N}_seed{S}.tsv        # campione condiviso (row_id, split, document, summary) — solo input
  summaries/{metodo}_{scope}.tsv       # riassunti generati (row_id, generated_summary); scope = test o full
  metrics/{metodo}_{scope}_per_example.csv   # ROUGE-1/2/L (F1,P,R), BLEU, METEOR per esempio
                                              # (+ BERTScore F1/P/R sulla split test, dopo 13_bertscore.ipynb)
  metrics/{metodo}_{scope}_aggregate.json    # medie complessive e per split + configurazione usata
  metrics/{metodo}_{scope}_geval_per_example.csv  # G-Eval 1-5 per esempio (14_geval.ipynb), file
  metrics/{metodo}_{scope}_geval_aggregate.json   # SEPARATI: vedi la sezione G-Eval per il perché
  metrics/geval_cache_{scope}.jsonl          # cache dei giudizi (una riga per metodo+row_id, con i
                                              # conteggi di token): è l'artefatto PAGATO, va committato
  figures/{metodo}/*.png                     # figure della sezione esplicativa dei notebook 11 e 15-17
                                              # (solo con SALVA_FIGURE; illustrative, non usate dalle metriche)
  metrics/analisi_lunghezze_{ambito}.json    # dispersione per cluster + T(n_articoli) (18_analisi_lunghezze.ipynb)
  metrics/geval_cache_test_deepseek.jsonl    # cache del SECONDO giudice (pilota, 6.997 giudizi): artefatto
                                              # pagato come l'altra cache, va committato
  metrics/confronto_giudici_test.json        # esito del confronto fra i due giudici (19_confronto_giudici.ipynb)
  summaries/{metodo}_test_budgetref.tsv      # riassunti rigenerati a lunghezza del riferimento (issue #16):
                                              # gli 11 estrattivi via driver --scope, gli LLM col budget nel prompt
  metrics/{metodo}_test_budgetref_*          # metriche post-troncamento per tutti e 18 (scripts/applica_budget.py)
  notebook_runs/{ambito}/*.ipynb             # notebook eseguiti dal driver negli ambiti a budget (in .gitignore)
scripts/
  budget_lunghezza.json                      # T(n_articoli): budget per-cluster (18_analisi_lunghezze.ipynb)
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
| First-k, `test` (5.610, **entrambe le varianti**) | ~3 min | ~3 min (nessun modello) |
| Centroid+MMR, `test` (5.610, **entrambe le varianti**) | non misurata (TF-IDF rapida, BERT lenta senza GPU) | ~8 min (corsa reale 2026-07-25, encoding BERT su GPU) |
| LSA (15) e LDA (17), `test` (5.610) | durata non registrata; solo scikit-learn, nessun modello da caricare | — (non serve la GPU) |
| SBERT clustering (16), `test` (5.610, **entrambe le varianti**) | durata non registrata; corsa committata fatta **su CPU** (encoding MiniLM) | — (la GPU accelera solo l'encoding) |
| GPT-5-mini (12), campione 100 | ~5–15 min (dipende dalla latenza dell'API) | — |
| GPT-5-mini (12), split test 5.610 | ~8 h sequenziali (corsa reale 2026-07-17: ~5 s/esempio) | — |
| GPT-5-mini (12), `full` intero dataset | ~2–4 giorni di chiamate sequenziali (riprendibile) | — |
| BERTScore (13), tutti e 18 i metodi su `test` (5.610 righe ciascuno) | sconsigliata (`roberta-large`, migliaia di forward pass) | ~1 h per tredici metodi + 24 min per i cinque dei notebook 15–17 (misurato: caricamento ~6 s/metodo + ~19-21 righe/s di scoring — vedi sezione dedicata) |
| G-Eval (14), pilota 20 righe (260 giudizi) | ~1 min (misurato, 8 thread) | — |
| G-Eval (14), i tredici metodi originali su `test` (72.681 giudizi) | ~3 h a 8 thread (~6,7 giudizi/s misurati) ma **~$111**; ~12 h a 2 thread per **~$56** — il compromesso è costo/tempo, non CPU. Riprendibile in qualunque momento | — |
| G-Eval (14), backfill dei cinque metodi 15–17 su `test` (27.940 giudizi) | **€32** in ~5 h a 2 thread, in due sessioni (la prima fermata dal tetto `--budget` al ~60%) | — |

Al primo avvio vengono scaricati i modelli da Hugging Face (MiniLM ~90 MB; BART ~1,6 GB;
PEGASUS ~2,3 GB; PRIMERA ~1,8 GB; roberta-large, per il BERTScore del notebook 13, ~1,4 GB).

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
- **G-Eval, bias del giudice e copertura**: il G-Eval del notebook 14 è l'unica metrica non
  ancorata al riferimento umano, ma un LLM giudice ha bias noti — premia i testi **più lunghi** e
  quelli generati da altri LLM — quindi il confronto estrattivi vs astrattivi su questa metrica
  va preso con cautela. La copertura è inoltre **parziale** (content filter di Azure, risposte
  non conformi): le medie vanno lette insieme alla colonna `n_geval` del notebook 05, che può
  essere minore di `n_esempi`. Il giudice è comunque **indipendente da tutti e 18 i metodi**
  valutati, quindi non c'è self-judging.
- **Lunghezza dei riassunti LDA**: i notebook 15/16/17 condividono lo stesso budget nominale
  (`N_SENTENCES = 11`), ma le frasi che l'allocazione proporzionale ai topic di LDA seleziona
  sono in media molto più lunghe: **477 parole** per riassunto contro 249 (`lsa`), 257
  (`lsa_steinberger`), 264 (`sbert_kmeans`) e 216 (`sbert_agglom`). È questo divario, più che la
  qualità della selezione, a spingere in alto le metriche sensibili alla lunghezza: LDA ha il
  ROUGE-1 **recall** più alto del gruppo (0,499 contro 0,357 di `lsa`) e il METEOR più alto
  (0,511), ma in **F1** resta alla pari con `lsa` (0,351) e sotto `lsa_steinberger` (0,376). Su
  questi cinque metodi conviene quindi leggere l'F1, non il recall; la colonna `parole_generate`
  del notebook 05 rende il confronto esplicito. Il confronto a lunghezza pari è stato poi
  fatto (ambito `test_budgetref`, issue #16, Vista 3 del notebook 05) e **chiude la questione**:
  con il budget in parole `lda` scende all'ultimo posto degli estrattivi in F1 (0,338) e il suo
  recall cala da 0,499 a 0,348 — il vantaggio era interamente di lunghezza.
- **Il content filter di Azure non è stabile nel tempo**: BERTScore e G-Eval ci sono ora per
  tutti e 18 i metodi, ma i due backfill sono stati eseguiti in due momenti diversi (i tredici
  metodi il 2026-08-17, i cinque dei notebook 15–17 il 2026-08-31) e le righe che Azure respinge
  sono cambiate nel frattempo. Nella prima corsa il rifiuto era di fatto **determinato dalla
  sorgente** (unione 366 righe respinte sui tredici metodi, intersezione 358: quasi le stesse per
  ognuno, come ci si aspetta da un filtro che scatta sul testo condiviso). Nella seconda, 155 di
  quelle righe passano, 91 nuove vengono respinte, e fra i cinque metodi l'insieme dei rifiuti
  varia molto di più (unione 294, intersezione 187). Di qui la copertura **più alta** dei cinque
  (94,7–96,7%) rispetto ai tredici (93,5–93,6%): è una proprietà del filtro al momento della
  corsa, non dei metodi. L'effetto sul confronto è trascurabile — restringendo ogni media alle
  **5.091 righe giudicate per tutti e 18** i metodi nessuna si sposta di più di **0,010**, contro
  un IC 95% di ±0,03 — ma va tenuto presente prima di leggere le differenze di `n_geval` come se
  dicessero qualcosa sui metodi. Riallineare i tredici alla nuova versione del filtro
  costerebbe una nuova corsa completa (~€70) e cambierebbe numeri già pubblicati: non è stato
  fatto. È stato invece fatto, il 2026-09-19 per €2,09, **il ritentativo dei soli giudizi
  falliti** (`run_geval.py --scope test --riprova-errori`): i 5.855 fallimenti stavano su 457
  righe, 381 delle quali erano passate due giorni prima nella corsa `test_budgetref` sulla
  stessa risorsa. Nessun giudizio esistente è stato toccato; i fallimenti sono scesi a 1.071, i
  riusciti a **99.550 su 100.621 (98,9%)**, le righe giudicate per tutti e 18 da 5.091 a
  **5.363**. Graduatoria invariata, spostamento massimo delle medie 0,010 (pegasus): le righe
  che il filtro aveva bloccato non sono sistematicamente diverse dalle altre. Il divario di
  copertura «tredici contro cinque» descritto sopra è quindi storico.
- **La logica «sottoinsieme» del notebook 05 va lasciata com'è**: ogni grafico BERTScore/G-Eval
  disegna i metodi che hanno quella metrica, elencando gli esclusi in un avviso, invece di
  pretenderla da tutti. Adesso che tutti e 18 le hanno l'avviso non compare mai, ma è ciò che
  evita di far sparire i grafici la prossima volta che si aggiunge un metodo prima del backfill.
- **METEOR non affidabile per output degeneri**: la formula `meteor()` di pyAutoSummarizer
  (`meteor = fmean * (1 - penalty**3)`) non è limitata a [0,1]. Nella corsa `test`, PEGASUS
  produce due riassunti patologici (un loop di ripetizione del beam search e un probabile
  mismatch sorgente/riferimento) con METEOR rispettivamente -1959.12 e -2.10, che trascinano la
  sua media riportata da ~0.42 a 0.079 — solo quella colonna va letta con questa avvertenza (vedi
  il dettaglio nel notebook 05). LexRank ha un caso molto più lieve, con effetto trascurabile.
