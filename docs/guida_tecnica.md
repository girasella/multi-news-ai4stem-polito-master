# Guida tecnica: come è costruito il benchmark

*Project work finale — Master di II livello "Artificial Intelligence for STEM", Politecnico
di Torino*

---

## Indice

1. [A cosa serve questa guida](#1-a-cosa-serve-questa-guida)
2. [Il flusso dei dati in una pagina](#2-il-flusso-dei-dati-in-una-pagina)
3. [Dai file grezzi a `complete.tab`](#3-dai-file-grezzi-a-completetab)
4. [Lo scheletro comune dei notebook dei metodi](#4-lo-scheletro-comune-dei-notebook-dei-metodi)
5. [Gli ambiti di esecuzione](#5-gli-ambiti-di-esecuzione)
6. [I metodi, notebook per notebook](#6-i-metodi-notebook-per-notebook)
7. [La valutazione](#7-la-valutazione)
8. [Il confronto a lunghezza pari, lato codice](#8-il-confronto-a-lunghezza-pari-lato-codice)
9. [I notebook di confronto e di analisi](#9-i-notebook-di-confronto-e-di-analisi)
10. [Gli script e i driver non presidiati](#10-gli-script-e-i-driver-non-presidiati)
11. [Riproducibilità](#11-riproducibilità)
12. [Scelte implementative e insidie note](#12-scelte-implementative-e-insidie-note)
13. [Come aggiungere un nuovo metodo](#13-come-aggiungere-un-nuovo-metodo)

---

## 1. A cosa serve questa guida

La [guida di studio](guida_studio.md) spiega il *perché* del progetto: la teoria dietro ogni
metodo e ogni metrica, e come leggere i risultati. Questa guida spiega il *come*: com'è
organizzato il codice, come sono costruiti i notebook, come girano gli script, e quali scelte
implementative stanno dietro ai numeri.

Le due guide hanno la stessa struttura per famiglie di metodi, così si possono leggere in
parallelo: la sezione 6 di questa guida corrisponde alla sezione 4 di quella di studio, la
sezione 7 alla 5, la sezione 8 alla 7.

Il riferimento operativo dettagliato resta nei README di cartella: parametri, tempi e comandi
in [notebooks/README.md](../notebooks/README.md), uso degli script in
[scripts/README.md](../scripts/README.md), formati dei dati in [data/README.md](../data/README.md).
Qui si spiega come i pezzi stanno insieme.

**Tre principi** attraversano tutto il codice e conviene averli in mente fin dall'inizio:

1. **Generare e valutare sono due passi separati.** I riassunti vengono scritti su file man
   mano che si generano; le metriche si calcolano dopo, leggendo *solo* quei file. La
   valutazione si può rifare in pochi secondi senza rigenerare nulla.
2. **Ogni corsa lunga è riprendibile.** Un'esecuzione interrotta riparte dalle righe mancanti,
   perché ogni riga prodotta è salvata subito su disco.
3. **Le routine comuni stanno in un solo posto.** Caricamento dei dati, ciclo di generazione,
   metriche, G-Eval e grafici di confronto vivono in
   [notebooks/summ_utils.py](../notebooks/summ_utils.py). Ogni notebook contiene solo ciò che
   è specifico del suo metodo.

---

## 2. Il flusso dei dati in una pagina

```
data/text/*.src.cleaned, *.tgt          testo come rilasciato (56.216 righe)
        |  scripts/convert_to_tab.py    pulizia: 115 righe scartate
        v
data/tab/complete.tab                   56.101 righe, colonna `split`  <-- unica sorgente dei notebook
        |
        |  notebook dei metodi (01-04, 06-12, 15-17, 21)   generazione
        v
results/summaries/{metodo}_{ambito}.tsv          un riassunto per riga, scritto in streaming
        |
        |  stessa notebook, sezione «Valutazione»          su.valuta_e_salva
        v
results/metrics/{metodo}_{ambito}_per_example.csv  +  _aggregate.json
        |
        |  backfill: 13 BERTScore (stessi file)   14 G-Eval (file separati *_geval_*)
        |  ambito a budget: scripts/applica_budget.py (tetto + nuove metriche)
        v
notebook di confronto e analisi (05a-05d, 18, 19)        tabelle e grafici, non generano nulla
        |
        |  scripts/estrai_figure_guida.py
        v
docs/resources/*.png                    figure delle guide
```

Gli script in `scripts/` fanno due mestieri diversi: preparano i dati (`convert_to_tab.py`,
`analyze_dataset.py`) oppure **guidano i notebook** senza presidio (`run_benchmark_test.py`,
`run_geval.py`), eseguendoli con `jupyter nbconvert` e le variabili d'ambiente giuste. I
notebook restano la fonte di verità: gli script non duplicano la logica dei metodi.

La struttura delle cartelle che contano per il codice:

| Percorso | Contenuto |
|---|---|
| `notebooks/summ_utils.py` | Tutte le routine condivise (circa 1.500 righe, in sezioni tematiche) |
| `notebooks/NN_*.ipynb` | Un notebook per metodo o per analisi |
| `notebooks/llm/` | Archivio dei notebook originali di Federica (LM Studio): solo consultazione |
| `notebooks_didattici/` | Versioni passo per passo di sei estrattivi, su una sola riga |
| `scripts/` | Preparazione dei dati e driver non presidiati |
| `results/sample/` | Il campione di 100 righe per le prove rapide |
| `results/summaries/` | I riassunti generati, un TSV per metodo e ambito |
| `results/metrics/` | Metriche per esempio e aggregate, cache del G-Eval, analisi delle lunghezze |
| `results/fewshot/` | I vicini recuperati per il few-shot e i risultati dei piloti |
| `results/figures/` | Figure delle sezioni esplicative dei notebook 11 e 15-17 |
| `sensibilita_estrattivi/` | Studio di sensibilità degli iperparametri, autonomo rispetto a `results/` |

---

## 3. Dai file grezzi a `complete.tab`

### 3.1 La conversione

I file canonici in `data/text/` hanno una riga per esempio: il file `.src.cleaned` contiene gli
articoli concatenati con il separatore `|||||` e il token `NEWLINE_CHAR` al posto degli a capo,
il file `.tgt` il riassunto di riferimento. Sono letti dallo script originale di Hugging Face
(`multi_news.py`), che resta invariato.

[scripts/convert_to_tab.py](../scripts/convert_to_tab.py) ne produce una copia nel formato
`.tab` di Orange Data Mining, una per split più `complete.tab` con tutte e tre le split e una
colonna `split`. Durante la conversione scarta tre tipi di righe:

- sorgenti sotto le 50 parole (comprese quelle vuote): scraping fallito;
- sorgenti sopra le 100.000 parole: gli outlier estremi sono disallineamenti
  sorgente/riassunto, non testi lunghi;
- sorgenti duplicate esatte (hash SHA-1 del testo normalizzato): si tiene la prima occorrenza
  nell'ordine train → val → test, il che elimina anche il rischio di leakage fra split.

Le 115 righe scartate sono elencate in `data/tab/excluded_rows.tsv`. Per questo `data/tab/`
**non è allineata riga per riga** con `data/text/`.

### 3.2 Il `row_id`

Tutto il benchmark identifica un esempio con il suo `row_id`: l'indice della riga dentro
`complete.tab`, contato da zero dopo le tre righe di intestazione di Orange. Le split sono in
ordine train, val, test, quindi:

| Split | Righe | `row_id` |
|---|---:|---|
| train | 44.880 | 0 – 44.879 |
| validation | 5.611 | 44.880 – 50.490 |
| test | 5.610 | 50.491 – 56.100 |

Il `row_id` è globale: la stessa riga ha lo stesso identificativo nel campione, nella split
test e nell'intero dataset. È ciò che permette di unire file prodotti da corse diverse.

### 3.3 Lettura in streaming

`complete.tab` pesa circa 658 MB, e nessun notebook lo carica in memoria per intero. Le
funzioni di lettura di `summ_utils` sono **generatori**:

```python
def itera_complete_tab(path):
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for _ in range(HEADER_LINES_TAB):       # le 3 righe di intestazione di Orange
            next(reader)
        for i, row in enumerate(reader):
            document, summary, split = row
            yield {'row_id': i, 'split': split, 'document': document, 'summary': summary}
```

`itera_split(path, 'test')` filtra lo stesso flusso su una split, `carica_campione(path)` legge
il TSV del campione prodotto dal notebook 00. Tutte restituiscono dizionari con le stesse
quattro chiavi, quindi il resto del codice non sa da dove arrivano le righe.

Due trasformazioni di testo sono comuni a quasi tutti i metodi:

- `prepara_documento(document)` sostituisce il separatore `|||||` con un a capo, perché
  altrimenti finirebbe dentro le «frasi» viste dai summarizer;
- `pulisci_riferimento(summary)` toglie il trattino iniziale `– ` che newser.com mette a ogni
  riassunto, prima di confrontarlo con quello generato.

### 3.4 Il campione

Il notebook [00](../notebooks/00_prepara_campione.ipynb) estrae 100 righe casuali da
`complete.tab` (`random.seed(42)`, `random.sample` sugli indici) e le salva in
`results/sample/sample_100_seed42.tsv`. Il campione serve solo per le prove rapide: i
risultati pubblicati sono tutti sulla split test intera.

---

## 4. Lo scheletro comune dei notebook dei metodi

### 4.1 Le sezioni

Tutti i notebook che generano riassunti hanno la stessa struttura:

1. **Installazione** — una cella che installa le dipendenze mancanti (serve su Colab).
2. **Configurazione** — costanti del metodo, ambito, percorsi, e la funzione `esempi_scope()`
   che restituisce le righe da elaborare.
3. **Generazione** — la definizione di `genera()` e il ciclo che la applica a ogni riga.
4. **Valutazione** — rilegge i riassunti salvati e calcola le metriche.
5. **Ispezione qualitativa** — stampa qualche riassunto accanto al riferimento.

I notebook degli estrattivi 11 e 15-17 hanno in più una sezione **«Come funziona, passo per
passo»**, che apre la pipeline su un solo documento e ne disegna le figure (sezione 6.7).

### 4.2 La configurazione

La cella di configurazione legge l'ambito e il limite di righe dalle variabili d'ambiente, con
un default che va bene quando il notebook si apre a mano:

```python
SCOPE  = os.environ.get('SUMM_SCOPE', 'sample')
LIMIT  = int(os.environ['SUMM_LIMIT']) if 'SUMM_LIMIT' in os.environ else None
BUDGET = su.budget_riferimento if su.budget_attivo(SCOPE) else None
```

È il meccanismo con cui i driver (sezione 10) pilotano i notebook senza modificarli: impostano
`SUMM_SCOPE=test` e lanciano l'esecuzione. Aperto in Jupyter, lo stesso notebook gira sul
campione.

`esempi_scope()` traduce l'ambito in un iterabile di righe:

```python
def esempi_scope():
    if SCOPE == 'sample':
        return su.carica_campione(SAMPLE_PATH)
    if su.split_base(SCOPE) == 'test':      # 'test' e 'test_budgetref' leggono le stesse righe
        return su.itera_split(P['complete_tab'], 'test')
    if SCOPE == 'full':
        return su.itera_complete_tab(P['complete_tab'])
    raise ValueError(f'SCOPE non valido: {SCOPE!r}')
```

È una funzione e non una variabile perché un generatore si consuma: la sezione di valutazione
la richiama per avere un flusso nuovo.

### 4.3 Il contratto di `genera()`

Ogni notebook definisce una funzione

```python
def genera(documento, budget=None) -> str
```

che riceve il testo di un cluster e restituisce il riassunto. Tutto ciò che distingue un
metodo dall'altro sta lì dentro. Il secondo argomento esiste solo negli ambiti a lunghezza del
riferimento (sezione 8): con `budget=None` il metodo si comporta come sempre.

I notebook con due varianti (10, 11, 15, 16) usano una **fabbrica**, `make_genera(...)`, che
restituisce una `genera` diversa per ciascuna variante. Le due varianti condividono tutto
tranne il pezzo che le distingue: il segmentatore di frasi, il vettorizzatore, la regola di
selezione o l'algoritmo di clustering. Così il confronto fra varianti isola davvero
quell'unica scelta.

### 4.4 Il ciclo con ripresa

La generazione passa sempre per due oggetti di `summ_utils`:

```python
scrittore = su.ScrittoreRiassunti(OUT_PATH)
errori = su.ciclo_summarization(esempi_scope(), scrittore, genera,
                                limit=LIMIT, etichetta='TextRank ', budget=BUDGET)
scrittore.chiudi()
```

`ScrittoreRiassunti` apre il TSV in *append*, scrive una riga `(row_id, riassunto)` alla volta
e fa `flush()` dopo ogni scrittura. Alla costruzione rilegge le righe già presenti: un
`row_id` già nel file viene saltato. Un'esecuzione interrotta, anche dopo giorni, riparte
quindi dalla prima riga mancante.

`ciclo_summarization` scorre le righe, salta quelle già fatte, applica `prepara` al documento
(di default `prepara_documento`), chiama `genera` e scrive il risultato. Tre dettagli:

- un'eccezione su una riga viene **registrata e non ferma il ciclo**: la riga semplicemente
  manca dal TSV, e i notebook di confronto lavorano sull'intersezione delle righe coperte;
- `limit` conta le righe *nuove* elaborate, non quelle saltate: `LIMIT=3` su un file già
  avviato fa davvero tre generazioni;
- con `budget` non nullo la chiamata diventa `genera(documento, budget(esempio))`: è l'unico
  punto in cui il ciclo sa qualcosa dell'ambito a budget.

Il TSV usa `csv.QUOTE_ALL`, così un riassunto che contiene tabulazioni o a capo non rompe il
formato.

### 4.5 La valutazione

La sezione di valutazione non usa nulla della generazione, solo il file:

```python
riassunti   = su.carica_riassunti(OUT_PATH)
riferimenti = esempi_scope()
config = {...}          # i parametri del metodo, salvati nel JSON aggregato
righe, aggregato = su.valuta_e_salva(riferimenti, riassunti, METODO, SCOPE,
                                     P['metrics_dir'], config)
```

`valuta_e_salva` (sezione 7.1) calcola le metriche per ogni riga presente nel TSV e scrive due
file: le metriche per esempio in CSV e le medie, complessive e per split, in JSON. Il
dizionario `config` finisce nel JSON: ogni aggregato documenta i parametri con cui è stato
prodotto.

---

## 5. Gli ambiti di esecuzione

Un **ambito** (`SCOPE`) dice su quali righe gira un notebook e con quale protocollo. Ogni
ambito scrive file propri, con il nome dell'ambito nel suffisso, quindi le corse non si
sovrascrivono a vicenda.

| Ambito | Righe | A cosa serve |
|---|---|---|
| `sample` | 100 del campione | Prove rapide in locale; i risultati non sono versionati |
| `test` | 5.610 della split test | **I risultati del benchmark** |
| `full` | 56.101 (tutte) | Corse sull'intero dataset (TextRank, LexRank) |
| `test_budgetref` | le stesse 5.610 | Confronto a lunghezza del riferimento (sezione 8) |

Non tutti i notebook supportano tutti gli ambiti:

| Notebook | `sample` | `test` | `full` | `test_budgetref` |
|---|:-:|:-:|:-:|:-:|
| 01, 02 (TextRank, LexRank) | ✓ | derivato da `full` | ✓ | ✓ |
| 03, 04, 06 (BART, PEGASUS, PRIMERA) | ✓ | ✓ | | solo tetto |
| 07, 08, 09 (LLM locali) | ✓ | ✓ | | ✓ |
| 10, 11, 15, 16, 17 (estrattivi) | ✓ | ✓ | ✓ | ✓ |
| 12 (GPT-5-mini) | ✓ | ✓ | ✓ | ✓ |
| 21 (Qwen few-shot) | ✓ | ✓ | | ✓ |

Due funzioni di `summ_utils` gestiscono il suffisso `_budgetref`:

```python
SUFFISSO_BUDGET = '_budgetref'

def budget_attivo(scope):      # 'test_budgetref' -> True
    return scope.endswith(SUFFISSO_BUDGET)

def split_base(scope):         # 'test_budgetref' -> 'test'
    return scope[:-len(SUFFISSO_BUDGET)] if budget_attivo(scope) else scope
```

Un ambito a budget legge le righe del suo ambito base e scrive file con il proprio nome. Un
ambito futuro come `val_budgetref` funzionerebbe senza cambiare codice.

**TextRank e LexRank sull'ambito `test`** non hanno una corsa dedicata: le loro metriche sono
ricavate filtrando sulla split test le metriche per esempio della corsa `full`. È
numericamente identico, perché le metriche si calcolano riga per riga. Lo fa
`deriva_metriche_test` nel driver (sezione 10.1).

---

## 6. I metodi, notebook per notebook

Questa sezione descrive, per ogni metodo, il codice dentro `genera()` e le scelte che lo
distinguono. La teoria di ciascun metodo è nella sezione 4 della guida di studio.

### 6.1 Il ruolo di pyAutoSummarizer

La libreria [pyAutoSummarizer](https://github.com/Valdecy/pyAutoSummarizer) ha tre ruoli
diversi nel progetto, e conviene tenerli distinti:

1. **Segmentazione in frasi** per tutti gli estrattivi. L'oggetto `psr.summarization(testo)`
   divide il testo in frasi e ne espone la lista originale in `s.original`. Usarla ovunque
   garantisce che tutti gli estrattivi scelgano dallo **stesso insieme di frasi candidate**.
2. **Generazione** solo per TextRank e LexRank (`summ_text_rank`, `summ_lex_rank`).
3. **Valutazione** per tutti i metodi: ROUGE, BLEU e METEOR (sezione 7.1).

Tutti gli altri metodi (Centroid+MMR, LSA, clustering, LDA, First-k) sono implementati a mano
con scikit-learn, e i modelli neurali sono caricati direttamente con `transformers`.

### 6.2 First-k / Lead (notebook 10)

Per ogni articolo del cluster prende le prime `K_SENTENCES = 3` frasi e le concatena. Il
documento viene diviso in articoli sul separatore `|||||`; per questo il ciclo riceve
`prepara=str.strip` invece di `prepara_documento`, che il separatore lo cancellerebbe.

Le due varianti differiscono solo nel segmentatore:

- `firstk_psr` usa `psr.summarization(...).original`, lo stesso degli altri estrattivi;
- `firstk_nltk` usa `nltk.sent_tokenize`, il modello Punkt, che gestisce meglio le
  abbreviazioni.

A budget di parole l'ordine di priorità diventa **a giri fra gli articoli**: la prima frase di
ogni articolo, poi la seconda di ogni articolo, e così via, fino al budget. Il riassunto resta
in ordine di documento.

### 6.3 TextRank e LexRank (notebook 01 e 02)

Sono gli unici metodi generati dalla libreria:

```python
modello_st = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)   # caricato una volta

def genera(documento, budget=None):
    s = psr.summarization(documento, stop_words=['en'])
    s.loaded_models['all-MiniLM-L6-v2'] = modello_st     # iniezione del modello già caricato
    rank = s.summ_text_rank(model='all-MiniLM-L6-v2')
    if budget is None:
        return s.show_summary(rank, n=N_SENTENCES)
    ordine = np.argsort(rank)[::-1]
    return ' '.join(s.original[i] for i in su.seleziona_per_budget(s.original, ordine, budget))
```

La riga dell'iniezione è una scelta di prestazioni: la libreria caricherebbe il modello di
embedding a ogni chiamata, cioè 56.101 volte sull'intero dataset. Scrivendolo nel suo
dizionario interno `loaded_models` il modello resta in memoria (e sulla GPU) per tutta la
corsa. LexRank non usa embedding (`summ_lex_rank()`, su TF-IDF) e non ne ha bisogno.

`show_summary` restituisce le frasi **in ordine di rank**, non di documento. Nell'ambito a
budget si mantiene lo stesso ordine, così il metodo non cambia natura.

### 6.4 Centroid + MMR (notebook 11)

Implementazione scritta a mano, attorno a una funzione che restituisce tutti gli artefatti
intermedi:

```python
def analizza_mmr(frasi, vettorizza=vettorizza_tfidf, lam=None, k=None):
    X = vettorizza(frasi)                                       # (n_frasi, dim)
    centroide = np.asarray(X.mean(axis=0)).reshape(1, -1)
    rilevanza = cosine_similarity(X, centroide).ravel()         # vicinanza al centroide
    sim = cosine_similarity(X)                                  # similarità frase-frase
    scelti, candidati = [], list(range(len(frasi)))
    while len(scelti) < k and candidati:
        # punteggio MMR: λ·rilevanza − (1−λ)·massima similarità con le già scelte
        migliore = max(candidati, key=lambda i:
                       lam * rilevanza[i] - (1 - lam) * max((sim[i][j] for j in scelti), default=0))
        scelti.append(migliore)
        candidati.remove(migliore)
    return {'X': X, 'centroide': centroide, 'rilevanza': rilevanza, 'sim': sim, 'scelti': scelti}
```

(Il notebook scrive il ciclo in forma esplicita; qui è compattato.) `LAMBDA = 0.7`, e le frasi
scelte vengono rimesse in ordine di documento prima di unirle.

Le due varianti cambiano solo `vettorizza`:

- `centroid_mmr`: `TfidfVectorizer(stop_words='english')`, matrice sparsa;
- `centroid_mmr_bert`: embedding `all-MiniLM-L6-v2` normalizzati, il modello caricato una
  sola volta in un dizionario di modulo.

Nell'ambito a budget il greedy produce un ordine lungo fino a `K_ORDINE_BUDGET = 50` frasi, poi
tagliato al budget in parole. È equivalente a fermare il greedy al budget, perché le prime *t*
scelte non dipendono da quante se ne chiedono in totale; il limite di 50 copre anche i
riferimenti più lunghi senza far esplodere il costo quadratico del greedy.

### 6.5 LSA (notebook 15)

Le frasi vengono vettorizzate con TF-IDF e proiettate con `TruncatedSVD`:

```python
X = TfidfVectorizer(stop_words='english').fit_transform(frasi)
k = max(1, min(K_LATENTE, min(X.shape) - 1))        # non oltre il rango della matrice
Z = TruncatedSVD(n_components=k, random_state=SEED).fit_transform(X)   # Z = U·Σ
```

Ogni riga di `Z` è una frase nello spazio dei concetti latenti, già pesata dai valori
singolari. Le due varianti condividono `Z` e cambiano la regola di selezione:

- `lsa` prende le frasi con la norma di riga più alta: `np.argsort(-norm(Z, axis=1))`;
- `lsa_steinberger` sceglie una frase alla volta e **sottrae la sua direzione** da tutte le
  altre:

```python
R = Z.copy()
for _ in range(n):
    lunghezze = np.linalg.norm(R, axis=1)
    lunghezze[scelti] = -1.0                 # una frase scelta non torna candidata
    i = int(np.argmax(lunghezze))
    if lunghezze[i] <= EPS:                  # spazio residuo esaurito
        break
    scelti.append(i)
    u = R[i] / np.linalg.norm(R[i])
    R -= np.outer(R @ u, u)                  # z_j ← z_j − (z_j·u)u
```

Dopo la sottrazione, una frase che ripete quella appena scelta ha residuo quasi nullo e non
viene più presa. Se lo spazio si esaurisce prima di *n* frasi, si completa con la salienza
originale. `K_LATENTE = 11` resta fisso anche nell'ambito a budget: è un parametro della
decomposizione, non della lunghezza.

### 6.6 Clustering su embedding SBERT (notebook 16) e LDA (notebook 17)

**Clustering.** Le frasi sono codificate con `all-MiniLM-L6-v2` e normalizzate, così il
prodotto scalare coincide con la similarità coseno. Le due varianti cambiano solo
l'algoritmo:

```python
KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(E)               # sbert_kmeans
AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(E)  # sbert_agglom
```

Per ogni gruppo si calcola il centroide (media degli embedding, rinormalizzata) e si sceglie il
**medoide**, la frase vera più vicina al centroide, perché un estrattivo può solo copiare frasi
esistenti. A budget di parole le frasi entrano **a giri**: prima il medoide di ogni gruppo, dal
gruppo più grande al più piccolo, poi la seconda frase più vicina al centroide di ogni gruppo,
e così via (`ordine_rappresentanti`). Il primo giro coincide con la selezione senza budget.

**LDA.** Le frasi vengono contate con `CountVectorizer(stop_words='english')` (LDA vuole
conteggi, non pesi TF-IDF) e date a `LatentDirichletAllocation(n_components=5)`. La matrice
`theta` dà la distribuzione sui topic di ogni frase; la sua somma per colonna è il **peso** di
ogni topic nel cluster. `alloca_quote` divide il numero di frasi fra i topic in proporzione al
peso, arrotondando per difetto e assegnando le unità mancanti ai topic con la parte frazionaria
più alta, così il totale torna esatto. Dentro ogni topic si prendono le frasi con appartenenza
più alta, saltando quelle già prese da un altro topic. A budget, `seleziona_per_quote_parole`
fa la stessa ripartizione in parole invece che in frasi.

### 6.7 Una funzione per la produzione e per le figure

I notebook 11 e 15-17 seguono uno schema comune: la funzione `analizza_*` (`analizza_mmr`,
`analizza_lsa`, `analizza_clustering`, `analizza_lda`) restituisce tutti gli artefatti
intermedi, e la stessa funzione serve sia alla generazione sia alla sezione «Come funziona,
passo per passo». Le figure mostrano quindi esattamente ciò che il metodo calcola in
produzione, non una sua ricostruzione.

La sezione esplicativa si configura con `RIGA_DEMO` (quale documento del campione aprire) e
`SALVA_FIGURE` (se salvare i PNG in `results/figures/{metodo}/`). Le figure della guida di
studio vengono da lì.

Un passo oltre lo fanno i [notebook didattici](../notebooks_didattici/README.md): per i sei
estrattivi rifanno a mano i calcoli principali (PageRank, SVD, greedy MMR, inferenza di LDA) su
una sola riga, e verificano con `assert` che il risultato coincida con il codice del benchmark,
copiato alla lettera.

### 6.8 BART e PEGASUS (notebook 03 e 04)

I modelli sono caricati con `transformers` **una sola volta** all'inizio del notebook (la
libreria li ricaricherebbe a ogni chiamata e ignorerebbe la GPU) e usati con i parametri di
generazione di pyAutoSummarizer:

```python
inputs = tokenizer([documento], max_length=1024, truncation=True, return_tensors='pt').to(DEVICE)
with torch.no_grad():
    out = modello.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'],
                           num_beams=4, max_length=MAX_LEN, early_stopping=True)
```

| | BART | PEGASUS |
|---|---|---|
| Checkpoint | `facebook/bart-large-cnn` | `google/pegasus-multi_news` |
| Input | 1.024 token, troncato | 1.024 token, troncato |
| Output massimo | 300 token | 256 token |
| Beam | 4 | 4 |

Il troncamento a 1.024 token vale per entrambi: vedono solo l'inizio del cluster.

### 6.9 PRIMERA (notebook 06)

PRIMERA accetta 4.096 token e si aspetta gli articoli separati dal token speciale
`<doc-sep>`. Il notebook costruisce l'input a mano:

```python
def costruisci_input(documento):
    articoli = [a for a in re.split(r'\s*\|\|\|\|\|\s*', documento) if a.strip()]
    budget = max((MAX_INPUT - 2) // len(articoli), 8)          # token uguali per articolo
    ids = []
    for articolo in articoli:
        ids.extend(tokenizer.encode(articolo, truncation=True, max_length=budget)[1:-1])
        ids.append(DOCSEP_ID)
    return [bos] + ids[:MAX_INPUT - 2] + [eos]
```

Ogni articolo riceve la stessa quota di token, così un articolo lungo non toglie spazio agli
altri. L'**attenzione globale** è attiva sul primo token e su ogni `<doc-sep>`: sono i punti da
cui il Longformer vede l'intero input. Per costruire l'input il notebook ha bisogno del
separatore originale `|||||`, e quindi passa `prepara=str.strip` al ciclo.

### 6.10 LLM locali via ollama (notebook 07-09)

Il modello gira nel server ollama; il notebook usa il client `openai` puntato all'endpoint
compatibile di ollama:

```python
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

def genera(documento, budget=None):
    prompt = PROMPT_USER if budget is None else PROMPT_USER_BUDGET
    risposta = client.chat.completions.create(
        model=MODELLO,
        messages=[{'role': 'system', 'content': PROMPT_SYSTEM},
                  {'role': 'user', 'content': prompt.format(documento=documento, richiesto=...)}],
        max_tokens=MAX_TOKENS if budget is None else MAX_TOKENS_BUDGET,
        temperature=TEMPERATURE)
    contenuto = risposta.choices[0].message.content
    if not contenuto or not contenuto.strip():
        raise RuntimeError('risposta vuota')      # la riga non viene scritta
    return contenuto.strip()
```

| | Qwen (07) | Gemma (08) | Mistral (09) |
|---|---|---|---|
| Tag ollama | `qwen2.5:7b-instruct` | `gemma4:latest` | `mistral:7b-instruct-v0.3-q4_K_M` |
| `max_tokens` | 200 | 1.500 | 300 |
| `temperature` | 0,3 | 0,3 | 0,3 |
| Fattore di richiesta a budget | 1,2 | 0,9 | 1,0 |

Il prompt è zero-shot e in inglese, ripreso dalla corsa originale. Gemma ha 1.500 token perché
i suoi token di ragionamento consumano il limite prima della risposta visibile: con 300 la
risposta usciva vuota nell'81 % dei casi. Una risposta vuota **solleva un'eccezione**, così la
riga non viene scritta invece di entrare nei risultati come riassunto vuoto.

Il prompt a budget non è riscritto a mano: è **derivato** da quello base con una sostituzione,
e un `assert` controlla che la sostituzione sia avvenuta. Se qualcuno cambiasse il prompt base,
la derivazione fallirebbe rumorosamente invece di produrre un prompt incoerente:

```python
PROMPT_USER_BUDGET = PROMPT_USER.replace(
    'a comprehensive summary:',
    'a comprehensive summary of about {richiesto} words. '
    'The summary must be at least {richiesto} words long:')
assert PROMPT_USER_BUDGET != PROMPT_USER, 'PROMPT_USER cambiato: aggiornare la derivazione'
```

Il numero di parole chiesto è il budget moltiplicato per `FATTORE_RICHIESTA`, calibrato per
ogni modello su un pilota di 30 righe.

### 6.11 GPT-5-mini su Azure (notebook 12)

Stessa struttura dei notebook 07-09, con tre differenze:

- **Credenziali solo da variabili d'ambiente**: `AZURE_OPENAI_ENDPOINT` (la radice della
  risorsa, senza percorso) e `AZURE_OPENAI_API_KEY`. Il client usa la rotta `v1` di Azure
  OpenAI, `<endpoint>/openai/v1/`, senza versione datata dell'API.
- **`model=` è il nome del deployment**, non il nome del modello.
- **Parametri da modello di ragionamento**: niente `temperature` (accetta solo il default),
  `max_completion_tokens=1500` e `reasoning_effort='minimal'`.

Il filtro di contenuti di Azure respinge alcuni cluster con un errore `content_filter` prima
che il modello li veda; il ciclo li registra come errori e la riga manca dal TSV.

### 6.12 Few-shot con esempi recuperati (notebook 20, 20b, 21)

Il notebook 21 aggiunge al prompt di Qwen quattro esempi del train simili al cluster da
riassumere. Il codice condiviso sta nella sezione *Few-shot* di `summ_utils` e si divide in
recupero e generazione.

**Recupero.** `calcola_vicini` calcola gli embedding `all-mpnet-base-v2` di tutto il train e
delle righe da riassumere, cerca i *k* vicini per coseno con `NearestNeighbors` di
scikit-learn e salva il risultato in `results/fewshot/qwen_fewshot_vicini_{ambito}.tsv`. Il
file è versionato: chi riesegue il notebook rilegge i vicini invece di ricalcolarli. Gli
embedding del train (circa 140 MB) sono in una cache `.npy` fuori da git, con il nome che
contiene un hash del modello e dei testi. Una `assert` impedisce che fra le righe da riassumere
ci siano righe del train, che ritroverebbero se stesse come esempio.

**Generazione.** Il prompt di ogni riga dipende dai suoi vicini, quindi va costruito per
`row_id`, mai cercando gli esempi a partire dal testo. Il notebook lo fa con un generatore che
sostituisce il documento con il prompt già pronto, e passa al ciclo una `prepara` identità:

```python
def righe_con_prompt():
    for es in righe:
        yield {**es, 'document': prompt_riga(es)}      # prompt costruito per row_id

errori = su.ciclo_summarization(righe_con_prompt(), scrittore, genera,
                                prepara=lambda p: p)
```

La chiamata passa per l'**API nativa** di ollama (`/api/chat`, funzione `genera_ollama_nativo`)
invece che per l'endpoint compatibile con OpenAI, perché solo quella accetta `num_ctx` per
richiesta e restituisce `prompt_eval_count`. Il contesto è fissato a 32.768 token, e se il
prompt lo satura la funzione solleva un'eccezione: un prompt troncato in silenzio non produce
mai un riassunto.

I testi dei due prompt (`old` e `new`) sono copiati alla lettera dai notebook originali di
Federica, refusi compresi: cambiarli significherebbe cambiare il metodo.

---

## 7. La valutazione

### 7.1 Le metriche lessicali

`crea_valutatore()` crea un'istanza di `psr.summarization` su un testo fittizio, usata solo per
le sue funzioni di metrica. Le opzioni di pre-processing dell'istanza decidono come vengono
normalizzati i testi prima del confronto: minuscole, punteggiatura tolta, stopword e numeri
mantenuti, come nel ROUGE standard.

`metriche_esempio` calcola per una coppia (generato, riferimento) ROUGE-1, ROUGE-2 e ROUGE-L
(F1, precisione, recall), BLEU-4 e METEOR, più `parole_generate`, la lunghezza del riassunto in
parole. `valuta_e_salva` la applica a ogni riga coperta e scrive:

- `results/metrics/{metodo}_{ambito}_per_example.csv` — una riga per esempio, con `row_id` e
  `split`;
- `results/metrics/{metodo}_{ambito}_aggregate.json` — `n_esempi`, medie in `overall`, medie
  per split in `per_split`, il `config` del notebook e un timestamp.

Due caratteristiche dell'implementazione della libreria vanno tenute presenti: il ROUGE-N usa
insiemi di n-grammi *unici* invece dei conteggi «clipped» dello standard, e il METEOR non è
limitato inferiormente su testi degeneri (sezione 12).

### 7.2 BERTScore (notebook 13)

pyAutoSummarizer ha una funzione `bert_score()`, ma ricarica `roberta-large` a ogni chiamata.
`calcola_bertscore_batch` usa invece direttamente `bert_score.BERTScorer`: un caricamento del
modello per metodo, poi tutte le coppie in un'unica chiamata a lotti (`batch_size=64`).

Il notebook è un **backfill incrementale**: salta i metodi il cui CSV ha già le colonne
`bertscore_*`, e per gli altri riscrive CSV e JSON con le colonne aggiunte
(`valuta_e_salva(..., extra_metriche=..., colonne_extra=...)`), conservando il `config`
originale. Con `BERTSCORE_FORZA=1` ricalcola tutto.

### 7.3 G-Eval (notebook 14)

Il G-Eval di pyAutoSummarizer non è utilizzabile qui per quattro motivi: non accetta un
endpoint diverso da OpenAI, ha `max_tokens=5` e `temperature=0` fissi (inaccettabili per un
modello di ragionamento), legge la sorgente da un attributo dell'istanza e fa una chiamata per
dimensione. È quindi reimplementato in `summ_utils`, ma con le **rubriche verbatim**:
`RUBRICHE_GEVAL` non è scritto a mano, è ricavato dai quattro template originali tagliandone la
coda «Reply with a single digit only.»

**Una chiamata, quattro punteggi.** Ogni giudizio è una sola chiamata che restituisce un
oggetto JSON con le quattro dimensioni. Il formato è imposto lato server con
`response_format={'type': 'json_schema', 'strict': True, ...}`; poiché in modalità strict lo
schema non ammette `minimum`/`maximum`, la scala 1-5 è espressa con `enum: [1,2,3,4,5]`.

**L'ordine dei messaggi è un contratto.** Il messaggio `system` (rubriche e formato) è
identico in tutte le chiamate; il messaggio `user` comincia con la sorgente troncata a 3.500
parole e mette il riassunto in fondo:

```python
def costruisci_messaggi_geval(sorgente, riassunto, max_parole=3500):
    return [{'role': 'system', 'content': PROMPT_GEVAL_SISTEMA},
            {'role': 'user',
             'content': f'Source:\n{tronca_parole(sorgente, max_parole)}\n\nSummary:\n{riassunto}'}]
```

Così tutti i giudizi di una stessa riga condividono un prefisso lungo, che la **prompt cache**
di Azure serve a tariffa ridotta dalla seconda chiamata in poi. Invertire l'ordine azzererebbe
il risparmio.

**La riga è l'unità di lavoro.** `giudica_geval_concorrente` usa un pool di thread, ma ogni
thread prende una riga intera ed esegue i giudizi dei suoi metodi **in sequenza**: il primo
riempie la cache, gli altri la usano. Il parallelismo è fra righe diverse. Il pool viene
rifornito di nuove righe man mano che le precedenti finiscono (`wait(..., FIRST_COMPLETED)`),
tenendo in volo circa il doppio delle righe rispetto ai thread. Più thread non sono gratis: il deployment smista le
richieste su istanze diverse e la cache è per istanza, quindi la quota servita dalla cache
scende con la concorrenza (65 % delle chiamate a 2 thread, 46 % a 8).

**La cache dei giudizi.** `CacheGiudizi` ha lo stesso contratto di `ScrittoreRiassunti`, ma
indicizzato su `(metodo, row_id)` e protetto da un lock perché ci scrivono più thread. Ogni
giudizio è una riga JSON in `results/metrics/geval_cache_{ambito}.jsonl`, con i punteggi e i
conteggi di token, scritta e flushata subito. Il file è **l'artefatto di riproducibilità**: le
chiamate a un modello di ragionamento non sono ripetibili bit per bit, e le metriche si
ricavano gratis dalla cache (`run_geval.py --solo-metriche`).

**Errori.** Due categorie, trattate in modo opposto:

- **permanenti** (filtro di contenuti, HTTP 400/401/403/404): vengono scritti in cache con il
  campo `errore`, così un rilancio non li ripaga. Siccome la sorgente è la stessa per tutti i
  metodi di una riga, un `content_filter` sul primo giudizio viene propagato agli altri senza
  chiamare l'API;
- **transitori** (429, errori 5xx, timeout): non vengono scritti, e `chiama_con_backoff` li
  ritenta con attesa esponenziale e jitter, rispettando l'header `Retry-After`.

`--riprova-errori` rimuove dalla cache i giudizi falliti per ritentarli.

**Costi.** Azure non ha un'API del costo in tempo reale, quindi la fonte di verità è l'oggetto
`usage` di ogni risposta. `ContatoreCosti` somma i token (input, input in cache, output,
ragionamento), li moltiplica per i prezzi unitari della Azure Retail Prices API e stampa ogni
1.500 giudizi ritmo, quota in cache e proiezione a fine corsa. Il limite di spesa (`--budget`)
è **complessivo**: somma quanto è già in cache, altrimenti ogni rilancio ripartirebbe da zero.

**File separati.** I punteggi vanno in `{metodo}_{ambito}_geval_per_example.csv` e
`..._geval_aggregate.json`, mai nel CSV standard. `valuta_e_salva` fa la media sommando ogni
colonna su ogni riga, e alcune righe non hanno giudizio: unire i file romperebbe la media, o
costringerebbe a riscrivere metriche già pubblicate su un insieme di righe diverso.

**Il secondo giudice.** [scripts/pilota_giudice_deepseek.py](../scripts/pilota_giudice_deepseek.py)
rigiudica con DeepSeek-V3.2 un campione appaiato di 1.000 righe su sette metodi, riusando le
stesse funzioni di `summ_utils` ma con cache e JSON propri
(`geval_cache_test_deepseek.jsonl`, `confronto_giudici_test.json`). Il notebook
[19](../notebooks/19_confronto_giudici.ipynb) legge le due cache e confronta i giudici: non
genera e non spende nulla.

---

## 8. Il confronto a lunghezza pari, lato codice

La sezione 7 della guida di studio spiega perché il benchmark ha un secondo ambito,
`test_budgetref`, in cui ogni metodo ha come budget la lunghezza del riassunto di riferimento.
Qui si vede come è realizzato.

### 8.1 Il budget

```python
def budget_riferimento(esempio):
    return len(pulisci_riferimento(esempio['summary']).split())
```

Il budget di una riga è il numero di parole del suo riferimento, contato come
`parole_generate`: un metodo che centra il budget ha rapporto lunghezza/riferimento pari a 1.
La funzione viene passata al ciclo come `budget=su.budget_riferimento`, e il ciclo la chiama
riga per riga.

### 8.2 Dove fermarsi: `seleziona_per_budget`

Gli estrattivi producono un **ordine** di frasi (per rank, per punteggio MMR, a giri fra
articoli o gruppi). Il budget non cambia l'ordine, decide solo dove fermarsi:

```python
def seleziona_per_budget(frasi, ordine, budget_parole):
    scelti, totale = [], 0
    for i in ordine:
        n = len(frasi[i].split())
        if n == 0:
            continue
        if totale + n <= budget_parole:          # la frase entra
            scelti.append(i)
            totale += n
            continue
        if budget_parole - totale > n / 2:       # la frase «a cavallo» entra se ne sta dentro
            scelti.append(i)                     # più della metà
        break
    ...                                          # almeno una frase, sempre
    return scelti
```

La regola della frase a cavallo centra la lunghezza sul budget invece di lasciarla sempre sotto.
La funzione non salta una frase lunga per cercarne una più corta più in basso: farlo
cambierebbe il criterio del metodo.

### 8.3 Il tetto: `scripts/applica_budget.py`

Il tetto si applica dopo la generazione, fuori dai notebook, a tutti i 19 metodi. Per ogni
metodo lo script:

1. sceglie la sorgente dei riassunti: il TSV rigenerato a budget se esiste, altrimenti quello
   dell'ambito `test` (BART, PEGASUS e PRIMERA, che ricevono il solo tetto);
2. tronca ogni riassunto alle prime `tetto_parole(R)` parole, dove *R* è la lunghezza del
   riferimento;
3. ricalcola metriche lessicali e BERTScore con le stesse funzioni dei notebook, e scrive
   `{metodo}_test_budgetref_per_example.csv` e `_aggregate.json`.

I notebook che rigenerano a budget scrivono già metriche sui testi grezzi; lo script le
sovrascrive con quelle post-tetto, che sono quelle da leggere. Il JSON registra in
`config.ambito_budget` quale sorgente è stata usata e quante righe il tetto ha tagliato. I TSV
`_test_budgetref.tsv` restano pre-tetto: il testo troncato esiste solo nelle metriche.

Il fattore del tetto vive in `summ_utils` e non nello script:

```python
FATTORE_TETTO_BUDGET = 1.25

def tetto_parole(parole_riferimento):
    return int(round(FATTORE_TETTO_BUDGET * parole_riferimento))
```

Il motivo è che più consumatori devono vedere esattamente lo stesso testo: lo script, il
G-Eval dell'ambito a budget (che riapplica il tetto prima di giudicare) e il notebook 18 (che
verifica quanti riassunti stanno nella banda). Tutti usano questo stesso intero. Il notebook 18
in passato confrontava con 1,25 × *R* esatto, e una riga troncata proprio su un tetto
arrotondato per eccesso (388 contro 387,5) risultava fuori banda per mezza parola: è l'errore
corretto con la issue #21.

### 8.4 Il G-Eval a budget

Con `GEVAL_SCOPE=test_budgetref` il notebook 14 legge i riassunti dal TSV a budget se esiste,
altrimenti da quello `test`, e applica `tetto_riferimento` a ogni riga prima di giudicare. I
giudizi dei testi rimasti identici a quelli dell'ambito `test` (il 94 % per BART, l'84 % per
PEGASUS) vengono copiati dalla cache `test` invece di essere ripagati, con il campo
`riuso_da: 'test'` e zero token.

---

## 9. I notebook di confronto e di analisi

I notebook 05a-05d, 18 e 19 non generano nulla: leggono i file di `results/metrics/`. Le
funzioni che condividono sono nella sezione *Confronto fra metodi* di `summ_utils`.

### 9.1 `carica_scope`: unione e intersezione

```python
dati = su.carica_scope(P['metrics_dir'], 'test', su.METODI_BENCHMARK)
```

Per ogni metodo legge CSV per esempio e JSON aggregato. Se esiste il file G-Eval lo aggancia con
un merge **LEFT** su `row_id`, che non cambia il numero di righe: le righe senza giudizio
restano `NaN` e pandas le ignora nelle medie.

Poi restringe tutti i metodi all'**intersezione** dei `row_id` coperti, perché il confronto sia
sulle stesse righe. Partecipano all'intersezione solo i metodi con almeno `COPERTURA_MINIMA`
(50) righe: una corsa quasi vuota non deve azzerare il confronto degli altri. Sulla split test
l'intersezione è di 5.437 righe.

### 9.2 Tabelle e grafici

- `tabella_medie` produce la tabella delle medie con `n_esempi` e `n_geval` (le righe giudicate,
  che possono essere meno).
- `barre_metriche` disegna un pannello di barre orizzontali per metrica, con i metodi ordinati e
  le etichette sull'asse. Non c'è legenda: con 19 metodi il colore da solo non basta.
- `COLORI_METODI` fissa un colore per metodo, uguale in tutti i grafici del progetto.
- `con_metrica` restringe un grafico ai metodi che hanno una certa colonna (BERTScore, G-Eval),
  elencando gli esclusi in un avviso. È ciò che evita di far sparire un grafico quando si
  aggiunge un metodo prima del suo backfill.
- `viste_metriche` mette insieme tutto ciò ed è chiamata sia da 05b (ambito `test`) sia da 05c
  (`test_budgetref`): i due ambiti sono presentati in modo identico.

### 9.3 Gli altri notebook di analisi

- **05a** confronta TextRank e LexRank sull'ambito `full`; **05d** mette `test` e
  `test_budgetref` uno accanto all'altro.
- **18** analizza le lunghezze cluster per cluster. Legge `parole_generate` dai CSV per esempio
  di entrambi gli ambiti e la incrocia, in un'unica passata su `complete.tab`, con lunghezza del
  riferimento e numero di articoli. Per misurare quanto ogni metodo rispetta il budget *da
  solo*, legge anche i TSV pre-tetto. Scrive `analisi_lunghezze_{ambito}.json`.
- **19** confronta i due giudici G-Eval (sezione 7.3).
- **20** e **20b** sono i piloti del few-shot, su 100 righe della split validation: scelgono
  numero di esempi, variante del prompt e modello di embedding prima della corsa sul test.

---

## 10. Gli script e i driver non presidiati

| Script | Che cosa fa |
|---|---|
| `convert_to_tab.py` | Genera `data/tab/` da `data/text/` (sezione 3.1) |
| `analyze_dataset.py` | Calcola le statistiche dell'EDA in un'unica passata e scrive `scripts/dataset_stats.json` |
| `run_benchmark_test.py` | Esegue in sequenza i notebook dei metodi su un ambito |
| `applica_budget.py` | Applica il tetto e ricalcola le metriche dell'ambito a budget (sezione 8.3) |
| `run_geval.py` | Guida il notebook 14 del G-Eval |
| `pilota_giudice_deepseek.py` | Il secondo giudice (sezione 7.3) |
| `estrai_figure_guida.py` | Copia le figure delle guide in `docs/resources/` |
| `import_llm_results.py` | Importatore storico dei risultati LM Studio di Federica, superato |

### 10.1 `run_benchmark_test.py`

Esegue i notebook dei metodi **dal più veloce al più lento** (First-k, LDA, LSA, clustering,
Centroid+MMR, BART, PEGASUS, Qwen, Mistral, Gemma, PRIMERA, Qwen few-shot), così i risultati
rapidi arrivano subito e un problema si vede presto. Per ogni notebook:

```python
env = dict(os.environ, SUMM_SCOPE=scope)        # più SUMM_LIMIT con --limit
subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook',
                '--execute', '--inplace', '--ExecutePreprocessor.timeout=-1', nome],
               cwd=NOTEBOOKS_DIR, env=env, capture_output=True, text=True)
```

- Nell'ambito `test` il notebook viene eseguito **in place**: gli output committati
  documentano la corsa di riferimento. Negli ambiti a budget l'eseguito va in
  `results/notebook_runs/{ambito}/`, fuori da git, così il notebook committato conserva gli
  output della corsa `test`.
- Un notebook che fallisce viene registrato in `run_benchmark_test.log` e **non ferma** i
  successivi; grazie alla ripresa, un rilancio completa solo ciò che manca.
- Prima di partire, i **controlli preliminari** verificano che ollama sia attivo con i tag
  giusti e che ci sia una GPU, ma solo se un notebook selezionato ne ha bisogno.
- Alla fine deriva le metriche `test` di TextRank e LexRank dalla corsa `full` e riesegue i
  notebook di confronto.

Opzioni: `--limit N` (prova su N righe per metodo), `--only 10,11` (solo alcuni notebook),
`--scope test_budgetref` (i sette notebook estrattivi a budget).

### 10.2 `run_geval.py`

Guida il notebook 14 impostando le variabili `GEVAL_*`; il notebook resta la fonte di verità.
È pensato per una corsa a pagamento di molte ore, quindi procede per gradi:

```bash
python scripts/run_geval.py --righe 1              # una chiamata per metodo: verifica cache e prompt
python scripts/run_geval.py --pilota 20            # 20 righe: misura il costo per giudizio
python scripts/run_geval.py --budget 120 --valuta EUR   # corsa completa con limite di spesa
python scripts/run_geval.py --costo                # rapporto dei costi dalla cache, anche a corsa in corso
python scripts/run_geval.py --solo-metriche        # ricava le metriche dalla cache, senza chiamate
```

Prima di partire fa una chiamata da un token al deployment, così un nome sbagliato costa
secondi e non ore.

Due protezioni nascono da un incidente reale. Su Windows, Ctrl-C al driver **non** terminava il
kernel Jupyter avviato da nbconvert, che è in un altro gruppo di processi: il kernel rimasto
orfano ha continuato a giudicare per ore accanto alla corsa rilanciata, pagando due volte gli
stessi giudizi. Ora:

- al Ctrl-C il driver termina l'intero albero di processi (`termina_albero`, `taskkill /T`);
- ogni cache ha un file di lock con il PID del proprietario (`geval_cache_{ambito}.lock`), e il
  driver rifiuta di partire se un'altra corsa viva lo detiene.

---

## 11. Riproducibilità

### 11.1 Cosa è versionato

| Artefatto | Versionato | Perché |
|---|:-:|---|
| `data/text/`, `data/tab/` | ✓ | I dati |
| `results/summaries/*_test.tsv`, `*_full.tsv`, `*_test_budgetref.tsv` | ✓ | Grandi ma costosi da rigenerare |
| `results/summaries/*_sample.tsv` | | Solo prove locali |
| `results/metrics/` (CSV, JSON) | ✓ | I risultati |
| `results/metrics/geval_cache_*.jsonl` | ✓ | Giudizi pagati, non ripetibili |
| `results/fewshot/*_vicini_*.tsv` | ✓ | Il recupero degli esempi, riproducibile senza embedding |
| `results/embeddings_cache/` | | Rigenerabile (circa 140 MB) |
| `results/notebook_runs/` | | Notebook eseguiti dai driver negli ambiti a budget |
| Notebook con output | ✓ | Documentano la corsa che ha prodotto i numeri |

### 11.2 Determinismo

- Tutto ciò che ha un seme lo riceve fisso (`SEED = 42`): campione, `TruncatedSVD`, `KMeans`,
  `LatentDirichletAllocation`, ordine casuale delle righe del G-Eval.
- Gli LLM locali girano con `temperature=0.3`: una nuova corsa non darebbe testi identici.
  Per questo i riassunti sono versionati.
- Il giudice G-Eval è un modello di ragionamento e non accetta `temperature`: le chiamate non
  sono ripetibili. La cache dei giudizi è l'artefatto da cui si ricavano i numeri.
- Lo studio di sensibilità ha mostrato che il clustering SBERT è sensibile a differenze
  numeriche minime negli embedding: ricalcolati in un'altra corsa, differiscono nelle ultime
  cifre decimali, e la frase scelta cambia nel 26 % (K-Means) e nel 46 % (agglomerativo) delle
  righe. Le metriche medie restano le stesse entro 0,001, perché gli scambi avvengono quasi
  sempre fra frasi quasi identiche.

### 11.3 La ripresa può mescolare corse

La ripresa salta le righe già presenti nel TSV, senza chiedersi come siano state prodotte. Se
si cambiano modello, prompt o parametri e si rilancia sopra un file esistente, il risultato
mescola due corse. Per una corsa nuova bisogna **cancellare prima il TSV** (o la cache, per il
G-Eval).

### 11.4 L'ambiente

- Dipendenze in [requirements-notebooks.txt](../requirements-notebooks.txt). Sulla macchina con
  GPU NVIDIA va installata prima la build CUDA di PyTorch adatta alla scheda.
- Modelli ollama: `qwen2.5:7b-instruct`, `gemma4:latest`, `mistral:7b-instruct-v0.3-q4_K_M`
  (verificare con `ollama list`).
- Azure: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` e, per il giudice,
  `AZURE_GEVAL_DEPLOYMENT`. Mai chiavi nel codice.
- Modelli Hugging Face scaricati al primo avvio: MiniLM, BART, PEGASUS, PRIMERA, `roberta-large`
  e `all-mpnet-base-v2`.

Prima di una corsa lunga conviene sempre una prova con `LIMIT=3` (o `--limit 3` nel driver),
che attraversa generazione e valutazione su tre righe.

---

## 12. Scelte implementative e insidie note

Una raccolta delle cose che non si vedono leggendo il codice una funzione alla volta.

- **Righe saltate dagli estrattivi.** Su 22 righe della split test il costruttore di
  `psr.summarization` solleva un `IndexError` (bug della libreria). Il ciclo registra l'errore e
  prosegue: quei metodi hanno 5.588 righe invece di 5.610.
- **METEOR non limitato.** La formula della libreria (`fmean * (1 - penalty**3)`) può diventare
  molto negativa su testi degeneri: due righe di PEGASUS (un loop di ripetizione e un probabile
  disallineamento) portano la sua media da circa 0,42 a 0,079. È documentato, nessun file è
  stato corretto a mano.
- **La derivazione di TextRank e LexRank perde il BERTScore.** `deriva_metriche_test` ricalcola
  le medie su `COLONNE_METRICHE`, che non include le colonne BERTScore: un rilancio del driver
  le toglie dagli aggregati di quei due metodi. Il notebook 13 le ricalcola.
- **Il G-Eval non va mai unito al CSV standard** (sezione 7.3).
- **Il filtro di contenuti di Azure non è stabile.** Le righe respinte cambiano fra una corsa e
  l'altra, e la severità del filtro è configurata per risorsa: la stessa corsa su due risorse
  diverse ha coperto 5.250 e 5.562 righe. Va tenuto presente prima di leggere differenze di
  copertura come differenze fra metodi.
- **Contesto di ollama.** L'endpoint compatibile con OpenAI non permette di fissare `num_ctx`;
  per prompt lunghi (few-shot) serve l'API nativa con il controllo su `prompt_eval_count`
  (sezione 6.12).
- **Token di ragionamento.** Gemma e GPT-5-mini consumano il limite di token in ragionamento
  prima della risposta: con un limite basso la risposta è vuota. Per questo il limite è 1.500.
- **Prompt derivati, non riscritti.** I prompt a budget sono ottenuti per sostituzione da quelli
  base, con un `assert` che fallisce se la sostituzione non avviene (sezione 6.10).
- **Un solo intero per il tetto.** Ogni controllo di banda usa `tetto_parole`, mai 1,25 × *R*
  (sezione 8.3).
- **Ctrl-C su Windows.** Interrompere un driver non garantisce che il kernel del notebook muoia:
  prima di rilanciare una corsa a pagamento va verificato che non ci siano processi orfani
  (sezione 10.2).

---

## 13. Come aggiungere un nuovo metodo

1. **Copiare il notebook più simile** (un estrattivo, un modello `transformers`, un LLM) e
   cambiarne `METODO`, cioè lo slug che dà il nome a tutti i file.
2. **Scrivere `genera(documento, budget=None)`**. Se il metodo deve vedere il separatore fra
   articoli, passare `prepara=str.strip` al ciclo. Per supportare l'ambito a budget, produrre un
   ordine di frasi e chiamare `su.seleziona_per_budget`, oppure mettere la lunghezza nel prompt.
3. **Riempire `config`** con tutti i parametri che servono a capire come è stato prodotto il
   risultato: finisce nel JSON aggregato.
4. **Provare sul campione** con `LIMIT=3`, poi lanciare l'ambito `test`, a mano o aggiungendo il
   notebook a `NOTEBOOKS` in `run_benchmark_test.py` (con `METODI_PER_NOTEBOOK` se genera più
   slug, e nei set `NOTEBOOKS_OLLAMA`/`NOTEBOOKS_GPU` se serve).
5. **Aggiungere lo slug** a `METODI_BENCHMARK` e un colore a `COLORI_METODI` in `summ_utils`,
   e alle liste `METODI` di `applica_budget.py` e `run_geval.py`.
6. **Fare i backfill**: notebook 13 per il BERTScore, `run_geval.py` per il G-Eval (giudica solo
   ciò che manca in cache), `applica_budget.py --solo <slug>` per l'ambito a budget.
7. **Rieseguire i notebook di confronto** (05b-05d, 18) e aggiornare la documentazione.
