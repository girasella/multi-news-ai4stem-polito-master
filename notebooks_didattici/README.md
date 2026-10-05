# Notebook didattici

Versioni **didattiche** dei notebook del benchmark. Non producono risultati per il progetto: servono a
**capire** cosa fanno gli algoritmi seguendo **una sola riga** di Multi-News (split di test, `row_id`
50492 di default) dal testo originale al riassunto e alle metriche.

| Notebook | Metodo | Corrisponde a |
|---|---|---|
| [01_textrank_didattico.ipynb](01_textrank_didattico.ipynb) | TextRank (embedding MiniLM + PageRank) | [notebooks/01_textrank.ipynb](../notebooks/01_textrank.ipynb) |
| [02_lexrank_didattico.ipynb](02_lexrank_didattico.ipynb) | LexRank (TF-IDF + PageRank), con un confronto finale con TextRank | [notebooks/02_lexrank.ipynb](../notebooks/02_lexrank.ipynb) |
| [11_centroid_mmr_didattico.ipynb](11_centroid_mmr_didattico.ipynb) | Centroid (MEAD) + MMR (TF-IDF o BERT), con un confronto finale TF-IDF vs BERT | [notebooks/11_centroid_mmr.ipynb](../notebooks/11_centroid_mmr.ipynb) |
| [15_lsa_didattico.ipynb](15_lsa_didattico.ipynb) | LSA (TF-IDF + SVD), varianti `lsa` (top-k) e `lsa_steinberger` (deflazione), con un confronto finale tra le due | [notebooks/15_lsa.ipynb](../notebooks/15_lsa.ipynb) |
| [16_sbert_clustering_didattico.ipynb](16_sbert_clustering_didattico.ipynb) | Clustering su embedding SBERT + medoidi, varianti `sbert_kmeans` (K-means) e `sbert_agglom` (agglomerativo, coseno), con un confronto finale tra le due | [notebooks/16_sbert_clustering.ipynb](../notebooks/16_sbert_clustering.ipynb) |
| [17_lda_didattico.ipynb](17_lda_didattico.ipynb) | LDA (conteggi + temi latenti), allocazione proporzionale delle frasi ai temi, con un confronto concettuale finale con LSA | [notebooks/17_lda.ipynb](../notebooks/17_lda.ipynb) |

## Percorso (01 e 02)
1. selezione della riga · 2. suddivisione in frasi · 3. rappresentazioni (pre-processing +
   embedding / TF-IDF) · 4. similarità (coseno, calcolato anche a mano) · 5. heatmap · 6. grafo ·
7. PageRank passo per passo (con convergenza) · 8. selezione delle frasi · 9. riassunto finale ·
10. confronto con il riferimento (11 frasi vs **rigenerazione a budget** + **tetto** 1,25×, issue #16) ·
11. metriche (ROUGE-1/2/L, BLEU, METEOR; ROUGE-1 anche a mano) · 12. interpretazione.

## Percorso (11 Centroid+MMR)
Non ha né grafo né PageRank: 1. riga · 2. frasi · 3. vettori (TF-IDF o BERT, scelti da `VETTORIZZAZIONE`) ·
4. **centroide** (media dei vettori) · 5. **rilevanza** (coseno col centroide) · 6. matrice di similarità
(la misura della ridondanza) · 7. **greedy MMR** passo per passo, con l'effetto di `LAMBDA` ·
8. selezione · 9. riassunto (frasi rimesse in **ordine di documento**) · 10-12 come sopra ·
13. confronto TF-IDF vs BERT sulla stessa riga. A differenza di 01/02, qui la segmentazione usa solo
`s.original` (niente `s.corpus`), e le `assert` verificano il greedy contro una copia fedele di
`analizza_mmr`/`genera` del notebook 11 (le due funzioni non stanno in `summ_utils`).

## Percorso (15 LSA)
Niente grafo né PageRank: una fattorizzazione di matrici. 1. riga · 2. frasi · 3. matrice **frase × termine**
TF-IDF · 4. **SVD** completa con `numpy` (A = U·Σ·Vᵀ, scree plot, errore di troncatura, parole dei concetti) e
verifica che `TruncatedSVD` ne dia le prime `K` colonne a meno del segno · 5. **salienza** ‖z‖ = √(Σ σ²u²),
calcolata a mano · 6. intermezzo sulla similarità nello spazio dei concetti ("sinonimia") · 7. selezione
**top-k** (`lsa`) e **greedy con deflazione** passo per passo (`lsa_steinberger`), con l'effetto di
`K_LATENTE` · 8-12 come sopra · 13. confronto `lsa` vs `lsa_steinberger` sulla stessa riga. La variante si
sceglie con `VARIANTE`. Le `assert` verificano tutto contro una copia fedele di `segmenta` / `seleziona_topk` /
`seleziona_deflazione` / `genera` del notebook 15. Sulla riga 50492 il notebook mostra un limite del metodo:
con ~20 frasi lo spettro è piatto e la salienza premia frasi **corte e isolate** (la prima è "Oh the irony").

## Percorso (16 clustering SBERT)
Niente grafo né PageRank: un partizionamento. 1. riga · 2. frasi · 3. **embedding SBERT** normalizzati (coseno = prodotto
scalare, ‖a − b‖² = 2 − 2·coseno, verificati a mano; heatmap e scatter della similarità semantica contro quella lessicale
TF-IDF) · 4. **K-means**: Lloyd scritto a mano da 10 partenze casuali (inerzia per iterazione, minimi locali) e verifica che
il risultato di sklearn sia un punto fisso di Lloyd · 5. **agglomerativo** coseno + average linkage scritto a mano con
l'aggiornamento di Lance-Williams (tabella delle fusioni, dendrogramma `scipy`, taglie dei gruppi a confronto) ·
6. **centroidi e medoidi** (tabella per gruppo, mappa PCA) e la trappola dei gruppi di due frasi · 7. **ordine a giri**
dell'ambito a budget (medoidi dei gruppi grandi prima, poi i secondi rappresentanti) e intermezzo sull'effetto di `k` ·
8-12 come sopra · 13. confronto `sbert_kmeans` vs `sbert_agglom` (più la selezione "solo salienza" senza clustering) sulla
stessa riga e numeri del benchmark letti da `results/metrics/`, con `centroid_mmr_bert` (stesso encoder) come termine di
paragone. La variante si sceglie con `VARIANTE`. Le `assert` verificano tutto contro una copia fedele di `segmenta` /
`codifica` / `cluster_kmeans` / `cluster_agglomerativo` / `centroidi_e_medoidi` / `analizza_clustering` /
`ordine_rappresentanti` / `make_genera` del notebook 16. Sulla riga 50492 il notebook mostra due fatti: con 11 gruppi su 19
frasi la maggior parte dei gruppi sono **singoletti** (7 con K-means, 8 con l'agglomerativo), che valgono un posto ciascuno
come il gruppo del fatto principale; e nei gruppi di **due** frasi il medoide è un **pareggio esatto** (i due coseni col
centroide sono matematicamente uguali), deciso dall'arrotondamento: per questo il riassunto `test_budgetref` salvato nel
benchmark per questa riga differisce da quello ricalcolato qui (medoide 18 invece di 15 nel gruppo {15, 18}).

## Percorso (17 LDA)
Anche qui niente grafo né PageRank, ma un modello probabilistico. 1. riga · 2. frasi · 3. matrice frase × termine di
**conteggi** (`CountVectorizer`, non TF-IDF: LDA modella occorrenze intere) · 4. il **modello generativo** (una frase
finta campionata da θ e β) e i **temi** β · 5. le frasi come miscele di temi **θ**, con l'inferenza riscritta a mano
(E-step variazionale con la digamma, verificato contro `lda.transform`), la nettezza di θ e la sua relazione con la
lunghezza della frase · 6. **allocazione proporzionale** (peso Σθ, parte intera + residuo, selezione per tema) e mappa
tematica · 7. intermezzo su `N_TOPICS` e `random_state` (LDA non è deterministico come la SVD) · 8. selezione per
`MODALITA` · 9-12 come sopra · 13. **confronto concettuale con LSA** (tabella delle differenze, concetti vs temi, θ vs
|Z|, bias di lunghezza opposti, numeri del benchmark letti da `results/metrics/`). Le `assert` verificano
segmentazione, θ, quote e selezione contro una copia fedele di `segmenta` / `alloca_quote` / `analizza_lda` /
`seleziona_per_quote_parole` / `genera` del notebook 17. Sulla riga 50492 il notebook mostra due fatti: θ è quasi
**dura** e più netta sulle frasi lunghe (Spearman 0,97 con la lunghezza), quindi il criterio "θ più alto" sceglie frasi
lunghe (528 parole a 11 frasi contro 168 del riferimento); una verifica leave-one-out mostra però che quella nettezza è
in buona parte **circolare** (tolta la frase dall'addestramento la correlazione con la lunghezza passa da +1,00 a −0,61);
a budget un tema può restare senza frasi.

## Lunghezza del riassunto: parametro `MODALITA`
Nella sezione 1 di ogni notebook si sceglie come fissare la lunghezza (i metodi producono solo un
ordinamento delle frasi: per 11 è l'ordine di selezione del greedy; **fa eccezione LDA**, in cui la lunghezza
entra nelle quote per tema e 'frasi' e 'budget' sono due allocazioni diverse, non due tagli dello stesso ordine):

| `MODALITA` | regola | ambito del benchmark |
|---|---|---|
| `'frasi'` (default) | prime `N_SENTENCES` = 11 frasi per punteggio | `test` |
| `'budget'` | frasi in ordine di punteggio fino alla lunghezza in parole del riferimento, poi **tetto** 1,25× | `test_budgetref` (issue #16) |

Le sezioni 8-12 seguono la modalità scelta; le tabelle delle sezioni 10-11 mostrano comunque entrambe, con
◀ sulla scelta. Attenzione: su righe con frasi molto lunghe (come la 50492, ~160 parole a frase) il budget
può corrispondere a sole 2-3 frasi.

## Fedeltà al benchmark
Le parti calcolate "a mano" (coseno, IDF, PageRank, centroide, greedy MMR, SVD, K-means, fusioni agglomerative, medoidi,
selezione, ROUGE-1) sono verificate con `assert`
contro `pyAutoSummarizer` (01/02) o contro il codice dei notebook 11, 15, 16 e 17 (11, 15, 16, 17), cioè ciò che usa il benchmark: i numeri qui mostrati sono quelli che il
codice di produzione calcolerebbe per la stessa riga. Funzioni di caricamento, budget e metriche sono
quelle di [`notebooks/summ_utils.py`](../notebooks/summ_utils.py).

## Uso
- Requisiti: gli stessi di [`requirements-notebooks.txt`](../requirements-notebooks.txt) (più `networkx`,
  `seaborn`, `matplotlib`; 16 e 17 usano anche `scipy`, già dipendenza di scikit-learn). Dati: basta **`data/tab/test.tab`** (68 MB); in alternativa
  `data/tab/complete.tab` (690 MB). I `.tab` sono su **Git LFS**: se su Colab clonare il repository dà
  file di pochi byte, o l'upload di `complete.tab` si interrompe (errore `not enough values to unpack`),
  carica solo `test.tab` in `<radice>/data/tab/` insieme a `notebooks/summ_utils.py`. Il notebook
  legge prima `test.tab` e ripiega su `complete.tab`.
- Si possono aprire dalla cartella `notebooks_didattici/` o dalla radice del repository. I notebook
  01, 11 (variante BERT) e 16 scaricano `all-MiniLM-L6-v2` alla prima esecuzione.
- Cambiando `ROW_ID` (sezione 1) tutto il resto si adatta; con documenti molto lunghi heatmap e grafo
  diventano fitti. Una riga **non è un benchmark**: per i confronti tra metodi vedere
  [`notebooks/`](../notebooks/README.md).
- I notebook sono consegnati con gli output già calcolati.
