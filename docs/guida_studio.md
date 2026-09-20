# Guida di studio: riassunto automatico multi-documento su Multi-News

*Project work finale — Master di II livello "Artificial Intelligence for STEM", Politecnico
di Torino*

---

## Indice

1. [Introduzione e obiettivi](#1-introduzione-e-obiettivi)
2. [Il punto di partenza: il paper e il dataset Multi-News](#2-il-punto-di-partenza-il-paper-e-il-dataset-multi-news)
3. [Approccio del progetto](#3-approccio-del-progetto)
4. [I metodi di summarization](#4-i-metodi-di-summarization)
5. [Le metriche di valutazione](#5-le-metriche-di-valutazione)
6. [Risultati principali](#6-risultati-principali)
7. [Il confondente della lunghezza: il confronto a lunghezza pari](#7-il-confondente-della-lunghezza-il-confronto-a-lunghezza-pari)
8. [Quanto ci si può fidare del giudice](#8-quanto-ci-si-può-fidare-del-giudice)
9. [Limiti e avvertenze di lettura](#9-limiti-e-avvertenze-di-lettura)
10. [Bibliografia](#10-bibliografia)

---

## 1. Introduzione e obiettivi

Questo progetto affronta il problema del **riassunto automatico multi-documento**
(*Multi-Document Summarization*, MDS): dato un insieme di articoli di cronaca che trattano lo
stesso evento, generare un unico testo di sintesi che ne riporti le informazioni salienti,
così come farebbe un redattore umano che deve produrre una rassegna stampa a partire da più
fonti.

Il lavoro si articola in tre parti:

1. **Analisi esplorativa (EDA)** — un audit quantitativo della qualità e della struttura
   dell'intero corpus Multi-News (56.216 esempi), per capire che tipo di dati si stanno
   maneggiando prima di usarli per addestrare o valutare modelli: quanto sono lunghi gli
   articoli sorgente, quanto sono lunghi i riassunti di riferimento, quanti articoli sorgente
   compongono in media un cluster, quali problemi di qualità (righe vuote, duplicati, mismatch
   sorgente/riassunto) sono presenti nel dataset così come pubblicato.
2. **Curation del dataset** — una pulizia mirata (rimozione di righe palesemente
   problematiche: sorgenti troppo corte, troppo lunghe, o duplicate) per ottenere una versione
   più affidabile del corpus da usare nelle fasi successive.
3. **Benchmark di summarization** — il cuore del progetto: il confronto sistematico di
   **18 metodi diversi** di riassunto automatico multi-documento, dalle tecniche classiche
   basate su statistiche testuali fino ai grandi modelli linguistici (LLM), tutti valutati
   sullo stesso split di test con lo stesso insieme di metriche.
4. **Controllo del confondente della lunghezza** — su richiesta del relatore, la verifica di
   quanto le differenze di lunghezza fra i riassunti dei diversi metodi influenzino le
   metriche, e il rifacimento del confronto *a lunghezza pari*, con una validazione
   indipendente del giudice usato per la metrica reference-free.

Lo scopo non è proporre un nuovo metodo di summarization, ma **mappare lo stato dell'arte** —
dalle tecniche pre-neurali degli anni 2000 ai modelli Transformer specializzati e ai moderni
LLM general-purpose — su un unico dataset e con un protocollo di valutazione condiviso,
per capire quali famiglie di metodi funzionano meglio, secondo quali metriche, e perché.

Questo documento è pensato come **guida di studio e di presentazione**: si concentra sul
*perché* di ogni scelta metodologica (la teoria dietro ciascun metodo e ciascuna metrica, con
riferimenti bibliografici per approfondire) e sui risultati, non sui dettagli implementativi
del codice — quelli sono documentati separatamente nei notebook del progetto.

---

## 2. Il punto di partenza: il paper e il dataset Multi-News

### 2.1 Il dataset

Il progetto si basa su **Multi-News** (Fabbri et al., 2019, *"Multi-News: A Large-Scale
Multi-Document Summarization Dataset and Abstractive Hierarchical Model"*, ACL 2019), il primo
dataset su larga scala per la summarization multi-documento in ambito news. Prima di
Multi-News, i dataset di riferimento per l'MDS (come DUC o TAC) erano piccoli (poche centinaia
di cluster) e costruiti ad hoc per campagne di valutazione; Multi-News porta l'MDS nell'era del
deep learning fornendo decine di migliaia di esempi realistici.

Caratteristiche principali:

- **56.216 coppie** (insieme di articoli sorgente, riassunto di riferimento), raccolte dal sito
  *newser.com*, dove redattori professionisti scrivono un riassunto citando esplicitamente gli
  articoli sorgente da cui attingono.
- In media **~2,75 articoli sorgente** per esempio (mediana 2; l'82% dei cluster ha al massimo
  3 fonti), provenienti da oltre 1.500 siti di news diversi — una diversità di stile e fonte
  molto maggiore rispetto ai dataset precedenti.
- I riassunti di riferimento sono **sensibilmente più lunghi** di quelli tipici della
  summarization a documento singolo (es. CNN/DailyMail, dove il riassunto medio è ~56 parole):
  qui la lunghezza media è di circa 260 parole su ~10 frasi, il che rende il compito più
  impegnativo anche solo in termini di coerenza testuale su un output lungo.
- Split originali: train/validation/test in proporzione 80/10/10 (44.972/5.622/5.622 esempi).
- Il progetto lavora sia sulla versione "grezza" del dataset (`data/text/`, così come
  rilasciata) sia su una versione derivata e ripulita (115 righe scartate per problemi di
  qualità: sorgenti vuote o duplicate, mismatch sorgente/riassunto).

### 2.2 Il modello proposto nel paper: Hi-MAP

È importante chiarire cosa proponesse *esattamente* il paper originale, perché è un punto di
partenza concettuale per buona parte dei metodi usati in questo progetto. Il paper **non**
propone semplicemente un Transformer: propone un modello ad hoc chiamato **Hi-MAP**
(*Hierarchical MMR-Attention Pointer-generator network*), che combina due ingredienti teorici
distinti già noti in letteratura:

1. Un **pointer-generator network** (See et al., 2017) — un modello encoder-decoder ricorrente
   che, ad ogni passo di generazione, può scegliere se generare una parola dal proprio
   vocabolario oppure "copiarla" direttamente dal testo sorgente (utile per nomi propri, cifre,
   entità che il modello altrimenti non saprebbe generare correttamente).
2. Un modulo di **Maximal Marginal Relevance (MMR)** (Carbonell & Goldstein, 1998) integrato
   *dentro* il modello neurale: ad ogni passo di decodifica, le frasi candidate del documento
   vengono ripesate in base a un compromesso tra rilevanza rispetto allo stato corrente del
   riassunto e ridondanza rispetto a quanto già generato, e questo punteggio MMR viene usato per
   modulare l'attenzione del decoder sulle parole del documento sorgente.

Il modello costruisce rappresentazioni **gerarchiche** delle frasi (un livello a parole con
Bi-LSTM, un livello a frasi con LSTM), da cui il nome "hierarchical". Hi-MAP viene confrontato
con diversi baseline: metodi estrattivi classici (First-3, LexRank, TextRank, MMR "puro"),
varianti neurali pointer-generator precedenti (PG-Original, PG-MMR, PG-BRNN) e un
**CopyTransformer** (un Transformer con un meccanismo di copia, usato come termine di
confronto per la famiglia di architetture "Transformer puro"). Sul test set di Multi-News,
Hi-MAP ottiene il miglior punteggio ROUGE-2 e ROUGE-SU tra i sistemi automatici, secondo solo
al CopyTransformer su ROUGE-1; il confronto con First-3 (che da solo ottiene un ROUGE-1
sorprendentemente alto, 39.41) mostra già nel paper originale quanto sia forte il "lead bias"
giornalistico su questo dataset — un tema che ritorna anche nei risultati di questo progetto
(sezione 4.1).

### 2.3 Perché questo paper è il punto di partenza

Il paper Multi-News fissa sia il **dataset** sia un primo **insieme di baseline e metriche**
(ROUGE) contro cui confrontarsi. Questo progetto eredita il dataset e amplia enormemente
l'insieme di metodi confrontati (da 8 sistemi nel paper originale a 18 in questo lavoro),
aggiungendo famiglie di metodi che nel 2019 non erano ancora disponibili o non erano state
provate su Multi-News (LLM general-purpose, modelli specializzati come PEGASUS e PRIMERA), e
amplia anche l'insieme di metriche (BERTScore, valutazione con LLM-as-judge) per ottenere un
quadro più ricco di quanto il solo ROUGE possa offrire.

---

## 3. Approccio del progetto

Il progetto ha selezionato i metodi da confrontare partendo da una **mappatura sistematica**
delle tecniche di MDS non basate su LLM insegnate nel corso di Deep NLP (PoliTO, prof.
L. Cagliero), verificando quali fossero disponibili in librerie mature (`sumy`,
`pyAutoSummarizer`) e quali richiedessero un'implementazione originale. Da questa mappatura
sono stati scelti i metodi che coprono le famiglie concettualmente più rilevanti — baseline
posizionale, metodi a grafo, centroid-based, fattorizzazione latente (LSA), clustering su
embedding, topic modeling — evitando di implementare tecniche marginali o non validate
direttamente su dataset di news multi-documento.

A questi metodi "classici" (non neurali o pre-neurali) si affiancano:

- tre modelli **abstractive specializzati**, pre-addestrati specificamente per il riassunto
  (uno dei quali, PRIMERA, nativamente per l'MDS);
- tre **LLM general-purpose** eseguiti localmente via Ollama, usati *zero-shot* (senza alcun
  fine-tuning sul task di summarization);
- un **LLM cloud** (GPT-5-mini su Azure AI Foundry), anch'esso zero-shot.

Un vincolo metodologico importante: la libreria `pyAutoSummarizer`, pur offrendo anche
implementazioni di alcuni algoritmi estrattivi, viene usata in questo progetto **solo come
motore di valutazione** (ROUGE, BLEU, METEOR), non come motore di generazione — per evitare
ambiguità sull'esatta variante di algoritmo effettivamente eseguita (es. "TextRank" indica
implementazioni algoritmicamente diverse in librerie diverse).

Tutti i 18 metodi sono valutati sullo **stesso split di test pulito** (5.610 esempi su 5.622
originali, dopo l'esclusione delle righe con problemi di qualità), con lo stesso protocollo di
valutazione, per garantire un confronto equo. I risultati aggregati di questo confronto sono
presentati nella sezione 6.

---

## 4. I metodi di summarization

I metodi sono presentati per famiglia concettuale, dal più semplice al più sofisticato. Per
ciascuno: cosa fa (a livello di principio, non di implementazione), su quale idea teorica si
basa, e riferimenti per approfondire.

### 4.1 Baseline posizionale (First-k / Lead)

**Idea**: selezionare semplicemente le prime *k* frasi dei documenti sorgente, senza alcuna
analisi del contenuto. Sfrutta il cosiddetto **lead bias**: nella scrittura giornalistica, le
informazioni più importanti tendono a comparire nelle prime righe dell'articolo (la piramide
rovesciata). È il baseline più semplice possibile, ma su corpora di news è sorprendentemente
difficile da battere — lo stesso paper Multi-News lo mostra chiaramente (First-3 ottiene un
ROUGE-1 superiore a LexRank, TextRank e MMR "puri").

**Perché è utile come termine di paragone**: qualunque metodo più sofisticato deve giustificare
la propria complessità superando questo baseline, altrimenti l'informazione aggiuntiva che
introduce (analisi semantica, ranking, ecc.) non sta producendo valore reale.

**Riferimenti**:
- A. Fabbri, I. Li, T. She, S. Li, D. Radev, *"Multi-News: A Large-Scale Multi-Document
  Summarization Dataset and Abstractive Hierarchical Model"*, ACL 2019 — discute
  esplicitamente la forza del lead bias su questo dataset.
- A. Nenkova, K. McKeown, *"Automatic Summarization"*, Foundations and Trends in Information
  Retrieval, 2011 — survey di riferimento sulla summarization automatica, inquadra il ruolo
  storico dei baseline posizionali.

### 4.2 Metodi a grafo: TextRank e LexRank

**Idea**: rappresentare le frasi di un documento (o di un cluster di documenti) come nodi di un
grafo, con archi pesati dalla similarità testuale tra coppie di frasi. Applicando un algoritmo
di **centralità su grafo** (nello spirito di PageRank), le frasi "più centrali" — cioè simili a
molte altre frasi importanti — ricevono un punteggio più alto e vengono selezionate per il
riassunto. L'intuizione è che le frasi che riassumono meglio il contenuto sono quelle
semanticamente più "rappresentative" dell'insieme.

- **TextRank** (Mihalcea & Tarau, 2004) applica l'idea del *random surfer* di PageRank
  direttamente a un grafo di frasi con pesi di similarità testuale.
- **LexRank** (Erkan & Radev, 2004) formalizza lo stesso principio tramite **centralità
  autovettoriale** su un grafo pesato da similarità TF-IDF/coseno, ed è stato proposto
  esplicitamente per l'MDS (a differenza di TextRank, nato per documento singolo).

Entrambi condividono la radice algoritmica in PageRank, l'algoritmo alla base del motore di
ricerca Google originale, applicato qui non a pagine web ma a frasi.

**Riferimenti**:
- R. Mihalcea, P. Tarau, *"TextRank: Bringing Order into Text"*, EMNLP 2004.
- G. Erkan, D. Radev, *"LexRank: Graph-based Lexical Centrality as Salience in Text
  Summarization"*, Journal of Artificial Intelligence Research (JAIR), 2004.
- S. Brin, L. Page, *"The Anatomy of a Large-Scale Hypertextual Web Search Engine"*, WWW 1998
  — il paper originale di PageRank, base algoritmica comune ai due metodi precedenti.

### 4.3 Centroid-based + MMR

**Idea**: rappresentare ogni frase come un vettore (TF-IDF o embedding neurale) e calcolare il
**centroide** dell'intero cluster di documenti (la media dei vettori-frase, o un vettore che
rappresenta i termini/temi più salienti dell'insieme). La rilevanza di ogni frase si misura
come similarità (coseno) rispetto al centroide — è l'idea alla base del sistema **MEAD**, uno
dei primi sistemi di MDS.

A questo si aggiunge la **Maximal Marginal Relevance (MMR)**: invece di selezionare
semplicemente le frasi più vicine al centroide (rischiando ridondanza, dato che più fonti
tendono a ripetere le stesse informazioni salienti), si seleziona in modo greedy massimizzando
a ogni passo un compromesso tra rilevanza (similarità al centroide/query) e **diversità**
(bassa similarità con le frasi già selezionate), controllato da un parametro λ. Questo è
particolarmente rilevante nell'MDS, dove più articoli sulla stessa notizia contengono
naturalmente informazioni ripetute.

**Riferimenti**:
- D. Radev, H. Jing, M. Styś, D. Tam, *"Centroid-based summarization of multiple documents"*,
  Information Processing & Management, 2004 — il sistema MEAD.
- J. Carbonell, J. Goldstein, *"The Use of MMR, Diversity-Based Reranking for Reordering
  Documents and Producing Summaries"*, SIGIR 1998 — paper originale della MMR.

### 4.4 LSA (Latent Semantic Analysis)

**Idea**: costruire una matrice termine-frase (righe = termini del vocabolario, colonne =
frasi) e applicarvi la **Singular Value Decomposition (SVD)**, ottenendo una rappresentazione
in uno spazio di "concetti latenti" a dimensionalità ridotta. L'assunzione è che i concetti
latenti principali corrispondano ai temi più importanti del testo, e che le frasi che
proiettano fortemente su questi concetti siano le più rappresentative da includere nel
riassunto.

Il progetto confronta due varianti di selezione delle frasi a partire dalla stessa
decomposizione SVD:
- selezione per **norma nello spazio latente** (le frasi con proiezione più forte sui concetti
  principali, secondo la formulazione originale di Gong & Liu);
- una variante **greedy con deflazione** (Steinberger & Ježek), pensata specificamente per
  l'MDS: dopo aver selezionato una frase, il suo contributo viene "sottratto" dalla
  rappresentazione latente residua, per favorire esplicitamente la copertura di temi diversi ed
  evitare ridondanza tra frasi provenienti da fonti diverse.

**Riferimenti**:
- Y. Gong, X. Liu, *"Generic Text Summarization Using Relevance Measure and Latent Semantic
  Analysis"*, SIGIR 2001.
- J. Steinberger, K. Ježek, *"Using Latent Semantic Analysis in Text Summarization and
  Summary Evaluation"*, Proc. ISIM 2004.

### 4.5 Clustering su sentence embedding (SBERT)

**Idea**: rappresentare ogni frase come un embedding semantico denso, prodotto da un modello
Sentence-BERT (SBERT) piuttosto che da un semplice TF-IDF — catturando quindi similarità di
*significato* anche tra frasi che non condividono parole in comune. Le frasi vengono poi
raggruppate in cluster (ogni cluster corrisponde idealmente a un sotto-tema del cluster di
articoli), e da ogni cluster si estrae la frase più rappresentativa (il **medoide**, la frase
più vicina al centro del cluster).

Il progetto confronta due algoritmi di clustering sugli stessi embedding:
- **K-Means** (clustering a partizione, numero di cluster fissato a priori);
- **clustering agglomerativo gerarchico** (bottom-up, unisce iterativamente i cluster più
  simili).

Concettualmente, questo approccio appartiene alla famiglia dei metodi che il corso PoliTO
classifica come "summarization neurale self-supervised": non richiede addestramento
specifico per il task di summarization, ma sfrutta rappresentazioni semantiche pre-addestrate
(un modello linguistico pre-addestrato, ma usato solo come *encoder* di frasi, non come
generatore).

**Riferimenti**:
- N. Reimers, I. Gurevych, *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"*,
  EMNLP-IJCNLP 2019 — il paper che introduce SBERT.
- J. MacQueen, *"Some Methods for Classification and Analysis of Multivariate Observations"*,
  Berkeley Symposium on Mathematical Statistics, 1967 — formulazione classica di K-Means, alla
  base della variante di clustering usata.

### 4.6 LDA (Latent Dirichlet Allocation) / topic modeling

**Idea**: modellare ogni documento come una **mistura di argomenti latenti** (*topic*), dove
ogni argomento è a sua volta una distribuzione di probabilità su parole del vocabolario. È un
modello generativo probabilistico bayesiano (non un metodo di riduzione dimensionale
deterministico come LSA). Applicato alla summarization, si stima la distribuzione dei topic nel
cluster di articoli, e si selezionano le frasi da includere nel riassunto in proporzione al
peso di ciascun topic — con l'obiettivo di garantire che il riassunto copra tutti i temi
principali nella proporzione in cui compaiono nelle fonti, non solo il tema dominante.

**Riferimenti**:
- D. Blei, A. Ng, M. Jordan, *"Latent Dirichlet Allocation"*, Journal of Machine Learning
  Research (JMLR), 2003 — il paper originale di LDA.

### 4.7 Modelli abstractive specializzati: BART, PEGASUS, PRIMERA

A differenza dei metodi precedenti (tutti **estrattivi**: selezionano frasi esistenti dal
testo sorgente), questi tre modelli sono **abstractive**: generano un testo nuovo, parola per
parola, potenzialmente riformulando il contenuto invece di copiarlo letteralmente. Sono tutti
basati sull'architettura **Transformer** (Vaswani et al., 2017) in configurazione
encoder-decoder, ma differiscono per come sono stati pre-addestrati:

- **BART** (Lewis et al., 2019) è pre-addestrato come *denoising autoencoder*: il testo viene
  corrotto artificialmente (frasi permutate, span di token mascherati) e il modello impara a
  ricostruire il testo originale. È un modello general-purpose per generazione di testo, non
  specifico per la summarization, qui usato nella variante fine-tuned su CNN/DailyMail
  (`bart-large-cnn`).
- **PEGASUS** (Zhang et al., 2020) introduce un obiettivo di pre-training pensato
  *specificamente* per la summarization: **Gap Sentence Generation (GSG)** — durante il
  pre-training, intere frasi vengono rimosse dal documento (scelte perché ritenute
  "importanti" secondo un punteggio ROUGE rispetto al resto del documento) e il modello deve
  generarle, imitando così direttamente il compito di produrre un riassunto. Il modello usato
  in questo progetto (`pegasus-multi_news`) è stato specificamente fine-tuned sullo split di
  training di Multi-News.
- **PRIMERA** (Xiao et al., 2022) estende l'idea di PEGASUS al caso **multi-documento**:
  introduce un obiettivo di pre-training basato su **Pyramid ROUGE** (le frasi da mascherare
  sono scelte considerando l'intero cluster di documenti correlati, non un singolo documento) e
  usa un'architettura **Longformer** con *sparse global attention*, che permette di processare
  input molto più lunghi (fino a 4096 token) rispetto ai 1024 tipici di BART/PEGASUS —
  particolarmente adatto quando l'input è la concatenazione di più articoli.

**Riferimenti**:
- A. Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017 — l'architettura Transformer
  di base.
- A. See, P. Liu, C. Manning, *"Get To The Point: Summarization with Pointer-Generator
  Networks"*, ACL 2017 — meccanismo di copia, ingrediente teorico condiviso con Hi-MAP
  (sezione 2.2).
- M. Lewis et al., *"BART: Denoising Sequence-to-Sequence Pre-training for Natural Language
  Generation, Translation, and Comprehension"*, ACL 2020.
- J. Zhang, Y. Zhao, M. Saleh, P. Liu, *"PEGASUS: Pre-training with Extracted Gap-sentences
  for Abstractive Summarization"*, ICML 2020.
- W. Xiao, I. Beltagy, G. Carenini, A. Cohan, *"PRIMERA: Pyramid-based Masked Sentence
  Pre-training for Multi-document Summarization"*, ACL 2022.
- I. Beltagy, M. Peters, A. Cohan, *"Longformer: The Long-Document Transformer"*, arXiv 2020 —
  l'architettura alla base dell'attenzione sparsa di PRIMERA.

### 4.8 LLM general-purpose: Qwen2.5, Gemma, Mistral (locali) e GPT-5-mini (cloud)

**Idea**: usare un modello linguistico di grandi dimensioni, addestrato in modo general-purpose
(non specializzato sulla summarization né su Multi-News), e chiedergli di produrre un
riassunto tramite un **prompt zero-shot** — senza alcun fine-tuning aggiuntivo, senza esempi
nel prompt (*few-shot*), solo un'istruzione in linguaggio naturale. È l'approccio oggi più
comune nella pratica industriale: si sfrutta la capacità generale del modello di seguire
istruzioni, appresa durante l'**instruction tuning** in fase di addestramento, applicandola a
un compito specifico senza addestramento dedicato.

Nel progetto sono confrontati quattro modelli di questa famiglia:
- **Qwen2.5-7B-Instruct** e **Mistral-7B-Instruct-v0.3**, eseguiti localmente via Ollama;
- **Gemma 4 E4B**, eseguito localmente via Ollama;
- **GPT-5-mini**, eseguito in cloud su Azure AI Foundry.

Tutti e quattro sono modelli "instruct/chat", cioè sottoposti dopo il pre-training a una fase
di *instruction tuning* (e tipicamente di allineamento tramite feedback umano) che li rende
capaci di seguire istruzioni in linguaggio naturale invece di limitarsi a completare testo.

**Riferimenti**:
- A. Yang et al. (Qwen Team, Alibaba), *"Qwen2.5 Technical Report"*, arXiv 2024.
- Gemma Team, Google DeepMind, *"Gemma: Open Models Based on Gemini Research and
  Technology"*, technical report, 2024.
- A. Jiang et al. (Mistral AI), *"Mistral 7B"*, arXiv 2023.
- L. Ouyang et al., *"Training language models to follow instructions with human feedback"*
  (InstructGPT), NeurIPS 2022 — riferimento concettuale per il paradigma di instruction
  tuning/RLHF condiviso (in varie forme) da tutti i moderni LLM instruct, incluso GPT-5-mini.

---

## 5. Le metriche di valutazione

Le prime quattro metriche (ROUGE, BLEU, METEOR, BERTScore) sono tutte **reference-based**:
misurano quanto il riassunto generato assomiglia al riassunto di riferimento scritto da un
umano. La quinta (G-Eval) è **reference-free**: valuta la qualità del riassunto guardando solo
il testo sorgente, senza confrontarlo con nessun riassunto "corretto" predefinito. Questa
distinzione è la chiave per interpretare correttamente i risultati della sezione 6.

### 5.1 ROUGE-1 / ROUGE-2 / ROUGE-L

**Cosa misura**: la sovrapposizione lessicale tra il riassunto generato e quello di
riferimento, in termini di n-grammi condivisi. ROUGE-1 conta parole singole in comune,
ROUGE-2 conta bigrammi (coppie di parole consecutive) in comune, ROUGE-L si basa sulla
**Longest Common Subsequence** (la più lunga sottosequenza comune, non necessariamente
contigua) tra i due testi. Per ciascuna variante si riportano tipicamente precisione, recall
e F1.

È la metrica storicamente più usata in summarization: semplice, veloce da calcolare, ma
puramente lessicale — non valuta la correttezza semantica o fattuale del contenuto, solo la
sovrapposizione di parole/sequenze di superficie con il riferimento.

**Nota di lettura**: nell'implementazione usata in questo progetto (libreria
pyAutoSummarizer), ROUGE-N è calcolato su **insiemi di n-grammi unici**, non sul conteggio
"clippato" (*clipped count*) previsto dalla definizione standard di Lin (2004). Questo rende i
valori confrontabili *internamente* fra i 18 metodi del progetto, ma **non direttamente
paragonabili** ai numeri ROUGE riportati in letteratura (compreso il paper Multi-News stesso).

**Riferimento**: C.-Y. Lin, *"ROUGE: A Package for Automatic Evaluation of Summaries"*, ACL
Workshop "Text Summarization Branches Out", 2004.

### 5.2 BLEU

**Cosa misura**: nato per la valutazione della traduzione automatica, BLEU calcola una media
geometrica delle precisioni su n-grammi (tipicamente da 1 a 4, da cui "BLEU-4") tra testo
generato e riferimento, moltiplicata per una **brevity penalty** che penalizza output troppo
corti rispetto al riferimento (per evitare che un output brevissimo ma "puro" nelle parole
usate ottenga un punteggio artificialmente alto). Come ROUGE, è una metrica puramente
lessicale/di sovrapposizione di superficie.

**Riferimento**: K. Papineni, S. Roukos, T. Ward, W.-J. Zhu, *"BLEU: a Method for Automatic
Evaluation of Machine Translation"*, ACL 2002.

### 5.3 METEOR

**Cosa misura**: come BLEU e ROUGE, confronta generato e riferimento, ma con un allineamento
più flessibile — oltre alla corrispondenza esatta di parole, considera **stemming** (radice
comune) e **sinonimia** (tramite risorse come WordNet), e combina precisione e recall in una
media armonica pesata (che dà più peso al recall), penalizzata da un fattore di
**frammentazione** se le parole allineate non sono in ordine contiguo tra i due testi. È
storicamente nata per superare alcuni limiti di BLEU nella valutazione della traduzione
automatica, mostrando correlazione più alta con i giudizi umani.

**Nota di lettura**: la formula usata dalla libreria di questo progetto
(`meteor = fmean * (1 - penalty³)`) non è limitata inferiormente in casi patologici. Su un
numero molto ridotto di righe (2 su 5.610 per PEGASUS, dovute a un loop di ripetizione in fase
di generazione e a un probabile mismatch sorgente/riassunto nel dataset), il punteggio METEOR
calcolato può risultare fortemente negativo, distorcendo la media aggregata di quel metodo
verso il basso in modo non rappresentativo della sua reale qualità media.

**Riferimento**: S. Banerjee, A. Lavie, *"METEOR: An Automatic Metric for MT Evaluation with
Improved Correlation with Human Judgments"*, ACL Workshop on Intrinsic and Extrinsic
Evaluation Measures for MT, 2005.

### 5.4 BERTScore

**Cosa misura**: a differenza delle tre metriche precedenti, non lavora su corrispondenze
lessicali esatte ma su **similarità semantica**, sfruttando gli embedding contestuali di un
modello linguistico pre-addestrato (in questo progetto, RoBERTa-large). Ogni token del
riassunto generato viene messo in corrispondenza (via similarità coseno tra embedding) con il
token più simile del riferimento, e viceversa, ottenendo precisione, recall e F1 "semantiche".
Questo permette di riconoscere come corrette anche riformulazioni che non condividono le
stesse parole del riferimento — un limite importante di ROUGE/BLEU/METEOR quando si valutano
modelli abstractive che parafrasano.

**Riferimento**: T. Zhang, V. Kishore, F. Wu, K. Weinberger, Y. Artzi, *"BERTScore:
Evaluating Text Generation with BERT"*, ICLR 2020.

### 5.5 G-Eval (valutazione tramite LLM-as-a-judge)

**Cosa misura**: a differenza di tutte le metriche precedenti, **G-Eval non confronta il
riassunto generato con un riferimento umano**. Un modello linguistico di grandi dimensioni
(il "giudice") legge il testo sorgente e il riassunto generato, e assegna un punteggio da 1 a
5 su quattro dimensioni di qualità indipendenti:

- **Coherence** (coerenza): quanto il riassunto è ben strutturato e organizzato in un
  discorso logico, non solo una sequenza di informazioni slegate.
- **Consistency** (consistenza fattuale): quanto il riassunto è fedele ai fatti riportati nel
  testo sorgente, senza contraddizioni o informazioni inventate (*hallucination*).
- **Fluency** (fluidità): la qualità grammaticale e di scorrevolezza del testo, valutata sul
  solo riassunto (senza guardare la sorgente).
- **Relevance** (rilevanza): quanto il riassunto cattura le informazioni davvero importanti
  del testo sorgente, senza includere dettagli marginali.

Il vantaggio concettuale di questo approccio è che misura la **qualità intrinseca** del
riassunto rispetto al testo sorgente, indipendentemente da come un particolare essere umano
abbia scelto di riassumere lo stesso contenuto (mentre ROUGE penalizza qualunque
riformulazione valida che si discosti dalle parole scelte nel riferimento). Lo svantaggio è
che introduce una nuova fonte di variabilità e di possibile bias (il giudice può avere
preferenze sistematiche, ad esempio per output più lunghi o per uno stile simile al proprio) e
non è perfettamente riproducibile bit-per-bit tra un'esecuzione e l'altra.

In questo progetto il giudice è un modello (GPT-5.4-mini su Azure) scelto per essere
indipendente da tutti i 18 metodi valutati (nessun metodo valutato è generato dallo stesso
modello) e più recente di ciascuno di essi. L'indipendenza a livello di *modello* non
garantisce però quella a livello di *famiglia*: uno dei metodi valutati (GPT-5-mini) è della
stessa famiglia del giudice, e la letteratura documenta che gli LLM tendono a preferire i
testi generati da modelli simili a sé (*self-preference bias*). Questo rischio è stato
verificato sperimentalmente con un secondo giudice di lignaggio diverso — sezione 8.

**Riferimento**: Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, C. Zhu, *"G-Eval: NLG Evaluation
using GPT-4 with Better Human Alignment"*, EMNLP 2023 — il paper che ha proposto e validato
l'uso di LLM come valutatori (*LLM-as-a-judge*) per la generazione di linguaggio naturale,
mostrando correlazione con i giudizi umani superiore alle metriche reference-based classiche.

---

## 6. Risultati principali

Tutti i valori seguenti sono calcolati sullo split di **test pulito** (5.610 esempi), lo
stesso per tutti i 18 metodi, a garanzia di un confronto equo.

### 6.1 Metriche reference-based (ROUGE, BLEU, METEOR, BERTScore)

| Metodo | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BLEU-4 | METEOR | BERTScore F1 |
|---|---|---|---|---|---|---|
| First-k (PSR) | 0.355 | 0.121 | 0.180 | 0.082 | 0.389 | 0.840 |
| First-k (NLTK) | 0.356 | 0.123 | 0.181 | 0.083 | 0.391 | 0.843 |
| Centroid+MMR (TF-IDF) | 0.383 | 0.144 | 0.189 | 0.097 | 0.500 | 0.842 |
| Centroid+MMR (BERT) | 0.373 | 0.138 | 0.185 | 0.094 | 0.498 | 0.841 |
| TextRank | 0.370 | 0.135 | 0.176 | 0.088 | 0.487 | 0.841 |
| LexRank | 0.365 | 0.137 | 0.169 | 0.083 | 0.504 | 0.837 |
| BART | 0.265 | 0.081 | 0.147 | 0.017 | 0.169 | 0.854 |
| PEGASUS | 0.424 | 0.180 | 0.240 | 0.131 | 0.079* | 0.871 |
| PRIMERA | **0.445** | **0.197** | **0.248** | **0.147** | 0.475 | **0.874** |
| Qwen2.5-7B | 0.344 | 0.097 | 0.177 | 0.057 | 0.321 | 0.857 |
| Gemma 4 | 0.335 | 0.099 | 0.176 | 0.065 | 0.435 | 0.839 |
| Mistral-7B | 0.354 | 0.111 | 0.190 | 0.067 | 0.361 | 0.860 |
| GPT-5-mini | 0.348 | 0.103 | 0.179 | 0.066 | 0.402 | 0.852 |
| LSA (Gong & Liu) | 0.351 | 0.127 | 0.183 | 0.091 | 0.400 | 0.833 |
| LSA (Steinberger) | 0.376 | 0.137 | 0.188 | 0.103 | 0.452 | 0.842 |
| SBERT + K-Means | 0.359 | 0.125 | 0.183 | 0.095 | 0.443 | 0.840 |
| SBERT + Agglomerativo | 0.329 | 0.106 | 0.171 | 0.085 | 0.378 | 0.833 |
| LDA | 0.351 | 0.130 | 0.171 | 0.081 | 0.511† | 0.835 |

\* Valore deflazionato da due righe patologiche (sezione 5.3, nota METEOR); la media reale di
PEGASUS sulle righe non degeneri è ≈0.42, in linea con gli altri metodi abstractive forti.

† Il valore di LDA è gonfiato dalla lunghezza dei riassunti prodotti (in media 477 parole,
contro 216-264 degli altri metodi estrattivi non supervisionati): a parità di soglia di
lunghezza LDA cattura più contenuto e quindi più recall/METEOR "gratuiti". Per un confronto
equo tra i cinque metodi non supervisionati (Centroid+MMR, LSA, SBERT, LDA), guardare la F1
di ROUGE, non il recall o il METEOR.

### 6.2 G-Eval — valutazione LLM-as-judge (scala 1-5, senza riferimento)

| Metodo | Coherence | Consistency | Fluency | Relevance | Media |
|---|---|---|---|---|---|
| First-k (PSR) | 2.98 | 2.93 | 2.38 | 3.14 | 2.86 |
| First-k (NLTK) | 3.52 | 3.12 | 3.74 | 3.11 | 3.37 |
| Centroid+MMR (TF-IDF) | 2.56 | 2.63 | 2.15 | 3.63 | 2.74 |
| Centroid+MMR (BERT) | 2.53 | 2.61 | 2.16 | 3.67 | 2.74 |
| TextRank | 1.95 | 2.05 | 1.89 | 3.33 | 2.30 |
| LexRank | 1.81 | 1.78 | 1.77 | 3.12 | 2.12 |
| BART | 4.79 | 4.64 | 4.85 | 3.56 | 4.46 |
| PEGASUS | 4.28 | 3.48 | 4.39 | 4.04 | 4.05 |
| PRIMERA | 4.59 | 3.74 | 4.48 | 4.32 | 4.28 |
| Qwen2.5-7B | 4.61 | 4.24 | 4.80 | 4.49 | 4.54 |
| Gemma 4 | 4.94 | 4.34 | 4.99 | 4.83 | 4.77 |
| Mistral-7B | 4.82 | 4.49 | 4.90 | 4.68 | 4.72 |
| GPT-5-mini | **4.98** | **4.65** | **5.00** | **4.93** | **4.89** |
| LSA (Gong & Liu) | 2.34 | 2.54 | 2.07 | 3.11 | 2.52 |
| LSA (Steinberger) | 2.50 | 2.70 | 2.17 | 3.44 | 2.70 |
| SBERT + K-Means | 2.46 | 2.57 | 2.16 | 3.37 | 2.64 |
| SBERT + Agglomerativo | 2.26 | 2.34 | 2.11 | 2.95 | 2.42 |
| LDA | 2.37 | 2.36 | 2.06 | 3.54 | 2.58 |

I giudizi coprono il 98,9 % delle coppie (metodo, esempio): il resto è stato respinto dal
filtro di sicurezza dei contenuti del servizio cloud prima che il giudice potesse leggerlo
(sezione 9).

### 6.3 Il risultato chiave: due metriche, due classifiche opposte

Confrontando le due tabelle emerge il risultato più interessante del progetto: le metriche
**reference-based** e quella **reference-free** producono classifiche quasi opposte.

- Su **ROUGE/BERTScore** (ancorate al riassunto umano di riferimento), vincono **PRIMERA** e
  **PEGASUS** — i due modelli specializzati pre-addestrati specificamente su Multi-News.
- Su **G-Eval** (senza riferimento, giudicata da un LLM indipendente), vincono **GPT-5-mini**,
  **Gemma** e **Mistral** — LLM general-purpose usati zero-shot, senza alcun addestramento
  specifico sul task o sul dataset.

L'interpretazione più plausibile: PRIMERA e PEGASUS, essendo stati addestrati (o
fine-tuned) proprio sul train split di Multi-News, hanno imparato a **riprodurre lo stile
lessicale e strutturale** dei riassunti di newser.com — cosa che le metriche reference-based
premiano direttamente, per costruzione. Un valutatore LLM indipendente, che giudica invece la
qualità *intrinseca* del riassunto (coerenza, fedeltà ai fatti, fluidità, rilevanza) senza
guardare come un umano specifico abbia scelto di formulare la stessa informazione, trova
invece che i riassunti degli LLM general-purpose moderni sono percepiti come più fluenti e
completi. Le due famiglie di metriche, in altre parole, **non misurano la stessa cosa**: una
misura l'aderenza a un particolare stile di riferimento, l'altra la qualità percepita in
assoluto.

Questa lettura lascia aperte due obiezioni, entrambe affrontate nelle sezioni seguenti: che
parte delle differenze fra metodi sia dovuta alla **lunghezza** dei riassunti e non alla loro
qualità (sezione 7), e che il giudice LLM favorisca i metodi della **propria famiglia**
(sezione 8).

---

## 7. Il confondente della lunghezza: il confronto a lunghezza pari

### 7.1 Il problema

Le metriche reference-based non sono neutrali rispetto alla lunghezza. Il **recall** di ROUGE
premia chi include più parole del riferimento: un riassunto più lungo, a parità di qualità
della selezione, ne cattura di più «gratis». La **precisione** fa l'opposto. La F1 le combina,
ma non annulla l'effetto, perché le due componenti non sono simmetriche rispetto alla
lunghezza. Il problema è noto in letteratura (Sun et al., 2019): confrontare sistemi che
producono output di lunghezza diversa senza controllare la lunghezza rende le conclusioni
ambigue, e buona parte dei confronti pubblicati ne è affetta.

Nel benchmark il problema è concreto. Le medie per metodo vanno da 55 parole (BART) a 477
(LDA), ma il dato rilevante è quello **dentro il singolo cluster**: per lo stesso insieme di
articoli, il più lungo dei 18 riassunti è in mediana **8,5 volte** il più corto, e la
distanza fra i due è il doppio del riassunto di riferimento. La lunghezza media spiega da sola
gran parte del recall ROUGE fra metodi (correlazione 0,89) e assai meno della F1 (0,26): il
sospetto che alcune posizioni in classifica fossero «comprate» con la lunghezza era fondato.
Nessun metodo, inoltre, adatta la propria lunghezza al cluster: gli estrattivi a budget di
frasi producono la stessa quantità di testo qualunque sia la dimensione dell'input, e la
correlazione fra lunghezza generata e lunghezza del riferimento non supera 0,51.

### 7.2 Quale lunghezza «giusta»?

La prima idea — un tetto fisso uguale per tutti — è sbagliata per due ragioni. Il riferimento
umano non ha lunghezza costante: va da circa 130 a circa 290 parole fra il decimo e il
novantesimo percentile e cresce col numero di articoli del cluster, quindi un tetto unico
sarebbe troppo largo per i cluster piccoli e troppo stretto per quelli grandi. E il
troncamento agisce solo verso il basso: un riassunto da 55 parole non si allunga troncandolo,
per cui il tetto da solo riduce la dispersione dentro il cluster da 8,5× a circa 4×, non
oltre.

Si è allora verificato se la lunghezza del riferimento si lasci **prevedere dal solo input**
(numero di articoli, lunghezza delle fonti, combinazioni e regressioni). La risposta è
negativa: il miglior predittore colloca il riferimento entro ±25 % nel 58 % dei cluster
contro il 54 % di un valore costante — un guadagno di quattro punti — e l'intuizione più
naturale, «proporzionale alla lunghezza delle fonti», è la peggiore, perché il rapporto di
compressione varia di cinque volte fra un cluster e l'altro. La lunghezza del riassunto umano
è in larga parte una scelta di chi lo scrive, non una funzione misurabile del cluster.

La scelta è quindi caduta sul protocollo **length-matched** o **oracle-length** (Sun et al.,
2019): a ogni metodo si assegna, cluster per cluster, **la lunghezza del riassunto di
riferimento** come budget di parole. Si usa un'informazione «gold» — la sola lunghezza, mai
il contenuto — e questo va dichiarato ogni volta che i numeri compaiono: rispondono alla
domanda «quanto è buona la selezione dei contenuti a parità di lunghezza», non a «cosa
produrrebbe il metodo in condizioni reali». Non sostituiscono i risultati della sezione 6: ne
sono il controllo.

### 7.3 Come si applica

Servono due interventi distinti, perché il vincolo va imposto in entrambe le direzioni:

- un **tetto** per tutti e 18 i metodi: ogni riassunto viene troncato a 1,25 volte la
  lunghezza del riferimento del suo cluster;
- una **rigenerazione a budget** per i metodi che producono meno del riferimento. Per gli
  undici estrattivi significa sostituire il budget in *frasi* («le prime 11») con un budget in
  *parole*, mantenendo intatto il criterio di ordinamento di ciascun metodo: cambia dove ci
  si ferma, non cosa si preferisce. Per gli LLM significa mettere la lunghezza richiesta nel
  prompt; un LLM la rispetta solo approssimativamente, e ogni modello sbaglia in una
  direzione propria (chi accorcia, chi eccede), per cui l'istruzione è stata calibrata modello
  per modello.

Tre metodi ricevono il solo tetto, per scelta: BART, PEGASUS e PRIMERA. Allungare un modello
abstractive richiede di imporgli una lunghezza minima di generazione, forzando la ricerca a
continuare oltre il punto in cui il modello si fermerebbe da sé — ed è proprio il meccanismo
che produce i loop di ripetizione già osservati per PEGASUS. PEGASUS e PRIMERA sono comunque
già vicini alla lunghezza del riferimento; BART, che dovrebbe quadruplicare il proprio
output, resta l'**eccezione dichiarata** del confronto.

Il risultato del contenimento: i 15 metodi rigenerati cadono nella banda [0,8; 1,25] × il
riferimento nel 71–100 % dei cluster, e la dispersione dentro il cluster passa da 8,5× a 4,7×
sui 18 metodi — ma quel 4,7 è quasi interamente BART; **senza BART il rapporto passa da 3,8×
a 1,6×, e sui soli 15 rigenerati da 3,7× a 1,4×**.

### 7.4 Che cosa cambia nei risultati

| Metodo | Parole prima → dopo | ROUGE-1 recall prima → dopo | ROUGE-1 F1 prima → dopo | G-Eval prima → dopo |
|---|---|---|---|---|
| PRIMERA | 211 → 201 | 0.441 → 0.433 | **0.445 → 0.446** | 4.28 → 4.22 |
| PEGASUS | 178 → 175 | 0.390 → 0.387 | 0.424 → 0.424 | 4.05 → 4.03 |
| Centroid+MMR (TF-IDF) | 362 → 215 | 0.465 → 0.365 | 0.383 → 0.378 | 2.74 → 2.79 |
| LSA (Steinberger) | 257 → 215 | 0.419 → 0.366 | 0.376 → 0.369 | 2.70 → 2.61 |
| First-k (PSR) | 217 → 215 | 0.347 → 0.354 | 0.355 → 0.368 | 2.86 → 2.86 |
| First-k (NLTK) | 219 → 215 | 0.348 → 0.352 | 0.356 → 0.368 | 3.37 → 3.38 |
| Mistral-7B | 161 → 233 | 0.325 → 0.367 | 0.354 → 0.365 | 4.72 → 4.59 |
| TextRank | 368 → 215 | 0.448 → 0.343 | 0.370 → 0.359 | 2.30 → 2.57 |
| LexRank | 450 → 216 | 0.482 → 0.338 | 0.365 → 0.352 | 2.12 → 2.46 |
| Qwen2.5-7B | 140 → 216 | 0.309 → 0.347 | 0.344 → 0.351 | 4.54 → 4.68 |
| GPT-5-mini | 241 → 209 | 0.380 → 0.363 | 0.348 → 0.351 | 4.89 → 4.89 |
| LDA | 477 → 230 | **0.499 → 0.348** | 0.351 → **0.338** | 2.58 → 2.54 |
| Gemma 4 | 292 → 219 | 0.390 → 0.337 | 0.335 → 0.328 | 4.77 → 4.83 |
| BART (solo tetto) | 55 → 55 | 0.181 → 0.181 | 0.265 → 0.265 | 4.46 → 4.46 |

(Le righe omesse — Centroid+MMR BERT, LSA, le due varianti SBERT — si muovono di meno di
0,01 in F1.)

Sulle metriche reference-based la graduatoria si riassesta in quattro modi:

- **LDA perde tutto il suo vantaggio**: il recall crolla da 0,499 a 0,348 e in F1 scende
  all'ultimo posto fra gli estrattivi. Il suo primato nella sezione 6.1 era interamente
  lunghezza.
- **Le baseline posizionali salgono** dal 9°–10° al 5°–6° posto in F1, sopra LSA, SBERT,
  TextRank e LexRank: a parità di parole, prendere le prime frasi di ogni articolo regge il
  confronto con metodi di selezione molto più elaborati. È il *lead bias* del paper originale
  (sezione 4.1), che il confondente della lunghezza aveva nascosto.
- **I metodi corti guadagnano** — Qwen, Mistral, le First-k — perché la rigenerazione dà loro
  lo spazio che non usavano: il confondente tagliava in entrambe le direzioni.
- **PRIMERA e PEGASUS restano in testa**, appena toccati dal tetto: il loro vantaggio non era
  di lunghezza.

Il G-Eval racconta una storia diversa, ed è il risultato più istruttivo. Dove il
riallineamento ha rimescolato il ROUGE, **le prime undici posizioni del G-Eval restano
identiche** e il massimo spostamento è di 0,34 punti su 5. La lunghezza non era il suo
confondente — coerentemente con ciò che misura: qualità della prosa rispetto alla fonte, non
sovrapposizione con il riferimento. Tre osservazioni:

- i due estrattivi più lunghi **guadagnano tagliando** (LexRank +0,34, TextRank +0,27): per
  il giudice un riassunto estrattivo da 450 parole è un testo *meno* coerente, non più
  completo;
- la **pertinenza** è l'unica dimensione sensibile alla lunghezza — chi ha dovuto tagliare di
  più la perde (LDA −0,50) mentre coerenza, consistenza e fluenza salgono. È la controparte
  del recall: i due effetti che la F1 di ROUGE somma in un numero solo, qui separati;
- fra gli LLM allungati, Qwen aggiunge contenuto pertinente e supera Mistral, che perde tutto
  sulla *consistenza*: costretto a scrivere di più, aggiunge cose che nella fonte non ci sono.

La conclusione della sezione 6.3 regge, e ne esce rafforzata: la divergenza fra metriche
reference-based e reference-free non è un artefatto della lunghezza.

---

## 8. Quanto ci si può fidare del giudice

### 8.1 Il dubbio

La validità di G-Eval come metrica dipende dall'imparzialità del giudice. La letteratura sul
paradigma *LLM-as-a-judge* documenta un **self-preference bias**: gli LLM tendono ad
assegnare punteggi più alti ai testi che riconoscono come simili al proprio stile (Zheng et
al., 2023; Panickssery et al., 2024), e la somiglianza è massima per i modelli della stessa
famiglia. In questo benchmark il giudice (GPT-5.4-mini) e uno dei metodi valutati (GPT-5-mini)
sono entrambi modelli OpenAI della stessa generazione. E i numeri sollevano il dubbio da soli:
fra i quattro LLM, l'ordine G-Eval è esattamente **invertito** rispetto a quello di BERTScore
e ROUGE, e GPT-5-mini — terzo su quattro sulle metriche ancorate al riferimento — è primo
assoluto sul G-Eval.

Due spiegazioni si adattano ugualmente bene a questo quadro: (a) un bias di famiglia del
giudice; (b) nessun bias — G-Eval misura una cosa diversa, e GPT-5-mini scrive davvero in
prosa migliore pur aderendo meno al riferimento. **Un solo giudice non può separarle.**

### 8.2 Il disegno sperimentale

Gli stessi riassunti, già generati e già giudicati, sono stati **rigiudicati da un secondo
giudice di lignaggio completamente diverso** (DeepSeek-V3.2), con istruzioni identiche, su un
campione casuale di circa mille cluster. È un confronto **appaiato sullo stesso testo**: la
sola variabile che cambia è il giudice, e qualunque differenza sistematica è attribuibile a
lui.

Il campione comprende i quattro LLM e **tre metodi di controllo non-LLM** (PRIMERA, First-k,
LexRank), presi a quote diverse della graduatoria. I controlli sono ciò che rende il disegno
conclusivo: permettono di distinguere un effetto generico «al giudice piace la prosa degli
LLM» — reale, ma condiviso e non distorsivo per il confronto — da un effetto **specifico
della famiglia** OpenAI. Il test è direzionale: se il giudice di famiglia gonfiasse
GPT-5-mini, il giudice estraneo gli toglierebbe qualcosa che agli altri LLM non toglie, e il
contrasto «variazione di GPT-5-mini meno variazione media degli altri LLM» risulterebbe
**negativo**.

### 8.3 L'esito

- I due giudici producono **lo stesso ordinamento** dei sette metodi: correlazione di rango
  pari a 1, nessuna inversione.
- Il contrasto che misurerebbe il bias vale **+0,05** (intervallo di confidenza al 95 % da
  +0,02 a +0,08): positivo, cioè di **segno opposto** a quello previsto dall'ipotesi. Con un
  giudice estraneo GPT-5-mini guadagna sugli altri LLM, non perde. L'ipotesi del bias di
  famiglia è respinta nella sua stessa direzione, non semplicemente «non confermata».
- I due giudici **non sono intercambiabili**: DeepSeek è sistematicamente più severo con i
  metodi non-LLM (circa −0,3 punti), soprattutto su coerenza e pertinenza, e allarga quindi il
  distacco fra prosa LLM e prosa estrattiva. La preferenza per la prosa degli LLM è una
  proprietà del paradigma, condivisa da entrambi i giudici, e va tenuta presente leggendo
  qualunque valutazione LLM-as-a-judge — ma non è una distorsione di famiglia.

Una precisazione statistica utile in sede di presentazione: con circa mille cluster appaiati
anche cinque centesimi di punto risultano statisticamente distinguibili da zero, quindi il
risultato *non* va descritto come «compatibile con zero». Ciò che lo rende conclusivo è il
**segno**, non la significatività.

---

## 9. Limiti e avvertenze di lettura

Alcuni aspetti da tenere presenti quando si presentano questi risultati, per evitare
conclusioni affrettate:

- **Data leakage per PEGASUS e PRIMERA**: entrambi i modelli sono stati pre-addestrati (anche)
  sul train split di Multi-News. Questo non invalida il confronto sul test split (che è
  comunque un dato mai visto durante l'addestramento), ma va ricordato quando si spiega perché
  questi due modelli dominano le metriche reference-based: parte del loro vantaggio deriva
  dall'aver imparato lo stile specifico di questo dataset, non da capacità di summarization
  generiche superiori. Il confronto a lunghezza pari (sezione 7) mostra che quel vantaggio
  non è comunque di lunghezza.
- **ROUGE non standard**: come indicato nella sezione 5.1, i valori ROUGE di questo progetto
  usano una variante a n-grammi unici, confrontabile solo *internamente* tra i 18 metodi, non
  con valori ROUGE pubblicati altrove (incluso il paper Multi-News originale).
- **Outlier METEOR**: la formula usata non è limitata inferiormente per input degeneri. La
  media METEOR di PEGASUS è artificialmente bassa a causa di due righe patologiche su 5.610
  (sezione 5.3); nel confronto a lunghezza pari lo stesso accade a LDA e LexRank su una riga
  la cui sorgente è un elenco di nomi. In nessun caso i valori sono stati corretti: sono
  segnalati.
- **Copertura del G-Eval e filtro di sicurezza del servizio cloud**: il filtro di contenuti
  di Azure respinge alcuni esempi (cronaca violenta o d'odio) prima che il giudice li legga,
  e lo fa in modo non stabile nel tempo. Le prime sessioni avevano lasciato una copertura fra
  il 93,5 % e il 96,7 % a seconda del metodo; un ritentativo successivo dei soli giudizi
  respinti l'ha portata al 96,8–99,0 %, senza toccare i giudizi esistenti e senza spostare
  nessuna media di più di 0,01 punti né l'ordinamento. Le righe che il filtro blocca non sono
  quindi sistematicamente diverse dalle altre; resta il fatto che i metodi non sono giudicati
  esattamente sullo stesso insieme di esempi.
- **Riassunti LDA più lunghi**: la lunghezza sistematicamente maggiore dei riassunti LDA
  gonfia recall e METEOR nella sezione 6.1. La questione è chiusa dal confronto a lunghezza
  pari (sezione 7): a parità di lunghezza LDA è l'ultimo degli estrattivi.
- **La lunghezza «oracolare» è un'informazione gold**: il confronto della sezione 7 usa la
  lunghezza del riassunto umano come budget. È il protocollo standard per neutralizzare la
  lunghezza, ma i suoi numeri descrivono la qualità della selezione a parità di lunghezza, non
  ciò che un metodo produrrebbe da solo; i risultati di testa restano quelli della sezione 6.
- **Un giudice non è un umano**: la validazione della sezione 8 esclude un bias di famiglia,
  non stabilisce che il G-Eval coincida con il giudizio umano; la preferenza dei giudici LLM
  per la prosa degli LLM è reale e condivisa. Le due famiglie di metriche vanno lette insieme,
  non l'una al posto dell'altra.

---

## 10. Bibliografia

**Dataset e modello di origine**
- A. Fabbri, I. Li, T. She, S. Li, D. Radev, *"Multi-News: A Large-Scale Multi-Document
  Summarization Dataset and Abstractive Hierarchical Model"*, ACL 2019 (arXiv:1906.01749).

**Metodi estrattivi classici**
- A. Nenkova, K. McKeown, *"Automatic Summarization"*, Foundations and Trends in Information
  Retrieval, 2011.
- R. Mihalcea, P. Tarau, *"TextRank: Bringing Order into Text"*, EMNLP 2004.
- G. Erkan, D. Radev, *"LexRank: Graph-based Lexical Centrality as Salience in Text
  Summarization"*, JAIR 2004.
- S. Brin, L. Page, *"The Anatomy of a Large-Scale Hypertextual Web Search Engine"*, WWW 1998.
- D. Radev, H. Jing, M. Styś, D. Tam, *"Centroid-based summarization of multiple documents"*,
  Information Processing & Management, 2004.
- J. Carbonell, J. Goldstein, *"The Use of MMR, Diversity-Based Reranking for Reordering
  Documents and Producing Summaries"*, SIGIR 1998.
- Y. Gong, X. Liu, *"Generic Text Summarization Using Relevance Measure and Latent Semantic
  Analysis"*, SIGIR 2001.
- J. Steinberger, K. Ježek, *"Using Latent Semantic Analysis in Text Summarization and
  Summary Evaluation"*, Proc. ISIM 2004.
- N. Reimers, I. Gurevych, *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"*,
  EMNLP-IJCNLP 2019.
- J. MacQueen, *"Some Methods for Classification and Analysis of Multivariate Observations"*,
  Berkeley Symposium on Mathematical Statistics, 1967.
- D. Blei, A. Ng, M. Jordan, *"Latent Dirichlet Allocation"*, JMLR 2003.

**Modelli abstractive e LLM**
- A. Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017.
- A. See, P. Liu, C. Manning, *"Get To The Point: Summarization with Pointer-Generator
  Networks"*, ACL 2017.
- M. Lewis et al., *"BART: Denoising Sequence-to-Sequence Pre-training for Natural Language
  Generation, Translation, and Comprehension"*, ACL 2020.
- J. Zhang, Y. Zhao, M. Saleh, P. Liu, *"PEGASUS: Pre-training with Extracted Gap-sentences
  for Abstractive Summarization"*, ICML 2020.
- W. Xiao, I. Beltagy, G. Carenini, A. Cohan, *"PRIMERA: Pyramid-based Masked Sentence
  Pre-training for Multi-document Summarization"*, ACL 2022.
- I. Beltagy, M. Peters, A. Cohan, *"Longformer: The Long-Document Transformer"*, arXiv 2020.
- A. Yang et al. (Qwen Team, Alibaba), *"Qwen2.5 Technical Report"*, arXiv 2024.
- Gemma Team, Google DeepMind, *"Gemma: Open Models Based on Gemini Research and
  Technology"*, 2024.
- A. Jiang et al. (Mistral AI), *"Mistral 7B"*, arXiv 2023.
- L. Ouyang et al., *"Training language models to follow instructions with human feedback"*,
  NeurIPS 2022.

**Metriche di valutazione**
- C.-Y. Lin, *"ROUGE: A Package for Automatic Evaluation of Summaries"*, ACL Workshop "Text
  Summarization Branches Out", 2004.
- K. Papineni, S. Roukos, T. Ward, W.-J. Zhu, *"BLEU: a Method for Automatic Evaluation of
  Machine Translation"*, ACL 2002.
- S. Banerjee, A. Lavie, *"METEOR: An Automatic Metric for MT Evaluation with Improved
  Correlation with Human Judgments"*, ACL Workshop on Intrinsic and Extrinsic Evaluation
  Measures for MT, 2005.
- T. Zhang, V. Kishore, F. Wu, K. Weinberger, Y. Artzi, *"BERTScore: Evaluating Text
  Generation with BERT"*, ICLR 2020.
- Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, C. Zhu, *"G-Eval: NLG Evaluation using GPT-4 with
  Better Human Alignment"*, EMNLP 2023.

**Lunghezza dei riassunti e affidabilità dei giudici LLM**
- S. Sun, O. Shapira, I. Dagan, A. Nenkova, *"How to Compare Summarizers without Target
  Length? Pitfalls, Solutions and Re-Examination of the Neural Summarization Literature"*,
  Workshop on Methods for Optimizing and Evaluating Neural Language Generation (NeuralGen),
  NAACL 2019 — il problema della lunghezza nei confronti fra sistemi e il protocollo a
  lunghezza pari (sezione 7).
- L. Zheng et al., *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"*, NeurIPS 2023
  (Datasets and Benchmarks) — i bias sistematici dei giudici LLM, fra cui la preferenza per
  le proprie generazioni (sezione 8).
- A. Panickssery, S. R. Bowman, S. Feng, *"LLM Evaluators Recognize and Favor Their Own
  Generations"*, NeurIPS 2024 — il self-preference bias e il suo legame con la capacità del
  giudice di riconoscere il proprio stile (sezione 8).
