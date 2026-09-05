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
7. [Limiti e avvertenze di lettura](#7-limiti-e-avvertenze-di-lettura)
8. [Bibliografia](#8-bibliografia)

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
modello) e più recente di ciascuno di essi.

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
| First-k (PSR) | 2.98 | 2.94 | 2.39 | 3.14 | 2.86 |
| First-k (NLTK) | 3.52 | 3.13 | 3.74 | 3.11 | 3.37 |
| Centroid+MMR (TF-IDF) | 2.56 | 2.63 | 2.15 | 3.63 | 2.74 |
| Centroid+MMR (BERT) | 2.53 | 2.61 | 2.16 | 3.67 | 2.74 |
| TextRank | 1.95 | 2.05 | 1.88 | 3.33 | 2.31 |
| LexRank | 1.81 | 1.78 | 1.77 | 3.12 | 2.12 |
| BART | 4.79 | 4.65 | 4.85 | 3.56 | 4.46 |
| PEGASUS | 4.27 | 3.46 | 4.38 | 4.03 | 4.04 |
| PRIMERA | 4.59 | 3.74 | 4.48 | 4.32 | 4.28 |
| Qwen2.5-7B | 4.61 | 4.25 | 4.80 | 4.49 | 4.54 |
| Gemma 4 | 4.94 | 4.34 | 4.99 | 4.83 | 4.78 |
| Mistral-7B | 4.81 | 4.49 | 4.90 | 4.68 | 4.72 |
| GPT-5-mini | **4.98** | **4.66** | **5.00** | **4.93** | **4.89** |
| LSA (Gong & Liu) | 2.34 | 2.55 | 2.07 | 3.11 | 2.52 |
| LSA (Steinberger) | 2.50 | 2.70 | 2.17 | 3.44 | 2.70 |
| SBERT + K-Means | 2.46 | 2.57 | 2.16 | 3.37 | 2.64 |
| SBERT + Agglomerativo | 2.26 | 2.34 | 2.11 | 2.95 | 2.41 |
| LDA | 2.37 | 2.36 | 2.06 | 3.54 | 2.58 |

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

---

## 7. Limiti e avvertenze di lettura

Alcuni aspetti da tenere presenti quando si presentano questi risultati, per evitare
conclusioni affrettate:

- **Data leakage per PEGASUS e PRIMERA**: entrambi i modelli sono stati pre-addestrati (anche)
  sul train split di Multi-News. Questo non invalida il confronto sul test split (che è
  comunque un dato mai visto durante l'addestramento), ma va ricordato quando si spiega perché
  questi due modelli dominano le metriche reference-based: parte del loro vantaggio deriva
  dall'aver imparato lo stile specifico di questo dataset, non da capacità di summarization
  generiche superiori.
- **ROUGE non standard**: come indicato nella sezione 5.1, i valori ROUGE di questo progetto
  usano una variante a n-grammi unici, confrontabile solo *internamente* tra i 18 metodi, non
  con valori ROUGE pubblicati altrove (incluso il paper Multi-News originale).
- **Outlier METEOR**: la media METEOR di PEGASUS è artificialmente bassa a causa di due righe
  patologiche su 5.610 (sezione 5.3); non riflette la sua reale qualità media.
- **Instabilità del filtro di contenuto Azure tra le sessioni di valutazione G-Eval**: le
  giudicazioni G-Eval sono state raccolte in due sessioni separate (a distanza di circa due
  settimane) per due sottoinsiemi di metodi; il filtro di sicurezza dei contenuti di Azure ha
  rifiutato insiemi di righe leggermente diversi nelle due sessioni. Questo produce una
  copertura (`n_geval`) diversa tra metodi (93,5%-96,7%), ma l'impatto sulle medie riportate è
  trascurabile (nessuno spostamento superiore a 0,01 punti, confrontando le sole righe
  giudicate per tutti e 18 i metodi).
- **Riassunti LDA più lunghi**: come indicato nella sezione 6.1, la lunghezza sistematicamente
  maggiore dei riassunti LDA gonfia recall e METEOR in modo non comparabile agli altri
  metodi; la F1 di ROUGE resta la metrica corretta per confrontarlo con gli altri quattro
  metodi non supervisionati.

---

## 8. Bibliografia

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
