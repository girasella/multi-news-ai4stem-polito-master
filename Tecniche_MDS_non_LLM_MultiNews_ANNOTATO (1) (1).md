**Tecniche di Multi-Document Summarization non basate su LLM**

Documento di valutazione per il progetto Multi-News (MDS)

**VERSIONE ANNOTATA**

_Confronto con la lezione "Summarization" (Prof. L. Cagliero, Deep Natural Language Processing, PoliTO)_

Ogni tecnica riporta un'annotazione che indica se è stata trattata a lezione.

La Sezione 4 raccoglie le tecniche presenti nella lezione ma assenti dalla prima versione del documento.

# Indice

[Indice 2](#_Toc234769319)

[1\. Obiettivo del documento e legenda delle annotazioni 3](#_Toc234769320)

[2\. Contesto del progetto e criteri di valutazione 3](#_Toc234769321)

[2.1 Il dataset Multi-News 3](#_Toc234769322)

[2.2 Obiettivo sperimentale 3](#_Toc234769323)

[2.3 Criteri di valutazione delle tecniche 3](#_Toc234769324)

[2.4 Quadro concettuale ripreso dalla lezione (nuovo) 3](#_Toc234769325)

[3\. Metodologie già presenti nel documento (annotate) 5](#_Toc234769326)

[3.1 Baseline posizionale (Lead / First-k) 5](#_Toc234769327)

[3.2 Metodi statistici a frequenza (Luhn, Edmundson, SumBasic, KL-Sum) 5](#_Toc234769328)

[3.3 Metodi graph-based (LexRank, TextRank) 6](#_Toc234769329)

[3.4 Centroid-based (MEAD) + Maximal Marginal Relevance (MMR) 7](#_Toc234769330)

[3.5 LSA (Latent Semantic Analysis) 8](#_Toc234769331)

[3.6 Topic modeling e clustering tematico (LDA + Orange) 9](#_Toc234769332)

[3.7 Clustering su sentence embedding (BERT / SBERT) 10](#_Toc234769333)

[3.8 SummPip (sentence graph + spectral clustering + compressione) 11](#_Toc234769334)

[3.9 Ottimizzazione combinatoria (occams) 11](#_Toc234769335)

[3.10 pyAutoSummarizer (libreria aggregatrice) 12](#_Toc234769336)

[4\. Tecniche presenti nella lezione e assenti dal documento (da aggiungere) 14](#_Toc234769337)

[4.1 Itemset-based summarization (MWI-Sum) 14](#_Toc234769338)

[4.2 Optimization-based: funzioni submodulari (Lin & Bilmes) 15](#_Toc234769339)

[4.3 Neural summarization supervisionata (BERTSum, MatchSum) 16](#_Toc234769340)

[4.4 COSUM (clustering + ottimizzazione) 17](#_Toc234769341)

[5\. Tabella di corrispondenza lezione ↔ documento 18](#_Toc234769342)

[6\. Sintesi del confronto e shortlist aggiornata 19](#_Toc234769343)

[6.1 Cosa è emerso dal confronto 19](#_Toc234769344)

[6.2 Shortlist aggiornata 19](#_Toc234769345)

[6.3 Prossimi passi 20](#_Toc234769346)

[7\. Approfondimento sulle librerie: sumy e pyAutoSummarizer 21](#_Toc234769347)

[7.1 sumy 21](#_Toc234769348)

[7.2 pyAutoSummarizer 21](#_Toc234769349)

[7.3 Confronto sintetico 23](#_Toc234769350)

[7.4 Raccomandazione operativa per il progetto 23](#_Toc234769351)

[7.5 Avvertenze metodologiche 23](#_Toc234769352)

[8\. Riferimenti bibliografici 25](#_Toc234769353)

# 1\. Obiettivo del documento e legenda delle annotazioni

Questo documento raccoglie i risultati di una ricerca sulle metodologie di Multi-Document Summarization (MDS) non basate su LLM, applicabili al dataset Multi-News. Lo scopo è fornire al team le informazioni necessarie - descrizione tecnica, modalità di implementazione, libreria di riferimento, vantaggi e svantaggi - per decidere collegialmente quali tecniche includere nella pipeline sperimentale di confronto con l'output del LLM locale (LM Studio), tramite metriche ROUGE.

Questa versione è stata annotata a seguito del confronto sistematico con le slide della lezione del Master dedicata alla summarization tradizionale. Ogni tecnica è preceduta da un riquadro che ne dichiara la copertura nel programma del corso, con riferimento alle slide. La Sezione 4 introduce le tecniche trattate a lezione e assenti dalla prima versione del documento; la Sezione 5 fornisce la tabella di corrispondenza completa lezione ↔ documento; la Sezione 7 approfondisce le due librerie generaliste (sumy e pyAutoSummarizer) e il loro ruolo rispettivo nel progetto.

**Legenda dei riquadri di annotazione**

- SÌ (bordo verde) - la tecnica è trattata esplicitamente nella lezione: massima aderenza al programma, facilmente difendibile in relazione.
- PARZIALE (bordo arancio) - la famiglia o i concetti sottostanti sono trattati, ma non lo specifico algoritmo/libreria.
- NO / SOLO NELLA LEZIONE (bordo rosso) - tecnica assente da uno dei due documenti: richiede una decisione esplicita del gruppo.

# 2\. Contesto del progetto e criteri di valutazione

## 2.1 Il dataset Multi-News

Multi-News è composto da 56.216 esempi (train/val/test), ciascuno formato da 2 a 10 articoli sorgente relativi allo stesso evento e da un summary umano di riferimento. Nel formato originale gli articoli sono separati dal token "|||||" e i riassunti iniziano tipicamente con il prefisso "- ". Dall'EDA già condotta è emerso che i summary di riferimento sono parzialmente abstractive (novel n-grams): questo pone un limite strutturale alla qualità massima ottenibile dai soli metodi estrattivi, aspetto utile da richiamare nella discussione dei risultati.

## 2.2 Obiettivo sperimentale

Confrontare i riassunti generati dalle tecniche non-LLM con i summary di riferimento tramite metriche ROUGE, e mettere questi risultati a confronto con l'output dell'LLM locale (LM Studio) sugli stessi esempi.

## 2.3 Criteri di valutazione delle tecniche

- Aderenza al problema multi-documento: la tecnica gestisce nativamente la ridondanza tra fonti o richiede una semplice concatenazione?
- Copertura nel programma del corso (criterio aggiunto in questa versione): la tecnica è stata trattata a lezione?
- Complessità implementativa: libreria pronta all'uso oppure sviluppo custom?
- Maturità della libreria: diffusione, documentazione, manutenzione.
- Qualità attesa: risultati riportati in letteratura, in particolare su Multi-News.
- Costo computazionale: tempo di esecuzione e risorse (CPU vs GPU, modelli da scaricare).

# 3\. Metodologie già presenti nel documento (annotate)

Le dieci tecniche della prima versione del documento, ciascuna preceduta dal riquadro di annotazione che ne dichiara la copertura nella lezione.

## 3.1 Baseline posizionale (Lead / First-k)

**ANNOTAZIONE - Copertura nella lezione: PARZIALE (concetto, non come baseline)**

La lezione non presenta Lead/First-k come baseline autonoma, ma introduce la posizione della frase come feature classica di scoring (slide 33, "Sentence position score", con la formula SPᵢ = 1 − (i−1)/N). Il razionale è quindi coperto, il metodo come baseline no.

Conseguenza per il gruppo: la baseline resta valida e va mantenuta (è la baseline del paper Multi-News), ma nella relazione conviene citarla collegandola alla feature "sentence position" vista a lezione, mostrando che si tratta della sua forma più semplice.

**Descrizione**

Il metodo più semplice concettualmente: per ogni articolo del cluster si estraggono le prime k frasi (tipicamente 1-3) e si concatenano a formare il riassunto. Si basa sul cosiddetto "lead bias" tipico della scrittura giornalistica, in cui le informazioni più salienti tendono a comparire in apertura dell'articolo (piramide rovesciata).

**Come implementarla**

Nessuna libreria di summarization è necessaria: è sufficiente uno script Python che tokenizza il testo in frasi (es. nltk.sent_tokenize o spaCy) ed estrae le prime k frasi di ciascun articolo del cluster, concatenandole nell'ordine desiderato.

**Libreria / Framework:** nltk o spaCy (solo per la tokenizzazione in frasi)

**Vantaggi**

- Estremamente semplice e veloce da implementare, nessuna dipendenza pesante e nessun training.
- Nel paper originale di Multi-News (Fabbri et al., 2019) la baseline First-3 supera empiricamente, in termini di ROUGE, metodi più sofisticati come LexRank, TextRank e MMR, a causa del lead bias delle notizie.
- Termine di paragone imprescindibile: qualunque tecnica più complessa dovrebbe giustificare la propria adozione superando questa baseline.

**Svantaggi / limiti**

- Nessuna reale comprensione del contenuto: ignora ridondanza tra fonti e rilevanza semantica.
- Funziona bene solo su testi con struttura giornalistica marcata.
- Va etichettata chiaramente come baseline di controllo, non come tecnica sperimentale a sé stante.

## 3.2 Metodi statistici a frequenza (Luhn, Edmundson, SumBasic, KL-Sum)

**ANNOTAZIONE - Copertura nella lezione: PARZIALE (famiglia sì, algoritmi specifici no)**

La lezione colloca esplicitamente la famiglia "Statistical" nella tassonomia dei metodi pre-LLM (slide 30, suddivisa in Heuristic / Optimization / Graph) e dedica due slide alle feature classiche di scoring - frequenza di parole, keyword, TF-IDF, title word, sentence position (slide 32-33) - che sono esattamente i mattoni su cui si basano Luhn ed Edmundson.

Tuttavia i quattro algoritmi specifici (Luhn, Edmundson, SumBasic, KL-Sum) non vengono mai nominati né descritti nelle slide.

Conseguenza per il gruppo: si possono mantenere come baseline storiche a costo quasi nullo (sono già tutti in sumy), ma nella relazione vanno presentati come istanze concrete della categoria "Statistical / Heuristic" della tassonomia vista a lezione, e non come metodi trattati dal corso.

**Descrizione**

Famiglia storica di metodi che assegnano un punteggio di importanza alle frasi in base alla frequenza delle parole significative nel testo (esclusi gli stopword):

**Come implementarla**

- Luhn (1958): individua le parole più frequenti, seleziona le "top words", e assegna un punteggio alle frasi in base a quante top words contengono.
- Edmundson: estende Luhn aggiungendo euristiche di posizione e "cue words".
- SumBasic: nato specificamente per il multi-documento; calcola la distribuzione di frequenza sull'intero cluster e aggiorna iterativamente i pesi dopo ogni frase selezionata, riducendo la ridondanza.
- KL-Sum: aggiunge frasi in modo greedy finché diminuisce la divergenza di Kullback-Leibler tra distribuzione del riassunto e del documento sorgente.
- Implementazione pratica: concatenare gli articoli del cluster in un unico pseudo-documento e applicare i summarizer di sumy, che espone un'API unificata per tutti e quattro.

**Libreria / Framework:** sumy (pip install sumy) - richiede nltk per la tokenizzazione

**Vantaggi**

- SumBasic è nativamente pensato per il multi-documento (a differenza degli altri tre, nati per il single-document).
- Implementazione già pronta in sumy, confrontabile in poche righe per tutti e quattro i metodi.
- Molto veloci: nessun training, nessuna GPU.
- Ben documentati come baseline storiche, utili per una relazione comparativa completa.

**Svantaggi / limiti**

- Luhn ed Edmundson non gestiscono nativamente più fonti: sulla semplice concatenazione non affrontano esplicitamente la ridondanza inter-documento.
- Punteggio basato solo su frequenza superficiale, nessuna informazione semantica.
- Edmundson richiede tipicamente un training supervisionato dei pesi delle euristiche; con parametri di default l'efficacia reale si riduce.

## 3.3 Metodi graph-based (LexRank, TextRank)

**ANNOTAZIONE - Copertura nella lezione: SÌ - trattata in modo approfondito**

È la famiglia più ampiamente trattata a lezione (slide 38-53). TextRank è presentato con i suoi 4 step (costruzione dei vertici, definizione delle relazioni/archi, iterazione dell'algoritmo di ranking fino a convergenza, ordinamento dei vertici per score). LexRank è sviluppato passo-passo: matrice di similarità coseno tra vettori TF-IDF, pruning degli archi sotto soglia, generazione del grafo di similarità (con esempi a soglia 0.1 / 0.2 / 0.3), calcolo della degree centrality e infine applicazione di PageRank, con anche la variante pesata con damping factor.

Pro/contro dalla lezione: alta efficacia ed easy-to-use; contro: alta complessità computazionale e metodo non incrementale.

Spunto aggiuntivo dalla lezione: la slide 50 segnala che esistono varianti di LexRank che usano BM25 o similarità BERT al posto del coseno TF-IDF - un'estensione a basso costo che il gruppo potrebbe testare per differenziarsi dalla baseline standard.

**Descrizione**

Rappresentano le frasi come nodi di un grafo, con archi pesati in base alla similarità (tipicamente cosine similarity su vettori TF-IDF), e applicano un algoritmo di tipo PageRank per calcolare un punteggio di centralità per ciascuna frase.

**Come implementarla**

- LexRank: usa la eigenvector centrality del grafo di similarità; nato esplicitamente per applicazioni multi-documento.
- TextRank: variante che utilizza un modello "random-surfer" fino a convergenza; nato per il single-document ma usato anche in contesti multi-doc.
- Implementazione pratica: concatenare gli articoli, costruire il grafo di similarità tra tutte le frasi ed estrarre quelle con punteggio più alto.
- Estensione suggerita dalla lezione: sostituire la similarità coseno TF-IDF con BM25 o con similarità su embedding BERT nella costruzione del grafo.

**Libreria / Framework:** sumy (LexRankSummarizer, TextRankSummarizer); in alternativa il pacchetto standalone lexrank su PyPI o pytextrank (plugin spaCy). Sconsigliato gensim.summarization, rimosso dalla versione 4.x.

**Vantaggi**

- LexRank è la baseline extractive più citata nei paper su Multi-News: utile per confrontare i risultati con la letteratura.
- Cattura relazioni tra frasi basate su similarità, non solo frequenza pura.
- Implementazione matura e stabile in sumy; riconosciuta a lezione come "high effectiveness" ed "easy-to-use".

**Svantaggi / limiti**

- Nei paper su Multi-News, LexRank (R-1 ≈ 40.3) è superato sia dalla baseline First-3 sia da MMR (R-1 ≈ 44.7).
- Non gestisce esplicitamente la ridondanza tra articoli diversi: tende a selezionare frasi simili quando il contenuto è ripetuto in più fonti.
- Alta complessità computazionale (O(n²) nella costruzione del grafo) e metodo non incrementale, come sottolineato a lezione.

## 3.4 Centroid-based (MEAD) + Maximal Marginal Relevance (MMR)

**ANNOTAZIONE - Copertura nella lezione: SÌ - MMR trattato esplicitamente; centroide coperto come concetto**

MMR è trattato a lezione tra gli approcci optimization-based (slide 58), con la formula completa del punteggio della frase, che bilancia salienza rispetto al documento e ridondanza rispetto alle frasi già selezionate tramite il parametro λ. Compare inoltre nella tassonomia dei metodi statistici/euristici (slide 30).

Il concetto di centroide è coperto negli approcci clustering-based (slide 34): raggruppare documenti/frasi omogenei e scegliere i rappresentanti dei cluster (centroidi o medoidi). Il sistema MEAD in quanto tale non è citato.

Conseguenza per il gruppo: questa tecnica è pienamente allineata al programma del corso ed è quindi tra le più difendibili in sede di valutazione. Resta la scelta più forte come contributo implementativo originale.

**Descrizione**

Il metodo centroid-based (introdotto dal sistema MEAD) calcola un vettore "centroide" che rappresenta il contenuto medio del cluster di documenti, e assegna un punteggio alle frasi in base alla similarità con tale centroide. Per gestire esplicitamente la ridondanza tra fonti si applica poi il Maximal Marginal Relevance (MMR): ad ogni passo si seleziona la frase che massimizza un compromesso tra rilevanza e dissimilarità rispetto alle frasi già selezionate.

**Come implementarla**

- Calcolare rappresentazioni vettoriali delle frasi (TF-IDF, oppure sentence embedding per una versione più moderna).
- Calcolare il centroide come media dei vettori di tutte le frasi del cluster.
- Ad ogni iterazione selezionare la frase che massimizza: λ · S(frase, D) − (1−λ) · max R(frase, frasi già scelte) - formula presentata a lezione (slide 58).
- Iterare finché non si raggiunge la lunghezza di riassunto desiderata.

**Libreria / Framework:** Nessuna libreria consolidata plug-and-play per MMR: si implementa da zero con scikit-learn (TfidfVectorizer + cosine_similarity), circa 30-40 righe. Esistono repository di riferimento su GitHub (es. K-means + centroid + MMR + posizione frase).

**Vantaggi**

- MMR affronta direttamente il problema centrale della MDS: la ridondanza tra fonti multiple sullo stesso evento.
- Su Multi-News, MMR ottiene punteggi ROUGE nettamente superiori a LexRank (R-1 44,72 vs 40,27; R-2 14,92 vs 12,63).
- Ideale come contributo originale del gruppo: dimostra comprensione del problema, non solo uso di librerie esterne.
- Pienamente allineato al programma del corso (trattato a lezione con formula esplicita).
- Il parametro λ è interpretabile e facilmente tarabile empiricamente.

**Svantaggi / limiti**

- Richiede implementazione manuale, quindi più tempo di sviluppo e test rispetto a sumy.
- La qualità dipende dalla rappresentazione vettoriale: TF-IDF cattura solo similarità lessicale.
- Come tutti gli optimization-based, la lezione ne segnala i limiti generali: scalabilità limitata ed explainability limitata; il greedy non garantisce l'ottimo globale.

## 3.5 LSA (Latent Semantic Analysis)

**ANNOTAZIONE - Copertura nella lezione: SÌ - sezione dedicata**

Trattata a lezione con slide dedicate (60-62), riferite specificamente alla LSA-based Multi-Document Summarization di Steinberger: si trasforma la matrice frase-per-termine in una matrice frase-per-topic latente tramite SVD, si filtrano gli autovettori tenendo solo le feature più significative e si scelgono le frasi che meglio coprono i topic latenti, aggiornando iterativamente la matrice approssimata per evitare ridondanza.

Utile correzione rispetto al nostro documento: la lezione elenca la scalabilità tra i PRO di LSA (insieme a semplicità e indipendenza dalla lingua), mentre il nostro documento la indicava come costosa. L'unico contro citato a lezione è che considera solo relazioni a livello di parola. Il testo della sezione è stato aggiornato di conseguenza.

Da notare che la variante di Steinberger include un meccanismo esplicito anti-ridondanza (aggiornamento iterativo di A'), assente nell'implementazione standard di sumy: è un possibile spunto di approfondimento.

**Descrizione**

Applica la Singular Value Decomposition (SVD) alla matrice frase-per-termine per individuare i "topic" latenti, e seleziona le frasi che meglio rappresentano le dimensioni semantiche principali. Nella variante multi-documento di Steinberger la matrice approssimata viene aggiornata iterativamente per evitare di selezionare frasi ridondanti.

**Come implementarla**

Concatenare gli articoli del cluster e utilizzare direttamente LsaSummarizer di sumy, con la stessa interfaccia usata per gli altri metodi della libreria. Per avvicinarsi alla variante vista a lezione occorrerebbe aggiungere manualmente l'aggiornamento iterativo della matrice approssimata (anti-ridondanza), non presente in sumy.

**Libreria / Framework:** sumy (LsaSummarizer); eventualmente estensione custom con numpy/scipy per la variante Steinberger

**Vantaggi**

- Cattura relazioni semantiche a livello di topic, andando oltre la frequenza di parole.
- Secondo la lezione: semplicità, scalabilità e indipendenza dalla lingua.
- Implementazione pronta in sumy, coerente con gli altri metodi: facilita un confronto sistematico nella stessa pipeline.
- Metodo esplicitamente trattato nel corso, quindi facilmente difendibile in relazione.

**Svantaggi / limiti**

- Considera solo relazioni a livello di parola (unico contro segnalato a lezione): non cattura co-occorrenze complesse tra più termini - è proprio il limite che motiva i metodi itemset-based (cfr. nuova sezione 4.1).
- L'implementazione standard di sumy non include il meccanismo anti-ridondanza della variante multi-documento.
- Meno citato come baseline specifica nei paper su Multi-News rispetto a LexRank/TextRank/MMR.

## 3.6 Topic modeling e clustering tematico (LDA + Orange)

**ANNOTAZIONE - Copertura nella lezione: PARZIALE (clustering sì, LDA no)**

Gli approcci clustering-based sono trattati a lezione (slide 34-37): raggruppamento di documenti/frasi/parole omogenei e selezione dei rappresentanti dei cluster (centroidi, medoidi). Come esempi vengono portati il modello FGB di Wang et al. (2011), che integra document clustering e summarization tramite fattorizzazione non-negativa, e la update summarization basata su clustering gerarchico incrementale (Wang & Li, 2010).

LDA in quanto tale non è nominato nelle slide; il topic modeling è coperto indirettamente via NMF (FGB) e via LSA.

Pro/contro dalla lezione, da recepire: PRO - language-agnostic, incrementale (via clustering gerarchico), abbastanza robusto al rumore (con clustering density-based); CONTRO - efficacia limitata su collezioni documentali complesse rispetto ad altre tecniche, in particolare rispetto ai modelli itemset-based e LSA-based. Questo è un segnale importante: il docente considera il clustering puro il meno performante tra i metodi classici.

**Descrizione**

Si individuano i temi/topic latenti presenti nell'insieme di articoli (tipicamente con LDA) e si estraggono le frasi più rappresentative per ciascun topic, in modo da garantire copertura di tutti gli argomenti trattati dalle fonti. Variante affine vista a lezione: il modello FGB, che ottiene simultaneamente cluster di documenti e riassunti tramite fattorizzazione non-negativa della matrice documento-termine usando la matrice frase-termine come base.

**Come implementarla**

È l'approccio che si integra più naturalmente con Orange, indicato tra gli strumenti del progetto. Il modulo Text Mining di Orange offre componenti visuali per LDA e clustering gerarchico. Workflow tipico: Corpus → Preprocess Text → Topic Modelling (LDA) → Corpus Viewer / Word Cloud. Per l'estrazione delle frasi rappresentative serve un passaggio custom in Python: assegnare ogni frase al topic dominante e selezionare quelle con punteggio di appartenenza più alto.

**Libreria / Framework:** Orange (clustering tematico visuale/esplorativo) + gensim.models.LdaModel o sklearn LatentDirichletAllocation / NMF (per l'estrazione delle frasi)

**Vantaggi**

- Si aggancia direttamente a Orange, valorizzando quella parte della consegna.
- Ottimo per l'analisi esplorativa e per la presentazione (word cloud, cluster visivi).
- Aiuta a garantire diversità tematica nel riassunto finale.
- Language-agnostic e potenzialmente incrementale (pro segnalati a lezione).

**Svantaggi / limiti**

- La lezione segnala esplicitamente l'efficacia limitata del clustering puro su collezioni complesse rispetto ai metodi itemset-based e LSA-based: da tenere presente nelle aspettative sui risultati ROUGE.
- Da solo LDA non produce un riassunto: va combinato con una fase di estrazione delle frasi, con più lavoro di integrazione rispetto ai metodi "pronti" di sumy.
- Il numero di topic è un iperparametro da validare.
- Meno standard come baseline in letteratura, difficile da confrontare con risultati pubblicati.

## 3.7 Clustering su sentence embedding (BERT / SBERT)

**ANNOTAZIONE - Copertura nella lezione: SÌ - coperta come "neural summarization self-supervised"**

La lezione dedica alla neural summarization le slide 73-77 e distingue nettamente due modalità. La prima è self-supervised: si imparano rappresentazioni vettoriali del testo (es. BERT), si calcola la similarità semantica con cosine similarity e si applicano i metodi tradizionali ai vettori così ottenuti - l'esempio esplicito del docente è "applicare TextRank alle codifiche BERT". È esattamente l'approccio della nostra sezione, quindi pienamente allineato al corso.

Questo chiarisce anche il dubbio terminologico sollevato nella nota: la lezione classifica questi metodi come pre-LLM (BERT è nella categoria "Pre-trained LM" della tassonomia di slide 30, distinta da "LLM"), confermando che usarli non viola il vincolo "non-LLM" del progetto.

La seconda modalità, supervisionata (BERTSum, MatchSum), è invece assente dal nostro documento: è stata aggiunta come nuova sezione 4.3.

Pro/contro dalla lezione: PRO - alte prestazioni, semantic-aware; CONTRO - necessità di dataset annotati su larga scala (solo per la variante supervisionata), risorse hardware costose, bassa spiegabilità.

**Descrizione**

Si calcolano rappresentazioni vettoriali (embedding) delle frasi tramite modelli encoder pre-addestrati (BERT o Sentence-BERT), si applica un clustering (tipicamente K-means) sugli embedding, e si seleziona la frase più vicina al centroide di ciascun cluster. Variante equivalente suggerita a lezione: applicare TextRank/LexRank direttamente sopra le codifiche BERT invece che sui vettori TF-IDF.

**Come implementarla**

Libreria pip-installabile che richiede poche righe: si istanzia il Summarizer (oppure SBertSummarizer per usare sentence-transformers) e si passa il testo concatenato del cluster, specificando ratio o num_sentences. In alternativa, pipeline custom: sentence-transformers per gli embedding + K-means di scikit-learn, oppure sentence-transformers + grafo di similarità + PageRank (networkx) per la variante "TextRank su BERT" indicata a lezione.

**Libreria / Framework:** bert-extractive-summarizer; in alternativa sentence-transformers + scikit-learn / networkx

**Vantaggi**

- Qualità semantica nettamente superiore a TF-IDF: cattura sinonimi e parafrasi tra fonti diverse, aspetto molto rilevante in Multi-News.
- API molto semplice, poche righe di codice per iniziare.
- Alte prestazioni e semantic-awareness (pro segnalati a lezione).
- La variante self-supervised non richiede alcun dataset annotato.

**Svantaggi / limiti**

- Richiede il download di un modello pre-addestrato e più risorse computazionali (la lezione segnala "expensive HW resources").
- Bassa spiegabilità: difficile motivare perché una specifica frase sia stata selezionata (contro esplicito nella lezione).
- Rimane la scelta terminologica da chiarire nella relazione (encoder pre-addestrato vs LLM generativo), sebbene la tassonomia della lezione li tenga distinti.

**_Nota:_** _La lezione tiene esplicitamente distinte le categorie "Pre-trained LM" (BERT, BART, PEGASUS) e "LLM" nella tassonomia dei paradigmi (slide 23 e 30): questo dà una base autorevole per giustificare l'inclusione di BERT/SBERT in un progetto dichiaratamente "non-LLM"._

## 3.8 SummPip (sentence graph + spectral clustering + compressione)

**ANNOTAZIONE - Copertura nella lezione: NO - non trattata a lezione**

SummPip non compare nelle slide. Tuttavia i suoi ingredienti sono tutti coperti separatamente: grafo di frasi (approcci graph-based), clustering (approcci clustering-based) e sentence compression, che la lezione elenca tra le "key operations" della summarization (slide 26-27, rimozione delle parti non importanti per accorciare la frase originale).

Conseguenza per il gruppo: resta uno stretch goal interessante (è l'unico metodo validato dagli autori proprio su Multi-News), ma non essendo trattato a lezione richiederebbe uno sforzo di studio autonomo aggiuntivo, oltre al costo di integrazione già segnalato. Da valutare solo se il tempo lo consente.

**Descrizione**

Metodo unsupervised pensato per la MDS e validato su Multi-News. Converte gli articoli in un grafo di frasi che incorpora sia conoscenza linguistica sia rappresentazioni neurali profonde, applica spectral clustering per ottenere gruppi di frasi affini, e comprime ciascun cluster per produrre la frase di sintesi finale.

**Come implementarla**

Codice originale disponibile sul repository degli autori (Zhao et al., SIGIR 2020); richiede l'integrazione di più componenti: grafo linguistico, embedding, spectral clustering (scikit-learn) e algoritmo di compressione basato su shortest path in un word graph.

**Libreria / Framework:** Repository ufficiale degli autori su GitHub - verificare la compatibilità delle dipendenze (codice di ricerca del 2020).

**Vantaggi**

- Unico metodo della lista validato sperimentalmente su Multi-News dagli stessi autori, con risultati competitivi rispetto a modelli neurali supervisionati.
- Va oltre la pura estrazione: la compressione produce output più simili a un vero riassunto.
- Affronta esplicitamente sia ridondanza (clustering) sia sintesi (compressione).

**Svantaggi / limiti**

- Non trattato a lezione: richiede studio autonomo aggiuntivo rispetto al programma del corso.
- Implementazione complessa; codice di ricerca del 2020 che può richiedere adattamento alle versioni attuali delle librerie.
- Tempo di sviluppo/integrazione significativamente più alto.

## 3.9 Ottimizzazione combinatoria (occams)

**ANNOTAZIONE - Copertura nella lezione: PARZIALE (famiglia trattata, libreria no)**

Gli approcci optimization-based sono una famiglia centrale della lezione (slide 54-59): si selezionano le frasi esplorando lo spazio di ricerca e ottimizzando una funzione obiettivo, tipicamente tramite Integer Linear Programming o funzioni submodulari. La formulazione a knapsack/budgeted coverage su cui si basa occams è quindi pienamente coperta.

La libreria occams in sé non è citata; la lezione presenta invece due riferimenti specifici - le funzioni submodulari di Lin & Bilmes e COSUM - che abbiamo aggiunto come nuove sezioni 4.2 e 4.4.

Conseguenza per il gruppo: se si vuole coprire la famiglia "optimization-based" del programma, conviene privilegiare l'implementazione della funzione submodulare di Lin & Bilmes (sezione 4.2), che è quella effettivamente vista a lezione, rispetto alla libreria occams.

**Descrizione**

Modella l'estrazione del riassunto come un problema di "budgeted maximum coverage": dati pesi/importanza dei termini, si seleziona il sottoinsieme di frasi che massimizza la copertura dei termini importanti rispetto a un budget di lunghezza, con un algoritmo di approssimazione ottimale.

**Come implementarla**

Libreria Python con estensione in Rust per le parti computazionalmente critiche; supporta nativamente single e multi-document, ed è multilingue (nltk o stanza per la tokenizzazione).

**Libreria / Framework:** occams (PyPI, paper MDPI 2023)

**Vantaggi**

- Formulazione rigorosa con garanzie teoriche di approssimazione, a differenza degli euristici greedy come MMR.
- Supporto nativo multi-documento e multilingue, con parte critica in Rust per le performance.
- La famiglia di appartenenza (optimization-based) è pienamente coperta dal programma del corso.

**Svantaggi / limiti**

- Libreria poco diffusa e poco documentata: minor supporto della community.
- Non usata come baseline nei paper su Multi-News: risultati non confrontabili con la letteratura.
- Ridondante rispetto alla sezione 4.2 (funzioni submodulari), che copre la stessa famiglia ed è quella effettivamente trattata a lezione.

## 3.10 pyAutoSummarizer (libreria aggregatrice)

**ANNOTAZIONE - Copertura nella lezione: NO - non trattata a lezione (è solo una libreria, non un metodo)**

La lezione non cita pyAutoSummarizer, ma questo non è un problema: non si tratta di una metodologia bensì di una libreria che re-implementa metodi (TextRank, LexRank, LSA, KL-Sum) già trattati a lezione.

RIVALUTAZIONE RISPETTO ALLA PRIMA VERSIONE: la libreria è stata aggiornata (v1.2.0, aprile 2026) e ora include, oltre alle metriche classiche, anche BERTScore, SummaC, AlignScore e G-Eval. Questo la rende molto più interessante di quanto valutato inizialmente - non come motore di summarization, ma come motore di valutazione del progetto, perché quelle metriche rispondono direttamente al problema della penalizzazione di ROUGE sui reference parzialmente abstractive di Multi-News.

La trattazione completa e il confronto con sumy sono nella nuova Sezione 7, a cui si rimanda.

**Descrizione**

Libreria "ombrello" che integra metodi classici non-LLM (TextRank, LexRank, LSA, KL-Sum), modelli deep generativi (BART, T5, PEGASUS, ChatGPT) e soprattutto una suite di valutazione molto ampia: ROUGE-N/L/S, BLEU, METEOR, BERTScore, SummaC, AlignScore e G-Eval.

**Come implementarla**

Da usare come motore di valutazione unico per tutti i riassunti prodotti nel progetto (metodi sumy, metodi custom e output dell'LLM di LM Studio), garantendo che tutti siano misurati con la stessa implementazione delle metriche. Dettagli operativi nella Sezione 7.

**Libreria / Framework:** pyautosummarizer (pip install pyautosummarizer) - v1.2.0, licenza GPLv3

**Vantaggi**

- Suite di valutazione molto completa: affianca a ROUGE anche metriche semantiche (BERTScore) e di faithfulness (SummaC, AlignScore), essenziali per un confronto equo tra metodi estrattivi e LLM.
- Usare un'unica libreria per valutare tutti i metodi garantisce il rigore del confronto.
- Il paper della libreria documenta il fenomeno per cui i modelli abstractive ottengono punteggi inferiori agli estrattivi sulle metriche tradizionali: citazione molto utile per la discussione dei nostri risultati.

**Svantaggi / limiti**

- Progetto giovane e con pochi contributor rispetto a sumy: minore maturità.
- Come motore di summarization offre un sottoinsieme di sumy (4 metodi contro 7): non va usata a quello scopo.
- Attenzione: il suo TextRank usa sentence embeddings, quello di sumy usa TF-IDF - stesso nome, algoritmo diverso (cfr. sez. 7.5).

# 4\. Tecniche presenti nella lezione e assenti dal documento (da aggiungere)

Le quattro tecniche seguenti sono trattate nella lezione del Master ma non comparivano nella prima versione del documento. Sono ordinate per priorità decrescente di inclusione nel progetto. La sezione 4.1 in particolare merita una discussione dedicata del gruppo.

## 4.1 Itemset-based summarization (MWI-Sum)

**ANNOTAZIONE - Copertura nella lezione: PRESENTE SOLO NELLA LEZIONE - da aggiungere**

PRIORITÀ ALTA. È la tecnica più significativa emersa dal confronto: la lezione le dedica ben 10 slide (63-72), il massimo spazio dopo i metodi graph-based, e soprattutto il riferimento è MWI-Sum di Baralis, Cagliero, Fiori e Garza - cioè ricerca del docente stesso. Includerla nel progetto è quindi la scelta con il maggiore ritorno atteso in sede di valutazione.

La lezione la contrappone esplicitamente ai limiti di LSA (che "considera solo relazioni a livello di parola") e del clustering puro ("efficacia limitata su collezioni complesse... rispetto ai modelli itemset-based e LSA-based"): nel quadro concettuale del corso è presentata come uno dei metodi classici più performanti.

**Descrizione**

Esegue la selezione delle frasi sopra un modello a itemset. Un itemset è un insieme di termini di lunghezza arbitraria (es. {treat, diseas}) che "ricorre" in una frase se tutti i suoi termini vi sono contenuti. Le occorrenze dei termini nelle frasi sono pesate (tipicamente con TF-IDF), e il peso di occorrenza di un itemset in una frase è il minore tra i pesi dei suoi termini (scelta conservativa); il supporto pesato dell'itemset è la media dei suoi pesi di occorrenza sull'intero dataset.

Il punto chiave è che gli itemset rappresentano co-occorrenze tra più termini, andando quindi oltre le associazioni a livello di singola parola: è precisamente il limite attribuito a LSA nella lezione. Un ulteriore vantaggio è che la dimensione del riassunto è adattiva, cioè determinata dalla distribuzione degli itemset e non fissata a priori.

**Come implementarla**

- Preprocessing dei documenti e filtraggio delle frasi; costruzione di una matrice frase × termine con pesi (TF-IDF).
- Weighted itemset mining: estrarre tutte le combinazioni di termini il cui supporto pesato supera una soglia minima. Si usa un algoritmo di frequent itemset mining (Apriori o FP-Growth) adattato ai pesi.
- Sentence selection: selezionare l'insieme minimale di frasi che massimizza la copertura degli itemset (minima ridondanza = numero minimo di frasi; massima informazione = numero massimo di itemset per frase).
- Euristica greedy indicata a lezione: iterare sulle frasi, ad ogni passo scegliere quella coperta dal massimo numero di itemset, fermarsi quando tutti gli itemset estratti coprono una frase.
- Sentence ordering: la prima frase selezionata dal greedy è potenzialmente la più informativa e va posta all'inizio del riassunto; le frasi a basso rank possono essere post-potate se serve rispettare un limite di lunghezza.

**Libreria / Framework:** Non esiste una libreria pronta: MWI-Sum va implementato. La parte di itemset mining è però coperta da librerie mature - mlxtend (mlxtend.frequent_patterns: apriori, fpgrowth) oppure la libreria PAMI; il resto (pesatura, copertura greedy, ordinamento) è codice custom con scikit-learn/numpy. Sforzo stimato: paragonabile o leggermente superiore a MMR.

**Vantaggi**

- Va oltre le associazioni a livello di parola, catturando co-occorrenze tra più termini (superando il limite principale di LSA).
- Pro segnalati a lezione: alte prestazioni, spiegabilità del modello (gli itemset selezionati sono ispezionabili e interpretabili) e indipendenza dalla lingua.
- Dimensione del riassunto adattiva, non fissata a priori.
- Massima aderenza al programma del corso: è la tecnica su cui la lezione insiste di più tra i metodi classici, ed è ricerca del docente.
- Ottimo secondo contributo implementativo originale del gruppo, accanto a MMR.

**Svantaggi / limiti**

- Contro segnalati a lezione: alta complessità computazionale e scalabilità limitata - l'itemset mining è esponenziale nel numero di termini distinti, quindi la soglia di supporto minimo va tarata con attenzione sui cluster più grandi di Multi-News (fino a 10 articoli).
- Nessuna libreria di summarization pronta all'uso: richiede implementazione custom (mitigata dall'uso di mlxtend per il mining).
- Non è una baseline standard nei paper su Multi-News: risultati difficilmente confrontabili con la letteratura pubblicata su questo dataset.

## 4.2 Optimization-based: funzioni submodulari (Lin & Bilmes)

**ANNOTAZIONE - Copertura nella lezione: PRESENTE SOLO NELLA LEZIONE - da aggiungere**

PRIORITÀ MEDIO-ALTA. La lezione dedica agli approcci optimization-based le slide 54-59 e presenta le funzioni submodulari di Lin & Bilmes (ACL 2011) con le formule complete. Il nostro documento copriva questa famiglia solo indirettamente tramite la libreria occams, che però non è quella vista a lezione: conviene sostituire/affiancare occams con questa formulazione.

Nota concettuale utile per la relazione: la funzione obiettivo F(S) = R(S) + λD(S) esplicita la stessa tensione rilevanza/diversità di MMR, ma con garanzie teoriche di approssimazione grazie alla submodularità (rendimenti decrescenti), invece che come pura euristica greedy. Confrontare MMR e submodular sullo stesso dataset è quindi un esperimento concettualmente elegante e ben allineato al corso.

**Descrizione**

La summarization è modellata come un problema di knapsack: dato un insieme di frasi V e una funzione F che assegna un valore reale a ogni sottoinsieme S, si cerca il sottoinsieme di dimensione limitata (|S| ≤ K) che massimizza F(S).

La funzione obiettivo combina due termini: F(S) = R(S) + λD(S), dove R(S) misura la rilevanza/copertura - quanto bene le frasi del documento sono "coperte" da quelle selezionate, con saturazione tramite un min{Cᵢ(S), αCᵢ(V)} che impedisce a una singola frase di monopolizzare il punteggio - e D(S) misura la diversità, premiando la selezione di frasi provenienti da cluster diversi tramite una radice quadrata che introduce rendimenti decrescenti all'interno dello stesso cluster.

**Come implementarla**

- Calcolare la matrice di similarità ωᵢⱼ tra tutte le frasi (cosine su TF-IDF o su embedding).
- Clusterizzare le frasi (K-means) per definire le partizioni Pᵢ usate nel termine di diversità.
- Implementare R(S) e D(S) secondo le formule della lezione; l'ottimizzazione avviene con un algoritmo greedy, che per funzioni submodulari monotone garantisce un'approssimazione con fattore noto rispetto all'ottimo.
- Tarare λ (bilanciamento rilevanza/diversità) e α (soglia di saturazione della copertura).

**Libreria / Framework:** Implementazione custom con scikit-learn + numpy (matrice di similarità, K-means, greedy). Esistono librerie generiche di ottimizzazione submodulare (es. apricot-select) riusabili per la parte di massimizzazione greedy.

**Vantaggi**

- Pro segnalati a lezione: buone prestazioni e possibilità di combinare machine learning e ottimizzazione.
- Garanzie teoriche di approssimazione (proprietà di submodularità), a differenza dell'euristica pura di MMR.
- Gestisce esplicitamente e simultaneamente copertura e diversità: entrambe cruciali nella MDS.
- Metodo trattato a lezione con formule esplicite: facilmente difendibile e ben documentabile in relazione.
- Confronto naturale e istruttivo con MMR (stessa tensione, formalizzazione diversa).

**Svantaggi / limiti**

- Contro segnalati a lezione: scalabilità limitata e spiegabilità limitata.
- Richiede implementazione custom e la taratura di due iperparametri (λ, α), oltre al numero di cluster K.
- Costo computazionale della matrice di similarità O(n²) sulle frasi del cluster.

## 4.3 Neural summarization supervisionata (BERTSum, MatchSum)

**ANNOTAZIONE - Copertura nella lezione: PRESENTE SOLO NELLA LEZIONE - da valutare**

PRIORITÀ DA DISCUTERE. La lezione presenta la modalità supervisionata della neural summarization (slide 73-77), che il nostro documento non contemplava: BERTSum (Liu & Lapata, EMNLP 2019) e MatchSum (Zhong et al., ACL 2020).

Rilevanza specifica per il nostro progetto: MatchSum è il precedente stato dell'arte extractive proprio su Multi-News (R-1 46,20 / R-2 16,51 / R-L 41,89 secondo la letteratura già raccolta), quindi rappresenterebbe il tetto superiore di riferimento per qualunque metodo estrattivo del gruppo.

Attenzione però al perimetro del progetto: si tratta di modelli neurali supervisionati basati su transformer pre-addestrati. Non sono LLM generativi (nella tassonomia della lezione stanno in "Pre-trained LM", categoria distinta da "LLM"), ma sono sicuramente lontani dai metodi "tradizionali". La lezione stessa ne segnala i costi: necessità di dataset annotati su larga scala e risorse hardware costose.

Raccomandazione: valutarli come termine di confronto bibliografico (citare i loro punteggi ROUGE nella relazione senza rieseguirli) piuttosto che implementarli, a meno che il gruppo non disponga di GPU e tempo.

**Descrizione**

BERTSum modella l'extractive summarization come un problema di classificazione delle frasi: per ogni frase si predice True se appartiene al riassunto target, False altrimenti; l'architettura usa interval segment embeddings per separare le frasi in input e aggiunge summarization layers sopra BERT.

MatchSum riformula il problema come semantic text matching: un buon riassunto dovrebbe essere, nel suo complesso, semanticamente più vicino al documento sorgente rispetto ai riassunti candidati non validi. Le frasi vengono pre-potate (tipicamente con BERTSum) per limitare le combinazioni, i riassunti candidati sono combinazioni delle frasi shortlistate, e si sceglie il candidato più vicino al documento nello spazio semantico.

**Come implementarla**

Entrambi hanno codice ufficiale su GitHub (nlpyang/PreSumm per BERTSum; maszhongming/MatchSum). Richiedono GPU e fine-tuning su dati annotati - anche se per Multi-News esistono checkpoint pre-addestrati resi disponibili dagli autori, il che renderebbe possibile la sola inferenza.

**Libreria / Framework:** PreSumm (BERTSum) e MatchSum - repository ufficiali degli autori; PyTorch + HuggingFace transformers

**Vantaggi**

- MatchSum è il riferimento SOTA extractive su Multi-News: fornisce il tetto superiore rispetto a cui misurare i metodi classici del gruppo.
- Pro segnalati a lezione: alte prestazioni e semantic-awareness.
- Trattati esplicitamente nel programma del corso.

**Svantaggi / limiti**

- Contro segnalati a lezione: richiedono dataset annotati su larga scala, risorse hardware costose (GPU) e hanno bassa spiegabilità.
- Si allontanano dallo spirito "metodi tradizionali / non-LLM" dell'obiettivo di ricerca del gruppo: da chiarire se rientrano nel perimetro concordato.
- Costo di setup elevato (codice di ricerca, dipendenze datate) rispetto al valore aggiunto, se l'alternativa è semplicemente citarne i risultati pubblicati.

## 4.4 COSUM (clustering + ottimizzazione)

**ANNOTAZIONE - Copertura nella lezione: PRESENTE SOLO NELLA LEZIONE - opzionale**

PRIORITÀ BASSA/MEDIA. Presentato a lezione (slide 57) come metodo che combina esplicitamente clustering e ottimizzazione per la multi-document summarization (Alguliyev et al., Expert Systems).

Nel nostro documento questa combinazione non è formalizzata come metodo a sé, anche se il repository GitHub citato nella sezione 3.4 (K-means + centroid + MMR + posizione) segue una logica molto simile. Potrebbe quindi essere assorbito come variante della pipeline MMR anziché come tecnica separata, riducendo il costo di sviluppo.

**Descrizione**

Combina clustering e ottimizzazione per la MDS: le frasi vengono prima raggruppate in cluster omogenei tramite K-Means, poi le frasi salienti vengono selezionate all'interno dei cluster tramite tecniche di ottimizzazione. La funzione obiettivo è la media armonica di due funzioni che impongono rispettivamente la copertura e la diversità delle frasi selezionate nel riassunto.

**Come implementarla**

Implementazione custom: K-Means su vettori TF-IDF (o embedding) delle frasi con scikit-learn, poi ottimizzazione della funzione obiettivo (media armonica di copertura e diversità) con un algoritmo greedy o metaeuristico. In pratica è una variante "strutturata" della pipeline centroid+MMR già prevista nella sezione 3.4.

**Libreria / Framework:** Nessuna libreria dedicata: scikit-learn (K-Means) + implementazione custom della funzione obiettivo

**Vantaggi**

- Unisce i vantaggi di clustering (copertura tematica di tutte le fonti) e ottimizzazione (selezione formalmente motivata).
- Concettualmente molto vicino alla pipeline centroid+MMR già in programma: il costo incrementale è basso se si riusa lo stesso codice.
- Trattato a lezione, quindi coerente con il programma del corso.

**Svantaggi / limiti**

- Ampiamente sovrapposto a quanto già previsto nelle sezioni 3.4 e 4.2: rischio di ridondanza sperimentale senza reale valore aggiunto.
- Eredita i contro degli optimization-based segnalati a lezione (scalabilità e spiegabilità limitate) e la sensibilità del K-Means alla scelta di K.
- Non è una baseline standard su Multi-News.

# 5\. Tabella di corrispondenza lezione ↔ documento

La tabella mostra la mappatura completa tra gli argomenti della lezione e le sezioni del documento, evidenziando le tecniche presenti in uno solo dei due.

| **Argomento della lezione (slide)**                                                             | **Nel nostro documento**                   | **Stato**                           |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------ | ----------------------------------- |
| Tassonomia e classificazione dei metodi (23-25, 30)                                             | Implicita nella struttura del doc          | Da esplicitare (nuova sez. 2.4)     |
| Key operations: compressione, fusione, riordino, selezione, clustering di frasi (26-28)         | Assente come quadro concettuale            | Da aggiungere (nuova sez. 2.4)      |
| Text preprocessing: stopword removal, stemming, tokenizing (32)                                 | Sez. 6.3 (pipeline comune)                 | Presente                            |
| Features: title word, keyword, sentence position (33)                                           | Sez. 3.1, 3.2 (parziale)                   | Parziale                            |
| Clustering-based approaches: centroidi/medoidi, FGB, clustering gerarchico incrementale (34-37) | Sez. 3.6 (topic modeling), 3.7             | Parziale                            |
| Graph-based: TextRank, LexRank (38-53)                                                          | Sez. 3.3                                   | Presente (allineato)                |
| Optimization-based: ILP, funzioni submodulari (54-56)                                           | Assente (solo occams, sez. 3.9)            | **DA AGGIUNGERE (sez. 4.2)**        |
| Optimization-based: COSUM (57)                                                                  | Assente                                    | **DA AGGIUNGERE (sez. 4.4)**        |
| Optimization-based: MMR (58-59)                                                                 | Sez. 3.4                                   | Presente (allineato)                |
| LSA-based summarization, Steinberger (60-62)                                                    | Sez. 3.5                                   | Presente (pro/contro da correggere) |
| Itemset-based summarization, MWI-Sum (63-72)                                                    | Assente                                    | **DA AGGIUNGERE (sez. 4.1)**        |
| Neural summarization self-supervised: TextRank su BERT (73)                                     | Sez. 3.7                                   | Presente (allineato)                |
| Neural summarization supervisionata: BERTSum, MatchSum (74-77)                                  | Assente                                    | DA VALUTARE (sez. 4.3)              |
| Abstractive neural: BART, PEGASUS (78-81)                                                       | Fuori perimetro (generativi)               | Escluso per scelta di progetto      |
| - (non trattato a lezione)                                                                      | Sez. 3.1 Baseline Lead/First-k             | Solo nel documento                  |
| - (non trattato a lezione)                                                                      | Sez. 3.2 Luhn, Edmundson, SumBasic, KL-Sum | Solo nel documento                  |
| - (non trattato a lezione)                                                                      | Sez. 3.8 SummPip                           | Solo nel documento                  |
| - (non trattato a lezione)                                                                      | Sez. 3.9 occams (libreria)                 | Solo nel documento                  |
| - (non trattato a lezione)                                                                      | Sez. 3.10 pyAutoSummarizer (libreria)      | Solo nel documento                  |

# 6\. Sintesi del confronto e shortlist aggiornata

## 6.1 Cosa è emerso dal confronto

Il confronto ha prodotto quattro risultati principali. Primo: le tecniche centrali del nostro documento (graph-based, MMR, LSA, clustering su embedding) sono tutte trattate a lezione, spesso in modo approfondito, quindi l'impostazione della ricerca è sostanzialmente confermata. Secondo: manca una tecnica importante, l'itemset-based summarization (MWI-Sum), a cui la lezione dedica dieci slide e che è ricerca del docente stesso - è la lacuna più rilevante da colmare. Terzo: la famiglia optimization-based era coperta solo tramite la libreria occams, mentre la lezione presenta le funzioni submodulari di Lin & Bilmes e COSUM: conviene riallineare il documento sostituendo occams con la formulazione submodulare. Quarto: la lezione tratta anche la neural summarization supervisionata (BERTSum, MatchSum), assente dal documento; MatchSum in particolare è il SOTA extractive su Multi-News e va almeno citato come tetto superiore di riferimento.

Un'osservazione trasversale utile per la relazione: la lezione fornisce, per ciascuna famiglia, una scheda esplicita di pro e contro (efficacia, scalabilità, spiegabilità, indipendenza dalla lingua, incrementalità). Adottare le stesse dimensioni di valutazione nella nostra relazione garantirebbe coerenza col linguaggio del corso e renderebbe il confronto tra metodi immediatamente leggibile per chi valuta il progetto.

## 6.2 Shortlist aggiornata

| **Categoria**                       | **Tecniche**                                                                                            | **Motivazione (aggiornata post-lezione)**                                                                                                                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Imprescindibili                     | Baseline First-k; suite sumy (Luhn, SumBasic, KL-Sum, LSA, LexRank, TextRank)                           | Costo implementativo molto basso, confronto ampio immediato. LexRank/TextRank/LSA sono trattati a lezione in modo approfondito; gli altri sono baseline storiche a costo quasi nullo.                                        |
| Fortemente consigliate              | Centroid-based + MMR (custom); Itemset-based MWI-Sum (custom)                                           | Entrambe trattate a lezione (MMR con formula esplicita; MWI-Sum con 10 slide ed è ricerca del docente). Sono i due contributi implementativi originali con il maggior ritorno atteso in valutazione.                         |
| Consigliate                         | Funzioni submodulari (Lin & Bilmes)                                                                     | Copre la famiglia optimization-based del programma con garanzie teoriche; confronto elegante con MMR. Da preferire a occams.                                                                                                 |
| Da valutare in base al tempo        | Clustering su embedding BERT/SBERT (incl. variante "TextRank su BERT"); Topic modeling (LDA) con Orange | La prima è coperta a lezione come neural self-supervised e ha buon rapporto qualità/sforzo; la seconda valorizza Orange ma la lezione ne segnala l'efficacia limitata su collezioni complesse.                               |
| Stretch goal (opzionali)            | SummPip; COSUM                                                                                          | SummPip: massima aderenza a Multi-News ma non trattato a lezione e costoso. COSUM: trattato a lezione ma ampiamente sovrapposto a MMR/submodular.                                                                            |
| Solo come riferimento bibliografico | BERTSum, MatchSum                                                                                       | MatchSum è il SOTA extractive su Multi-News: citarne i punteggi ROUGE come tetto superiore, senza rieseguirli (richiedono GPU e si allontanano dal perimetro non-LLM).                                                       |
| Infrastruttura (non tecniche)       | sumy come motore di summarization; pyAutoSummarizer come motore di valutazione                          | Vanno usate entrambe con ruoli separati: sumy copre 7 metodi classici con API unificata; pyAutoSummarizer fornisce ROUGE + BERTScore + SummaC per valutare TUTTI i metodi con la stessa implementazione. Dettagli in Sez. 7. |
| Da escludere                        | occams                                                                                                  | Sostituito dalla sez. 4.2 (funzioni submodulari), che copre la stessa famiglia, è quella vista a lezione ed è confrontabile con MMR.                                                                                         |

## 6.3 Prossimi passi

- Decidere se includere l'itemset-based (MWI-Sum): è la scelta con il maggior ritorno atteso in valutazione, ma richiede implementazione custom (mitigabile con mlxtend per l'itemset mining).
- Decidere se sostituire occams con le funzioni submodulari di Lin & Bilmes per coprire la famiglia optimization-based in linea col programma del corso.
- Decidere il perimetro rispetto ai metodi neurali supervisionati (BERTSum/MatchSum): implementarli, citarli soltanto, o escluderli.
- Definire la pipeline comune di preprocessing (gestione del separatore "|||||", tokenizzazione, stopword removal, stemming) condivisa da tutte le tecniche, per un confronto equo.
- Decidere il set di metriche di valutazione: la proposta è ROUGE + BERTScore + SummaC via pyAutoSummarizer, per non penalizzare l'LLM sulle parafrasi e per misurare la fedeltà alle fonti (cfr. Sez. 7.4).
- Adottare le dimensioni di valutazione della lezione (efficacia, scalabilità, spiegabilità, indipendenza dalla lingua) come griglia comune nella relazione finale.
- Assegnare le tecniche ai membri del gruppo in base a interesse e competenze.

# 7\. Approfondimento sulle librerie: sumy e pyAutoSummarizer

Questa sezione approfondisce le due librerie "generaliste" citate nel documento e le mette a confronto in relazione agli obiettivi specifici del progetto. La conclusione anticipata è che non sono alternative in concorrenza tra loro, ma strumenti complementari: sumy è la scelta migliore per la fase di summarization, pyAutoSummarizer per la fase di valutazione.

## 7.1 sumy

**Cosa è:** Libreria Python matura (licenza Apache-2.0, richiede Python 3.8+) per l'estrazione di riassunti da testi o pagine HTML, con anche una utility da riga di comando. Dipende da nltk per la tokenizzazione.

**Algoritmi implementati (7)**

- Luhn - metodo euristico basato sulla frequenza delle parole
- Edmundson - metodo euristico che estende Luhn (richiede la specifica di bonus_words, stigma_words e null_words, a differenza degli altri summarizer)
- LSA - Latent Semantic Analysis via SVD
- LexRank - approccio unsupervised ispirato a PageRank e HITS
- TextRank - ranking graph-based delle frasi
- SumBasic - metodo frequentemente usato come baseline in letteratura
- KL-Sum - aggiunge frasi in modo greedy finché diminuisce la divergenza KL

**Perché è rilevante per il progetto**

Copre da sola sei delle tecniche della nostra Sezione 3 (3.2, 3.3, 3.5) con una API unificata: si cambia una sola riga di codice per passare da un algoritmo all'altro, il che la rende ideale per un benchmark comparativo sistematico. È l'unica libreria che offre Luhn, Edmundson e SumBasic - quest'ultimo particolarmente interessante perché è l'unico dei metodi statistici pensato nativamente per il multi-documento. Include inoltre un semplice framework di valutazione interno, meno completo però di quello di pyAutoSummarizer.

**Come usarla nel progetto**

Il flusso è sempre lo stesso: si costruisce un PlaintextParser sul testo (per noi: la concatenazione degli articoli del cluster Multi-News, dopo aver gestito il separatore "|||||"), si istanzia lo Stemmer per la lingua, si imposta la lista di stop_words, e si invoca il summarizer specificando il numero di frasi desiderate. Poiché tutti e sette i summarizer condividono la stessa firma, è immediato costruire un dizionario {nome_metodo: summarizer} e iterare su tutti i metodi in un unico ciclo, producendo la tabella comparativa dei risultati ROUGE. Attenzione all'unica eccezione: EdmundsonSummarizer richiede parametri aggiuntivi (bonus_words, stigma_words, null_words) e non funziona con i soli valori di default.

**Limiti**

- Tutti i summarizer sono nativamente single-document: il multi-documento si ottiene solo per concatenazione, quindi nessuno di essi (tranne SumBasic) gestisce esplicitamente la ridondanza tra fonti diverse.
- Non include MMR, né metodi itemset-based, né ottimizzazione submodulare: tutte le tecniche "native MDS" del nostro piano (sez. 3.4, 4.1, 4.2) restano da implementare a mano.
- Il framework di valutazione integrato è minimale: per ROUGE conviene comunque affiancare rouge-score di Google o il modulo di pyAutoSummarizer.

## 7.2 pyAutoSummarizer

**Cosa è:** Libreria Python (licenza GPLv3, Python 3.9+) che copre summarization estrattiva e astrattiva e - soprattutto - offre una suite di metriche di valutazione molto ampia, dalle classiche basate su overlap di n-grammi fino a misure semantiche e di faithfulness. Versione corrente 1.2.0 (aprile 2026); il riferimento è Pereira et al. (2026), Springer.

**Metodi di summarization**

- Estrattivi: TextRank (ranking graph-based su sentence embeddings e similarità coseno), LexRank (ranking graph-based su similarità coseno TF-IDF), LSA (via SVD su embeddings o matrice TF-IDF), KL-Sum.
- Astrattivi / deep: BART (facebook/bart-large-cnn), T5 (t5-base), PEGASUS (google/pegasus-xsum), ChatGPT via API OpenAI - questi ultimi fuori dal perimetro non-LLM del progetto.

**Suite di valutazione (il vero punto di forza)**

- Metriche classiche lessicali (reference-based): ROUGE-N, ROUGE-L, ROUGE-S (con skip-distance configurabile), BLEU, METEOR - tutte restituiscono F1, precision e recall.
- Metrica semantica (reference-based): BERTScore, che confronta gli embedding contestualizzati dei token e cattura le parafrasi che ROUGE non vede.
- Metriche di faithfulness / consistenza fattuale (source-based, non richiedono il reference): SummaC, basata su un modello NLI incluso, e AlignScore (Zha et al., ACL 2023). Servono a rilevare le allucinazioni, invisibili alle metriche lessicali.
- LLM-as-judge: G-Eval, che usa un modello OpenAI per assegnare un punteggio 1-5 su coerenza, consistenza, fluency e rilevanza (richiede API key).

**Preprocessing**

Pipeline configurabile con lowercasing, rimozione di accenti, caratteri speciali e numeri, rimozione di parole custom, stopword removal in 26 lingue (italiano incluso) e segmentazione delle frasi per punteggiatura, numero di parole o di caratteri.

**Perché è molto rilevante per il progetto**

La suite di valutazione risponde direttamente al problema metodologico centrale del nostro progetto. Sappiamo dall'EDA che i reference di Multi-News sono parzialmente abstractive: questo significa che ROUGE penalizza strutturalmente qualunque metodo estrattivo, e allo stesso tempo penalizza l'LLM locale quando riformula correttamente il contenuto con parole diverse. Valutare l'LLM di LM Studio contro i metodi classici usando solo ROUGE produrrebbe quindi un confronto sistematicamente distorto.

Le metriche aggiuntive di pyAutoSummarizer permettono di correggere questa distorsione: BERTScore misura la vicinanza semantica al reference anche in presenza di parafrasi, mentre SummaC/AlignScore misurano se il riassunto è fedele alle fonti - dimensione su cui l'LLM è a rischio (allucinazioni) mentre i metodi estrattivi sono fedeli per costruzione, dato che copiano frasi esistenti. È esattamente l'asse su cui il confronto LLM vs non-LLM diventa interessante e non banale.

A supporto di questa impostazione, il paper della libreria riporta un risultato molto pertinente: nei loro esperimenti i modelli astrattivi ottengono punteggi INFERIORI ai metodi estrattivi come TextRank e LexRank sulle metriche tradizionali, e gli autori attribuiscono il fenomeno al fatto che tali metriche si basano su corrispondenze esatte di frasi e struttura, e non colgono l'accuratezza semantica e contestuale dei riassunti astrattivi. È molto probabile che il gruppo osservi lo stesso fenomeno con l'LLM di LM Studio: avere una citazione pubblicata che documenta e spiega l'effetto è un ottimo appiglio per la discussione dei risultati.

**Limiti**

- Progetto piccolo (una manciata di star su GitHub, pochi contributor): minore maturità e minore supporto della community rispetto a sumy.
- Sui metodi classici offre un sottoinsieme di sumy (4 contro 7): mancano Luhn, Edmundson e soprattutto SumBasic.
- Licenza GPLv3 (copyleft), contro l'Apache-2.0 di sumy: irrilevante per un progetto accademico interno, ma da tenere presente se il codice venisse ridistribuito.
- Alcune metriche hanno dipendenze o costi aggiuntivi: BERTScore richiede il pacchetto bert-score, AlignScore un extra e un modello spaCy, G-Eval una API key OpenAI a pagamento.

## 7.3 Confronto sintetico

| **Criterio**                       | **sumy**                                                      | **pyAutoSummarizer**                                                |
| ---------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------- |
| Metodi classici estrattivi         | 7 (Luhn, Edmundson, LSA, LexRank, TextRank, SumBasic, KL-Sum) | 4 (TextRank, LexRank, LSA, KL-Sum)                                  |
| Metodi esclusivi                   | Luhn, Edmundson, SumBasic (unico nativo multi-doc)            | BART, T5, PEGASUS, ChatGPT (fuori perimetro non-LLM)                |
| Metriche di valutazione            | Framework interno minimale                                    | ROUGE-N/L/S, BLEU, METEOR, BERTScore, SummaC, AlignScore, G-Eval    |
| Gestione nativa multi-doc          | No (solo via concatenazione)                                  | No (solo via concatenazione)                                        |
| Maturità / community               | Alta (libreria di riferimento, molto usata e citata)          | Bassa (progetto giovane, pochi contributor)                         |
| Licenza                            | Apache-2.0 (permissiva)                                       | GPLv3 (copyleft)                                                    |
| Ultimo aggiornamento               | Progetto stabile e mantenuto                                  | v1.2.0 (aprile 2026) - in evoluzione attiva                         |
| **Ruolo consigliato nel progetto** | **Motore di SUMMARIZATION per i metodi classici**             | **Motore di VALUTAZIONE per tutti i metodi (inclusi custom e LLM)** |

## 7.4 Raccomandazione operativa per il progetto

La raccomandazione è usarle entrambe, ma con ruoli nettamente separati, evitando di scegliere l'una "contro" l'altra.

- sumy come motore di summarization per tutti i metodi classici della Sezione 3 (Luhn, Edmundson, SumBasic, KL-Sum, LSA, LexRank, TextRank), sfruttando l'API unificata per generare in un solo ciclo i riassunti di tutti e sette i metodi su ciascun cluster.
- Implementazione custom (scikit-learn) per le tecniche native MDS non coperte da nessuna libreria: MMR (sez. 3.4), MWI-Sum (sez. 4.1), funzioni submodulari (sez. 4.2).
- pyAutoSummarizer come motore di valutazione unico per TUTTI i riassunti prodotti - quelli di sumy, quelli custom e quelli dell'LLM di LM Studio - così che tutti i metodi siano misurati con la stessa identica implementazione delle metriche. Questo è metodologicamente importante: implementazioni diverse di ROUGE danno risultati leggermente diversi, e un confronto tra metodi valutati con librerie differenti non sarebbe rigoroso.
- In particolare, affiancare a ROUGE almeno BERTScore (per non penalizzare l'LLM sulle parafrasi) e SummaC (per misurare la fedeltà alle fonti, dove i metodi estrattivi hanno un vantaggio strutturale). Questo trasforma il confronto da una singola classifica ROUGE a un'analisi multi-dimensionale molto più interessante da discutere.
- G-Eval è opzionale e va valutato con cautela: richiede una API key OpenAI a pagamento e introdurrebbe un LLM proprietario nella fase di valutazione di un progetto che confronta metodi non-LLM - una scelta da motivare esplicitamente se adottata.

## 7.5 Avvertenze metodologiche

- Stesso nome, algoritmo diverso: il TextRank di pyAutoSummarizer usa sentence embeddings, quello di sumy usa la similarità su TF-IDF. Applicare "TextRank" con le due librerie NON produce gli stessi risultati. Va quindi scelta una sola libreria per la generazione dei riassunti (raccomandata: sumy) e dichiarato esplicitamente nella relazione quale implementazione è stata usata.
- Preprocessing e ROUGE: la pipeline di pyAutoSummarizer può applicare lowercasing e rimozione di numeri e caratteri speciali. Se il riassunto restituito eredita queste trasformazioni mentre il reference di Multi-News resta grezzo, i punteggi ROUGE ne risultano falsati. Va verificato sperimentalmente su qualche esempio e, se necessario, disattivate le opzioni più aggressive o ricostruito il riassunto a partire dalle frasi originali non processate.
- Preprocessing condiviso: qualunque sia la scelta, il preprocessing deve essere identico per tutti i metodi confrontati (compreso l'LLM), altrimenti il confronto perde validità. Conviene definire una funzione di preprocessing unica nel repository del gruppo, a monte sia di sumy sia dei metodi custom.
- Nessuna delle due librerie gestisce nativamente il multi-documento: entrambe lavorano su un testo unico. La gestione del separatore "|||||" e la strategia di concatenazione (ordine degli articoli, eventuale deduplicazione) sono decisioni di progetto che vanno prese esplicitamente e documentate, perché influenzano i risultati di tutti i metodi.

# 8\. Riferimenti bibliografici

**Fonti citate nella lezione del Master**

- Erkan, G., Radev, D. (2004). LexPageRank: Prestige in Multi-Document Text Summarization. EMNLP 2004.
- Mihalcea, R., Tarau, P. (2004). TextRank: Bringing Order into Texts.
- Carbonell, J., Goldstein, J. (1998). The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.
- Lin, H., Bilmes, J. (2011). A Class of Submodular Functions for Document Summarization. ACL 2011.
- Alguliyev, R. M. et al. COSUM: Text summarization based on clustering and optimization. Expert Systems, Wiley.
- Steinberger, J. LSA-based Multi-Document Summarization (2011).
- Baralis, E., Cagliero, L., Fiori, A., Garza, P. (2015). MWI-Sum: A Multilingual Summarizer Based on Frequent Weighted Itemsets. ACM TOIS 34(1).
- Wang, D., Zhu, S., Li, T., Chi, Y., Gong, Y. (2011). Integrating Document Clustering and Multidocument Summarization. ACM TKDD 5(3).
- Wang, D., Li, T. (2010). Document update summarization using incremental hierarchical clustering. CIKM '10.
- Liu, Y., Lapata, M. (2019). Text Summarization with Pretrained Encoders (BERTSum). EMNLP/IJCNLP.
- Zhong, M. et al. (2020). Extractive Summarization as Text Matching (MatchSum). ACL 2020.
- El-Kassas, W. S. et al. (2021). Automatic text summarization: A comprehensive survey. ESWA.
- Widyassari, A. P. et al. Review of automatic text summarization techniques & methods. J. King Saud University.
- Survey sui metodi di summarization pre-LLM e con LLM: arxiv.org/pdf/2406.11289.

**Fonti aggiuntive del nostro documento**

- Fabbri, A. et al. (2019). Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model. ACL 2019.
- Radev, D. et al. (2000/2004). Centroid-based summarization of multiple documents (sistema MEAD).
- Zhao, J. et al. (2020). SummPip: Unsupervised Multi-Document Summarization with Sentence Graph Compression. SIGIR 2020.
- occams: A Text Summarization Package. MDPI, 2023.
- Miso Belica - sumy: Module for automatic summarization of text documents and HTML pages (GitHub, Apache-2.0); Miller, D. - bert-extractive-summarizer (GitHub).
- Pereira, V., de Lima Porto, R. C., Figueira, L. A. A., Ferreira, R. A. C. A. (2026). Unveiling pyAutoSummarizer: An Extractive and Abstractive Summarization Library Powered with Artificial Intelligence. In: Technology Mining, Springer, Cham.
- Zhang, T. et al. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR 2020.
- Laban, P. et al. (2022). SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization. TACL.
- Zha, Y. et al. (2023). AlignScore: Evaluating Factual Consistency with A Unified Alignment Function. ACL 2023.
- Liu, Y. et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.