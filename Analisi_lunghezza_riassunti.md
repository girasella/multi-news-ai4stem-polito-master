# Lunghezza dei riassunti: analisi per cluster e confronto a lunghezza pari

**Richiesta del relatore.** *«Verificate le lunghezze e, in caso di disomogeneità significative,
riallineate i modelli troncando ad una lunghezza massima comparabile per tutti»* — precisata poi
in: analizzare, **per ogni gruppo Multi-News (cluster)**, le differenze di lunghezza fra i
riassunti dei diversi metodi, e contenere la distribuzione delle lunghezze dentro un intervallo.

Riferimenti nel repository: `notebooks/18_analisi_lunghezze.ipynb` (analisi), Vista 3 di
`notebooks/05_confronto.ipynb` (confronto prima/dopo), issue #15 e #16. Tutte le lunghezze sono
conteggi di parole (`str.split()`, la convenzione del repository), sulla split test.

---

## 1. Verifica: la disomogeneità è grave, e dentro il singolo cluster

Le medie per metodo (55 parole per `bart` → 477 per `lda`) nascondono il livello a cui il
confronto avviene davvero. Sui 5.437 cluster della split test coperti da tutti e 18 i metodi:

| statistica per cluster | p10 | p25 | **mediana** | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| spread (più lungo − più corto), parole | 265 | 329 | **409** | 503 | 609 |
| rapporto più lungo / più corto | 5,6 | 6,8 | **8,5×** | 10,6 | 13,1 |
| spread ÷ lunghezza del riferimento | 1,23 | 1,52 | **1,95** | 2,53 | 3,26 |

Nel cluster tipico il riassunto più lungo è **8,5 volte** il più corto, e la distanza fra i due è
**il doppio** del riassunto di riferimento. Nessun metodo si adatta al cluster: la correlazione
fra lunghezza generata e lunghezza del riferimento arriva al massimo a 0,51 (`primera`) ed è ≈ 0
per gli estrattivi a budget di frasi (`lsa`, `lsa_steinberger`, `sbert_agglom`), che con «11
frasi» producono la stessa lunghezza qualunque sia la dimensione del cluster. La lunghezza
spiega da sola gran parte del ROUGE-1 *recall* fra metodi (r = 0,89) ma poco dell'F1 (r = 0,26).

## 2. Quale intervallo: il riferimento stesso, cluster per cluster

Il riferimento non è costante (p10 129 → p90 292 parole; mediana da ~164 parole per i cluster
da 1 articolo a ~327 per quelli da 8+). Prima di scegliere un budget si è verificato quanto la
lunghezza del riferimento si lasci prevedere dal solo input, tarando ogni candidato sulla split
train e valutandolo su test:

| budget candidato (dal solo input) | MAE (parole) | riferimento entro ±25% | Spearman |
|---|---:|---:|---:|
| costante (mediana train, 220) | 53,2 | 54,2% | — |
| mediana per numero di articoli | 48,1 | 56,7% | 0,36 |
| mediana per decili di lunghezza della sorgente | 48,0 | 57,7% | 0,40 |
| sorgente ÷ compressione mediana | **137,7** | 27,6% | 0,41 |
| numero di articoli × terzile di sorgente | 46,5 | 58,0% | 0,44 |
| regressione log-log su entrambi | 46,7 | 56,7% | 0,46 |
| *oracolo: il riferimento stesso* | 0 | 100% | 1,00 |

**Nessun predittore dal solo input funziona**: il migliore batte un budget costante di 4 punti,
e l'idea intuitiva «scala con la lunghezza della sorgente» è la peggiore (il rapporto di
compressione varia di 5× fra cluster). Il budget scelto è quindi la **lunghezza del riferimento
di ciascun cluster**, con banda [0,80, 1,25]×: il protocollo *length-matched* / *oracle-length*
(Sun et al., 2019). Con `|candidato| ≈ |riferimento|` richiamo e precisione ROUGE coincidono, e
l'obiezione «i riassunti lunghi vincono sul recall» non può più nascere per costruzione.

**Caveat, da dichiarare sempre.** Il budget usa un'informazione gold — la sola *lunghezza* del
riferimento, mai il contenuto — quindi i numeri rispondono a «quanto è buona la selezione dei
contenuti a parità di lunghezza», non a «cosa otterrebbe il metodo in produzione». Non
sostituiscono i numeri di testa del benchmark: sono il controllo che li rende leggibili senza il
confondente della lunghezza.

## 3. Riallineamento: soffitto e pavimento

Il solo troncamento non basta: porta il rapporto max/min mediano da 8,5× a ~4×, perché un
riassunto *corto* non si tronca verso l'alto (`bart` resta sotto banda nel 100% dei cluster,
`qwen` nel 99%). Servono due interventi:

- **Soffitto, tutti e 18 i metodi** — troncamento a 1,25 × budget e ricalcolo delle metriche.
- **Pavimento, gli 11 estrattivi** — rigenerati con un budget in **parole** al posto del budget
  in **frasi**: il criterio di ordinamento di ciascun metodo è invariato, cambia solo dove ci si
  ferma (la frase che sfora entra se ne entra più della metà).
- **Pavimento, gli LLM** (`qwen`, `gemma`, `mistral`, `gpt5mini`) — rigenerabili con il budget
  nel prompt. Un LLM lo segue solo approssimativamente: con «of approximately *N* words» `qwen`
  scrive ~0,65 N; con «about *N* words, at least *N* words» e *N* = 1,2 × budget (fattore
  calibrato su un pilota di 30 righe) arriva a 1,01 × riferimento. **Eseguito per tutti e
  quattro.** Il fattore è calibrato **per modello**, con un pilota da 30 righe ciascuno, perché
  i modelli sbagliano in direzioni opposte: `qwen` sotto-genera (1,2), `gemma` e `gpt5mini`
  eccedono il vincolo «at least» (0,9), `mistral` è centrato ma dispersivo (1,0).
- **Solo soffitto per `bart`, `pegasus`, `primera`**, per scelta e non solo per costo. Per questi
  tre il pavimento passerebbe da `min_length` in token, cioe' costringendo il beam search a
  continuare a generare oltre la lunghezza naturale del modello: e' il meccanismo che produce i
  loop di ripetizione gia' osservati in questo benchmark (PEGASUS, riga 51178, METEOR −1959 alla
  sua lunghezza *naturale*). Applicato a `bart`, che dovrebbe quadruplicare l'output, il rischio
  e' di misurare il degrado da allungamento forzato invece della qualita' della selezione.
  `pegasus` (0,83×) e `primera` (0,97×) sono inoltre gia' dentro o al bordo della banda senza
  alcun intervento — una proprieta' dei modelli, non del contenimento. `bart` (0,26×, 0% in
  banda) resta l'**eccezione dichiarata** del confronto.

## 4. Risultato

Tabella finale (5.527 cluster comuni a tutti e 18 i metodi; «banda» = quota di cluster con
lunghezza entro [0,80, 1,25] × riferimento).

| metodo | rigenerato | banda | parole prima → dopo | R-1 recall prima → dopo | R-1 F1 prima → dopo |
|---|:-:|---:|---:|---:|---:|
| primera | no | 66% | 211 → 201 | 0,441 → 0,433 | **0,445 → 0,446** |
| pegasus | no | 52% | 178 → 175 | 0,390 → 0,387 | 0,424 → 0,424 |
| centroid_mmr | sì | 98% | 362 → 215 | 0,465 → 0,365 | 0,383 → **0,378** |
| lsa_steinberger | sì | 99% | 257 → 215 | 0,419 → 0,366 | 0,376 → 0,369 |
| firstk_psr | sì | 98% | 217 → 215 | 0,347 → 0,354 | 0,355 → 0,368 |
| firstk_nltk | sì | 98% | 219 → 215 | 0,348 → 0,352 | 0,356 → 0,368 |
| mistral | sì | 71% | 161 → 233 | 0,325 → 0,367 | 0,354 → 0,365 |
| centroid_mmr_bert | sì | 98% | 361 → 215 | 0,460 → 0,357 | 0,373 → 0,364 |
| sbert_kmeans | sì | 98% | 264 → 215 | 0,404 → 0,358 | 0,359 → 0,359 |
| textrank | sì | 98% | 368 → 215 | 0,448 → 0,343 | 0,370 → 0,359 |
| lexrank | sì | 97% | 450 → 216 | 0,482 → 0,338 | 0,365 → 0,352 |
| qwen | sì | 79% | 140 → 216 | 0,309 → 0,347 | 0,344 → 0,351 |
| lsa | sì | 98% | 249 → 215 | 0,357 → 0,322 | 0,351 → 0,350 |
| gpt5mini | sì | 100% | 241 → 209 | 0,380 → 0,363 | 0,348 → 0,351 |
| sbert_agglom | sì | 99% | 216 → 215 | 0,350 → 0,342 | 0,328 → 0,339 |
| lda | sì | 93% | 477 → 230 | **0,499 → 0,348** | 0,351 → **0,338** |
| gemma | sì | 99% | 292 → 219 | 0,390 → 0,337 | 0,335 → 0,328 |
| bart | no | 0,3% | 55 → 55 | 0,181 → 0,181 | 0,265 → 0,265 |

**Il contenimento ha funzionato, e il numero complessivo lo nasconde.** Il rapporto fra il
riassunto più lungo e il più corto dentro lo stesso cluster passa da 8,5× a 4,7× se si contano
tutti e 18 i metodi — ma quel 4,7 è quasi interamente `bart`, che con 0,26× il riferimento è il
minimo in quasi ogni cluster mentre il soffitto tiene il massimo a 1,25× (1,25/0,26 ≈ 4,8):

| insieme di metodi | prima | dopo |
|---|---:|---:|
| tutti e 18 | 8,5× | 4,7× |
| senza `bart` (17) | 3,8× | **1,6×** |
| i soli 15 rigenerati | 3,7× | **1,4×** |

Fra i metodi effettivamente riportati a lunghezza pari la dispersione dentro il cluster è quindi
quasi annullata. I 15 rigenerati stanno in banda nel **71–100%** dei cluster (11 estrattivi
93–99%, `gpt5mini` 100%, `gemma` 99%, `qwen` 79%, `mistral` 71%).

**La graduatoria si riassesta, e il quadro cambia in tre punti.**

- **`lda` perde tutto il suo vantaggio**: recall 0,499 → 0,348 e ultimo posto fra gli estrattivi
  in F1 (0,338). Era interamente lunghezza, come sospettato nella issue #14.
- **Le baseline `firstk_*` salgono dal 9°–10° al 5°–6° posto** (F1 0,368), sopra `lda`, `lsa`,
  `sbert_*`, `textrank` e `lexrank`. A parità di parole, prendere le prime frasi di ogni
  articolo regge il confronto con metodi di selezione molto più elaborati — il risultato più
  scomodo dell'esercizio, ma anche il più interessante.
- **I metodi corti guadagnano**: `qwen` (F1 0,344 → 0,351), `mistral` (0,354 → 0,365) e le due
  `firstk_*` migliorano, perché il pavimento dà loro lo spazio che prima non usavano. È la prova
  che il confondente tagliava in entrambe le direzioni, non solo a favore dei lunghi.
- **Chi era già a lunghezza di riferimento resta in testa**: `primera` (0,446) e `pegasus`
  (0,424), non rigenerati e appena toccati dal soffitto, restano primo e secondo del benchmark.
  Il loro vantaggio non era di lunghezza.

**Avvertenze.**

- `bart` è l'**eccezione dichiarata**: 0,3% in banda, invariato, e va letto come tale (§3).
- `mistral` segue l'istruzione di lunghezza in modo molto disperso (p10 0,76×, p90 1,98×): 71%
  in banda contro il 99–100% di `gemma` e `gpt5mini`. È una proprietà del modello, misurata.
- **Content filter e filtro custom**: la severità del filtro è configurata **per risorsa**. Sul
  nuovo account la prima corsa copriva solo 5.250 righe su 5.610 (360 rifiuti, tutti
  `content_filter`) contro le 5.471 della corsa `test` sul vecchio. Attaccando al deployment un
  **filtro custom permissivo** e rilanciando le sole righe mancanti se ne sono recuperate 312:
  copertura finale **5.562/5.610 = 99,1%**, la più alta fra tutte le corse Azure del progetto, e
  intersezione comune ai 18 metodi a **5.527** cluster (sopra i 5.437 di partenza). Le 48 righe
  residue sono respinte anche dalla policy permissiva. La provenienza resta pulita: stessa
  risorsa, stesso modello, stesso prompt e stessi parametri — cambia solo la policy del filtro,
  che regola l'*accesso* e non altera la generazione. **Per riprodurre la corsa serve il filtro
  custom**, altrimenti si torna a ~93,6% di copertura.
- Sulla riga 50540 (sorgente degenere, un elenco di nomi di birrifici) il METEOR non limitato di
  pyAutoSummarizer esplode per `lda` e `lexrank`; stessa patologia già documentata per PEGASUS,
  stessa scelta: nessun numero alterato, la riga è segnalata.

## 5. Prossimi passi

1. Valutare se rigiudicare il **G-Eval** a lunghezza pari: è la metrica più contaminata dalla
   lunghezza (il giudice premia i testi lunghi), quindi quella che guadagnerebbe di più dal
   confronto a parità di parole. Riutilizzando i giudizi già in cache per le righe rimaste
   identiche (`bart` 100%, `pegasus` 90%, `primera` 79%) e campionando le righe in ordine
   casuale, il costo scende da ~€95 a ~€25–30. Dettagli e due bug da correggere prima: issue #16.
2. PR di chiusura della issue #16.
