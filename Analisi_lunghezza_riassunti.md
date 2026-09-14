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
  calibrato su un pilota di 30 righe) arriva a 1,01 × riferimento. **Eseguito per `qwen`**
  (8 h); `gemma`, `mistral`, `gpt5mini` ricevono per ora il solo soffitto.
- **Solo soffitto per `bart`, `pegasus`, `primera`** (decine di ore di GPU per rigenerarli).

## 4. Risultato

| metodo | rigenerato | in banda | parole prima → dopo | R-1 recall prima → dopo | R-1 F1 prima → dopo |
|---|:-:|---:|---:|---:|---:|
| primera | no | 66% | 211 → 201 | 0,441 → 0,433 | **0,445 → 0,446** |
| pegasus | no | 52% | 178 → 175 | 0,390 → 0,387 | 0,424 → 0,424 |
| centroid_mmr | sì | 98% | 362 → 215 | 0,465 → 0,365 | 0,383 → **0,378** |
| lsa_steinberger | sì | 99% | 257 → 215 | 0,419 → 0,366 | 0,376 → 0,369 |
| firstk_psr | sì | 98% | 217 → 215 | 0,347 → 0,354 | 0,355 → 0,368 |
| firstk_nltk | sì | 98% | 219 → 215 | 0,348 → 0,352 | 0,356 → 0,368 |
| centroid_mmr_bert | sì | 98% | 361 → 215 | 0,460 → 0,357 | 0,373 → 0,364 |
| sbert_kmeans | sì | 98% | 264 → 215 | 0,404 → 0,358 | 0,359 → 0,359 |
| textrank | sì | 98% | 368 → 215 | 0,448 → 0,343 | 0,370 → 0,359 |
| mistral | no | 40% | 161 → 158 | 0,325 → 0,322 | 0,354 → 0,355 |
| gpt5mini | no | 64% | 241 → 210 | 0,380 → 0,364 | 0,348 → 0,352 |
| lexrank | sì | 97% | 450 → 216 | 0,482 → 0,338 | 0,365 → 0,352 |
| lsa | sì | 98% | 249 → 215 | 0,357 → 0,322 | 0,351 → 0,350 |
| qwen | sì | 79% | 140 → 216 | 0,309 → 0,347 | 0,344 → 0,351 |
| gemma | no | 73% | 292 → 242 | 0,390 → 0,363 | 0,335 → 0,340 |
| sbert_agglom | sì | 99% | 216 → 215 | 0,350 → 0,342 | 0,328 → 0,339 |
| lda | sì | 93% | 477 → 230 | **0,499 → 0,348** | 0,351 → **0,338** |
| bart | no | 0% | 55 → 55 | 0,181 → 0,181 | 0,265 → 0,265 |

- **Contenimento riuscito dove applicato**: gli 11 estrattivi cadono in banda nel 93–98,5% dei
  cluster, con lunghezza mediana pari al riferimento (~215 parole per tutti); `qwen`, col
  budget nel prompt, nel 79% (mediana 1,01×) e ne guadagna in F1 (0,344 → 0,351) e METEOR
  (0,32 → 0,41). Il rapporto max/min dentro il cluster scende da 8,5× a 4,7×, e il residuo è
  interamente dei metodi corti non rigenerati (`bart` 0,26×, `mistral` 0,75× del riferimento).
- **La graduatoria degli estrattivi si riassesta.** `lda` perde il primato di recall
  (0,499 → 0,348) e finisce **ultimo in F1**: il suo vantaggio era interamente di lunghezza.
  `lexrank`, `textrank`, `centroid_mmr` perdono ~0,10 di recall ma guadagnano precisione;
  `centroid_mmr` resta il miglior estrattivo in F1 (0,378), e le due baseline `firstk_*`
  salgono dal 9°–10° al 3°–4° posto (0,368).
- **Chi era già sotto il riferimento e non è stato rigenerato non cambia**: `primera` (0,446) e `pegasus` (0,424) restano
  in testa al benchmark — un vantaggio che quindi non era di lunghezza.
- Avvertenza: sulla riga 50540 (sorgente degenere, un elenco di nomi di birrifici) il METEOR
  non limitato di pyAutoSummarizer esplode per `lda` e `lexrank`; stessa patologia già
  documentata per PEGASUS, stessa scelta: nessun numero alterato, la riga è segnalata.

## 5. Prossimi passi

1. Pilota di 30 righe e corsa completa per `gemma` e `mistral` (ore per modello) e `gpt5mini`
   (Azure), ricalibrando il fattore per modello se necessario.
2. Eventualmente `bart`/`pegasus` con `min_length`/`max_length` (2 h e 5–9 h di GPU).
