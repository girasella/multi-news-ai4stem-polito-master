# Contenuto di `data/`

File di dati grezzi su cui poggia il loader del dataset Multi-News
([multi_news.py](../multi_news.py)). Questo documento descrive che cosa c'è davvero in questi
file, sulla base di un'ispezione diretta — vedi il [README.md](../README.md) per la citazione e la
sintesi della licenza, e [Multi-News_paper.md](../Multi-News_paper.md) (Fabbri et al., 2019,
arXiv:1906.01749) per il paper originale che ha introdotto il dataset. Questo repository ospita
solo il dataset in sé (questi file più lo script di caricamento), non il modello di summarization
Hi-MAP né il codice di addestramento che il paper descrive. Per le statistiche sull'intero corpus
con i grafici, vedi la dashboard EDA autoconsistente
[multi_news_dashboard.html](../multi_news_dashboard.html) (testo del report in italiano); i suoi
risultati principali sono ripresi nelle sezioni che seguono.

## Provenienza (dal paper)

- Sorgenti e riassunti sono raccolti via scraping da **newser.com**: ogni `summary` è una sintesi
  di un fatto di cronaca scritta professionalmente da un redattore, e ogni articolo sorgente in
  `document` è uno dei pezzi che il redattore ha citato. Circa 20 redattori firmano l'85% di tutti
  i riassunti.
- I documenti sorgente provengono da **oltre 1.500 testate distinte** (ciascuna presente almeno 5
  volte), il che rende questo dataset più vario, come sorgenti, dei precedenti dataset di
  summarization giornalistica (CNN/DailyMail, per esempio, attinge da due sole testate).
- Il paper riporta di aver raccolto oltre 250.000 link archiviati su Wayback, tenendo poi solo i
  cluster con **da 2 a 10 articoli sorgente** per riassunto, per un totale di 56.216 coppie
  divise 80/10/10 in train/validation/test (44.972 / 5.622 / 5.622) — esattamente le dimensioni
  delle split presenti in questo repository.
- La distribuzione del numero di sorgenti pubblicata nel paper (la sua Tabella 2, la cui somma fa
  esattamente 44.972, cioè descrive la split **train**) è vicina ma non identica bit per bit a
  quella misurata direttamente su `text/train.src.cleaned` qui sotto (per esempio il paper riporta
  23.894 esempi con 2 sorgenti contro i 23.741 trovati qui; il paper non riporta alcun esempio con
  0 o 1 sorgente, avendo filtrato a 2–10). I dati di questo repository non sono quindi
  un'istantanea congelata dei numeri esatti pubblicati nel paper: comprendono un piccolo numero di
  esempi con 0 e 1 sorgente, fuori dal filtro 2–10 dichiarato (vedi sotto), il che è coerente con
  ulteriori rielaborazioni e con il *link rot* intercorso fra il paper e questo rilascio.
- Le medie sull'intero corpus riportate dal paper (Tabella 3, dipendenti da spaziatura e
  tokenizer): i documenti hanno in media **2.103 parole / 82,7 frasi**, i riassunti **263,7 parole
  / 10,0 frasi**, con un vocabolario di 666.515 elementi. I conteggi di parole ottenuti con un
  semplice `str.split()` sui file di questo repository risultano un po' più bassi (vedi
  «Lunghezza della sorgente» e «Lunghezza del riassunto» sotto): i numeri del paper vanno
  considerati il riferimento normalizzato dal tokenizer e quelli qui un controllo rapido e
  indipendente dal tokenizer, non una contraddizione.
- Il paper riporta inoltre che questi riassunti sono astrattivi in misura paragonabile ai dataset
  giornalistici a documento singolo: solo il 17,8% degli unigrammi e l'82,3% dei 4-grammi di un
  riassunto sono *nuovi* (cioè assenti dai documenti sorgente), il che significa che i riassunti
  si appoggiano molto a espressioni copiate dal testo sorgente invece di essere pure parafrasi.

## File

`data/` è organizzata in due sottocartelle:

- `data/text/` — il formato originale a sei file: un `.src.cleaned` (articoli sorgente) e un
  `.tgt` (riassunto) per ogni split, per tutti i 56.216 esempi. È la copia canonica e autorevole,
  mantenuta **come rilasciata** (righe sporche note comprese).
- `data/tab/` — file `.tab` di Orange Data Mining generati da `data/text/` con
  [scripts/convert_to_tab.py](../scripts/convert_to_tab.py): uno per split (`train.tab`,
  `val.tab`, `test.tab`) più `complete.tab`, con le tre split unite e una colonna `split` che ne
  registra l'origine. Questa copia è **pulita**: 115 righe con problemi noti di qualità della
  sorgente sono escluse (56.101 esempi in totale), quindi *non* è allineata riga per riga con
  `data/text/`. Vedi «Formato di `data/tab/*.tab`» più sotto.

### `data/text/`

| file                        |  righe | dimensione (byte) | dimensione |
|-----------------------------|-------:|------------------:|------------|
| `text/train.src.cleaned`    | 44.972 |       547.512.283 | ~522 MB    |
| `text/train.tgt`            | 44.972 |        58.793.912 | ~56 MB     |
| `text/val.src.cleaned`      |  5.622 |        66.875.522 | ~64 MB     |
| `text/val.tgt`              |  5.622 |         7.295.302 | ~7,0 MB    |
| `text/test.src.cleaned`     |  5.622 |        68.999.509 | ~66 MB     |
| `text/test.tgt`             |  5.622 |         7.309.099 | ~7,0 MB    |

Tutti i file sono testo UTF-8 con terminatori di riga Unix (`\n`), senza BOM. All'interno di una
split, `.src.cleaned` e `.tgt` sono allineati 1:1 riga per riga — la riga *i* del file sorgente si
accoppia con la riga *i* del file target a formare un esempio (è esattamente ciò su cui si basa
`_generate_examples` in `multi_news.py`, che fa lo `zip()` dei due file riga per riga).

### `data/tab/`

| file                    | righe  | dimensione (byte) | dimensione |
|-------------------------|-------:|------------------:|------------|
| `tab/train.tab`         | 44.880 |       554.038.300 | ~528 MB    |
| `tab/val.tab`           |  5.611 |        67.629.026 | ~64 MB     |
| `tab/test.tab`          |  5.610 |        67.824.837 | ~65 MB     |
| `tab/complete.tab`      | 56.101 |       689.811.869 | ~658 MB    |
| `tab/excluded_rows.tsv` |    115 |             3.761 | ~4 KB      |

I conteggi sono di righe di dati (escluse le 3 righe di intestazione Orange). `complete.tab` è
l'unione delle tre split in un solo file, con una colonna `split` in più che registra l'origine di
ogni riga (vedi sotto). Le 115 righe scartate rispetto a `data/text/` (92 train, 11 val, 12 test)
sono elencate una per una, con il motivo, in `tab/excluded_rows.tsv` — vedi «Pulizia» dentro
«Formato di `data/tab/*.tab`» più sotto.

## Formato di `data/text/*.src.cleaned`

Ogni riga è l'intero input sorgente di un esempio: uno o più articoli di cronaca sullo stesso
fatto, concatenati con il token separatore `` ||||| `` (cinque pipe, con spazi ai lati). Il
separatore resta come sottostringa letterale nel campo `document` prodotto dal loader: non viene
spezzato in una lista.

Dentro il testo di un singolo articolo, i ritorni a capo veri sono stati sostituiti con il token
letterale `NEWLINE_CHAR`, così che ogni esempio occupi comunque esattamente una riga fisica del
file. Il loader li riporta a veri caratteri `\n` in fase di lettura (`multi_news.py:109`).

**Articoli per esempio** (separando su `` ||||| ``), su tutte e tre le split:

| n. articoli | train  | val   | test  |
|------------:|-------:|------:|------:|
| 1           |    498 |    58 |    71 |
| 2           | 23.741 | 3.066 | 3.022 |
| 3           | 12.577 | 1.555 | 1.540 |
| 4           |  4.921 |   610 |   609 |
| 5           |  1.846 |   195 |   219 |
| 6           |    706 |    79 |    96 |
| 7           |    371 |    38 |    40 |
| 8           |    194 |    13 |    15 |
| 9           |     81 |     7 |     8 |
| 10          |     29 |     0 |     1 |
| 0 (vuoto)   |      8 |     1 |     1 |

La mediana è di 2 articoli per esempio, la media ~2,7–2,8. Corrisponde all'incirca alla
distribuzione riportata dal paper (vedi «Provenienza» sopra) ma non è identica e — a differenza
del filtro 2–10 sorgenti dichiarato nel paper — questo rilascio contiene anche righe con 1 o 0
articoli:

- Una manciata di esempi (8 in train, 1 in val e 1 in test) ha la riga sorgente **completamente
  vuota**: il riassunto corrispondente esiste ancora e si legge normalmente, quindi sembrano
  fallimenti di scraping a monte (per esempio un articolo citato che non risolve più) più che
  righe corrotte.
- Un gruppo più numeroso (498 train, 58 val, 71 test) ha esattamente 1 articolo: righe a documento
  singolo dentro un dataset nominalmente multi-documento.

Chi consuma questi dati dando per scontato un `document` non vuoto, o almeno 2 articoli sorgente
per esempio secondo la metodologia del paper, deve tenere conto di questi casi limite.

**La lunghezza della sorgente** è fortemente asimmetrica: aggregando tutte le split, la mediana è
di 1.319 parole per esempio (media ≈ 1.789, p95 ≈ 4.599 con una semplice separazione sugli spazi;
il paper riporta una media normalizzata dal tokenizer di 2.103 parole / 82,7 frasi per cluster di
documenti), ma una lunga coda di esempi arriva a decine o centinaia di migliaia di parole. Non
dare per scontata una dimensione di input limitata quando si scrivono strumenti su questo campo.

Gli outlier estremi di lunghezza non sono semplicemente «cluster di notizie lunghi»: l'ispezione
manuale mostra che sono **disallineamenti sorgente/riassunto**. Il più grande in assoluto (riga
22256 di train, 449.620 parole) è il programma completo di un convegno accademico (Society of
Biblical Literature Annual Meeting, Atlanta 2015 — centinaia di abstract concatenati), mentre il
riassunto abbinato è una normale sintesi di 319 parole su una storia del tutto diversa (un
frammento di papiro del Nuovo Testamento venduto su eBay): per quella riga non c'è alcuna
relazione semantica fra le due colonne. La causa probabile è un errore di scraping o di link a
monte (per esempio un link Wayback che risolve alla pagina sbagliata); gli altri outlier di testa
(tutti oltre le 100k parole, per esempio `train:26686` con 168.796 e `test:4403` con 145.130) sono
plausibilmente casi analoghi. Chi si addestra su questi dati dovrebbe valutare di *filtrare* gli
esempi con sorgente anomala, non solo di troncare la lunghezza dell'input: una porzione troncata
di un documento irrilevante non porta comunque alcun segnale verso il riassunto target.

## Formato di `data/text/*.tgt`

Ogni riga è l'unico riassunto multi-documento scritto da umani per la riga sorgente
corrispondente: testo semplice, senza token `NEWLINE_CHAR` (i riassunti sono di un solo
paragrafo). Ogni riassunto, in ogni split, comincia con una lineetta (`– `): è la convenzione
redazionale di newser.com per queste sintesi, non un artefatto di elaborazione.

La lunghezza dei riassunti è molto più uniforme di quella delle sorgenti: mediana di ~218–221
parole con una semplice separazione sugli spazi (il paper riporta una media normalizzata dal
tokenizer di 263,7 parole / 10,0 frasi), con un intervallo di circa 34–973 parole (train), più
stretto per val e test. Il paper osserva che questa media di ~260 parole è nettamente più alta di
quella dei dataset comparabili di summarization a documento singolo (CNN/DailyMail sta in media
sulle ~56 parole), il che rende la generazione fluente e coerente su un output lungo una sfida
specifica di questo dataset.

## Formato di `data/tab/*.tab`

È il formato nativo di Orange Data Mining, delimitato da tabulazioni — un file per split
(`train.tab`, `val.tab`, `test.tab`) più `complete.tab` che le unisce, generati dalle coppie di
`data/text/` da [scripts/convert_to_tab.py](../scripts/convert_to_tab.py). Ogni riga è un esempio:
lo stesso contenuto `document`/`summary` di `data/text/`, con `NEWLINE_CHAR` già riportato a veri
`\n` e il separatore di articoli `` ||||| `` ancora presente in `document`.

**`complete.tab`** contiene l'intero dataset (pulito) — le tre split concatenate nell'ordine
train → val → test, le stesse righe dei file per split — con una terza colonna `split` (dichiarata
`discrete`, valori `train`/`val`/`test`) che registra da quale split proviene ogni riga, così che
l'origine non si perda nell'unione e possa essere usata in Orange per filtrare o raggruppare:

```
document	summary	split
string	string	discrete
meta	meta	meta
testo del documento...	testo del riassunto...	train
```

**Pulizia.** Il convertitore scarta le righe la cui *sorgente* presenta i problemi di qualità
individuati dalla dashboard EDA (vedi «Statistiche sull'intero corpus» sotto), quindi i file
`.tab` contengono 56.101 dei 56.216 esempi e **non sono allineati riga per riga** con
`data/text/`. Una riga viene esclusa quando la sua sorgente (conteggi di parole via `str.split()`,
esclusi `NEWLINE_CHAR` e `` ||||| ``) è:

1. **più corta di 50 parole** (55 righe, comprese le 10 sorgenti completamente vuote) —
   probabili scraping falliti;
2. **più lunga di 100.000 parole** (8 righe) — gli outlier estremi la cui sorgente è
   semanticamente slegata dal riassunto (vedi la nota sul disallineamento più sopra);
3. **un duplicato esatto** (SHA-1 con spaziatura normalizzata) di una sorgente precedente,
   scandendo train → val → test: si tiene solo la prima occorrenza, il che elimina anche il
   *leakage* train/eval dei gruppi di duplicati a cavallo fra split (52 righe portano questa
   etichetta; le altre 25 righe ridondanti erano già escluse dalla regola 1, dato che le sorgenti
   vuote o troncate si duplicano fra loro).

Ogni riga esclusa è elencata in `tab/excluded_rows.tsv` (colonne: `split`, `line` — indice a base
0 nei file di `data/text/` — e `reason`). I riassunti non sono mai un criterio di scarto: i 12
riassunti più lunghi di 600 parole e i 637 esempi a sorgente singola segnalati dalla dashboard
vengono mantenuti, perché sono contenuto legittimo.

I file `.tab` di Orange hanno un'intestazione di 3 righe (nomi degli attributi, tipi, flag)
seguita dalle righe di dati. Entrambe le colonne sono dichiarate di tipo `string` con il flag
`meta` (campi di testo descrittivo, non una classe target — questo è un dataset di summarization,
non di classificazione):

```
document	summary
string	string
meta	meta
testo del documento...	testo del riassunto...
```

Le righe sono scritte con il modulo `csv` di Python (delimitate da tabulazioni, con quoting),
quindi i valori di `document` che contengono ritorni a capo o tabulazioni letterali sopravvivono
correttamente al giro di andata e ritorno — ed è una cosa da sapere, perché `Orange.data.Table`
interpreta i file `.tab` allo stesso modo (via `csv.reader`) e non spezzando ingenuamente sulle
righe.

Questi file sono **derivati, non mantenuti a mano**: se `data/text/` dovesse cambiare, rigenerare
`data/tab/` rilanciando `python scripts/convert_to_tab.py` dalla radice del repository, invece di
modificare direttamente i `.tab`. Lo script è documentato in
[scripts/README.md](../scripts/README.md).

## Statistiche sull'intero corpus (EDA)

I numeri qui sotto vengono dalla dashboard EDA
([multi_news_dashboard.html](../multi_news_dashboard.html)), calcolati in streaming su tutte e tre
le split aggregate: 56.216 esempi, 154.530 articoli sorgente. Metodologia: i conteggi di parole
sono indipendenti dal tokenizer (`str.split()`, con `NEWLINE_CHAR` ripristinato e `` ||||| ``
escluso prima del conteggio); i conteggi di frasi sono euristici (separazione su `[.!?]+`, quindi
le abbreviazioni li gonfiano leggermente); i duplicati sono rilevati con SHA-1 sul testo
normalizzato, i quasi-duplicati con un'impronta delle prime 15 parole; i riferimenti alle righe
usano indici `split:riga` a base 0. Questi valori sono rigenerabili con
`python scripts/analyze_dataset.py`, che scrive `scripts/dataset_stats.json`; il JSON incorporato
nella dashboard (`const D` nel suo script inline) è costruito da quel file e vi corrisponde.

| metrica                 |   media | mediana |  min |     max |    p05 |   p95 |
|-------------------------|--------:|--------:|-----:|--------:|-------:|------:|
| parole / input          | 1.788,8 |   1.319 |    0 | 449.620 |    356 | 4.599 |
| frasi / input           |   102,2 |      73 |    0 |  21.417 |     20 |   263 |
| parole / riassunto      |   218,0 |     220 |   34 |     973 |    109 |   318 |
| frasi / riassunto       |    11,1 |      11 |    1 |      55 |      5 |    18 |
| compressione (in÷out)   |   8,19× |   6,33× |   0× | 1.409×  | 2,00× | 19,4× |

Vocabolario stimato (token unici separati da spazi): 494.577 (il valore normalizzato dal tokenizer
riportato dal paper è 666.515). Le metriche di astrattività/estrattività (n-grammi nuovi,
copertura e densità dei frammenti) e il riconoscimento della lingua **non** sono stati ricalcolati:
dove rilevanti si citano i valori del paper. Lingua: sia le sorgenti sia i riassunti sono in
larghissima maggioranza (se non esclusivamente) in **inglese** (newser.com è un aggregatore
editoriale anglofono).

**Integrità e ridondanza** (percentuali sui 56.216 esempi del corpus):

- Righe sorgente vuote: 10 (0,018%); riassunti vuoti: 0.
- Riassunti duplicati esatti: 0 — tutti i 56.216 riassunti sono unici. Riassunti quasi duplicati
  (impronta delle prime 15 parole): 3 righe ridondanti in 3 gruppi.
- **Sorgenti** duplicate esatte: 77 righe ridondanti in 20 gruppi (0,137%) — solo 56.139 dei testi
  sorgente sono unici. Se si rifanno le split o si ricampiona questo dataset, deduplicare prima le
  sorgenti per evitare *leakage* train/eval.
- Esempi con ≤1 articolo sorgente: 637 (1,13%) — non multi-documento, nonostante la premessa del
  dataset.
- Rispetto a soglie di buon senso: 0 riassunti sotto le 20 parole, 12 sopra le 600; 55 righe
  sorgente sotto le 50 parole (probabili scraping falliti).

**Correlazioni** (su tutti gli esempi; le coppie sulla compressione solo dove il riassunto non è
vuoto):

| coppia di variabili                | Pearson r | Spearman ρ |
|------------------------------------|----------:|-----------:|
| n. sorgenti vs parole riassunto    |     0,428 |      0,360 |
| n. sorgenti vs compressione        |     0,190 |      0,316 |
| parole input vs parole riassunto   |     0,209 |      0,404 |

La relazione più netta è che i riassunti crescono con il *numero* di sorgenti (la lunghezza media
sale in modo grosso modo monotono da ~176 parole con 1 sorgente a ~372 con 9), e con esso cresce
anche la compressione (da ~3× a ~20×). Al contrario, la lunghezza del riassunto è in larga parte
indipendente dalla *dimensione* grezza dell'input: la r di Pearson è abbassata dalle sorgenti
enormemente fuori scala, abbinate a riassunti che restano nella fascia ~150–300 parole.

## Note pratiche

- I file sono grandi (`data/text/train.src.cleaned` da solo pesa ~522 MB, `data/tab/train.tab`
  ~528 MB): come indicato nel `CLAUDE.md` alla radice, conviene preferire letture in streaming
  riga per riga, o campionamenti, invece di caricare un intero file in memoria.
- Poiché `.src.cleaned` e `.tgt` sono accoppiati unicamente dalla posizione della riga, qualunque
  strumento che filtri, ordini o deduplichi uno dei due file deve applicare l'operazione identica
  anche all'altro, altrimenti l'allineamento si rompe silenziosamente.
- Nel calcolare statistiche o campioni sui file di `data/text/`, mettere in conto e gestire: righe
  sorgente vuote, il separatore `` ||||| `` che compare dentro `document` e il token
  `NEWLINE_CHAR` se si legge il file grezzo invece di passare da `multi_news.py`.
