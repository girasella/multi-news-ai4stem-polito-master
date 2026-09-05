# Multi-News — text mining e summarization

**Project work finale — master di II livello «Artificial Intelligence for STEM»,
Politecnico di Torino**

Questo repository ospita il project work costruito sul dataset di summarization
multi-documento **Multi-News** (Fabbri et al., 2019). È nato come copia del repository originale
del dataset, [`alexfabbri/multi_news`](https://huggingface.co/datasets/alexfabbri/multi_news)
sull'Hugging Face Hub, e lo estende con la curation del dataset, l'analisi esplorativa e gli
esperimenti di summarization. Il merito del dataset in sé va interamente agli autori originali —
vedi [Attribuzione, licenza e citazione](#attribuzione-licenza-e-citazione).

## Panoramica del progetto

Il progetto esplora Multi-News come corpus di text mining e summarization:

- **Analisi esplorativa dei dati** — un audit di qualità e struttura sull'intero corpus (tutti i
  56.216 esempi), pubblicato come dashboard autoconsistente
  ([multi_news_dashboard.html](multi_news_dashboard.html)) e ripreso nella documentazione.
- **Curation del dataset** — conversione dei file di testo grezzi nel formato `.tab` di Orange
  Data Mining, con un passo di pulizia che rimuove le righe con problemi noti di qualità della
  sorgente (scraping fallito, duplicati esatti, disallineamenti sorgente/riassunto) individuati
  dall'analisi esplorativa.
- **Esperimenti di summarization** — un benchmark di **18 metodi** sul corpus curato, valutati
  sull'intera split test pulita (5.610 esempi):
  - due baseline posizionali First-k / Lead (notebook 10);
  - nove estrattivi non supervisionati — TextRank e LexRank, Centroid-based (MEAD) + MMR in due
    varianti, LSA in due varianti, clustering su embedding SBERT in due varianti e LDA (notebook
    01/02, 11, 15–17);
  - tre abstractive specializzati — BART, PEGASUS, PRIMERA (notebook 03/04/06);
  - tre LLM generalisti eseguiti in locale via [ollama](https://ollama.com) — Qwen2.5-7B,
    Gemma 4 E4B, Mistral-7B (notebook 07–09);
  - un LLM cloud su Azure AI Foundry — GPT-5-mini (notebook 12).

  Un notebook per metodo in [notebooks/](notebooks/) (documentati in italiano), con i riassunti
  generati e le metriche salvati sotto [results/](results/), così che la valutazione possa essere
  rieseguita senza rigenerare i riassunti. Le metriche sono quelle di
  [pyAutoSummarizer](https://github.com/Valdecy/pyAutoSummarizer) — ROUGE-1/2/L, BLEU, METEOR —
  più due aggiunte in backfill separati: il **BERTScore** (`roberta-large`, notebook 13) e il
  **G-Eval** (LLM-as-a-Judge con giudice GPT-5.4-mini su Azure, notebook 14), l'unica metrica del
  benchmark che non passa dal riassunto di riferimento umano.

### Il risultato principale

Le due famiglie di metriche danno classifiche **opposte**, ed è il risultato più interessante del
benchmark. Sulle metriche ancorate al riferimento umano vincono gli abstractive specializzati,
addestrati proprio su Multi-News: PRIMERA ha il ROUGE-1 F1 più alto (0,444) e il BERTScore più
alto (0,874), seguito da PEGASUS (0,423 / 0,871). Sul G-Eval, che valuta coerenza, fedeltà
fattuale, fluidità e pertinenza **senza** guardare il riferimento, l'ordine si ribalta e passano
davanti gli LLM generalisti: GPT-5-mini 4,89 su 5, Gemma 4,78, Mistral 4,72, contro 4,28 di
PRIMERA e 4,04 di PEGASUS. In altre parole i modelli addestrati sul dataset riproducono bene lo
*stile* dei riassunti di riferimento, che è ciò che ROUGE misura, mentre un giudice indipendente
trova più leggibili e fedeli i riassunti degli LLM generalisti. Tabelle, grafici e avvertenze
metodologiche sono in [notebooks/05_confronto.ipynb](notebooks/05_confronto.ipynb).

## Contenuto del repository

| percorso | descrizione |
|------|-------------|
| [multi_news.py](multi_news.py) | Script di caricamento originale per `datasets` di Hugging Face (invariato) |
| [data/text/](data/) | File canonici del dataset, come rilasciati a monte — una coppia `.src.cleaned`/`.tgt` per split |
| [data/tab/](data/) | Copie Orange `.tab` **pulite**: una per split più `complete.tab` (tutte le split unite, con una colonna `split`) ed `excluded_rows.tsv` (elenco delle 115 righe scartate) |
| [scripts/convert_to_tab.py](scripts/convert_to_tab.py) | Rigenera `data/tab/` a partire da `data/text/` applicando i criteri di pulizia — documentato in [scripts/README.md](scripts/README.md) |
| [multi_news_dashboard.html](multi_news_dashboard.html) | Dashboard EDA autoconsistente — si apre direttamente nel browser (testo del report in italiano) |
| [notebooks/](notebooks/) | Notebook del benchmark di summarization: uno per metodo (18 slug), più il confronto (05) e i due backfill di metrica (13 BERTScore, 14 G-Eval), in italiano — vedi [notebooks/README.md](notebooks/README.md) |
| [scripts/run_benchmark_test.py](scripts/run_benchmark_test.py) | Driver non presidiato della corsa `test`: esegue in sequenza i notebook dei metodi, dal più veloce al più lento — documentato in [scripts/README.md](scripts/README.md) |
| [scripts/run_geval.py](scripts/run_geval.py) | Driver non presidiato del backfill G-Eval, con controllo di spesa e report di costo offline — documentato in [scripts/README.md](scripts/README.md) |
| [results/](results/) | Output del benchmark: campione di valutazione condiviso, riassunti generati, metriche per esempio e aggregate (comprese quelle G-Eval, in file dedicati, e la cache dei giudizi) |
| [requirements-notebooks.txt](requirements-notebooks.txt) | Dipendenze Python dei notebook del benchmark |
| [Multi-News_paper.md](Multi-News_paper.md) | Il paper originale (Fabbri et al., 2019), come riferimento — lasciato in inglese perché copia verbatim della pubblicazione |
| [Tecniche_MDS_non_LLM_MultiNews.md](Tecniche_MDS_non_LLM_MultiNews.md) | Rassegna ragionata delle tecniche di summarization multi-documento non-LLM rispetto alla lezione del PoliTO — il «documento-guida» citato dai notebook 11 e 15–17 |
| [data/README.md](data/README.md) | Documentazione dettagliata di formati dei file, statistiche e criteri di pulizia |

## Il dataset

Multi-News è composto da articoli di cronaca e dai relativi riassunti scritti professionalmente,
presi da [newser.com](https://www.newser.com). Ogni riassunto è redatto da redattori umani e cita
gli articoli originali. Ogni esempio ha due feature:

- `document`: il testo degli articoli sorgente, concatenati con il token separatore speciale
  `|||||`;
- `summary`: il riassunto multi-documento scritto da umani (comincia con `– `, la convenzione
  redazionale di newser.com).

| split | esempi (canonici, `data/text/`) | esempi (puliti, `data/tab/`) |
|-------|--------------------------------:|-----------------------------:|
| train      | 44.972 | 44.880 |
| validation |  5.622 |  5.611 |
| test       |  5.622 |  5.610 |
| **totale**  | **56.216** | **56.101** |

### In sintesi dall'analisi esplorativa

Dalla [dashboard](multi_news_dashboard.html) (calcolata su tutte le split aggregate; dettagli e
metodologia in [data/README.md](data/README.md)):

- 154.530 articoli sorgente in totale — media ≈ 2,75 per esempio, mediana 2; l'82% degli esempi
  ha ≤3 sorgenti.
- La lunghezza dell'input è fortemente asimmetrica a destra (mediana 1.319 parole, media ≈ 1.789,
  massimo 449.620), mentre i riassunti sono uniformi (mediana 220 parole, intervallo 34–973). Il
  rapporto di compressione mediano è ≈ 6,3×.
- La lunghezza del riassunto correla con il *numero* di sorgenti (la media cresce da ~176 a ~372
  parole passando da 1 a 9+ sorgenti) molto più che con la dimensione grezza dell'input.
- Problemi noti di qualità dei dati: 10 righe sorgente vuote, 637 esempi con ≤1 articolo
  sorgente, 77 righe sorgente duplicate esatte; gli outlier di lunghezza più estremi sono
  disallineamenti sorgente/riassunto dovuti a errori di scraping a monte. La copia Orange
  derivata in `data/tab/` esclude già queste righe sporche (115 scartate, elencate una per una in
  `data/tab/excluded_rows.tsv`); i file canonici in `data/text/` le mantengono.

## Usare i dati

**In Orange Data Mining** — caricare un qualsiasi file `data/tab/*.tab` con il widget *File* (il
widget *Corpus* dell'add-on [Orange3-Text](https://orangedatamining.com/widget-catalog/#text-mining)
abilita gli strumenti di text mining). Entrambe le colonne sono meta-attributi di tipo stringa;
`complete.tab` aggiunge una colonna discreta `split` (`train`/`val`/`test`), così l'intero corpus
può essere analizzato in una volta sola e filtrato o raggruppato per split.

**Con `datasets` di Hugging Face** — lo script di caricamento originale funziona ancora sui file
canonici:

```python
from datasets import load_dataset
dataset = load_dataset("path/to/multi_news.py")
```

**Dashboard EDA** — aprire [multi_news_dashboard.html](multi_news_dashboard.html) in un browser
qualsiasi; è completamente autoconsistente (non servono né server né rete).

Nota: i file di dati sono grandi (~750 MB in totale in `data/text/`, di cui ~520 MB il solo file
sorgente di train; ~1,3 GB la copia Orange in `data/tab/`); nello scrivere strumenti che li usano
conviene preferire letture in streaming, riga per riga.

## Attribuzione, licenza e citazione

Questo progetto si basa sul dataset **Multi-News** di Alexander R. Fabbri, Irene Li, Tianwei She,
Suyi Li e Dragomir R. Radev ([paper](https://arxiv.org/abs/1906.01749), LILY Lab, Yale
University). Il punto di partenza di questo repository è stato il repository del dataset su
Hugging Face [`alexfabbri/multi_news`](https://huggingface.co/datasets/alexfabbri/multi_news);
grazie a [@patrickvonplaten](https://github.com/patrickvonplaten),
[@lewtun](https://github.com/lewtun) e [@thomwolf](https://github.com/thomwolf) per averlo
originariamente aggiunto all'Hub.

Il dataset è rilasciato dal LILY Lab **esclusivamente per scopi di ricerca ed educativi non
commerciali**, "così com'è" e senza garanzie: questo repository lo usa rigorosamente entro quei
limiti, come project work didattico. Il **testo integrale e vincolante** del Dataset Usage
Agreement è in [LICENSE](LICENSE) e resta in inglese — questa è solo una sintesi di cortesia, in
caso di divergenza fa fede il testo originale.

```bibtex
@misc{alex2019multinews,
    title={Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model},
    author={Alexander R. Fabbri and Irene Li and Tianwei She and Suyi Li and Dragomir R. Radev},
    year={2019},
    eprint={1906.01749},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
