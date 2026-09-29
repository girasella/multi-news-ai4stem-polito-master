# Archivio: benchmark LLM locali di Federica (LM Studio)

Materiale **conservato come ricevuto** dalla corsa originale di Federica (Mac M4 24GB,
LM Studio, 2026-07-16). Non va eseguito né modificato: le versioni integrate nel benchmark
del repository sono i notebook [07_qwen](../07_qwen.ipynb), [08_gemma](../08_gemma.ipynb) e
[09_mistral](../09_mistral.ipynb) (adattati a ollama e alle convenzioni condivise di
`summ_utils`).

| File | Contenuto |
|---|---|
| `Summarization_LLM_Evaluation.docx` | Relazione di Federica: scelta dei modelli, parametri, risultati e commenti |
| `qwen2.5_7B.ipynb` | Corsa `qwen2.5-7b-instruct-mlx` (unico con `max_tokens=200`) |
| `gemma-4-e4b.ipynb` | Corsa `google/gemma-4-e4b` |
| `mistral-small3-7B.ipynb` | Corsa `mistralai/mistral-7b-instruct-v0.3` (⚠️ il nome del file è improprio: il modello invocato non è Mistral Small 3) |
| `{qwen,gemma,mistral}_summary_evaluation_results.csv` | Risultati per documento: riassunto generato, riferimento, metriche |
| `20_preliminary_fs_test.ipynb` | Few-shot, esplorazione (ollama, qwen): 4 embedding × k∈{2,4,6} sul campione da 100, token dei prompt, 3 varianti di prompt — vedi sotto |
| `21_fsprompt_qwen.ipynb` | Few-shot, corsa completa sul test (`sbert_mpnet_en`, k=4, prompt `new`; 39 h su Mac) — vedi sotto |

## Few-shot (notebook 20 e 21, aggiunti a settembre 2026)

Gli esempi del prompt sono le coppie (articolo, riassunto umano) dei k cluster del **train**
più simili a quello da riassumere. Le versioni integrate nel benchmark sono
[20_fewshot_pilota](../20_fewshot_pilota.ipynb) e [21_qwen_fewshot](../21_qwen_fewshot.ipynb)
(slug `qwen_fewshot`); il prompt resta quello di Federica, copiato alla lettera in
`su.prompt_fewshot`. La corsa del notebook 21 archiviato **non è stata importata** in
`results/`, per due motivi:

- **row_id non confrontabili.** `test.tab` letto con pandas trasforma le due righe di
  intestazione del formato Orange (`string/string`, `meta/meta`) in dati (5.612 righe invece di
  5.610) e il row_id è l'indice locale 0..5611, che nel resto del progetto indica righe del
  *train*: gli id del test sono gli indici globali di `complete.tab` (50491..56100). Lo stesso
  vale per `train-2.tab` (44.882 = 44.880 + 2), quindi fra gli esempi recuperabili c'erano
  anche le due righe d'intestazione.
- **Contesto di ollama non impostato.** L'endpoint OpenAI-compatibile non accetta `num_ctx`:
  vale il default del server. I prompt con k=4 hanno mediana ~11.800 token e p95 ~28.000
  (cella del conteggio token del notebook 20); con un default di 2–4k token ollama tronca il
  prompt in silenzio. Su questa macchina (ollama 0.34, default 32.768) un documento di
  10.856 token arriva intero, ma la versione e il default del Mac della corsa non sono noti.

Nel notebook 20 archiviato, inoltre, la configurazione migliore era `sbert_mpnet_en` con
**k=2**, e a k=2 il prompt `old` batteva `new` su ROUGE e METEOR per tutti e tre gli
embedding provati, mentre la corsa del 21 usa k=4 e `new`. Le differenze restano comunque
dentro il rumore (ROUGE-1 F1 fra 0,283 e 0,298 con errore standard ~0,009): il pilota
integrato le rimisura su righe di validation, con il controllo zero-shot e le differenze
appaiate.

## Provenienza dei dati e integrazione nel repository

I tre CSV sono stati la **fonte** della prima versione dei riassunti committati in
`results/summaries/{metodo}_sample.tsv`, importati tramite
[`scripts/import_llm_results.py`](../../scripts/README.md): le loro righe sono allineate
1:1, in ordine, con `results/sample/sample_100_seed42.tsv` (`doc_index` i = riga i del campione;
verificato confrontando tutti i `reference_summary`). I file committati in `results/` sono stati
poi **rigenerati da zero via ollama** dai notebook 07–09 (qwen/gemma 2026-07-16, mistral
2026-07-17) e non provengono più da questi CSV.

⚠️ La cella di caricamento dati nei notebook mostra una **variante precedente** che leggeva il
mirror HF `Awesome075/multi_news_parquet`: non riflette la corsa che ha prodotto i CSV, che è
avvenuta sul campione condiviso. Fa fede il contenuto dei CSV.

Avvertenze sui contenuti:

- **gemma**: 81 risposte vuote su 100 → solo 19 esempi valutati;
- le **metriche nei CSV** (ROUGE/BLEU/METEOR) sono state calcolate con impostazioni di
  normalizzazione diverse da quelle condivise del benchmark (stopword, accenti e numeri
  rimossi): i valori in `results/metrics/` (oggi riferiti alle corse ollama) usano la pipeline
  comune e i valori dei CSV non vanno mescolati con quelli;
- i CSV contengono anche **BERTScore** (roberta-large), metrica che la pipeline del repository
  non calcola: al momento non è stata portata in `results/` (possibile estensione futura per
  tutti i metodi).
