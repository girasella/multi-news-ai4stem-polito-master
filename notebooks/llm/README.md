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

## Provenienza dei dati e integrazione nel repository

I tre CSV sono stati la **fonte** della prima versione dei riassunti committati in
`results/summaries/{metodo}_sample.tsv`, importati tramite
[`scripts/import_llm_results.py`](../../scripts/README.md): le loro righe sono allineate
1:1, in ordine, con `results/sample/sample_100_seed42.tsv` (`doc_index` i = riga i del campione;
verificato confrontando tutti i `reference_summary`). I file committati in `results/` sono stati
poi **rigenerati da zero via ollama** dai notebook 07–09 (2026-07-16) e non provengono più da
questi CSV — per mistral con un modello diverso (Mistral Small ~24B invece del 7B v0.3, vedi
[09_mistral](../09_mistral.ipynb)).

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
