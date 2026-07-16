# coding=utf-8
"""One-off import of Federica's local-LLM benchmark results into results/.

Her LM Studio runs (Mac M4, 2026-07-16) live in notebooks/llm/ as
{qwen,gemma,mistral}_summary_evaluation_results.csv: 100 rows each, aligned 1:1
by position with results/sample/sample_100_seed42.tsv (verified here before
writing anything — every reference_summary must match the sample's summary).

For each model this script:

1. writes results/summaries/{method}_sample.tsv in the repo format
   (row_id, generated_summary — via summ_utils.ScrittoreRiassunti), skipping
   rows whose generation failed (empty or "Error:..." content; gemma has 81);
2. recomputes the metrics with the repo pipeline (summ_utils.valuta_e_salva),
   because her CSVs used a different pyAutoSummarizer normalization
   (stopwords/accents/numbers removed) and their ROUGE/BLEU/METEOR values are
   NOT comparable with the other methods of this benchmark. Her BERTScore
   columns are not carried over (the repo pipeline doesn't compute BERTScore);
   they remain available in the archived CSVs.

The script refuses to overwrite an existing results/summaries/{method}_sample.tsv:
those files may since contain rows regenerated via ollama (notebooks 07-09), and
appending imported LM Studio rows would silently mix backends. Delete the file
first if you really want to re-import.
"""

import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "notebooks"))

import summ_utils as su  # noqa: E402  (needs the sys.path tweak above)

LLM_DIR = BASE_DIR / "notebooks" / "llm"
SAMPLE_PATH = BASE_DIR / "results" / "sample" / "sample_100_seed42.tsv"
SUMMARIES_DIR = BASE_DIR / "results" / "summaries"
METRICS_DIR = BASE_DIR / "results" / "metrics"

PROMPT_SYSTEM = ("You are a helpful assistant that summarizes news articles "
                 "from different sources concisely.")
PROMPT_USER = "Summarize the following document into a comprehensive summary: {documento}"

# Provenance shared by all three runs; per-model fields below. The 'note'
# fields are in Italian for consistency with the other aggregate JSONs.
_BACKEND = "LM Studio (API OpenAI-compatibile), Mac M4 24GB — corsa di Federica, 2026-07-16"
_NOTE_COMUNE = ("riassunti importati da notebooks/llm/{csv} tramite "
                "scripts/import_llm_results.py; documento passato grezzo "
                "(separatore ||||| incluso); metriche ricalcolate con la "
                "normalizzazione condivisa del benchmark (i valori nel CSV "
                "originale usavano impostazioni diverse e non sono confrontabili)")

MODELLI = {
    "qwen": {
        "csv": "qwen_summary_evaluation_results.csv",
        "config": {
            "checkpoint": "qwen2.5-7b-instruct-mlx",
            "backend": _BACKEND,
            "max_tokens": 200,
            "temperature": 0.3,
            "prompt_system": PROMPT_SYSTEM,
            "prompt_user": "/no_think\n\n" + PROMPT_USER,
            "note": "unico modello con max_tokens=200 (gli altri 300)",
        },
    },
    "gemma": {
        "csv": "gemma_summary_evaluation_results.csv",
        "config": {
            "checkpoint": "google/gemma-4-e4b",
            "backend": _BACKEND,
            "max_tokens": 300,
            "temperature": 0.3,
            "prompt_system": PROMPT_SYSTEM,
            "prompt_user": "/no_think\n\n" + PROMPT_USER,
            "note": ("81 risposte vuote su 100: solo 19 esempi valutati — medie "
                     "NON confrontabili direttamente con gli altri metodi"),
        },
    },
    "mistral": {
        "csv": "mistral_summary_evaluation_results.csv",
        "config": {
            "checkpoint": "mistralai/mistral-7b-instruct-v0.3",
            "backend": _BACKEND,
            "max_tokens": 300,
            "temperature": 0.3,
            "prompt_system": PROMPT_SYSTEM + " (inviato come messaggio user, non system)",
            "prompt_user": PROMPT_USER,
            "note": ("il notebook di origine si chiama 'mistral-small3-7B' ma il "
                     "modello effettivamente invocato e' mistral-7b-instruct-v0.3"),
        },
    },
}


def _norm(text):
    return " ".join(text.split())


def _riga_valida(generated):
    return bool(generated) and generated.strip() and not generated.startswith("Error:")


def importa(metodo, cfg, campione):
    csv_path = LLM_DIR / cfg["csv"]
    out_path = SUMMARIES_DIR / f"{metodo}_sample.tsv"
    if out_path.exists():
        sys.exit(f"ERROR: {out_path} already exists — delete it first to re-import "
                 f"(refusing to mix imported rows into an existing file).")

    with open(csv_path, encoding="utf-8", newline="") as f:
        righe = list(csv.DictReader(f))
    assert len(righe) == len(campione), (
        f"{metodo}: {len(righe)} CSV rows vs {len(campione)} sample rows")

    # Positional alignment check: doc_index i must be the sample's i-th row.
    for i, (r, es) in enumerate(zip(righe, campione)):
        assert int(r["doc_index"]) == i, f"{metodo}: doc_index out of order at {i}"
        assert _norm(r["reference_summary"]) == _norm(es["summary"]), (
            f"{metodo}: reference mismatch at doc_index {i} "
            f"(row_id {es['row_id']}) — CSV is not aligned with the sample")

    scrittore = su.ScrittoreRiassunti(out_path)
    scartate = 0
    for r, es in zip(righe, campione):
        if _riga_valida(r["generated_summary"]):
            scrittore.scrivi(es["row_id"], r["generated_summary"])
        else:
            scartate += 1
    scrittore.chiudi()
    print(f"{metodo}: {len(righe) - scartate} riassunti importati, "
          f"{scartate} righe scartate (generazione fallita) -> {out_path}")

    config = dict(cfg["config"])
    config["note"] = (config["note"] + "; " +
                      _NOTE_COMUNE.format(csv=cfg["csv"]))
    riassunti = su.carica_riassunti(out_path)
    su.valuta_e_salva(campione, riassunti, metodo, "sample", METRICS_DIR, config)


def main():
    campione = su.carica_campione(SAMPLE_PATH)
    for metodo, cfg in MODELLI.items():
        importa(metodo, cfg, campione)


if __name__ == "__main__":
    main()
