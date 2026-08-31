# `scripts/` contents

## `run_benchmark_test.py`

Unattended driver for the `SCOPE='test'` benchmark stint: runs notebooks 10 (First-k), 17 (LDA),
15 (LSA), 16 (SBERT clustering), 11 (Centroid+MMR), 03 (BART), 04 (PEGASUS), 07 (Qwen),
09 (Mistral), 08 (Gemma) and 06 (PRIMERA)
— in that fastest-to-slowest order — against the full clean test split (5,610 rows), one after
another, without needing to reopen and re-run each notebook by hand. Notebooks 10, 11, 15 and 16
each generate two method variants (`firstk_psr`/`firstk_nltk`,
`centroid_mmr`/`centroid_mmr_bert`, `lsa`/`lsa_steinberger` and `sbert_kmeans`/`sbert_agglom`).

### Usage

```
python scripts/run_benchmark_test.py             # full run, all rows (~3.5-5 days on a CUDA GPU)
python scripts/run_benchmark_test.py --limit 2   # smoke test: 2 rows per method, end-to-end
python scripts/run_benchmark_test.py --only 10,11  # only the listed notebooks (05 still re-runs)
```

`--only` takes comma-separated notebook number prefixes; useful when the other notebooks have
already completed their test run (re-executing them would reload models and recompute metrics
for nothing). Preflight checks shrink to what the selection needs (ollama for 07-09, GPU for
03/04/06/11).

Run from anywhere — paths are resolved relative to the script's own location. Requires the
notebook dependencies (`pip install -r requirements-notebooks.txt`) plus `jupyter`/`nbconvert`.

### What it does

1. **Preflight**: checks a CUDA GPU is available (if any of 03/04/06/11 is selected), `ollama`
   is reachable at `http://localhost:11434` with the three required model tags pulled (if any
   of 07-09 is selected), `data/tab/complete.tab` exists, and `jupyter nbconvert` is importable
   — fails fast before committing to a multi-day run.
2. **Executes each selected notebook** via `jupyter nbconvert --to notebook --execute
   --inplace`, with `SUMM_SCOPE=test` (and `SUMM_LIMIT=N` if `--limit` was passed) in the
   subprocess environment. Each method notebook reads its scope from
   `os.environ.get('SUMM_SCOPE', 'sample')`, so opening them by hand in Jupyter without this
   env var still runs the default `sample` scope unchanged. A failed notebook is logged and
   does **not** stop the run — the shared resumable generation loop
   (`notebooks/summ_utils.py`) means re-running this script later just completes whatever rows
   are still missing.
3. **Derives TextRank/LexRank `test` metrics** by filtering their already-committed
   `SCOPE='full'` per-example CSV on `split == 'test'`, instead of re-running notebooks 01/02
   (which already cover the entire dataset, test split included). This is numerically
   identical to a dedicated test-scope run because metrics are computed per example.
4. **Re-executes notebook 05** so the comparison views reflect the new results.

### Output

- `results/summaries/{method}_test.tsv` and `results/metrics/{method}_test_{per_example.csv,aggregate.json}`
  for the fifteen generated method slugs (seven single-method notebooks + the two variants each
  of notebooks 10/11/15/16), plus the two derived ones (`textrank`, `lexrank`).
- `run_benchmark_test.log` in the repo root (gitignored) — append-only, timestamped, one line
  per notebook start/end plus a final summary.

### Before a long run

Disable Windows sleep (`powercfg /change standby-timeout-ac 0`), make sure `ollama serve` is
running with the required tags (`ollama list`), and expect the machine to be busy for several
days — the script does not throttle GPU/CPU usage.

## `run_geval.py`

Unattended driver for the **G-Eval (LLM-as-a-Judge)** backfill — notebook 14. Judges every
generated summary on the test split with `gpt-5.4-mini` on Azure, scoring coherence,
consistency, fluency and relevance on a 1–5 scale. **100,621 judgments** across the 18 methods,
hours of paid API calls; see the *G-Eval* section of `notebooks/README.md` for the methodology.
90,233 of them are in the committed cache (the 13 original methods in full, the five notebook
15-17 methods at ~60% — the Azure credit ran out); a relaunch judges only what is missing.

### Usage

```
python scripts/run_geval.py --righe 1        # smoke: one call per not-yet-cached method, proves
                                             # the prompt cache works
python scripts/run_geval.py --pilota 20      # pilot: 20 rows, measures cost/judgment
python scripts/run_geval.py --budget 120     # full run, hard stop once $120 has been spent
python scripts/run_geval.py --righe 500 --thread 12
python scripts/run_geval.py --solo-metriche  # rewrite CSV/JSON from the cache, ZERO API calls
python scripts/run_geval.py --costo          # cost report from the cache, ZERO API calls
python scripts/run_geval.py --riprova-errori # drop cached failures so they get retried
python scripts/run_geval.py --no-05          # skip re-running notebook 05 at the end
```

Run them in that order: `--righe 1`, then `--pilota 20`, then the full run. The pilot is what
turns the cost estimate into a measured number.

### What it does

Executes `notebooks/14_geval.ipynb` in place via `nbconvert`, passing the `GEVAL_*` environment
variables (`GEVAL_SCOPE`, `GEVAL_RIGHE`, `GEVAL_PILOTA`, `GEVAL_THREAD`, `GEVAL_BUDGET`,
`GEVAL_OGNI`, `GEVAL_RIPROVA_ERRORI`, `GEVAL_SOLO_METRICHE`) — the notebook stays the single
source of truth and the documentary artifact. Preflight fails fast on: missing
`AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_API_KEY`, a **1-token ping on the judge deployment** (so a
wrong deployment name costs seconds, not hours), a missing `complete.tab`, missing summary TSVs
for any of the 18 methods, and an unavailable `nbconvert`. Notebook 05 is re-executed at the end
unless `--pilota` or `--no-05`.

`run_benchmark_test.py` is untouched — G-Eval is not on the generation path.

### Output

- `results/metrics/geval_cache_test.jsonl` — one JSON line per `(method, row_id)` with the four
  scores (or the error) **and the token counts**. This is the paid artifact: it is committed, and
  the metric files are re-derived from it for free.
- `results/metrics/{method}_test_geval_{per_example.csv,aggregate.json}` for the 18 methods —
  deliberately **separate** from the standard metric files (see `notebooks/README.md` for why
  merging them would break `valuta_e_salva`).
- `run_geval.log` in the repo root (gitignored) — append-only, timestamped.

### Cost monitoring

**Azure has no real-time cost API.** Cost Management lags 8–24 h, so it is useless mid-run. The
source of truth is the `usage` object on every response: token counts are exact and immediate,
and unit prices come from the public, unauthenticated **Azure Retail Prices API**
(`su.prezzi_retail_azure()`, with `su.PREZZI_GEVAL` pinned as fallback).

During a run the notebook prints a cost block every 1,500 judgments (rate, observed TPM, ETA,
cached-input share, reasoning-token share, cost split by line item, and the **projected total**).
Because the counts are also in the cache, `--costo` recomputes the same figures offline — run it
**from a second terminal while the long run is in progress**. `--budget` is a hard ceiling
(cumulative across sessions, not per-launch: it adds up whatever is already in the cache before
comparing to the cap) and stops cleanly, resuming on the next launch.

**`--valuta` must match your subscription's billing currency — this is not a cosmetic default.**
The Azure Retail Prices API defaults to USD, but Azure's per-currency price lists are **not**
live FX conversions of each other: verified on this account (EUR billing), the EUR list is a
flat **~0.8776×** the USD number on every meter (input, cached, output alike), not whatever
today's EUR/USD rate is. Running with the wrong `--valuta` doesn't change the token accounting —
it's still exact — but the dollar/euro figure printed and compared against `--budget` will not
match what the subscription is actually charged. First discovered when a run tracked at $36.00
turned out, 24 h later against the actual EUR credit balance, to correspond to €31.59 — a 12%
gap explained entirely by this fixed EUR discount, not by Cost Management's reporting lag.

For next-day reconciliation against actual billing:

```
az costmanagement query --type ActualCost --timeframe MonthToDate \
  --scope "/subscriptions/<sub-id>/resourceGroups/rg-antonio.girasella-0716"
```

Note this needs `az login --tenant <tenant-id>` first: the CLI may be signed in to a different
tenant than the subscription owning the AI Foundry resource.

### Before a long run

Disable Windows sleep (`powercfg /change standby-timeout-ac 0`), check the deployment's **TPM
quota** in the Azure portal (it, not `--thread`, is the real bottleneck), and remember this
spends real money — watch the printed cost counter and set `--budget`.

## `import_llm_results.py`

One-off importer of Federica's local-LLM benchmark results (LM Studio runs, archived in
[`notebooks/llm/`](../notebooks/llm/README.md)) into the shared `results/` layout.

> **Historical note:** the committed `results/` files for `qwen`/`gemma`/`mistral` were later
> regenerated from scratch via local ollama runs of notebooks 07–09 (qwen/gemma 2026-07-16,
> mistral 2026-07-17) and no longer match this import. The script is kept to document/reproduce the original LM Studio import;
> its overwrite guard (below) prevents it from clobbering the ollama results.

### Usage

```
python scripts/import_llm_results.py
```

Run from anywhere — paths are resolved relative to the script's own location. Requires the
notebook dependencies (`pip install -r requirements-notebooks.txt`), since it reuses
`notebooks/summ_utils.py` for writing and scoring.

### What it does

For each of `qwen`, `gemma`, `mistral`:

1. reads `notebooks/llm/{name}_summary_evaluation_results.csv` and **verifies** it is aligned
   1:1, in order, with `results/sample/sample_100_seed42.tsv` (every `reference_summary` must
   match the sample's `summary`);
2. writes `results/summaries/{name}_sample.tsv` in the repo format (`row_id`,
   `generated_summary`), skipping rows whose generation failed (empty/`Error:` content —
   gemma has 81 such rows, leaving 19);
3. recomputes ROUGE/BLEU/METEOR with the shared benchmark normalization
   (`summ_utils.valuta_e_salva`) into `results/metrics/{name}_sample_per_example.csv` +
   `..._aggregate.json`, whose `config` records the original run's provenance (LM Studio,
   checkpoint, prompt, parameters). The metric values inside the source CSVs use different
   normalization settings and are NOT carried over; neither is their BERTScore column.

### Safety

The script **refuses to run** if a target `results/summaries/{name}_sample.tsv` already
exists: that file may since contain rows regenerated via ollama (notebooks 07–09), and
re-importing would silently mix backends. Delete the file first to re-import.

## `convert_to_tab.py`

Converts the canonical dataset files in [`data/text/`](../data/README.md) into cleaned
[Orange Data Mining](https://orangedatamining.com) `.tab` files in `data/tab/`.

### Usage

```
python scripts/convert_to_tab.py
```

Run from anywhere — paths are resolved relative to the script's own location. No third-party
dependencies (Python 3 standard library only: `csv`, `hashlib`, `os`). Expect a few minutes of
runtime: it streams ~680 MB of source text twice and writes ~1.3 GB of output. Existing files in
`data/tab/` are overwritten.

### Inputs

The six canonical files, which the script never modifies:

| file | content |
|------|---------|
| `data/text/{train,val,test}.src.cleaned` | source articles, one example per line, articles joined by `\|\|\|\|\|`, newlines encoded as `NEWLINE_CHAR` |
| `data/text/{train,val,test}.tgt` | one summary per line, line-aligned with the `.src.cleaned` pair |

### Outputs

| file | content |
|------|---------|
| `data/tab/{train,val,test}.tab` | one cleaned Orange `.tab` per split — columns `document`, `summary` (both `string`/`meta`) |
| `data/tab/complete.tab` | all three splits joined, in train → val → test order, with a third `split` column (`discrete`/`meta`, values `train`/`val`/`test`) recording each row's origin |
| `data/tab/excluded_rows.tsv` | manifest of dropped rows — columns `split`, `line` (0-based index into the `data/text/` files), `reason` |

In all `.tab` outputs, `NEWLINE_CHAR` is restored to real newlines and the `|||||` separator is
kept inside `document`. Rows are written with Python's `csv` module (tab-delimited, quoted), which
is how Orange itself parses `.tab` files, so embedded newlines and tabs round-trip correctly.

### Cleaning

The `.tab` output is *cleaned*: rows whose source text matches a known quality problem
(identified by the EDA dashboard, [`multi_news_dashboard.html`](../multi_news_dashboard.html))
are dropped. With the current data this removes 115 of the 56,216 examples (92 train, 11 val,
12 test), leaving 56,101. A row is excluded when its source is:

1. **shorter than `MIN_SRC_WORDS` (50) words** — includes fully empty sources; likely failed
   scrapes (55 rows);
2. **longer than `MAX_SRC_WORDS` (100,000) words** — the extreme outliers, whose source text is
   semantically unrelated to the summary (upstream scraping/link errors, not just long text)
   (8 rows);
3. **an exact duplicate of an earlier source** — whitespace-normalized SHA-1 hash, scanning
   train → val → test; only the first occurrence is kept, which also removes train/eval leakage
   from duplicate groups that span splits (52 rows labeled as duplicates; 25 more
   duplicate-redundant rows are already caught by rule 1, since empty/stub sources duplicate
   each other).

Word counts are tokenizer-free `str.split()` with `NEWLINE_CHAR` restored and `|||||` excluded,
matching the dashboard's methodology. The thresholds are the `MIN_SRC_WORDS` / `MAX_SRC_WORDS`
constants at the top of the script. Summaries are never a drop criterion.

**Consequence:** `data/tab/` is *not* line-aligned with `data/text/` — use `excluded_rows.tsv` to
map between the two.

### How it works

Two passes:

1. `find_dirty_rows()` streams every split's `.src.cleaned` file, computes each source's word
   count and hash, and returns the set of `(split, line)` rows to exclude with a reason.
2. `convert_split()` streams each `.src.cleaned`/`.tgt` pair again, zipping them line by line and
   writing every non-excluded row to the split's `.tab` file and to `complete.tab`
   simultaneously. `write_manifest()` then emits `excluded_rows.tsv`.

Everything is streamed line by line; no file is loaded into memory wholesale.

### When to rerun

`data/tab/` files are derived, never hand-edited. Rerun the script whenever `data/text/` changes
or the cleaning criteria are adjusted, and update the row counts/sizes in
[`data/README.md`](../data/README.md) if they change.

## `analyze_dataset.py`

Corpus-wide EDA over the **whole** Multi-News dataset (train+val+test aggregated, no per-split
breakdown): the script behind the figures embedded in
[`multi_news_dashboard.html`](../multi_news_dashboard.html).

### Usage

```
python scripts/analyze_dataset.py
```

Run from the repo root. Only third-party dependency is `numpy` (percentiles/histograms);
everything else is the standard library. Expect a few minutes: it streams ~680 MB of source text
once.

### Inputs

The six canonical files in [`data/text/`](../data/README.md), read line by line and never
modified — the same files, and the same `NEWLINE_CHAR` / `|||||` handling, that
`multi_news.py::_generate_examples` uses. Note this is the **uncleaned** canonical data, so the
dirty rows that `convert_to_tab.py` drops are still counted here (that is the point: the
dashboard reports them).

### Output

`scripts/dataset_stats.json` — one aggregated JSON object (committed), with top-level keys
`meta`, `structure`, `sources`, `lengths`, `hist`, `paper_reference`, `correlations` and
`quality`. The dashboard's inline `const D = {...}` literal is built from this file; editing the
dashboard's numbers by hand means editing that literal.

### What it does

A single streaming pass per split, computing only what is cheap:

- **Structure** — empty sources/targets, exact duplicates (SHA-1 over whitespace-normalized
  lowercase text) and near-duplicates (fingerprint of the first 15 normalized words), article
  counts per example.
- **Lengths** — word counts (tokenizer-free `str.split()`, with `NEWLINE_CHAR` restored and the
  `|||||` separator excluded), heuristic sentence counts (`[.!?]+`), compression ratios,
  percentiles and histograms, plus a global vocabulary set (`[a-z0-9']+`).
- **Quality** — counts against the anomaly thresholds declared at the top of the script
  (`SUM_MIN=20`, `SUM_MAX=600`, `SRC_MIN=50` words) and the extreme outlier rows, referenced as
  0-based `split:line`.

**Deliberately omitted** (agreed scope, stated in the module docstring): the heavy metrics —
novel n-gram percentages, extractive fragment coverage/density, and language detection. The
dashboard shows those only as static reference values from the paper, never as recomputed
figures. Expect systematic offsets from the paper's own table for anything tokenizer-dependent
(e.g. vocabulary 494,577 here vs 666,515 in Fabbri et al.) — different tokenization, not a bug.

### When to rerun

Whenever `data/text/` changes. After rerunning, port the new values into the dashboard's
`const D` literal and update the figures quoted in
[`data/README.md`](../data/README.md) and [`CLAUDE.md`](../CLAUDE.md) — nothing does that
automatically.
