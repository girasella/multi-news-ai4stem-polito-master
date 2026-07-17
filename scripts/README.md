# `scripts/` contents

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
