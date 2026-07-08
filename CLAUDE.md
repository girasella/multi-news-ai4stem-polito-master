# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This repository is the final project work for the master "Artificial Intelligence for STEM"
(Politecnico di Torino), built on top of the Multi-News summarization dataset. It started as a
copy of the Hugging Face dataset repo (`https://huggingface.co/datasets/alexfabbri/multi_news`,
whose `datasets`-library loader it retains) and adds dataset curation, EDA, and summarization
experiment work. All source and data live at the repo root.

```
multi_news.py         # HF `datasets` GeneratorBasedBuilder loader script (unchanged from upstream)
README.md              # Project README (AI4STEM final project) + dataset summary and condensed licensing; YAML frontmatter intentionally removed — this repo no longer targets HF Hub dataset-card compatibility
LICENSE                # Full upstream Dataset Usage Agreement (moved out of README.md)
Multi-News_paper.md    # Original paper (Fabbri et al., 2019) — background/context only, not consumed by any tooling
multi_news_dashboard.html  # Self-contained EDA dashboard (Italian) — see "EDA dashboard" section below
scripts/
  README.md            # Documentation for the scripts (usage, inputs/outputs, cleaning criteria)
  convert_to_tab.py    # Regenerates data/tab/ from data/text/ (Orange format), dropping dirty rows
data/
  README.md            # Detailed description of data/ file formats and content
  text/                # Canonical format — consumed by multi_news.py; kept as-released (dirty rows included)
    {train,val,test}.src.cleaned   # source documents, one example per line
    {train,val,test}.tgt           # target summaries, one example per line
  tab/                 # Derived, CLEANED Orange `.tab` copy — 115 dirty rows dropped, so NOT
                       # line-aligned with data/text/; regenerate via scripts/convert_to_tab.py
    {train,val,test}.tab
    complete.tab       # All three splits joined (56,101 rows) + a `split` origin column
    excluded_rows.tsv  # Manifest of the dropped rows (split, 0-based line in data/text/, reason)
```

## Architecture

- `multi_news.py` defines `MultiNews(datasets.GeneratorBasedBuilder)`, the standard three-method
  HF dataset script shape:
  - `_info()` — declares the two features: `document` (string) and `summary` (string).
  - `_split_generators()` — downloads/resolves the six `data/*` files (paired src/tgt per split)
    from the `_REPO` URL (`.../multi_news/resolve/main/data`) and wires them into
    train/validation/test `SplitGenerator`s.
  - `_generate_examples()` — zips a split's `.src.cleaned` and `.tgt` files line-by-line; each
    line pair is one example. The literal token `NEWLINE_CHAR` in source lines is restored to a
    real `\n` (that substitution exists so the raw data file can keep one example per physical
    line while document text still contains newlines separating individual news articles).
- Each `.src.cleaned` line is itself multiple news articles concatenated with the separator token
  `|||||`; each `.tgt` line is the corresponding human-written multi-document summary.
- `data/text/` files are line-aligned 1:1 across the src/tgt pair for a split — do not reorder or
  filter one file without the other. `data/tab/` files are NOT line-aligned with `data/text/`:
  the converter drops rows with dirty sources (<50 words, >100k words, or exact duplicates —
  criteria documented in `scripts/convert_to_tab.py` and `data/README.md`), listing them in
  `data/tab/excluded_rows.tsv`.
- `README.md` no longer carries HF Hub dataset-card YAML frontmatter (`dataset_info`,
  `train-eval-index`, etc.) — it was intentionally stripped since this repo doesn't need to
  maintain Hugging Face Hub compatibility. Don't reintroduce it or treat its absence as a bug.

## EDA dashboard (`multi_news_dashboard.html`)

A self-contained static HTML report (UI text in Italian, no external dependencies) with
corpus-wide exploratory statistics, computed in streaming over all three splits aggregated
(56,216 examples, 154,530 source articles). All numbers are embedded as a single JSON literal
(`const D = {...}` in the inline `<script>`) — read/edit that object, not the rendering code, to
get or change the stats. Key facts it establishes (details in `data/README.md`):

- Word counts are tokenizer-free (`str.split()`, with `NEWLINE_CHAR` and `|||||` excluded);
  sentence counts are heuristic (`[.!?]+` split). Paper values (Fabbri et al., Table 3) are shown
  as a reference column, not recomputed — expect systematic offsets (e.g. vocab 494,577 here vs
  666,515 in the paper).
- Data-quality caveats to respect in any tooling: 10 empty source lines, 637 examples with ≤1
  source article, 77 source rows that are exact duplicates of another row (20 groups — dedup
  before any re-splitting to avoid train/eval leakage), 0 duplicate summaries. These apply to
  `data/text/`; the derived `data/tab/` copy already excludes the dirty rows.
- Extreme source-length outliers (top one: `train:22256`, 449,620 words) are source/summary
  **mismatches** from upstream scraping errors, not just long text — the summary is unrelated to
  the source. Filtering/truncating by length alone doesn't fix them.
- Its footer says it's regenerable via `python scripts/analyze_dataset.py`, but that script is
  **not present in the repo** — only `scripts/convert_to_tab.py` exists. Treat the dashboard's
  embedded JSON as the current source of truth for these stats.

## Working with the data files

The files in `data/` are large (the train source file is ~500MB); avoid loading them
wholesale in tooling — prefer streaming/line-by-line reads (as `_generate_examples` does) or
sampling a subset of lines when inspecting content.

## Licensing

The dataset is released for **non-commercial research and educational purposes only** (full
Dataset Usage Agreement in `LICENSE`, condensed in `README.md`); keep this in mind before
proposing any commercial use of the data.
