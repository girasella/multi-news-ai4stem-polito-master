# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This repository (`multi_news/`) is its own git repo (remote:
`https://huggingface.co/datasets/alexfabbri/multi_news`) — a Hugging Face `datasets`-library
dataset loader for the Multi-News summarization dataset. All source and data live at the repo
root.

```
multi_news.py         # HF `datasets` GeneratorBasedBuilder loader script
README.md              # Dataset summary + license; YAML frontmatter intentionally removed — this repo no longer targets HF Hub dataset-card compatibility
Multi-News_paper.md    # Original paper (Fabbri et al., 2019) — background/context only, not consumed by any tooling
scripts/
  convert_to_tab.py    # Regenerates data/tab/ from data/text/ (Orange Data Mining format)
data/
  README.md            # Detailed description of data/ file formats and content
  text/                # Canonical format — consumed by multi_news.py
    {train,val,test}.src.cleaned   # source documents, one example per line
    {train,val,test}.tgt           # target summaries, one example per line
  tab/                 # Derived Orange `.tab` copy, regenerate via scripts/convert_to_tab.py
    {train,val,test}.tab
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
- `data/` files are line-aligned 1:1 across the src/tgt pair for a split — do not reorder or
  filter one file without the other.
- `README.md` no longer carries HF Hub dataset-card YAML frontmatter (`dataset_info`,
  `train-eval-index`, etc.) — it was intentionally stripped since this repo doesn't need to
  maintain Hugging Face Hub compatibility. Don't reintroduce it or treat its absence as a bug.

## Working with the data files

The files in `data/` are large (the train source file is ~500MB); avoid loading them
wholesale in tooling — prefer streaming/line-by-line reads (as `_generate_examples` does) or
sampling a subset of lines when inspecting content.

## Licensing

The dataset is released for **non-commercial research and educational purposes only** (see the
Licensing Information section of `README.md`); keep this in mind before proposing any
commercial use of the data.
