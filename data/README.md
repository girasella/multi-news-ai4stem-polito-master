# `data/` contents

Raw data files backing the Multi-News dataset loader ([multi_news.py](../multi_news.py)). This
document describes what's actually in these files, based on direct inspection — see the
[README.md](../README.md) for the citation and license summary, and
[Multi-News_paper.md](../Multi-News_paper.md) (Fabbri et al., 2019, arXiv:1906.01749) for the
original paper that introduced this dataset. This repo hosts only the dataset itself (these files
plus the loader script) — not the Hi-MAP summarization model or training code the paper also
describes.

## Provenance (from the paper)

- Source and summaries are scraped from **newser.com**: each `summary` is a professionally
  editor-written digest of a news event, and each source article in `document` is one of the
  pieces the editor cited. About 20 editors account for 85% of all summaries.
- Source documents come from **1,500+ distinct news sites** (each appearing 5+ times), making this
  more source-diverse than prior news summarization datasets (e.g. CNN/DailyMail draws from only
  two sites).
- The paper reports collecting 250,000+ Wayback-archived links, then keeping only clusters with
  **2 to 10 source articles** per summary, yielding 56,216 total pairs split 80/10/10 into
  train/validation/test (44,972 / 5,622 / 5,622) — matching the split sizes in this repo exactly.
- The paper's published per-split source-count distribution (its Table 2, which sums to exactly
  44,972 — i.e. it describes the **train** split) is close to, but not bit-identical to, what's
  measured directly from `text/train.src.cleaned` below (e.g. paper reports 23,894 examples with 2
  sources vs. 23,741 found here; paper reports no 0- or 1-source examples at all, since it filtered
  to 2–10). The data in this repo therefore isn't a frozen snapshot of the paper's exact reported
  numbers — it includes a small number of 0- and 1-source examples that fall outside the paper's
  stated 2–10 filter (see below), consistent with further reprocessing/link-rot between the paper
  and this release.
- The paper's own corpus-wide averages (its Table 3, whitespace/tokenizer-dependent): documents
  average **2,103 words / 82.7 sentences**, summaries average **263.7 words / 10.0 sentences**,
  vocabulary size 666,515. Simple `str.split()` word counts on the files in this repo run somewhat
  lower (see "Source length" and "Summary length" below); treat the paper's numbers as the
  tokenizer-normalized reference and the counts here as a quick, tokenizer-free sanity check, not a
  contradiction.
- The paper also reports these summaries are comparably abstractive to single-document news
  datasets: only 17.8% of unigrams and 82.3% of 4-grams in a summary are *novel* (i.e. absent from
  its source documents), meaning summaries lean heavily on phrases copied from the source text
  rather than being purely paraphrased.

## Files

`data/` is organized into two subfolders holding the same 56,216 examples in different formats:

- `data/text/` — the original six-file format: one `.src.cleaned` (source articles) and one `.tgt`
  (summary) file per split. This is the canonical/authoritative copy.
- `data/tab/` — one Orange Data Mining `.tab` file per split (`train.tab`, `val.tab`, `test.tab`),
  generated from `data/text/` by [scripts/convert_to_tab.py](../scripts/convert_to_tab.py). See
  "`data/tab/*.tab` format" below.

### `data/text/`

| file                        |  lines | size (bytes) | size    |
|-----------------------------|-------:|-------------:|---------|
| `text/train.src.cleaned`    | 44,972 |   547,512,283 | ~522 MB |
| `text/train.tgt`            | 44,972 |    58,793,912 | ~56 MB  |
| `text/val.src.cleaned`      |  5,622 |    66,875,522 | ~64 MB  |
| `text/val.tgt`              |  5,622 |     7,295,302 | ~7.0 MB |
| `text/test.src.cleaned`     |  5,622 |    68,999,509 | ~66 MB  |
| `text/test.tgt`             |  5,622 |     7,309,099 | ~7.0 MB |

All files are UTF-8 text with Unix (`\n`) line endings, no BOM. Within a split, `.src.cleaned` and
`.tgt` are line-aligned 1:1 — line *i* of the source file pairs with line *i* of the target file to
form one example (this is exactly what `_generate_examples` in `multi_news.py` relies on: it
`zip()`s the two files line-by-line).

### `data/tab/`

| file             |  rows  | size (bytes) | size    |
|------------------|-------:|-------------:|---------|
| `tab/train.tab`  | 44,972 |   559,677,315 | ~534 MB |
| `tab/val.tab`    |  5,622 |    68,430,683 | ~65 MB  |
| `tab/test.tab`   |  5,622 |    70,192,756 | ~67 MB  |

## `data/text/*.src.cleaned` format

Each line is one example's full source input: one or more news articles about the same event,
concatenated with the separator token `` ||||| `` (five pipes, space-padded). The separator is
kept as a literal substring in the `document` field produced by the loader — it is not split out
into a list.

Within a single article's text, real newlines have been replaced with the literal token
`NEWLINE_CHAR` so that each example still occupies exactly one physical line in the file. The
loader restores these to real `\n` characters at read time (`multi_news.py:109`).

**Articles per example** (splitting on `` ||||| ``), across all three splits:

| # articles | train  | val   | test  |
|-----------:|-------:|------:|------:|
| 1          |    498 |    58 |    71 |
| 2          | 23,741 | 3,066 | 3,022 |
| 3          | 12,577 | 1,555 | 1,540 |
| 4          |  4,921 |   610 |   609 |
| 5          |  1,846 |   195 |   219 |
| 6          |    706 |    79 |    96 |
| 7          |    371 |    38 |    40 |
| 8          |    194 |    13 |    15 |
| 9          |     81 |     7 |     8 |
| 10         |     29 |     0 |     1 |
| 0 (empty)  |      8 |     1 |     1 |

Median is 2 articles per example, mean ~2.7–2.8. This roughly matches the distribution the paper
reports (see "Provenance" above) but isn't identical, and — unlike the paper's stated 2–10 source
filter — this release also contains rows with 1 or 0 articles:

- A handful of examples (8 in train, 1 each in val/test) are **completely empty** source lines —
  the corresponding summary still exists and reads normally, so these look like upstream scraping
  failures (e.g. a cited article no longer resolving) rather than corrupt rows.
- A larger handful (498 train, 58 val, 71 test) have exactly 1 article — single-document rows in a
  nominally multi-document dataset.

Consumers that assume a non-empty `document`, or that every example has ≥2 source articles per the
paper's methodology, should account for these edge cases.

**Source length** is highly skewed: median is ~1,350–1,400 words per example (mean ~1,850–1,900 by
simple whitespace splitting; the paper reports a tokenizer-normalized mean of 2,103 words / 82.7
sentences per document cluster), but a long tail of examples run to tens or hundreds of thousands
of words (the single largest, train line 22256, is a concatenation of conference-abstract listings
totaling ~460k words). Don't assume a bounded input size when writing tooling against this field.

## `data/text/*.tgt` format

Each line is the single human-written multi-document summary for the corresponding source line —
plain text, no `NEWLINE_CHAR` tokens (summaries are single-paragraph). Every summary in every
split starts with an en dash (`– `), which is the newser.com editorial house style for these
digest summaries, not an artifact of processing.

Summary length is far more uniform than source length: median ~218–221 words by simple whitespace
splitting (the paper reports a tokenizer-normalized mean of 263.7 words / 10.0 sentences), with a
range of roughly 34–973 words (train), narrower for val/test. The paper notes this ~260-word
average is notably longer than comparable single-document summarization datasets (e.g. CNN/DailyMail
averages ~56 words), making fluent, coherent generation over a longer output a specific challenge
of this dataset.

## `data/tab/*.tab` format

Orange Data Mining's native tab-delimited format — one file per split (`train.tab`, `val.tab`,
`test.tab`), generated from the corresponding `data/text/` pair by
[scripts/convert_to_tab.py](../scripts/convert_to_tab.py). Each row is one example: the same
`document`/`summary` content as `data/text/`, with `NEWLINE_CHAR` already restored to real `\n` and
the `` ||||| `` article separator still present in `document`.

Orange `.tab` files use a 3-line header (attribute names, types, flags) followed by data rows. Both
columns are declared `string` type with the `meta` flag (descriptive text fields, not a class
target — this is a summarization dataset, not a classification one):

```
document	summary
string	string
meta	meta
some doc text...	some summary text...
```

Rows are written with Python's `csv` module (tab-delimited, quoted), so `document` values containing
embedded newlines or literal tab characters round-trip correctly — this is required reading, since
`Orange.data.Table` parses `.tab` files the same way (via `csv.reader`), not by naive line-splitting.

These files are **derived, not hand-maintained**: if `data/text/` ever changes, regenerate
`data/tab/` by rerunning `python scripts/convert_to_tab.py` from the repo root rather than editing
the `.tab` files directly.

## Practical notes

- The files are large (`data/text/train.src.cleaned` alone is ~522 MB, `data/tab/train.tab` ~534 MB);
  per the top-level `CLAUDE.md`, prefer streaming/line-by-line reads or sampling rather than loading
  a whole file into memory.
- Because `.src.cleaned` and `.tgt` are matched purely by line position, any tooling that filters,
  sorts, or dedupes one file must apply the identical operation to its pair, or the alignment
  breaks silently.
- When computing statistics or samples over `data/text/` files, expect and handle: empty source
  lines, the `` ||||| `` separator appearing inside `document`, and the `NEWLINE_CHAR` token if
  reading the raw file directly instead of through `multi_news.py`.
