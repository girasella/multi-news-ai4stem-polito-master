# `data/` contents

Raw data files backing the Multi-News dataset loader ([multi_news.py](../multi_news.py)). This
document describes what's actually in these files, based on direct inspection — see the
[README.md](../README.md) for the citation and license summary, and
[Multi-News_paper.md](../Multi-News_paper.md) (Fabbri et al., 2019, arXiv:1906.01749) for the
original paper that introduced this dataset. This repo hosts only the dataset itself (these files
plus the loader script) — not the Hi-MAP summarization model or training code the paper also
describes. For corpus-wide statistics with charts, see the self-contained EDA dashboard
[multi_news_dashboard.html](../multi_news_dashboard.html) (report text in Italian); its key
findings are folded into the sections below.

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

`data/` is organized into two subfolders:

- `data/text/` — the original six-file format: one `.src.cleaned` (source articles) and one `.tgt`
  (summary) file per split, all 56,216 examples. This is the canonical/authoritative copy, kept
  as-released (including its known dirty rows).
- `data/tab/` — Orange Data Mining `.tab` files generated from `data/text/` by
  [scripts/convert_to_tab.py](../scripts/convert_to_tab.py): one per split (`train.tab`,
  `val.tab`, `test.tab`) plus `complete.tab`, all three splits joined with a `split` origin
  column. This copy is **cleaned**: 115 rows with known source-quality problems are excluded
  (56,101 examples total), so it is *not* line-aligned with `data/text/`. See
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

| file                    |  rows  | size (bytes) | size    |
|-------------------------|-------:|-------------:|---------|
| `tab/train.tab`         | 44,880 |  554,038,300 | ~528 MB |
| `tab/val.tab`           |  5,611 |   67,629,026 | ~64 MB  |
| `tab/test.tab`          |  5,610 |   67,824,837 | ~65 MB  |
| `tab/complete.tab`      | 56,101 |  689,811,869 | ~658 MB |
| `tab/excluded_rows.tsv` |    115 |        3,761 | ~4 KB   |

Row counts are data rows (excluding the 3-line Orange header). `complete.tab` is the three splits
joined into one file, with an extra `split` column recording each row's origin (see below). The
115 rows dropped relative to `data/text/` (92 train, 11 val, 12 test) are itemized with reasons in
`tab/excluded_rows.tsv` — see "Cleaning" under "`data/tab/*.tab` format" below.

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

**Source length** is highly skewed: aggregated over all splits, the median is 1,319 words per
example (mean ≈ 1,789, p95 ≈ 4,599 by simple whitespace splitting; the paper reports a
tokenizer-normalized mean of 2,103 words / 82.7 sentences per document cluster), but a long tail
of examples run to tens or hundreds of thousands of words. Don't assume a bounded input size when
writing tooling against this field.

The extreme length outliers are not merely "long news clusters" — manual inspection shows they are
**source/summary mismatches**. The single largest (train line 22256, 449,620 words) is the full
program of an academic conference (Society of Biblical Literature Annual Meeting, Atlanta 2015 —
hundreds of concatenated paper abstracts), while its paired summary is a normal 319-word digest
about an entirely unrelated story (a New Testament papyrus fragment sold on eBay): no semantic
relation between the two columns for that row. The likely cause is an upstream scraping/link error
(e.g. a Wayback link resolving to the wrong page); the other top outliers (all >100k words, e.g.
`train:26686` at 168,796 and `test:4403` at 145,130) are plausibly analogous cases. Anyone
training on this data should consider *filtering* such anomalous-source examples, not just capping
input length — a truncated slice of an irrelevant document still carries no signal for the target
summary.

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
`test.tab`) plus the joined `complete.tab`, generated from the `data/text/` pairs by
[scripts/convert_to_tab.py](../scripts/convert_to_tab.py). Each row is one example: the same
`document`/`summary` content as `data/text/`, with `NEWLINE_CHAR` already restored to real `\n` and
the `` ||||| `` article separator still present in `document`.

**`complete.tab`** holds the whole (cleaned) dataset — the three splits concatenated in
train → val → test order, same rows as the per-split files — with a third column `split`
(declared `discrete`, values `train`/`val`/`test`) recording which split each row came from, so
the origin isn't lost after joining and can be used for filtering/grouping in Orange:

```
document	summary	split
string	string	discrete
meta	meta	meta
some doc text...	some summary text...	train
```

**Cleaning.** The converter drops rows whose *source* text matches the quality problems identified
by the EDA dashboard (see "Corpus-wide statistics" above), so the `.tab` files hold 56,101 of the
56,216 examples and are **not line-aligned** with `data/text/`. A row is excluded when its source
(word counts via `str.split()`, `NEWLINE_CHAR`/`` ||||| `` excluded) is:

1. **shorter than 50 words** (55 rows, including the 10 fully empty sources) — likely failed
   scrapes;
2. **longer than 100,000 words** (8 rows) — the extreme outliers whose source is semantically
   unrelated to the summary (see the mismatch note above);
3. **an exact duplicate** (whitespace-normalized SHA-1) of an earlier source, scanning
   train → val → test — only the first occurrence is kept, which also removes the train/eval
   leakage from duplicate groups spanning splits (52 rows carry this label; the other 25
   duplicate-redundant rows were already excluded by rule 1, since empty/stub sources duplicate
   each other).

Every excluded row is listed in `tab/excluded_rows.tsv` (columns: `split`, `line` — 0-based index
into the `data/text/` files — and `reason`). Summaries are never a drop criterion: the 12
longer-than-600-word summaries and the 637 single-source examples flagged by the dashboard are
retained, as they are legitimate content.

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

## Corpus-wide statistics (EDA)

Numbers below come from the EDA dashboard
([multi_news_dashboard.html](../multi_news_dashboard.html)), computed in streaming over all three
splits aggregated: 56,216 examples, 154,530 source articles. Methodology: word counts are
tokenizer-free (`str.split()`, with `NEWLINE_CHAR` restored and `` ||||| `` excluded before
counting); sentence counts are heuristic (split on `[.!?]+`, so abbreviations slightly inflate
them); duplicates detected via SHA-1 on normalized text, near-duplicates via a fingerprint of the
first 15 words; row references use 0-based `split:line` indices. Note: the dashboard's footer
mentions a `scripts/analyze_dataset.py` for regeneration, but that script is not currently in the
repo — the dashboard's embedded JSON (`const D` in its inline script) is the source of truth for
these figures.

| metric              |   mean | median |  min |     max |    p05 |   p95 |
|---------------------|-------:|-------:|-----:|--------:|-------:|------:|
| words / input       | 1,788.8 | 1,319 |    0 | 449,620 |   356 | 4,599 |
| sentences / input   |  102.2 |     73 |    0 |  21,417 |    20 |   263 |
| words / summary     |  218.0 |    220 |   34 |     973 |   109 |   318 |
| sentences / summary |   11.1 |     11 |    1 |      55 |     5 |    18 |
| compression (in÷out)|  8.19× |  6.33× |   0× | 1,409×  | 2.00× | 19.4× |

Estimated vocabulary (unique whitespace tokens): 494,577 (paper's tokenizer-normalized figure:
666,515). Abstractiveness/extractiveness metrics (novel n-grams, fragment coverage/density) and
language detection were *not* recomputed — the paper's values are quoted where relevant. Language:
both sources and summaries are overwhelmingly (if not exclusively) **English** (newser.com is an
English-language editorial aggregator).

**Integrity and redundancy** (percentages over the 56,216-example corpus):

- Empty source lines: 10 (0.018%); empty summaries: 0.
- Exact-duplicate summaries: 0 — all 56,216 summaries are unique. Near-duplicate summaries
  (first-15-words fingerprint): 3 redundant rows across 3 groups.
- Exact-duplicate **sources**: 77 redundant rows across 20 groups (0.137%) — only 56,139 of the
  source texts are unique. If you ever re-split or resample this data, de-duplicate sources first
  to avoid train/eval leakage.
- Examples with ≤1 source article: 637 (1.13%) — not multi-document despite the dataset's premise.
- Against simple sanity thresholds: 0 summaries under 20 words, 12 over 600 words; 55 source lines
  under 50 words (likely failed scrapes).

**Correlations** (over all examples; compression pairs only where the summary is non-empty):

| variable pair                  | Pearson r | Spearman ρ |
|--------------------------------|----------:|-----------:|
| n. sources vs summary words    |     0.428 |      0.360 |
| n. sources vs compression      |     0.190 |      0.316 |
| input words vs summary words   |     0.209 |      0.404 |

The clearest relationship is that summaries grow with the *number* of sources (mean summary length
rises roughly monotonically from ~176 words at 1 source to ~372 at 9 sources), and compression
grows with it too (from ~3× to ~20×). By contrast, summary length is largely independent of raw
input *size*: the Pearson r is depressed by huge outlier sources paired with summaries that stay
in the ~150–300-word band.

## Practical notes

- The files are large (`data/text/train.src.cleaned` alone is ~522 MB, `data/tab/train.tab` ~528 MB);
  per the top-level `CLAUDE.md`, prefer streaming/line-by-line reads or sampling rather than loading
  a whole file into memory.
- Because `.src.cleaned` and `.tgt` are matched purely by line position, any tooling that filters,
  sorts, or dedupes one file must apply the identical operation to its pair, or the alignment
  breaks silently.
- When computing statistics or samples over `data/text/` files, expect and handle: empty source
  lines, the `` ||||| `` separator appearing inside `document`, and the `NEWLINE_CHAR` token if
  reading the raw file directly instead of through `multi_news.py`.
