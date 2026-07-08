# Multi-News — text mining & summarization

**Final project work — 2nd-level specializing master "Artificial Intelligence for STEM",
Politecnico di Torino** 

This repository hosts the project work built on the **Multi-News** multi-document summarization
dataset (Fabbri et al., 2019). It started as a copy of the original dataset repository,
[`alexfabbri/multi_news`](https://huggingface.co/datasets/alexfabbri/multi_news) on the Hugging
Face Hub, and extends it with dataset curation, exploratory analysis, and (in progress)
summarization experiments. All credit for the dataset itself goes to the original authors — see
[Attribution, license & citation](#attribution-license--citation).

## Project overview

The project explores Multi-News as a text-mining and summarization corpus:

- **Exploratory data analysis** — a corpus-wide quality and structure audit of all 56,216
  examples, published as a self-contained dashboard
  ([multi_news_dashboard.html](multi_news_dashboard.html)) and folded into the documentation.
- **Dataset curation** — conversion of the raw text files into Orange Data Mining's `.tab`
  format, including a cleaning step that removes rows with known source-quality problems
  (failed scrapes, exact duplicates, source/summary mismatches) identified by the EDA.
- **Summarization experiments** — text-mining and summarization work on the curated corpus
  (in progress).

## Repository contents

| path | description |
|------|-------------|
| [multi_news.py](multi_news.py) | Original Hugging Face `datasets` loader script (unchanged) |
| [data/text/](data/) | Canonical dataset files, as released upstream — one `.src.cleaned`/`.tgt` pair per split |
| [data/tab/](data/) | **Cleaned** Orange `.tab` copies: one per split plus `complete.tab` (all splits joined, with a `split` column) and `excluded_rows.tsv` (manifest of the 115 dropped rows) |
| [scripts/convert_to_tab.py](scripts/convert_to_tab.py) | Regenerates `data/tab/` from `data/text/`, applying the cleaning criteria — documented in [scripts/README.md](scripts/README.md) |
| [multi_news_dashboard.html](multi_news_dashboard.html) | Self-contained EDA dashboard — open directly in a browser (report text in Italian) |
| [Multi-News_paper.md](Multi-News_paper.md) | The original paper (Fabbri et al., 2019), for reference |
| [data/README.md](data/README.md) | Detailed documentation of file formats, statistics, and cleaning criteria |

## The dataset

Multi-News consists of news articles and professionally written summaries of these articles from
[newser.com](https://www.newser.com). Each summary is written by editors and cites the original
articles. Each example has two features:

- `document`: the text of the source news articles, concatenated with the special separator
  token `|||||`;
- `summary`: the human-written multi-document summary (starts with `– `, the newser.com house
  style).

| split | examples (canonical `data/text/`) | examples (cleaned `data/tab/`) |
|-------|----------------------------------:|-------------------------------:|
| train      | 44,972 | 44,880 |
| validation |  5,622 |  5,611 |
| test       |  5,622 |  5,610 |
| **total**  | **56,216** | **56,101** |

### EDA highlights

From the [dashboard](multi_news_dashboard.html) (computed over all splits aggregated; details and
methodology in [data/README.md](data/README.md)):

- 154,530 source articles in total — mean ≈ 2.75 per example, median 2; 82% of examples have ≤3
  sources.
- Input length is heavily right-skewed (median 1,319 words, mean ≈ 1,789, max 449,620), while
  summaries are uniform (median 220 words, range 34–973). Median compression ratio ≈ 6.3×.
- Summary length correlates with the *number* of sources (mean grows from ~176 to ~372 words
  going from 1 to 9+ sources) far more than with raw input size.
- Known data-quality issues: 10 empty source lines, 637 examples with ≤1 source article, 77
  exact-duplicate source rows; the most extreme length outliers are source/summary mismatches
  from upstream scraping errors. The derived Orange copy in `data/tab/` already excludes these
  dirty rows (115 dropped, itemized in `data/tab/excluded_rows.tsv`); the canonical `data/text/`
  files retain them.

## Using the data

**In Orange Data Mining** — load any `data/tab/*.tab` file with the *File* widget (the
[Orange3-Text](https://orangedatamining.com/widget-catalog/#text-mining) add-on's *Corpus* widget
enables the text-mining tools). Both columns are string meta attributes; `complete.tab` adds a
discrete `split` column (`train`/`val`/`test`) so the whole corpus can be analyzed at once and
filtered or grouped by split.

**With Hugging Face `datasets`** — the original loader script still works against the canonical
files:

```python
from datasets import load_dataset
dataset = load_dataset("path/to/multi_news.py")
```

**EDA dashboard** — open [multi_news_dashboard.html](multi_news_dashboard.html) in any browser;
it is fully self-contained (no server or network needed).

Note: the data files are large (~1.3 GB total for the sources); prefer streaming/line-by-line
reads when writing tooling against them.

## Attribution, license & citation

This project builds on the **Multi-News** dataset by Alexander R. Fabbri, Irene Li, Tianwei She,
Suyi Li and Dragomir R. Radev ([paper](https://arxiv.org/abs/1906.01749), LILY Lab, Yale
University). The starting point of this repository was the Hugging Face dataset repo
[`alexfabbri/multi_news`](https://huggingface.co/datasets/alexfabbri/multi_news); thanks to
[@patrickvonplaten](https://github.com/patrickvonplaten), [@lewtun](https://github.com/lewtun)
and [@thomwolf](https://github.com/thomwolf) for originally adding the dataset to the Hub.

The dataset is released by LILY Lab for **non-commercial research and educational purposes
only**, "as is", without warranties — this repository uses it strictly within that scope, as
educational project work. The full Dataset Usage Agreement is in [LICENSE](LICENSE).

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
