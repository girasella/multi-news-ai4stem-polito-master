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
Tecniche_MDS_non_LLM_MultiNews.md  # Annotated survey of non-LLM MDS techniques vs the PoliTO lecture (Italian) — the "documento-guida" cited by notebooks/README.md; reference only, not consumed by tooling
multi_news_dashboard.html  # Self-contained EDA dashboard (Italian) — see "EDA dashboard" section below
scripts/
  README.md            # Documentation for the scripts (usage, inputs/outputs, cleaning criteria)
  convert_to_tab.py    # Regenerates data/tab/ from data/text/ (Orange format), dropping dirty rows
  import_llm_results.py  # One-off importer of the archived LM Studio LLM runs (notebooks/llm/*.csv) into results/ — superseded by the ollama re-runs, kept for provenance
  run_benchmark_test.py  # Unattended driver: runs notebooks 03-04/06-11 with SCOPE='test' back-to-back (--only N,N to select a subset), derives textrank/lexrank test metrics from their full run, re-runs notebook 05
  run_geval.py         # Unattended driver for the G-Eval backfill (notebook 14): staged --righe/--pilota/full run, --budget hard stop, --costo offline cost report, --solo-metriche re-derivation
requirements-notebooks.txt  # Dependencies for the benchmark notebooks (pyAutoSummarizer, openai etc.)
notebooks/             # Summarization benchmark — see "Summarization benchmark" section below
  README.md            # Run order, parameters, runtimes, Colab instructions (Italian)
  summ_utils.py        # Shared routines: data loading, resumable generation loop, metrics
  0X_*.ipynb           # 00 sample prep, 01-04 and 06-09 one method each, 05 comparison
  1X_*.ipynb           # 10 First-k baseline, 11 Centroid+MMR, 12 Azure AI Foundry GPT-5-mini (scopes sample/test/full), 13 BERTScore backfill, 14 G-Eval backfill; ex Azure 11-12 (Claude Haiku, DeepSeek) removed — recoverable from git history
  llm/                 # ARCHIVE (do not run/edit): Federica's original LM Studio notebooks,
                       # result CSVs (source of the originally imported qwen/gemma/mistral
                       # results, since replaced by local ollama runs) and docx report —
                       # see notebooks/llm/README.md
results/
  sample/              # Shared evaluation sample TSV (committed)
  summaries/           # Generated summaries per method, scope=test/full only (large but regenerable); scope=sample is a local smoke-test artifact, not committed
  metrics/             # Per-example CSVs + aggregate JSONs (committed); plus the SEPARATE
                       # G-Eval files `{method}_{scope}_geval_{per_example.csv,aggregate.json}`
                       # and `geval_cache_{scope}.jsonl` (the paid judgment cache — commit it)
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

## Summarization benchmark (`notebooks/` + `results/`)

Eleven method slugs: the First-k / Lead positional baseline in two sentence-segmentation
variants (notebook 10, slugs `firstk_psr`/`firstk_nltk`); TextRank, LexRank (extractive) plus
a custom scikit-learn Centroid-based (MEAD) + MMR in two vectorization variants (notebook 11,
slugs `centroid_mmr` TF-IDF / `centroid_mmr_bert` BERT); BART `facebook/bart-large-cnn`,
PEGASUS `google/pegasus-multi_news`, PRIMERA `allenai/PRIMERA-multinews` (specialized
abstractive); three local general-purpose LLMs — Qwen2.5-7B-Instruct, Gemma 4 E4B,
Mistral-7B-Instruct-v0.3 (notebooks 07/08/09, method slugs `qwen`/`gemma`/`mistral`); plus
one cloud LLM on Azure AI Foundry — GPT-5-mini (notebook 12, slug `gpt5mini`). TextRank/LexRank
and BART/PEGASUS
run via pyAutoSummarizer, PRIMERA directly via `transformers` (notebook 06), the local LLMs via
the `openai` client against ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`),
GPT-5-mini via `openai.OpenAI` against the Azure OpenAI **v1 route**
(`<endpoint>/openai/v1/`, no dated api-version) — all scored with pyAutoSummarizer's
ROUGE-1/2/L, BLEU, METEOR implementations. Two more Azure notebooks (Claude Haiku 4.5 and
DeepSeek-V3.2, slugs `haiku`/`deepseek`, formerly numbered 11/12) were removed because their
Foundry deployments could not be created; recover them from git history if retried (the
numbers were since reused: 10/11 are now the First-k and Centroid+MMR baselines, 12 the Azure
GPT-5-mini notebook). Baseline notebooks 10/11 support all three scopes
(`sample`/`test`/`full`, `SUMM_SCOPE`/`SUMM_LIMIT` env overrides like 03-04/06-09) and are
driven by `run_benchmark_test.py`, listed first as the fastest methods; each generates its two
variant slugs in one execution. Conventions to
respect:

- **All notebook documentation, comments and printed labels are in Italian** (consistent with the
  EDA dashboard). `README.md`, `CLAUDE.md`, `data/README.md` etc. stay in English.
- All method notebooks default to the same shared sample (`results/sample/sample_{N}_seed{S}.tsv`,
  default N=100 seed=42, drawn from `data/tab/complete.tab` by notebook 00, `split` column kept)
  as a **local smoke-test convenience only** — `SCOPE='sample'` results are no longer committed
  to `results/` (superseded by the `test`-scope run below, which now covers every method); the
  sample TSV itself stays versioned since notebooks still read it whenever `SCOPE='sample'`.
  Extractive notebooks (01/02) also support `SCOPE='full'` (all 56,101 rows, streamed).
  Notebooks 03-04 and 06-11 (BART/PEGASUS/PRIMERA/Qwen/Gemma/Mistral/First-k/Centroid+MMR)
  also support `SCOPE='test'` (the full clean test split, 5,610 rows = 5,622 − 12 dirty,
  streamed via `summ_utils.itera_split`; 10/11 support `'full'` too) — read from
  `os.environ.get('SUMM_SCOPE', 'sample')` so opening
  them by hand in Jupyter is unaffected; `LIMIT` is similarly overridable via `SUMM_LIMIT`.
  `scripts/run_benchmark_test.py` drives all eight unattended, fastest-to-slowest (10/11
  first), setting those
  env vars per subprocess (`jupyter nbconvert --execute --inplace`); `--only 10,11` restricts
  the run to the listed notebook numbers (preflight checks shrink to match) — see its
  docstring and
  `scripts/README.md`. TextRank/LexRank test-split metrics are *derived* by that script from
  their existing `full`-scope per-example CSV (filtered on `split == 'test'`) rather than
  re-run, since metrics are computed per example. The Azure notebook (12) also supports
  `SCOPE='test'` and `SCOPE='full'` (all 56,101 rows, **sequential at standard pricing** — the
  Azure OpenAI Batch API offers no gpt-5-mini in any region, so there is no 50% batch
  discount). The resumable loop makes the multi-day full run splittable across sessions.
- Generation is expensive and **resumable**: summaries append to
  `results/summaries/{method}_{scope}.tsv` one flushed row at a time, and re-runs skip row_ids
  already present. Metrics sections read ONLY saved files — never make evaluation depend on
  re-generating summaries.
- Known caveats (documented in the notebooks/README): `pegasus-multi_news` and
  `PRIMERA-multinews` were trained on this dataset's train split → leakage on train-split sample
  rows (aggregates include per-split means; clean comparison = test split only);
  pyAutoSummarizer's ROUGE uses unique-n-gram sets, not clipped counts, so values aren't
  comparable to the literature; BART/PEGASUS truncate input to 1024 tokens, while PRIMERA
  (notebook 06) takes 4096 with an equal per-article token budget and `<doc-sep>` separators
  (global attention on `<s>` and `<doc-sep>`), so it must see the raw `|||||` separator — it
  passes `prepara=str.strip` to `ciclo_summarization` instead of the default
  `prepara_documento`; the library reloads HF models per call and ignores CUDA, so notebooks
  03/04/06 load the model once themselves and notebook 01 injects a shared SentenceTransformer
  into `loaded_models`.
- **LLM results provenance (`qwen`/`gemma`/`mistral`), historical**: this describes the
  retired `sample`-scope validation, not the currently-committed `test`-scope results. The
  first (sample-scope) summaries/metrics came from local ollama runs of notebooks 07-09
  (qwen/gemma 2026-07-16, mistral 2026-07-17, 100/100 examples each). They replaced an earlier
  import of Federica's LM Studio runs (Mac M4, 2026-07-16; archived CSVs in `notebooks/llm/`,
  imported by
  `scripts/import_llm_results.py`, which verifies 1:1 alignment with the shared sample,
  refuses to overwrite existing summary TSVs, and recomputes metrics with the shared
  normalization — the CSVs' own metric values use different settings and must not be mixed
  in; their BERTScore column is not carried over). An interim ollama run of mistral
  (2026-07-16) mistakenly used Mistral Small ~24B (tag `mistral-small`) and was discarded and
  redone with the correct `mistral:7b-instruct-v0.3-q4_K_M` (documented in notebook 09).
  Deliberate deviations of the ollama runs from the original: documents pass through
  `prepara_documento` (separator → newline) instead of raw text; no LM Studio-specific
  `enable_thinking` extra_body; mistral's system prompt uses the real `system` role. Because
  of resumability, regenerating on top of a TSV from a different run would mix runs — delete
  the TSV first.
- **Azure notebook (12) conventions**: same zero-shot English prompt as 07-09, documents
  through `prepara_documento`, the `/no_think` prefix deliberately dropped (qwen artifact).
  **GPT-5-mini deviates from the 07-09 params** (documented in notebook 12): it is a reasoning
  model, so no `temperature` (only default accepted) and `max_completion_tokens=1500` with
  `reasoning_effort='minimal'` — reasoning tokens consume the completion budget before visible
  output, the same failure mode as gemma (notebook 08). GPT-5-mini was chosen because Azure
  retired the gpt-4o-mini family (deprecating state, no new deployments) and gpt-4.1-mini is on
  the same retirement path. Credentials come ONLY from environment variables
  (`AZURE_OPENAI_ENDPOINT` = the bare resource root, no path; `AZURE_OPENAI_API_KEY`) — never
  hardcode keys. For Azure OpenAI, `model=` in requests is the **deployment name**, not the
  model name. Azure's content filter deterministically rejects some news clusters
  (hate/violence at medium severity) with `content_filter` errors before the model sees them:
  those rows stay absent from the TSVs and are not retryable without a custom high-only filter
  attached to the deployment (139/5,610 test rows missing, 2026-07-17 run).
- **gemma coverage**: full (5,610/5,610 in the committed `test`-scope run; also full 100/100 in
  the retired sample-scope run), thanks to `MAX_TOKENS=1500` in notebook 08. The original LM
  Studio run had 81/100 empty responses (only 19 evaluated): Gemma 4 emits reasoning tokens
  that exhaust `max_tokens=300` before any visible content (`finish_reason=length`, empty
  content) — reproduced via ollama, and still occasionally seen even at 1500 (one `test`-scope
  retry needed on 2026-07-24). Notebook 05 still computes the shared row_id intersection only
  over methods with ≥`COPERTURA_MINIMA` (50) rows, as protection against future low-coverage
  runs (which are shown with their own `n_esempi`).
- **METEOR unreliable for degenerate output**: pyAutoSummarizer's `meteor()` formula
  (`meteor = fmean * (1 - penalty**3)`) is unbounded for pathological inputs. In the `test`-scope
  run, PEGASUS hits it on 2/5,610 rows — a beam-search repetition loop (row_id 51178, meteor
  -1959.12) and a likely source/summary mismatch (row_id 56099, meteor -2.10) — dragging its
  reported mean METEOR from a true ≈0.42 down to 0.079; ROUGE/BLEU/row counts are unaffected.
  LexRank has one much milder case (row_id 55805, meteor -1.24 out of 5,588 rows), negligible
  effect on its mean. Documented as a caveat in notebook 05 and the README — no per-example CSV
  or aggregate JSON was altered to compensate for it.

- **G-Eval / LLM-as-a-Judge (notebook 14 + `scripts/run_geval.py`)**: adds coherence /
  consistency / fluency / relevance scores (1-5) for the test split — the only metric here not
  anchored to the human reference. Judge is **`gpt-5.4-mini`** on Azure (its own GlobalStandard
  deployment, `AZURE_GEVAL_DEPLOYMENT`, same `AZURE_OPENAI_*` credentials as notebook 12), chosen
  because it is independent of all 13 benchmarked methods (no self-judging of `gpt5mini`) *and*
  newer than every generator — which matters mainly for the consistency dimension. Verified live:
  DeepSeek/Grok/Llama/Mistral are **not deployable** on this AIServices account, and
  **`gpt-5.1-mini` does not exist**. `psr.g_eval()` is unusable (no `base_url` → can't reach
  Azure; hardcoded `max_tokens=5`/`temperature=0.0`; reads the source from `self.full_txt`, which
  breaks the `crea_valutatore()` shared-instance pattern; one call per dimension), so it is
  reimplemented in `summ_utils.py` with the **rubrics kept verbatim** — `RUBRICHE_GEVAL` is
  derived mechanically from `PROMPT_GEVAL_ORIGINALI`, don't hand-edit it. Being a reasoning
  model, it follows notebook 12's rules (no `temperature`, `max_completion_tokens=1500`,
  `reasoning_effort='minimal'`); runs are **not bitwise reproducible** — the committed
  `geval_cache_{scope}.jsonl` is the reproducibility artifact and metrics re-derive from it for
  free (`--solo-metriche`). Hard rules to respect:
  - **Never merge G-Eval columns into the standard per-example CSV.** `valuta_e_salva`'s inner
    mean sums *every* column over *every* row, and the judge leaves some rows uncovered (Azure
    content filter, parse failures) → `KeyError`; restricting the evaluated rows instead would
    rewrite the 13 committed CSVs and change already-published `n_esempi` and means. Notebook 05
    attaches them with a LEFT merge and reports coverage as `n_geval`.
  - **Message order is a contract**: constant `system`, then the truncated source (shared by a
    row's 13 methods), summary last. That is what makes Azure's prompt cache hit; the work unit
    is the *row* (its 13 judgments run sequentially in one thread), parallelism is *between*
    rows. Reordering costs roughly 3.5x on input tokens.
  - Source truncated to **3,500 words**, which leaves 91.5% of test-split clusters intact
    (p90 = 3,244 words, max = 35,362). Scope is 72,681 judgments (not 13x5,610 — coverage is
    uneven).
  - **Cost is entirely input-bound.** Measured: `reasoning_effort='minimal'` emits **zero**
    reasoning tokens, so output is ~30 tokens/judgment (~$10 total). Everything else rides on the
    prompt-cache hit rate. Two things cap it, and neither is the truncation limit:
    (a) Azure does not cache prefixes under **1,024 tokens**, so **14.8% of test rows (830/5,610)
    never cache at all**; (b) even in cacheable rows, hit rate falls with concurrency because
    GlobalStandard routes each request to any backend instance and the cache is per-instance —
    measured **64.8% of calls at 2 threads vs 45.9% at 8**. Net: **~$87 at 2 threads (~12 h)
    vs ~$107 at 8 threads (~3 h)**. Do NOT extrapolate from a single isolated row: one row run
    alone hit 12/13 and suggests a ~$56 run, which is not reproducible at scale.
  - There is **no real-time Azure cost API** (Cost Management lags 8-24 h). Costing is done from
    each response's `usage`, with unit prices from the public Azure Retail Prices API. Note the
    pre-existing trap that `run_benchmark_test.py::deriva_metriche_test` recomputes textrank /
    lexrank aggregates using `COLONNE_METRICHE` only, so a driver re-run **drops their BERTScore
    columns** — G-Eval is immune because its files are separate.
  - **Currency is not cosmetic.** The Retail Prices API defaults to USD; a subscription billing
    in another currency gets a genuinely different, non-FX price list from Azure (verified: the
    EUR list here is a flat ~0.8776x the USD number on every meter, not a live exchange rate).
    A run tracked at $36.00 (USD default) corresponded to €31.59 of actual EUR credit consumed —
    a 12% gap from currency, not from Cost Management's reporting lag. `su.prezzi_retail_azure`
    and `scripts/run_geval.py` take `valuta`/`--valuta`; it must match the subscription's real
    billing currency, or the token accounting stays exact but the printed figure won't.

## Working with the data files

The files in `data/` are large (the train source file is ~500MB); avoid loading them
wholesale in tooling — prefer streaming/line-by-line reads (as `_generate_examples` does) or
sampling a subset of lines when inspecting content.

## Licensing

The dataset is released for **non-commercial research and educational purposes only** (full
Dataset Usage Agreement in `LICENSE`, condensed in `README.md`); keep this in mind before
proposing any commercial use of the data.
