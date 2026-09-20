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
  analyze_dataset.py   # Streaming corpus-wide EDA over data/text/ — writes dataset_stats.json, the figures multi_news_dashboard.html embeds
  dataset_stats.json   # Its output: one aggregated JSON over train+val+test (committed)
  convert_to_tab.py    # Regenerates data/tab/ from data/text/ (Orange format), dropping dirty rows
  import_llm_results.py  # One-off importer of the archived LM Studio LLM runs (notebooks/llm/*.csv) into results/ — superseded by the ollama re-runs, kept for provenance
  run_benchmark_test.py  # Unattended driver: runs notebooks 03-04/06-11/15-17 with SCOPE='test' back-to-back (--only N,N to select a subset), derives textrank/lexrank test metrics from their full run, re-runs 05b/05d; --scope test_budgetref runs the 7 extractive notebooks at reference-length budget (issue #16)
  applica_budget.py    # Ceiling (1.25x reference length) + re-scoring of all 18 methods into the test_budgetref scope (issue #16)
  budget_lunghezza.json  # T(n_articoli) table from notebook 18 — an analysis artifact, NOT the containment budget (which is the reference length)
  pilota_giudice_deepseek.py  # Paired second-judge pilot (DeepSeek vs gpt-5.4-mini) — tests the family-bias
                       # hypothesis on 7 methods; separate cache/JSON, touches nothing committed
  run_geval.py         # Unattended driver for the G-Eval backfill (notebook 14): staged --righe/--pilota/full run, --budget hard stop, --costo offline cost report, --solo-metriche re-derivation
requirements-notebooks.txt  # Dependencies for the benchmark notebooks (pyAutoSummarizer, openai etc.)
notebooks/             # Summarization benchmark — see "Summarization benchmark" section below
  README.md            # Run order, parameters, runtimes, Colab instructions (Italian)
  summ_utils.py        # Shared routines: data loading, resumable generation loop, metrics, G-Eval, and the comparison helpers of 05a-d (METODI_BENCHMARK, COLORI_METODI, carica_scope, tabella_medie, barre_metriche, viste_metriche)
  0X_*.ipynb           # 00 sample prep, 01-04 and 06-09 one method each; 05a/b/c/d comparison
                       # (full / test / test_budgetref / before-after), helpers in summ_utils
  1X_*.ipynb           # 10 First-k baseline, 11 Centroid+MMR, 12 Azure AI Foundry GPT-5-mini (scopes sample/test/full), 13 BERTScore backfill, 14 G-Eval backfill, 15 LSA (2 variants), 16 SBERT clustering (2 variants), 17 LDA, 18 per-cluster length analysis, both scopes test/test_budgetref side by side — budgetref lengths are POST-ceiling (from the CSVs applica_budget.py rewrites); Vista 3b reads the pre-ceiling `*_test_budgetref.tsv` of the 15 regenerated methods (issue #15/#16, generates nothing, writes `analisi_lunghezze_{scope}.json` per scope), 19 second-judge
                       # validation of the G-Eval judge (generates nothing, spends nothing); ex Azure 11-12 (Claude Haiku, DeepSeek) removed — recoverable from git history
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
  figures/{method}/    # PNGs from the "how it works" explanatory sections of notebooks 11 and
                       # 15-17 (written only when SALVA_FIGURE is on); illustrative, no metric reads them
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
- It is regenerable via `python scripts/analyze_dataset.py`, which streams `data/text/` in a
  single pass and writes `scripts/dataset_stats.json` — the aggregated JSON the dashboard's
  `const D` is built from (verified to match: 56,216 examples, 154,530 articles, vocab 494,577).
  The script deliberately omits the heavy metrics (novel n-grams, extractive fragment
  coverage/density, language detection); those appear in the dashboard only as static reference
  values from the paper. Editing the dashboard's numbers by hand means editing `const D`.

## Summarization benchmark (`notebooks/` + `results/`)

Eighteen method slugs: the First-k / Lead positional baseline in two sentence-segmentation
variants (notebook 10, slugs `firstk_psr`/`firstk_nltk`); TextRank, LexRank (extractive) plus
a custom scikit-learn Centroid-based (MEAD) + MMR in two vectorization variants (notebook 11,
slugs `centroid_mmr` TF-IDF / `centroid_mmr_bert` BERT); BART `facebook/bart-large-cnn`,
PEGASUS `google/pegasus-multi_news`, PRIMERA `allenai/PRIMERA-multinews` (specialized
abstractive); three local general-purpose LLMs — Qwen2.5-7B-Instruct, Gemma 4 E4B,
Mistral-7B-Instruct-v0.3 (notebooks 07/08/09, method slugs `qwen`/`gemma`/`mistral`); one
cloud LLM on Azure AI Foundry — GPT-5-mini (notebook 12, slug `gpt5mini`); plus five
unsupervised extractives built on scikit-learn — LSA/TruncatedSVD in two selection variants
(notebook 15, slugs `lsa` top-k by latent norm / `lsa_steinberger` greedy with deflation),
clustering over MiniLM sentence embeddings with medoid selection in two variants (notebook 16,
slugs `sbert_kmeans` / `sbert_agglom`), and LDA topic modeling with topic-proportional sentence
allocation (notebook 17, slug `lda`). TextRank/LexRank
and BART/PEGASUS
run via pyAutoSummarizer, PRIMERA directly via `transformers` (notebook 06), the local LLMs via
the `openai` client against ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`),
GPT-5-mini via `openai.OpenAI` against the Azure OpenAI **v1 route**
(`<endpoint>/openai/v1/`, no dated api-version) — all scored with pyAutoSummarizer's
ROUGE-1/2/L, BLEU, METEOR implementations. Two more Azure notebooks (Claude Haiku 4.5 and
DeepSeek-V3.2, slugs `haiku`/`deepseek`, formerly numbered 11/12) were removed because their
Foundry deployments could not be created; recover them from git history if retried (the
numbers were since reused: 10/11 are now the First-k and Centroid+MMR baselines, 12 the Azure
GPT-5-mini notebook). Baseline notebooks 10/11 and the unsupervised extractives 15-17 support
all three scopes (`sample`/`test`/`full`, `SUMM_SCOPE`/`SUMM_LIMIT` env overrides like
03-04/06-09) and are driven by `run_benchmark_test.py`, listed first as the fastest methods;
15/16 (like 10/11) generate their two variant slugs in one execution. Conventions to
respect:

- **All notebook documentation, comments and printed labels are in Italian** (consistent with the
  EDA dashboard). **So is every `.md` in the repo**: `README.md`, `data/README.md`,
  `scripts/README.md` and `notebooks/README.md` were translated to Italian, matching the language
  of the project work itself — keep new documentation in Italian. Two exceptions, both
  deliberate: **this file** (`CLAUDE.md`, agent-facing) stays in English, and
  `Multi-News_paper.md` stays in English because it is a verbatim copy of the published paper,
  cited as a primary source — never translate it. **Italian terminology for the length
  containment (issue #16)**: the truncation at 1.25x the reference is the **«tetto»** and the
  regeneration of short methods up to the budget is the **«rigenerazione a budget»** — never
  «soffitto»/«pavimento», which are calques of ceiling/floor the author rejected (2026-09-19;
  code identifiers follow: `FATTORE_TETTO_BUDGET`, `su.tetto_riferimento`, JSON key `tetto`).
  Likewise «limite» rather than «cap». This file, being English, keeps ceiling/floor. `LICENSE` likewise stays in its original
  English as the binding text; `README.md` carries only an Italian courtesy summary of it.
- All method notebooks default to the same shared sample (`results/sample/sample_{N}_seed{S}.tsv`,
  default N=100 seed=42, drawn from `data/tab/complete.tab` by notebook 00, `split` column kept)
  as a **local smoke-test convenience only** — `SCOPE='sample'` results are no longer committed
  to `results/` (superseded by the `test`-scope run below, which now covers every method); the
  sample TSV itself stays versioned since notebooks still read it whenever `SCOPE='sample'`.
  Extractive notebooks (01/02) also support `SCOPE='full'` (all 56,101 rows, streamed).
  Notebooks 03-04, 06-11 and 15-17 (BART/PEGASUS/PRIMERA/Qwen/Gemma/Mistral/First-k/
  Centroid+MMR/LSA/SBERT-clustering/LDA)
  also support `SCOPE='test'` (the full clean test split, 5,610 rows = 5,622 − 12 dirty,
  streamed via `summ_utils.itera_split`; 10/11 and 15-17 support `'full'` too) — read from
  `os.environ.get('SUMM_SCOPE', 'sample')` so opening
  them by hand in Jupyter is unaffected; `LIMIT` is similarly overridable via `SUMM_LIMIT`.
  `scripts/run_benchmark_test.py` drives all eleven unattended, fastest-to-slowest (10/17/15/16
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
- **Azure's content filter is not stable across runs, and `n_geval` shows it.** All 18 methods
  now have BERTScore and G-Eval (issue #12), but the two backfills ran two weeks apart — the 13
  original methods on 2026-08-17, the five notebook 15-17 methods on 2026-08-31 — and the rows
  Azure rejects changed in between. *Within* the first run rejection was effectively
  source-determined (union 366 rejected rows across the 13, intersection 358 — nearly the same
  rows for every method). *Across* runs it is not: 155 of those rows now pass for all five new
  methods, 91 new ones are rejected, and among the five the rejected set varies far more (union
  294, intersection 187). Hence coverage of 94.7-96.7% for the five vs 93.5-93.6% for the
  thirteen — a property of the filter at run time, not of the methods. Impact on the comparison
  is negligible: restricted to the 5,091 rows judged for all 18, no mean moves by more than
  0.010 (95% CI is ±0.03). Do not read `n_geval` differences as saying anything about the
  methods, and do not "fix" it by re-judging the 13 — that is a fresh ~€70 run that would change
  published numbers. **What WAS done (2026-09-19, €2.09): the 5,855 failed judgments — 457
  rows, of which 381 had passed the same filter two days earlier in the budgetref run — were
  retried with `run_geval.py --scope test --riprova-errori`.** That fills gaps without touching
  any existing judgment: failures fell to 1,071, success to **99,550/100,621 (98.9%)**, coverage
  per method **96.8–99.0%**, rows judged for all 18 **5,091 → 5,363**. Ranking unchanged, max
  mean shift 0.010 (pegasus) — the rows the filter had flagged are not systematically different.
  The old "13 vs 5" coverage gap described above is therefore historical.
- The comparison notebooks (05b/05c via `su.viste_metriche`) chart BERTScore/G-Eval over the *subset* of methods that have the column (naming
  the excluded ones) instead of requiring it from all — keep that even now that all 18 have both;
  don't "fix" it into an `all(...)` gate.
- **Notebook 05 was split into 05a/05b/05c/05d on 2026-09-19** (full / test / test_budgetref /
  before-after) because it had grown past 13 cells and 1 MB of outputs. Everything they share
  lives in `summ_utils` (section *Confronto fra metodi*): `METODI_BENCHMARK` (the 18 slugs in
  canonical order), `COLORI_METODI` + `INK/INK2/MUTED/GRID/SURFACE`, `METRICHE_CHIAVE`,
  `carica_scope(metrics_dir, scope, metodi)` (LEFT-merges the G-Eval files, intersects row_ids
  over methods with >= `COPERTURA_MINIMA`), `tabella_medie`, `con_metrica`, `barre_metriche`,
  and `viste_metriche` — the one function that renders the standard table + charts, called by
  05b and 05c so the two scopes are presented identically. 05c is deliberately "05b on the
  budget scope", 05d holds the before/after (the old Vista 3). Drivers re-run 05b+05d after a
  `test` run and 05c+05d after a `test_budgetref` run (`run_geval.py --no-05` skips both). Keep
  new comparison views in the notebook of their scope, not in a fifth file.
- **Notebook 13 is incremental**: it lists all 18 slugs and skips any method whose per-example
  CSV already has the `bertscore_*` columns, because `valuta_e_salva` rewrites that method's CSV
  and JSON wholesale. `BERTSCORE_FORZA=1` forces a full recompute. This is what let the five new
  slugs be backfilled without regenerating the 13 published files.
- **Reference-length scope `test_budgetref` (issue #16, length-matched / oracle-length)**:
  notebook 18 showed that inside a single cluster the longest of the 18 summaries is a median
  8.5x the shortest, and that no input-only predictor (article count, source length,
  regressions) approximates the reference length well — the best brackets it within ±25% in
  58% of clusters vs 54% for a constant (its Vista 5). So the containment budget is the
  **reference length itself**, per cluster (`su.budget_riferimento`): only the *length* is
  gold, never the content, and the numbers answer "content selection at matched length",
  never "production performance" — say so wherever they appear; they never replace the
  `test` headline numbers. Mechanics: `SUMM_SCOPE=test_budgetref` reads the test rows
  (`su.split_base`) and writes `*_test_budgetref*` files; `ciclo_summarization(budget=...)`
  passes the per-row budget to `genera(doc, budget)`; the 7 extractive notebooks
  (01/02/10/11/15/16/17, driver `run_benchmark_test.py --scope test_budgetref`, executed to
  `results/notebook_runs/` not in-place) swap the sentence budget for a word budget via
  `su.seleziona_per_budget` (keeps each method's own ordering — rank order for 01/02,
  document order elsewhere; crossing sentence kept if more than half fits; `K_LATENTE` /
  `N_CLUSTER` stay 11, they are structural); the LLM notebooks (07/08/09/12) put the budget
  in the prompt (`PROMPT_USER_BUDGET` derived from `PROMPT_USER`: "about N words ... at least N
  words" with N = `FATTORE_RICHIESTA`(1.2) x budget — calibrated on a 30-row qwen pilot, where
  the bare "approximately N" gave 0.65N and 13% in band; `MAX_TOKENS_BUDGET=1500` because
  qwen's 200 / mistral's 300 cannot reach a 300-word target) and follow it only approximately
  — re-pilot each model, the factor may differ; `scripts/applica_budget.py` then truncates everyone to 1.25x the budget and
  re-scores (overwriting the notebooks' raw `_test_budgetref_*` metrics — the post-ceiling
  numbers are the ones to read). bart/pegasus/primera get the ceiling only and stay below
  band — a deliberate choice, not just cost: their floor would come from `min_length` in tokens,
  which forces beam search past the model's natural length and is the very mechanism behind the
  repetition loops already documented for PEGASUS (row 51178, METEOR −1959, at its *natural*
  length); on bart, which would have to quadruple its output, that would measure
  forced-lengthening damage rather than content selection, while pegasus (0.83x) and primera
  (0.97x) already sit at or inside the band unaided. bart (0.26x, 0% in band) is the declared
  exception of the comparison — do not "fix" it by regenerating; G-Eval IS recomputed (see below). Notebook 05d is the before/after view, 05c the budget scope on its own. Outcome of
  the 2026-09-13/17 runs (15 of 18 regenerated, ceiling on all 18): the 15 are in band in
  71-100% of clusters (11 extractives 93-99%, gpt5mini 100%, gemma 99%, qwen 79%, mistral 71%).
  Per-cluster max/min goes 8.5x → 4.7x across all 18, but that 4.7 is almost entirely `bart`
  (0.26x = the minimum in nearly every cluster, while the ceiling caps the max at 1.25x):
  **without bart it is 3.8x → 1.6x, and over the 15 regenerated 3.7x → 1.4x** — quote those, the
  all-18 figure understates the containment. Ranking shifts: `lda` loses everything (recall
  0.499 → 0.348, last extractive in F1 at 0.338); `firstk_*` climb from 9th-10th to 5th-6th
  (0.368), above far more elaborate selectors; the short methods *gain* because the floor gives
  them room they weren't using (qwen 0.344 → 0.351, mistral 0.354 → 0.365) — the confound cut
  both ways; `primera` (0.446) and `pegasus` (0.424) stay on top, their advantage was not length.
  The prompt factor is calibrated PER MODEL on a 30-row pilot (qwen 1.2 undershoots, gemma 0.9
  and gpt5mini 0.9 overshoot the "at least" constraint, mistral 1.0 but very dispersed: p10 0.76x,
  p90 1.98x). NOTE on Azure's content filter: its severity is configured PER RESOURCE, not
  merely unstable across runs — gpt5mini's first budgetref run on the new resource covered only
  5,250/5,610 rows (360 rejections, all content_filter) vs 5,471 on the old one. Attaching a
  **custom permissive content filter** to the deployment and rerunning only the missing rows
  recovered 312 of the 360: final coverage **5,562/5,610 = 99.1%**, the highest of any Azure run
  in the project, and the 18-method common intersection rose to **5,527** clusters (above the
  5,437 it started from). The remaining 48 rows are refused even by the permissive policy.
  Provenance stays clean — same resource, model, prompt and params; only the filter policy
  differs, and a filter gates *access* without altering generation. **Reproducing this run
  requires the custom filter on the deployment**, otherwise coverage falls back to ~93.6%. The
  `test`-scope gpt5mini numbers were produced under the default policy. Row 50540 (a
  degenerate list-of-breweries source) blows up pyAutoSummarizer's unbounded METEOR for
  `lda` (−77,557) and `lexrank` (−3,074) — same pathology as PEGASUS row 51178, same policy:
  documented, no file altered.
- **LDA summary length**: notebooks 15/16/17 share `N_SENTENCES = 11`, but LDA's
  topic-proportional allocation picks much longer sentences — 477 words per summary vs 216-264
  for the other four. Its ROUGE-1 *recall* (0.499) and METEOR (0.511) lead the group for that
  reason alone, while its F1 (0.351) only ties `lsa` and trails `lsa_steinberger` (0.376).
  Compare these five on F1, not recall. Settled by the `test_budgetref` scope (below): at
  matched length `lda` drops to last among the extractives in F1 (0.338) and its recall falls
  0.499 → 0.348 — the advantage was entirely length.
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
  retry needed on 2026-07-24). `su.carica_scope` still computes the shared row_id intersection only
  over methods with ≥`COPERTURA_MINIMA` (50) rows, as protection against future low-coverage
  runs (which are shown with their own `n_esempi`).
- **METEOR unreliable for degenerate output**: pyAutoSummarizer's `meteor()` formula
  (`meteor = fmean * (1 - penalty**3)`) is unbounded for pathological inputs. In the `test`-scope
  run, PEGASUS hits it on 2/5,610 rows — a beam-search repetition loop (row_id 51178, meteor
  -1959.12) and a likely source/summary mismatch (row_id 56099, meteor -2.10) — dragging its
  reported mean METEOR from a true ≈0.42 down to 0.079; ROUGE/BLEU/row counts are unaffected.
  LexRank has one much milder case (row_id 55805, meteor -1.24 out of 5,588 rows), negligible
  effect on its mean. Documented as a caveat in notebook 05b and the README — no per-example CSV
  or aggregate JSON was altered to compensate for it.

- **G-Eval / LLM-as-a-Judge (notebook 14 + `scripts/run_geval.py`)**: adds coherence /
  consistency / fluency / relevance scores (1-5) for the test split — the only metric here not
  anchored to the human reference. Judge is **`gpt-5.4-mini`** on Azure (its own GlobalStandard
  deployment, `AZURE_GEVAL_DEPLOYMENT`, same `AZURE_OPENAI_*` credentials as notebook 12), chosen
  because it is independent of all 18 benchmarked methods (no self-judging of `gpt5mini`) *and*
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
    rewrite the committed CSVs and change already-published `n_esempi` and means — drastically so
    for the five notebook 15-17 methods, whose G-Eval coverage is only ~60%. `su.carica_scope`
    attaches them with a LEFT merge and reports coverage as `n_geval`.
  - **Message order is a contract**: constant `system`, then the truncated source (shared by a
    row's methods), summary last. That is what makes Azure's prompt cache hit; the work unit
    is the *row* (its judgments run sequentially in one thread), parallelism is *between*
    rows. Reordering costs roughly 3.5x on input tokens. The amortization weakens when a row has
    few methods to judge: the 15-17 backfill pays the full prefix once per 5 judgments instead of
    once per 13, and the cached-input share fell from 59% to 56% cumulative.
  - Source truncated to **3,500 words**, which leaves 91.5% of test-split clusters intact
    (p90 = 3,244 words, max = 35,362). Scope is 100,621 judgments (not 18x5,610 — coverage is
    uneven), all of them run, for €95.19 (94,766 succeeded, 5,855 hit the content filter); after the 2026-09-19 retry of
    the failures, €97.28 and 99,550 succeeded, 1,071 failed.
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
  - **The judge was validated against a second judge of different lineage, and it held**
    (notebook 19 + `scripts/pilota_giudice_deepseek.py`, 2026-09-17/18). The concern was real:
    `gpt-5.4-mini` is GPT-5-family and so is one of the *judged* methods, `gpt5mini`, which is
    3rd of four LLMs on BERTScore/ROUGE-1 F1 but 1st overall on G-Eval (4.89). 6,799 paired
    judgments over 983 clusters were re-judged by `DeepSeek-V3.2-Speciale` with an identical
    prompt, on the 4 LLMs plus **3 non-LLM controls** (the controls are what separate "the judge
    likes LLM prose" from "the judge likes its own family" — without them the design concludes
    nothing). Result: **rank order identical, Spearman rho = 1.000, zero inversions over seven
    positions**, and the family-bias contrast (`delta(gpt5mini) - mean(delta of the other 3
    LLMs)`, computed per row) is **+0.049, 95% CI [+0.022, +0.076]**. Family bias predicted a
    **negative** contrast, so the sign is wrong for the hypothesis — correcting for this effect
    would *widen* gpt5mini's G-Eval lead, not shrink it. Two things not to get wrong when
    quoting this: (a) **zero is outside the CI** — with ~970 paired clusters the SE is 0.014 and
    five hundredths of a point is statistically detectable, so do NOT write "compatible with
    zero"; what makes the result conclusive is the sign, not the significance; (b) the two
    judges are **not interchangeable** — DeepSeek is harsher on non-LLM methods (−0.276 vs
    +0.005 on LLMs), widening the LLM-vs-extractive gap by 0.28, concentrated in `coherence`
    (−0.32) and `relevance` (−0.30, almost all of it on the controls: −0.675 vs −0.017), while
    `fluency` is identical and `consistency` moves the other way (+0.19). The LLM-prose
    preference is a property of the LLM-as-a-Judge paradigm, shared by both, not a family
    distortion. **DeepSeek stays a pilot, for throughput not quality**: 20 RPM nominal but
    **6.5 judgments/min measured** (32% of nominal, rest is 429 backoff) — 6,997 judgments in
    17.99 h for €9.16, extrapolating to **10.8 days and €132 per scope** vs 3–12 h and €93 for
    `gpt-5.4-mini`, which is faster *and* cheaper per judgment because GlobalStandard has the
    **prompt cache** that MaaS deployments lack. Raising `--thread` does not help. So the
    `test_budgetref` G-Eval runs on `gpt-5.4-mini` (old account), keeping continuity with the
    published numbers. Its cache `geval_cache_test_deepseek.jsonl` is a paid artifact — commit
    it — and is kept SEPARATE from `geval_cache_test.jsonl`; never merge them.
  - **`GEVAL_SCOPE=test_budgetref` is supported** (`run_geval.py --scope test_budgetref`).
    Three things differ from `test`, all in notebook 14: rows come from `su.split_base(SCOPE)`;
    `percorso_riassunti` prefers `{m}_{scope}.tsv`, then `_test`, then `_full`; and the
    **1.25x ceiling is re-applied per row** (`su.tetto_riferimento`, the factor now lives in
    `summ_utils` and `applica_budget.py` imports it) because the budget TSVs are pre-ceiling and
    bart/pegasus/primera have none — the judge must see the exact text the metrics were scored
    on. Judgments whose post-ceiling text is byte-identical to the `test` text are **copied from
    `geval_cache_test.jsonl`** instead of re-bought (entries carry `riuso_da: 'test'` and zero
    tokens; `--costo` excludes them from the per-judgment cost): 15,061 of 100,712 — bart 94%,
    pegasus 84%, primera 74%. The notebook executes to `results/notebook_runs/test_budgetref/`,
    not in place, so the committed notebook 14 keeps its `test` outputs.
    `applica_budget.py::config_base` carries the *notebook's* config into the budget-scope
    aggregate (stripping its own `ambito_budget` block so `--forza` is idempotent) and falls
    back to the `test` config only for the ceiling-only methods; until 2026-09-19 it read the
    `test` config unconditionally, so the 15 budget aggregates documented test-scope params
    (fixed one-off by re-capturing each notebook's `config` via a stubbed `valuta_e_salva`;
    metrics untouched).
    **Outcome (run 2026-09-18/19, 100,712 judgments, 99,040 succeeded, €76.26 useful spend):
    length was NOT G-Eval's confound.** Where matched length overturned ROUGE recall and
    reshuffled the ROUGE ranking, **the top 11 G-Eval positions are identical** before/after
    and the largest move is 0.34 on a 5-point scale — consistent with G-Eval scoring against
    the *source*, not overlap with the reference, and retroactively justifying the choice not
    to recompute it for lda. Specifics worth quoting: the two longest extractives *gain* from
    being cut (`lexrank` +0.34 at 450→216 words, `textrank` +0.27 at 368→215; coherence /
    consistency / fluency all +0.4–0.6, relevance slightly down — they remain last);
    **relevance is the only length-sensitive dimension**, the counterpart of recall (`lda`
    −0.50, `centroid_mmr*` −0.27/−0.29, `lsa_steinberger` −0.23 on relevance while their other
    three dimensions rise) — the two effects ROUGE F1 sums into one number, here separated;
    the lengthened LLMs diverge: `qwen` (140→216) +0.15 and **overtakes `mistral`** (the only
    inversion in the top 11), `mistral` (161→233) −0.13 entirely on consistency (−0.24) — forced
    to write more, qwen adds pertinent content, mistral adds things not in the source; `gpt5mini`
    (241→209) +0.001. Coverage is *higher* than in `test` (5,483–5,554 vs 5,215–5,401 per
    method): same resource and filter, the content filter simply varies run to run, favourably
    this time. Notebook 05d includes `geval_media` and a before/after chart.
  - **Never advise "Ctrl-C and restart" on a running G-Eval without confirming the kernel died.**
    On Windows, Ctrl-C to the driver did NOT kill its `jupyter nbconvert` kernel (separate
    process group); on 2026-09-18 the orphan kept judging for 10 h alongside the relaunched run,
    both walking the same `--casuale` seed-42 order → **50,275 judgments paid twice (€44.42
    wasted; €116.40 billed vs €71.99 useful)** and a cache corrupted by interleaved appends. The
    driver now (a) kills the whole process tree on Ctrl-C (`termina_albero`, `taskkill /T`) and
    (b) holds a per-cache PID lock (`geval_cache_{scope}.lock`) and refuses to start while
    another live run owns it. The committed budgetref cache is the deduplicated rewrite (one
    entry per pair, success beats error, else first occurrence); `--costo` on it therefore
    reports the *useful* spend, not the bill. Judgments are not bitwise reproducible, so the two
    runs' duplicates could differ — the kept one is the first written, which is arbitrary but
    deterministic.

## Working with the data files

The files in `data/` are large (the train source file is ~500MB); avoid loading them
wholesale in tooling — prefer streaming/line-by-line reads (as `_generate_examples` does) or
sampling a subset of lines when inspecting content.

## Licensing

The dataset is released for **non-commercial research and educational purposes only** (full
Dataset Usage Agreement in `LICENSE`, condensed in `README.md`); keep this in mind before
proposing any commercial use of the data.
