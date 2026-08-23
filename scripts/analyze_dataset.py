#!/usr/bin/env python3
# coding=utf-8
"""Aggregated EDA over the whole Multi-News corpus (train+val+test combined).

Streams the six ``data/text/*`` files line-by-line (like ``_generate_examples`` in
``multi_news.py``) and emits a single aggregated JSON — no per-split breakdown — that
backs ``multi_news_dashboard.html``.

Scope note (agreed with the user): the *heavy* metrics are intentionally NOT computed
here — no novel n-gram %, no extractive-fragment coverage/density, no language detection.
Those appear in the dashboard only as static reference values from the paper. Everything
here is a light single streaming pass: splitting, sentence heuristics, hashing and a
global vocabulary set.

Run from the repo root:  python scripts/analyze_dataset.py
Writes: scripts/dataset_stats.json
"""

import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEXT = os.path.join(ROOT, "data", "text")
OUT = os.path.join(HERE, "dataset_stats.json")

SPLITS = ["train", "val", "test"]
SEP = "|||||"                      # article separator (space-padded in the file)
NLCHAR = "NEWLINE_CHAR"            # literal newline token inside articles

SENT_RE = re.compile(r"[.!?]+")
WORD_RE = re.compile(r"[a-z0-9']+")   # for the vocabulary estimate (C-speed findall)
WS_RE = re.compile(r"\s+")

# anomaly thresholds (words)
SUM_MIN, SUM_MAX = 20, 600
SRC_MIN = 50


def clean_source(raw):
    """Restore newlines, drop the article separator; return plain text."""
    return raw.replace(NLCHAR, "\n").replace(SEP, " ")


def n_sentences(text):
    return sum(1 for s in SENT_RE.split(text) if s.strip())


def norm_hash(text):
    """SHA-1 over whitespace-normalised lowercase text (exact-dup key)."""
    return hashlib.sha1(WS_RE.sub(" ", text.strip().lower()).encode("utf-8")).hexdigest()


def fingerprint(text, n=15):
    """Hash of the first n normalised words (cheap near-dup key)."""
    words = WS_RE.sub(" ", text.strip().lower()).split()[:n]
    if not words:
        return None
    return hashlib.sha1(" ".join(words).encode("utf-8")).hexdigest()


def main():
    # per-example arrays (aggregated across all splits)
    n_articles = []
    src_words = []
    src_sents = []
    tgt_words = []
    tgt_sents = []
    comp = []            # source_words / summary_words (only when tgt_words > 0)
    comp_full = []       # aligned with the arrays above; np.nan where undefined

    vocab = set()

    # exact / near duplicate bookkeeping
    tgt_exact = {}       # hash -> count
    tgt_finger = {}      # fingerprint -> count
    src_exact = {}       # hash -> count

    # anomaly counters
    empty_src = 0
    empty_tgt = 0
    le1_article = 0
    tgt_out_low = 0
    tgt_out_high = 0
    src_short = 0

    # outlier refs: (words, "split:line")
    src_ref = []   # for longest / shortest source
    tgt_ref = []   # for longest / shortest summary

    total_examples = 0

    for split in SPLITS:
        src_path = os.path.join(TEXT, f"{split}.src.cleaned")
        tgt_path = os.path.join(TEXT, f"{split}.tgt")
        print(f"[{split}] streaming {os.path.basename(src_path)} ...", flush=True)
        with open(src_path, encoding="utf-8") as sf, open(tgt_path, encoding="utf-8") as tf:
            for i, (sline, tline) in enumerate(zip(sf, tf)):
                total_examples += 1
                s_raw = sline.strip()
                t_raw = tline.strip()

                # --- articles ---
                arts = [a for a in s_raw.split(SEP) if a.strip()] if s_raw else []
                na = len(arts)
                n_articles.append(na)
                if na == 0:
                    empty_src += 1
                if na <= 1:
                    le1_article += 1

                # --- cleaned text ---
                s_clean = clean_source(s_raw)
                sw = len(s_clean.split())
                sc = n_sentences(s_clean)
                tw = len(t_raw.split())
                tc = n_sentences(t_raw)

                src_words.append(sw)
                src_sents.append(sc)
                tgt_words.append(tw)
                tgt_sents.append(tc)

                if tw == 0:
                    empty_tgt += 1

                # --- compression ---
                if tw > 0:
                    c = sw / tw
                    comp.append(c)
                    comp_full.append(c)
                else:
                    comp_full.append(np.nan)

                # --- vocabulary (alphanumeric lowercase tokens) ---
                vocab.update(WORD_RE.findall(s_clean.lower()))
                vocab.update(WORD_RE.findall(t_raw.lower()))

                # --- duplicates ---
                th = norm_hash(t_raw)
                tgt_exact[th] = tgt_exact.get(th, 0) + 1
                tfp = fingerprint(t_raw)
                if tfp is not None:
                    tgt_finger[tfp] = tgt_finger.get(tfp, 0) + 1
                sh = norm_hash(s_clean)
                src_exact[sh] = src_exact.get(sh, 0) + 1

                # --- anomalies ---
                if tw < SUM_MIN:
                    tgt_out_low += 1
                elif tw > SUM_MAX:
                    tgt_out_high += 1
                if sw < SRC_MIN:
                    src_short += 1

                ref = f"{split}:{i}"
                src_ref.append((sw, ref))
                tgt_ref.append((tw, ref))

    print(f"total examples: {total_examples}", flush=True)

    # ---------- numpy arrays ----------
    na_a = np.array(n_articles)
    sw_a = np.array(src_words, dtype=np.float64)
    sc_a = np.array(src_sents, dtype=np.float64)
    tw_a = np.array(tgt_words, dtype=np.float64)
    tc_a = np.array(tgt_sents, dtype=np.float64)
    comp_a = np.array(comp, dtype=np.float64)
    compf_a = np.array(comp_full, dtype=np.float64)

    def stats(arr):
        a = np.asarray(arr, dtype=np.float64)
        a = a[~np.isnan(a)]
        return {
            "n": int(a.size),
            "total": round(float(a.sum()), 3),
            "mean": round(float(a.mean()), 3),
            "median": round(float(np.median(a)), 3),
            "std": round(float(a.std()), 3),
            "min": round(float(a.min()), 3),
            "max": round(float(a.max()), 3),
            "p05": round(float(np.percentile(a, 5)), 3),
            "p25": round(float(np.percentile(a, 25)), 3),
            "p75": round(float(np.percentile(a, 75)), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
        }

    def hist(arr, edges):
        """Counts per [edge_i, edge_{i+1}) bin; last bin is [edge_last, inf)."""
        a = np.asarray(arr, dtype=np.float64)
        a = a[~np.isnan(a)]
        counts = []
        labels = []
        for j in range(len(edges)):
            lo = edges[j]
            hi = edges[j + 1] if j + 1 < len(edges) else float("inf")
            counts.append(int(((a >= lo) & (a < hi)).sum()))
            labels.append(fmt_bin(lo, hi))
        return {"counts": counts, "labels": labels}

    def fmt_bin(lo, hi):
        def f(x):
            if x == float("inf"):
                return ""
            if x >= 1000 and x % 1000 == 0:
                return f"{int(x/1000)}k"
            if x >= 1000:
                return f"{x/1000:.1f}k"
            return f"{int(x)}" if x == int(x) else f"{x:g}"
        if hi == float("inf"):
            return f"{f(lo)}+"
        return f"{f(lo)}-{f(hi)}"

    # ---------- correlations ----------
    def pearson(x, y):
        return round(float(np.corrcoef(x, y)[0, 1]), 4)

    def spearman(x, y):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return round(float(np.corrcoef(rx, ry)[0, 1]), 4)

    # align n_articles with compression-defined mask for the compression correlation
    mask = ~np.isnan(compf_a)
    corr = {
        "art_vs_tgt": {"pearson": pearson(na_a, tw_a), "spearman": spearman(na_a, tw_a)},
        "art_vs_comp": {
            "pearson": pearson(na_a[mask], compf_a[mask]),
            "spearman": spearman(na_a[mask].astype(float), compf_a[mask]),
        },
        "src_vs_tgt": {"pearson": pearson(sw_a, tw_a), "spearman": spearman(sw_a, tw_a)},
    }

    # binned means by article count (0..10) — for the correlation charts
    art_bins = {}
    for k in range(0, 11):
        m = na_a == k
        cnt = int(m.sum())
        if cnt == 0:
            continue
        cm = m & mask
        art_bins[str(k)] = {
            "count": cnt,
            "tgt_words_mean": round(float(tw_a[m].mean()), 1),
            "comp_mean": round(float(compf_a[cm].mean()), 2) if cm.any() else None,
        }

    # article-count distribution 0..10
    art_dist = {str(k): int((na_a == k).sum()) for k in range(0, 11)}

    # ---------- duplicates ----------
    def dup_stats(d):
        groups = sum(1 for v in d.values() if v > 1)
        dup_examples = sum(v for v in d.values() if v > 1)
        # "extra" copies beyond the first, i.e. how many rows are redundant
        redundant = sum(v - 1 for v in d.values() if v > 1)
        return {"groups": groups, "examples": dup_examples, "redundant": redundant}

    tgt_exact_s = dup_stats(tgt_exact)
    tgt_finger_s = dup_stats(tgt_finger)
    src_exact_s = dup_stats(src_exact)

    # ---------- outliers ----------
    src_sorted = sorted(src_ref, reverse=True)
    tgt_sorted = sorted(tgt_ref, reverse=True)

    def refs(sorted_list, top=8):
        longest = [{"words": w, "ref": r} for w, r in sorted_list[:top]]
        shortest = [{"words": w, "ref": r} for w, r in sorted_list[-top:][::-1]]
        return longest, shortest

    src_long, src_shortest = refs(src_sorted)
    tgt_long, tgt_shortest = refs(tgt_sorted)

    N = total_examples

    result = {
        "meta": {
            "n_examples": N,
            "total_articles": int(na_a.sum()),
            "generated_note": "Aggregated over train+val+test. Heavy metrics (novel n-grams, "
                              "fragment coverage/density, language detection) intentionally omitted.",
        },
        # ---- 1. structure ----
        "structure": {
            "n_examples": N,
            "empty_src_count": empty_src,
            "empty_src_pct": round(100 * empty_src / N, 4),
            "empty_tgt_count": empty_tgt,
            "empty_tgt_pct": round(100 * empty_tgt / N, 4),
            "tgt_exact_dup": tgt_exact_s,
            "tgt_exact_dup_pct": round(100 * tgt_exact_s["redundant"] / N, 4),
            "tgt_near_dup": tgt_finger_s,
            "tgt_near_dup_pct": round(100 * tgt_finger_s["redundant"] / N, 4),
            "src_exact_dup": src_exact_s,
            "src_exact_dup_pct": round(100 * src_exact_s["redundant"] / N, 4),
            "le1_article_count": le1_article,
            "le1_article_pct": round(100 * le1_article / N, 4),
            "unique_summaries": N - tgt_exact_s["redundant"],
            "unique_sources": N - src_exact_s["redundant"],
        },
        # ---- 2. source documents per example ----
        "sources": {
            "distribution": art_dist,
            "mean": round(float(na_a.mean()), 3),
            "median": float(np.median(na_a)),
            "cum_le3_pct": round(100 * float((na_a <= 3).sum()) / N, 2),
            "paper": {"mode2": 23894, "count3": 12707},
        },
        # ---- 3. lengths ----
        "lengths": {
            "src_words": stats(sw_a),
            "src_sents": stats(sc_a),
            "tgt_words": stats(tw_a),
            "tgt_sents": stats(tc_a),
            "compression": stats(comp_a),
            "vocab_size": len(vocab),
            "paper": {
                "src_words": 2103, "src_sents": 82.7,
                "tgt_words": 263.7, "tgt_sents": 9.97, "vocab_size": 666515,
            },
        },
        # ---- histograms (aggregated) ----
        "hist": {
            "src_words": hist(sw_a, [0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000]),
            "src_sents": hist(sc_a, [0, 20, 40, 60, 80, 100, 125, 150, 200, 300]),
            "tgt_words": hist(tw_a, [0, 100, 150, 200, 250, 300, 350, 450]),
            "tgt_sents": hist(tc_a, [0, 4, 6, 8, 10, 12, 14, 16, 20, 25]),
            "compression": hist(comp_a, [0, 3, 5, 7, 9, 11, 13, 16, 20, 30]),
            "articles": {
                "counts": [art_dist[str(k)] for k in range(0, 11)],
                "labels": [str(k) for k in range(0, 11)],
            },
        },
        # ---- 4/5. static reference from the paper (NOT recomputed) ----
        "paper_reference": {
            "novel_ngrams": {"unigram": 17.76, "bigram": 57.10, "trigram": 75.71, "fourgram": 82.30},
            "note": "Abstractiveness (novel n-grams) and extractiveness (fragment coverage/density) "
                    "are quoted from Fabbri et al. (2019) and NOT recomputed from this repo's data. "
                    "The paper places Multi-News near CNN/DailyMail on the coverage/density plane.",
        },
        # ---- 6. correlations ----
        "correlations": {
            "coefficients": corr,
            "by_articles": art_bins,
        },
        # ---- 7. quality / anomalies ----
        "quality": {
            "tgt_out_low_count": tgt_out_low,
            "tgt_out_low_pct": round(100 * tgt_out_low / N, 3),
            "tgt_out_high_count": tgt_out_high,
            "tgt_out_high_pct": round(100 * tgt_out_high / N, 3),
            "src_short_count": src_short,
            "src_short_pct": round(100 * src_short / N, 3),
            "thresholds": {"tgt_min": SUM_MIN, "tgt_max": SUM_MAX, "src_min": SRC_MIN},
            "outliers": {
                "longest_src": src_long,
                "shortest_src": src_shortest,
                "longest_tgt": tgt_long,
                "shortest_tgt": tgt_shortest,
            },
        },
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=None)
    print(f"wrote {OUT}", flush=True)

    # quick sanity to stderr-ish
    print(json.dumps({
        "n_examples": N,
        "total_articles": result["meta"]["total_articles"],
        "unique_summaries": result["structure"]["unique_summaries"],
        "vocab_size": result["lengths"]["vocab_size"],
        "median_src_words": result["lengths"]["src_words"]["median"],
        "median_comp": result["lengths"]["compression"]["median"],
    }, indent=2))


if __name__ == "__main__":
    sys.exit(main())
