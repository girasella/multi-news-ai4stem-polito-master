# coding=utf-8
"""Convert data/text/*.src.cleaned + *.tgt into Orange-readable data/tab/*.tab files.

Outputs one .tab per split (train.tab, val.tab, test.tab) plus complete.tab, which joins all
three splits into a single file with an extra ``split`` meta column recording each row's origin.

Unlike data/text/ (the canonical, untouched copy), the .tab output is *cleaned*: rows whose
source text is a known quality problem (see multi_news_dashboard.html and data/README.md) are
dropped, so data/tab/ is NOT line-aligned with data/text/. Dropped rows are listed in
data/tab/excluded_rows.tsv. Criteria:

- source < MIN_SRC_WORDS words (includes fully empty sources) — likely failed scrapes;
- source > MAX_SRC_WORDS words — the extreme outliers are source/summary mismatches from
  upstream scraping errors, not just long text;
- exact-duplicate source text (whitespace-normalized SHA-1, ``|||||`` separator excluded):
  only the first occurrence in train -> val -> test scan order is kept, which also removes
  train/eval leakage from duplicate groups that span splits.

Word counts are tokenizer-free str.split() with NEWLINE_CHAR restored and ``|||||`` excluded,
matching the EDA dashboard's methodology.
"""

import csv
import hashlib
import os

_SPLITS = ("train", "val", "test")
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_TEXT_DIR = os.path.join(_DATA_DIR, "text")
_TAB_DIR = os.path.join(_DATA_DIR, "tab")

_HEADER_NAMES = ["document", "summary"]
_HEADER_TYPES = ["string", "string"]
_HEADER_FLAGS = ["meta", "meta"]

_COMPLETE_HEADER_NAMES = ["document", "summary", "split"]
_COMPLETE_HEADER_TYPES = ["string", "string", "discrete"]
_COMPLETE_HEADER_FLAGS = ["meta", "meta", "meta"]

MIN_SRC_WORDS = 50
MAX_SRC_WORDS = 100_000


def _source_tokens(src_line):
    """Tokenize a raw .src.cleaned line the way the EDA dashboard counts words."""
    doc = src_line.strip().replace("NEWLINE_CHAR", " ")
    return [t for t in doc.split() if t != "|||||"]


def find_dirty_rows():
    """Pass 1: stream every split's source file and flag rows to exclude.

    Returns {(split, line_idx): reason}, line_idx 0-based within the split.
    """
    dirty = {}
    seen_hashes = set()
    for split in _SPLITS:
        src_path = os.path.join(_TEXT_DIR, f"{split}.src.cleaned")
        with open(src_path, encoding="utf-8") as src_f:
            for idx, line in enumerate(src_f):
                toks = _source_tokens(line)
                n = len(toks)
                h = hashlib.sha1(" ".join(toks).encode("utf-8")).hexdigest()
                if n < MIN_SRC_WORDS:
                    dirty[(split, idx)] = f"source<{MIN_SRC_WORDS}w ({n} words)"
                elif n > MAX_SRC_WORDS:
                    dirty[(split, idx)] = f"source>{MAX_SRC_WORDS}w ({n} words)"
                elif h in seen_hashes:
                    dirty[(split, idx)] = "duplicate source"
                seen_hashes.add(h)
    return dirty


def convert_split(split, dirty, complete_writer):
    src_path = os.path.join(_TEXT_DIR, f"{split}.src.cleaned")
    tgt_path = os.path.join(_TEXT_DIR, f"{split}.tgt")
    tab_path = os.path.join(_TAB_DIR, f"{split}.tab")

    kept = dropped = 0
    with open(src_path, encoding="utf-8") as src_f, \
         open(tgt_path, encoding="utf-8") as tgt_f, \
         open(tab_path, "w", encoding="utf-8", newline="") as tab_f:
        writer = csv.writer(tab_f, delimiter="\t")
        writer.writerow(_HEADER_NAMES)
        writer.writerow(_HEADER_TYPES)
        writer.writerow(_HEADER_FLAGS)
        for idx, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
            if (split, idx) in dirty:
                dropped += 1
                continue
            document = src_line.strip().replace("NEWLINE_CHAR", "\n")
            summary = tgt_line.strip()
            writer.writerow([document, summary])
            complete_writer.writerow([document, summary, split])
            kept += 1
    return kept, dropped


def write_manifest(dirty):
    manifest_path = os.path.join(_TAB_DIR, "excluded_rows.tsv")
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["split", "line", "reason"])
        for (split, idx), reason in sorted(
                dirty.items(), key=lambda kv: (_SPLITS.index(kv[0][0]), kv[0][1])):
            writer.writerow([split, idx, reason])
    return manifest_path


def main():
    os.makedirs(_TAB_DIR, exist_ok=True)
    dirty = find_dirty_rows()
    complete_path = os.path.join(_TAB_DIR, "complete.tab")
    total_kept = 0
    with open(complete_path, "w", encoding="utf-8", newline="") as complete_f:
        complete_writer = csv.writer(complete_f, delimiter="\t")
        complete_writer.writerow(_COMPLETE_HEADER_NAMES)
        complete_writer.writerow(_COMPLETE_HEADER_TYPES)
        complete_writer.writerow(_COMPLETE_HEADER_FLAGS)
        for split in _SPLITS:
            kept, dropped = convert_split(split, dirty, complete_writer)
            total_kept += kept
            print(f"wrote {os.path.join(_TAB_DIR, split + '.tab')} "
                  f"({kept} rows kept, {dropped} dropped)")
    print(f"wrote {complete_path} ({total_kept} rows, all splits joined)")
    manifest = write_manifest(dirty)
    print(f"wrote {manifest} ({len(dirty)} excluded rows)")


if __name__ == "__main__":
    main()
