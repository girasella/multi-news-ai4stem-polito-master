# coding=utf-8
"""Convert data/text/*.src.cleaned + *.tgt into Orange-readable data/tab/*.tab files."""

import csv
import os

_SPLITS = ("train", "val", "test")
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_TEXT_DIR = os.path.join(_DATA_DIR, "text")
_TAB_DIR = os.path.join(_DATA_DIR, "tab")

_HEADER_NAMES = ["document", "summary"]
_HEADER_TYPES = ["string", "string"]
_HEADER_FLAGS = ["meta", "meta"]


def convert_split(split):
    src_path = os.path.join(_TEXT_DIR, f"{split}.src.cleaned")
    tgt_path = os.path.join(_TEXT_DIR, f"{split}.tgt")
    tab_path = os.path.join(_TAB_DIR, f"{split}.tab")

    with open(src_path, encoding="utf-8") as src_f, \
         open(tgt_path, encoding="utf-8") as tgt_f, \
         open(tab_path, "w", encoding="utf-8", newline="") as tab_f:
        writer = csv.writer(tab_f, delimiter="\t")
        writer.writerow(_HEADER_NAMES)
        writer.writerow(_HEADER_TYPES)
        writer.writerow(_HEADER_FLAGS)
        for src_line, tgt_line in zip(src_f, tgt_f):
            document = src_line.strip().replace("NEWLINE_CHAR", "\n")
            summary = tgt_line.strip()
            writer.writerow([document, summary])


def main():
    os.makedirs(_TAB_DIR, exist_ok=True)
    for split in _SPLITS:
        convert_split(split)
        print(f"wrote {os.path.join(_TAB_DIR, split + '.tab')}")


if __name__ == "__main__":
    main()
