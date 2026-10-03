#!/usr/bin/env python3
"""Convert raw LoCoMo JSON into QA splits used by StateTree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statetree.locomo import export_locomo_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to locomo10.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--keep_adversarial", action="store_true")
    args = ap.parse_args()
    paths = export_locomo_splits(
        args.input,
        args.output_dir,
        skip_adversarial=not args.keep_adversarial,
    )
    for k, p in paths.items():
        n = sum(1 for _ in open(p, encoding="utf-8") if _.strip())
        print(f"{k}: {n} rows -> {p}")


if __name__ == "__main__":
    main()
