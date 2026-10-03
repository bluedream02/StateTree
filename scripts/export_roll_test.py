#!/usr/bin/env python3
"""Convert clean QA jsonl (no StateTree) into ROLL test format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statetree.formats import build_warmup_problem, to_roll_row  # noqa: E402
from statetree.io_utils import read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--subset", default="locomo")
    ap.add_argument("--id_prefix", default="")
    args = ap.parse_args()

    rows = read_jsonl(args.input_jsonl)
    out = []
    prefix = args.id_prefix or args.subset
    for i, row in enumerate(rows, 1):
        ctx = row.get("context", "")
        q = row.get("input", "")
        answers = row.get("answers") or []
        ans = answers[0] if isinstance(answers, list) and answers else str(row.get("answer") or "")
        problem = build_warmup_problem(ctx, q)
        out.append(
            to_roll_row(
                problem=problem,
                answer=ans,
                row_id=f"{prefix}_{i}",
                subset=args.subset,
                question=q,
                category=row.get("category"),
            )
        )
    write_jsonl(args.output_jsonl, out)
    print(f"Wrote {len(out)} rows -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
