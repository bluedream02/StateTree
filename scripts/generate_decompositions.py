#!/usr/bin/env python3
"""Generate Compositional StateTree decompositions via an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statetree.decompose import decompose_question, make_openai_chat_fn  # noqa: E402
from statetree.io_utils import read_jsonl, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True, help="Train QA jsonl")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--max_retries", type=int, default=3)
    args = ap.parse_args()

    rows = read_jsonl(args.input_jsonl)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r.get("sample_id")].append(r)

    chat_fn = make_openai_chat_fn(
        model=args.model,
        temperature=args.temperature,
        api_key=args.api_key or None,
        base_url=args.base_url or None,
    )

    out = {}
    for i, row in enumerate(rows):
        sid = row.get("sample_id", "")
        q = row.get("input", "")
        key = f"{sid}::{q}"
        print(f"[{i+1}/{len(rows)}] decomposing {key[:80]}...")
        decomp = decompose_question(
            target_question=q,
            sample_id=str(sid),
            row_index=i,
            speaker_a=str(row.get("speaker_a") or "SpeakerA"),
            speaker_b=str(row.get("speaker_b") or "SpeakerB"),
            pool_rows=by_sample[sid],
            chat_fn=chat_fn,
            max_retries=args.max_retries,
        )
        out[key] = decomp

    write_json(args.output_json, out)
    print(f"Wrote {len(out)} decompositions -> {args.output_json}")


if __name__ == "__main__":
    main()
