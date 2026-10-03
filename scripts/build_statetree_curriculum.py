#!/usr/bin/env python3
"""Build StateTree curriculum datasets from LoCoMo QA jsonl (ROLL format)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statetree.formats import (  # noqa: E402
    build_basic_problem,
    build_compositional_problem,
    build_warmup_problem,
    to_roll_row,
)
from statetree.insert import embed_records_in_dialogue  # noqa: E402
from statetree.io_utils import read_jsonl, write_jsonl  # noqa: E402
from statetree.tree import (  # noqa: E402
    build_basic_tree,
    build_compositional_tree,
    validate_basic_tree,
)


def _pool_by_sample(rows: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    by = defaultdict(list)
    for r in rows:
        by[r.get("sample_id")].append(r)
    return by


def _answer(row: Dict[str, Any]) -> str:
    answers = row.get("answers") or []
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(row.get("answer") or "")


def build_warmup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, row in enumerate(rows):
        ctx = row.get("context", "")
        q = row.get("input", "")
        ans = _answer(row)
        problem = build_warmup_problem(ctx, q)
        out.append(
            to_roll_row(
                problem=problem,
                answer=ans,
                row_id=f"warmup_{i+1}",
                subset="locomo",
                question=q,
                category=row.get("category"),
            )
        )
    return out


def build_basic_stage(
    rows: List[Dict[str, Any]],
    *,
    depth: int,
    seed: int,
    distract_pool_size: int = 64,
    uuid_format: str = "default",
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_sample = _pool_by_sample(rows)
    all_qs = [r.get("input", "") for r in rows if r.get("input")]
    out = []
    for i, row in enumerate(rows):
        local_rng = random.Random(rng.randint(0, 2**31 - 1))
        q = row.get("input", "")
        ans = _answer(row)
        same = [r.get("input", "") for r in by_sample.get(row.get("sample_id"), []) if r.get("input") != q]
        # Paper Appendix: distractors from the same conversation (and training split).
        n_need = 2**depth - 1
        if len(same) >= n_need:
            pool = same
        else:
            others = [x for x in all_qs if x != q and x not in same]
            pool = same + others
        if len(pool) > distract_pool_size:
            pool = local_rng.sample(pool, k=distract_pool_size)
        tree = build_basic_tree(
            target_question=q,
            target_answer=ans,
            distractor_questions=pool,
            depth=depth,
            rng=local_rng,
            uuid_format=uuid_format,
        )
        validate_basic_tree(tree)
        aug, place_audit = embed_records_in_dialogue(row.get("context", ""), tree.edges, rng=local_rng)
        problem = build_basic_problem(aug, tree.root_key)
        item = to_roll_row(
            problem=problem,
            answer=ans,
            row_id=f"basic_d{depth}_{i+1}",
            subset="locomo",
            question=q,
            category=row.get("category"),
        )
        item["meta"] = {
            "stage": f"basic_D{depth}",
            "sample_id": row.get("sample_id"),
            "original_input": q,
            "root_key": tree.root_key,
            "tree_audit": tree.audit,
            "placement": place_audit,
        }
        out.append(item)
    return out


def build_compositional_stage(
    rows: List[Dict[str, Any]],
    *,
    decompositions: Dict[str, Dict[str, Any]],
    seed: int,
    uuid_format: str = "default",
) -> List[Dict[str, Any]]:
    """
    `decompositions` maps a key "{sample_id}::{question}" -> decomposition JSON.
    Rows without a decomposition are skipped.
    """
    rng = random.Random(seed)
    out = []
    for i, row in enumerate(rows):
        q = row.get("input", "")
        key = f"{row.get('sample_id')}::{q}"
        decomp = decompositions.get(key)
        if decomp is None:
            continue
        local_rng = random.Random(rng.randint(0, 2**31 - 1))
        tree = build_compositional_tree(decomposition=decomp, rng=local_rng, uuid_format=uuid_format)
        aug, place_audit = embed_records_in_dialogue(row.get("context", ""), tree.edges, rng=local_rng)
        ans = tree.target_answer or _answer(row)
        problem = build_compositional_problem(aug, tree.root_key)
        item = to_roll_row(
            problem=problem,
            answer=ans,
            row_id=f"comp_d3_{i+1}",
            subset="locomo",
            question=q,
            category=row.get("category"),
        )
        item["meta"] = {
            "stage": "compositional_D3",
            "sample_id": row.get("sample_id"),
            "original_input": q,
            "root_key": tree.root_key,
            "path_steps": tree.path_steps,
            "tree_audit": tree.audit,
            "placement": place_audit,
        }
        out.append(item)
    return out


def load_decompositions(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    out = {}
    for item in data:
        key = f"{item['sample_id']}::{item['question']}"
        out[key] = item["decomposition"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build StateTree curriculum datasets (ROLL jsonl)")
    ap.add_argument("--input_jsonl", required=True, help="LoCoMo QA jsonl (train split)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stages", default="warmup,basic2,basic3,compositional", help="Comma-separated stages")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--uuid_format", default="default")
    ap.add_argument("--decompositions_json", default="", help="Required for compositional stage")
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl(args.input_jsonl)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    for stage in stages:
        if stage == "warmup":
            items = build_warmup(rows)
            write_jsonl(out_dir / "stage0_warmup.roll.jsonl", items)
            print(f"[ok] warmup -> {len(items)} rows")
        elif stage in {"basic2", "basic_d2", "D2"}:
            items = build_basic_stage(rows, depth=2, seed=args.seed, uuid_format=args.uuid_format)
            write_jsonl(out_dir / "stage1_basic_D2.roll.jsonl", items)
            print(f"[ok] basic D=2 -> {len(items)} rows")
        elif stage in {"basic3", "basic_d3", "D3"}:
            items = build_basic_stage(rows, depth=3, seed=args.seed + 1, uuid_format=args.uuid_format)
            write_jsonl(out_dir / "stage2_basic_D3.roll.jsonl", items)
            print(f"[ok] basic D=3 -> {len(items)} rows")
        elif stage in {"compositional", "comp", "stage3"}:
            if not args.decompositions_json:
                raise SystemExit("--decompositions_json is required for compositional stage")
            decomps = load_decompositions(Path(args.decompositions_json))
            items = build_compositional_stage(
                rows, decompositions=decomps, seed=args.seed + 2, uuid_format=args.uuid_format
            )
            write_jsonl(out_dir / "stage3_compositional_D3.roll.jsonl", items)
            print(f"[ok] compositional D=3 -> {len(items)} rows")
        else:
            raise SystemExit(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
