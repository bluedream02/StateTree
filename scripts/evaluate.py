#!/usr/bin/env python3
"""
Evaluate predictions on LoCoMo / EM test sets.

Paper LoCoMo metrics (Appendix): LLM-judged ACC (GPT-4o) + token-F1 + BLEU-1.
Judge prompt: statetree.prompts.LOCOMO_JUDGE_PROMPT (Appendix app:locomo_eval_prompt).

Examples:
  # Lexical only (no API)
  python scripts/evaluate.py \\
    --roll_test data/locomo_qa/locomo_qa_test.roll.jsonl \\
    --pred outputs/predictions/locomo_pred.jsonl \\
    --dataset locomo

  # Full paper metrics including GPT-4o ACC
  python scripts/evaluate.py \\
    --roll_test data/locomo_qa/locomo_qa_test.roll.jsonl \\
    --pred outputs/predictions/locomo_pred.jsonl \\
    --dataset locomo --judge_model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statetree.metrics import (  # noqa: E402
    bleu1,
    exact_match,
    extract_boxed,
    extract_prediction,
    extract_question,
    mean,
    summarize_by_category,
    token_f1,
)
from statetree.prompts import LOCOMO_JUDGE_PROMPT  # noqa: E402
from statetree.rewards import parse_judge_label  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def call_judge_api(
    prompt: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_retries: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url or None)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"judge API failed after {max_retries} retries: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roll_test", required=True, help="Gold ROLL jsonl (prefer export_roll_test.py output)")
    ap.add_argument("--pred", required=True, help="Predictions jsonl with id + prediction")
    ap.add_argument("--dataset", choices=["locomo", "em"], default="locomo")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_result", default="", help="Summary JSON path")
    ap.add_argument("--out_details", default="", help="Per-example JSONL path")
    # GPT-4o ACC (paper). Empty = skip ACC / only F1+BLEU (or EM).
    ap.add_argument(
        "--judge_model",
        default="",
        help="OpenAI-compatible judge model for ACC (paper: gpt-4o). Empty skips ACC.",
    )
    ap.add_argument("--judge_base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    ap.add_argument("--judge_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--judge_max_retries", type=int, default=3)
    ap.add_argument("--judge_sleep", type=float, default=0.0, help="Seconds between judge calls")
    args = ap.parse_args()

    gold_rows = read_jsonl(Path(args.roll_test))
    pred_rows = {str(r.get("id")): r for r in read_jsonl(Path(args.pred))}
    use_judge = bool(args.judge_model) and args.dataset == "locomo"

    details: List[Dict[str, Any]] = []
    missing = 0
    judge_errors = 0

    for i, g in enumerate(gold_rows):
        if args.limit and i >= args.limit:
            break
        gid = str(g.get("id"))
        if gid not in pred_rows:
            missing += 1
            continue

        raw_pred = extract_prediction(pred_rows[gid])
        pred = extract_boxed(raw_pred).strip() or raw_pred.strip()
        gold = str(g.get("ground_truth") or "")
        question = extract_question(g)
        category = g.get("category")

        row: Dict[str, Any] = {
            "id": gid,
            "question": question,
            "gold": gold,
            "prediction": pred,
            "category": category,
        }

        if args.dataset == "em":
            row["em"] = exact_match(pred, gold)
        else:
            row["f1"] = token_f1(pred, gold)
            row["bleu1"] = bleu1(pred, gold)
            if use_judge:
                if not question:
                    row["acc"] = 0.0
                    row["judge_raw"] = ""
                    row["judge_error"] = "missing_question"
                    judge_errors += 1
                else:
                    prompt = LOCOMO_JUDGE_PROMPT.format(
                        question=question,
                        gold_answer=gold,
                        generated_answer=pred,
                    )
                    try:
                        raw = call_judge_api(
                            prompt,
                            model=args.judge_model,
                            base_url=args.judge_base_url,
                            api_key=args.judge_api_key,
                            temperature=args.judge_temperature,
                            max_retries=args.judge_max_retries,
                        )
                        row["acc"] = parse_judge_label(raw)
                        row["judge_raw"] = raw
                        if args.judge_sleep > 0:
                            time.sleep(args.judge_sleep)
                    except Exception as e:  # noqa: BLE001
                        row["acc"] = 0.0
                        row["judge_raw"] = ""
                        row["judge_error"] = str(e)
                        judge_errors += 1

        details.append(row)

    if args.dataset == "em":
        result: Dict[str, Any] = {
            "dataset": "em",
            "n": len(details),
            "missing": missing,
            "em": mean([float(r["em"]) for r in details]),
        }
    else:
        result = {
            "dataset": "locomo",
            "n": len(details),
            "missing": missing,
            "f1": mean([float(r["f1"]) for r in details]),
            "bleu1": mean([float(r["bleu1"]) for r in details]),
        }
        if use_judge:
            result["acc"] = mean([float(r["acc"]) for r in details if "acc" in r])
            result["judge_model"] = args.judge_model
            result["judge_errors"] = judge_errors
            result["by_category"] = summarize_by_category(details, ("acc", "f1", "bleu1"))
        else:
            result["acc"] = None
            result["note"] = "Pass --judge_model gpt-4o for paper LLM-judged ACC."
            result["by_category"] = summarize_by_category(details, ("f1", "bleu1"))

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.out_result:
        Path(args.out_result).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_result).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_details:
        Path(args.out_details).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_details).open("w", encoding="utf-8") as f:
            for r in details:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
