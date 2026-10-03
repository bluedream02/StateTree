"""Combined Exact-Match + LLM-as-a-Judge reward (paper Section 3.2)."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional

from .prompts import LOCOMO_JUDGE_PROMPT


def extract_boxed(response: str) -> str:
    idx = response.find(r"\boxed{")
    if idx < 0:
        return ""
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(response) and depth > 0:
        if response[i] == "{":
            depth += 1
        elif response[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return ""
    return response[start : i - 1].strip()


def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


def exact_match_reward(pred: str, gold: str) -> float:
    if not gold:
        return 1.0
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def parse_judge_label(text: str) -> float:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        label = str(obj.get("label", "")).upper()
        if "CORRECT" in label and "WRONG" not in label:
            return 1.0
        return 0.0
    except Exception:
        pass
    upper = text.upper()
    if "CORRECT" in upper and "WRONG" not in upper:
        return 1.0
    if re.search(r"\bYES\b", upper):
        return 1.0
    return 0.0


JudgeFn = Callable[[str], str]


def combined_reward(
    response: str,
    gold: str,
    *,
    question: str = "",
    judge_fn: Optional[JudgeFn] = None,
) -> Dict[str, Any]:
    """
    r = max(r_EM, r_LLM) as in the paper.

    `judge_fn` should accept a prompt string and return the judge model output.
    If omitted, only EM is used (r_LLM=0).
    """
    y_ans = extract_boxed(response)
    # Fallback: if no box, try last non-empty line.
    if not y_ans:
        lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
        y_ans = lines[-1] if lines else ""

    r_em = exact_match_reward(y_ans, gold)
    r_llm = 0.0
    judge_raw = ""
    if judge_fn is not None and question:
        prompt = LOCOMO_JUDGE_PROMPT.format(
            question=question,
            gold_answer=gold,
            generated_answer=y_ans,
        )
        judge_raw = judge_fn(prompt)
        r_llm = parse_judge_label(judge_raw)

    return {
        "reward": max(r_em, r_llm),
        "r_em": r_em,
        "r_llm": r_llm,
        "extracted_answer": y_ans,
        "judge_raw": judge_raw,
    }
