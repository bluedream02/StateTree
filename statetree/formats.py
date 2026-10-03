"""Dataset formatting helpers (ROLL RLVR)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .prompts import SYSTEM_PROMPT, basic_statetree_prompt, compositional_statetree_prompt, warmup_prompt


def to_roll_row(
    *,
    problem: str,
    answer: str,
    row_id: str,
    subset: str = "locomo",
    system_prompt: Optional[str] = None,
    question: Optional[str] = None,
    category: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    ROLL RLVR row. Includes the paper system prompt (Appendix) so the model
    emits <think>...</think> and \\boxed{...} for reward extraction.
    """
    sys_msg = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": problem},
    ]
    row: Dict[str, Any] = {
        "id": row_id,
        "source": subset,
        "difficulty": "0",
        "prompt": problem,
        "messages": json.dumps(messages, ensure_ascii=False),
        "ground_truth": answer,
        "case_type": "",
        "test_case_function": "",
        "test_cases": "",
        "tag": subset,
    }
    if question is not None:
        row["question"] = question
    if category is not None:
        row["category"] = category
    return row


def build_warmup_problem(context: str, question: str) -> str:
    return context.rstrip() + "\n\n" + warmup_prompt(question)


def build_basic_problem(context: str, root_key: str) -> str:
    return context.rstrip() + "\n\n" + basic_statetree_prompt(root_key)


def build_compositional_problem(context: str, root_key: str) -> str:
    return context.rstrip() + "\n\n" + compositional_statetree_prompt(root_key)


def default_training_system_prompt() -> str:
    return SYSTEM_PROMPT
