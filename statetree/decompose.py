"""LLM-based question decomposition for Compositional StateTree (Stage 3)."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from .prompts import DECOMPOSITION_SYSTEM_PROMPT, decomposition_user_prompt

ChatFn = Callable[[str, str], str]  # (system, user) -> assistant text


def format_question_pool(rows: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(rows):
        q = r.get("input") or r.get("question") or ""
        ans = r.get("answers") or r.get("answer") or ""
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        evid = r.get("evidence", [])
        lines.append(f"{i}. Q: {q}\n   A: {ans}\n   evidence: {evid}")
    return "\n".join(lines)


def validate_decomposition(decomp: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    persons = decomp.get("persons")
    if not isinstance(persons, list) or len(persons) != 2:
        errors.append("need exactly 2 persons")
        return errors
    n_target = 0
    for pi, person in enumerate(persons):
        if "[A]" not in str(person.get("step_1", "")):
            errors.append(f"person[{pi}].step_1 must contain [A]")
        events = person.get("events") or []
        if len(events) != 2:
            errors.append(f"person[{pi}] must have 2 events")
            continue
        for ei, event in enumerate(events):
            if "[B]" not in str(event.get("step_2", "")):
                errors.append(f"person[{pi}].events[{ei}].step_2 must contain [B]")
            leaves = event.get("leaves") or []
            if len(leaves) != 2:
                errors.append(f"person[{pi}].events[{ei}] must have 2 leaves")
                continue
            steps = [str(leaf.get("step_3", "")) for leaf in leaves]
            if len(set(steps)) < 2:
                errors.append(f"sibling step_3 values must differ at ({pi},{ei})")
            for li, leaf in enumerate(leaves):
                s3 = str(leaf.get("step_3", ""))
                if "[A]" not in s3 or "[B]" not in s3:
                    errors.append(f"leaf ({pi},{ei},{li}) step_3 must contain [A] and [B]")
                if leaf.get("is_target"):
                    n_target += 1
    if n_target != 1:
        errors.append(f"need exactly 1 target leaf, found {n_target}")
    return errors


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def decompose_question(
    *,
    target_question: str,
    sample_id: str,
    row_index: int,
    speaker_a: str,
    speaker_b: str,
    pool_rows: Sequence[Dict[str, Any]],
    chat_fn: ChatFn,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Call an LLM to produce a validated 2x2x2 decomposition."""
    pool_text = format_question_pool(pool_rows)
    user = decomposition_user_prompt(
        target_question=target_question,
        sample_id=sample_id,
        row_index=row_index,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        question_pool=pool_text,
        num_questions=len(pool_rows),
    )
    last_err = ""
    for _ in range(max_retries):
        raw = chat_fn(DECOMPOSITION_SYSTEM_PROMPT, user)
        try:
            decomp = _extract_json(raw)
        except Exception as e:
            last_err = f"json parse error: {e}"
            continue
        errs = validate_decomposition(decomp)
        if not errs:
            return decomp
        last_err = "; ".join(errs)
        user = user + f"\n\nPrevious output failed validation: {last_err}. Please fix and output JSON only."
    raise ValueError(f"decomposition failed after retries: {last_err}")


def make_openai_chat_fn(
    *,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ChatFn:
    """Optional OpenAI-compatible client for Stage-3 data construction."""

    def _chat(system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install openai to use make_openai_chat_fn: pip install openai") from e
        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    return _chat
