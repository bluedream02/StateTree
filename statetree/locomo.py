"""LoCoMo conversation → long-context QA rows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io_utils import write_jsonl

SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


def extract_sessions(conv: Dict[str, Any]) -> List[Tuple[int, str, List[Dict[str, Any]]]]:
    sessions: List[Tuple[int, str, List[Dict[str, Any]]]] = []
    for k, v in conv.items():
        m = SESSION_KEY_RE.match(k)
        if not m or not isinstance(v, list):
            continue
        idx = int(m.group(1))
        date_time = conv.get(f"session_{idx}_date_time", "")
        sessions.append((idx, str(date_time), v))
    sessions.sort(key=lambda x: x[0])
    return sessions


def build_long_context(sample: Dict[str, Any]) -> str:
    """LoCoMo evaluation-style formatting with DATE / CONVERSATION markers."""
    conv = sample.get("conversation") or {}
    speaker_a = conv.get("speaker_a", "speaker_a")
    speaker_b = conv.get("speaker_b", "speaker_b")
    lines: List[str] = [
        "Below is a conversation between two people: "
        f"{speaker_a} and {speaker_b}. The conversation takes place over multiple days "
        "and the date of each conversation is wriiten at the beginning of the conversation."
    ]
    for _sess_idx, sess_time, turns in extract_sessions(conv):
        lines.append("")
        lines.append(f"DATE: {sess_time}")
        lines.append("CONVERSATION:")
        for t in turns:
            speaker = t.get("speaker", "") or "UNKNOWN"
            text = t.get("text", "")
            blip_caption = t.get("blip_caption")
            if blip_caption:
                lines.append(f'{speaker} said, "{text}" and shared {blip_caption}.')
            else:
                lines.append(f'{speaker} said, "{text}"')
    return "\n".join(lines).strip() + "\n"


def locomo_to_qa_rows(
    data: List[Dict[str, Any]],
    *,
    skip_adversarial: bool = True,
    skip_empty_answer: bool = True,
) -> List[Dict[str, Any]]:
    """Expand LoCoMo samples into one QA row per question."""
    rows: List[Dict[str, Any]] = []
    global_index = 0
    for sample in data:
        sample_id = sample.get("sample_id", "unknown")
        context = build_long_context(sample)
        conv = sample.get("conversation") or {}
        qa_list = sample.get("qa", []) or []
        for q in qa_list:
            question = (q.get("question") or "").strip()
            answer = q.get("answer", "")
            category = q.get("category", None)
            # LoCoMo adversarial / unanswerable often use category 5 or empty answers.
            if skip_empty_answer and (answer is None or str(answer).strip() == ""):
                continue
            if skip_adversarial and category == 5:
                continue
            rows.append(
                {
                    "index": global_index,
                    "sample_id": sample_id,
                    "input": question,
                    "context": context,
                    "answers": [answer if isinstance(answer, str) else str(answer)],
                    "length": len(context),
                    "category": category,
                    "evidence": q.get("evidence", []),
                    "speaker_a": conv.get("speaker_a", ""),
                    "speaker_b": conv.get("speaker_b", ""),
                }
            )
            global_index += 1
    return rows


def split_qa_rows(
    rows: List[Dict[str, Any]],
    *,
    train_ratio: float = 0.5,
    val_ratio_of_train: float = 0.2,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Paper split: filter then 50/50. First half → 80/20 train/val; second half → test.
    """
    n = len(rows)
    mid = n // 2
    first, second = rows[:mid], rows[mid:]
    n_val = int(len(first) * val_ratio_of_train)
    # Keep train contiguous from the start (matches paper's 616/154 when n≈1540).
    train = first[: len(first) - n_val] if n_val > 0 else first
    val = first[len(first) - n_val :] if n_val > 0 else []
    return {"train": train, "val": val, "test": second}


def load_locomo_json(path: Path | str) -> List[Dict[str, Any]]:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected top-level list in LoCoMo json.")
    return data


def export_locomo_splits(
    locomo_json: Path | str,
    out_dir: Path | str,
    *,
    skip_adversarial: bool = True,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = locomo_to_qa_rows(load_locomo_json(locomo_json), skip_adversarial=skip_adversarial)
    splits = split_qa_rows(rows)
    paths = {}
    for name, items in splits.items():
        p = out_dir / f"locomo_qa_{name}.jsonl"
        write_jsonl(p, items)
        paths[name] = p
    write_jsonl(out_dir / "locomo_qa_all.jsonl", rows)
    paths["all"] = out_dir / "locomo_qa_all.jsonl"
    return paths
