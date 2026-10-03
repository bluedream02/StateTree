"""LoCoMo evaluation metrics (paper: ACC / token-F1 / BLEU-1)."""

from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# LoCoMo category ids (after filtering adversarial=5).
LOCOMO_CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
}


def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not g:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def bleu1(pred: str, gold: str) -> float:
    """Sentence-level BLEU-1 with brevity penalty (unigram precision)."""
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not g:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    precision = overlap / len(p)
    if len(p) > len(g):
        bp = 1.0
    else:
        bp = math.exp(1.0 - len(g) / len(p))
    return bp * precision


def exact_match(pred: str, gold: str) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def extract_boxed(text: str) -> str:
    idx = text.find(r"\boxed{")
    if idx < 0:
        return text
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else text


def extract_prediction(row: Dict[str, Any]) -> str:
    for k in ("prediction", "pred", "output", "response", "hypothesis", "answer"):
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def extract_question(gold_row: Dict[str, Any]) -> str:
    for k in ("question", "input", "original_input"):
        if gold_row.get(k):
            return str(gold_row[k])
    prompt = str(gold_row.get("prompt") or "")
    m = re.search(r"(?m)^Question:\s*(.+)$", prompt)
    if m:
        return m.group(1).strip()
    return ""


def category_name(cat: Any) -> str:
    if cat is None or cat == "":
        return "unknown"
    try:
        return LOCOMO_CATEGORY_NAMES.get(int(cat), str(cat))
    except (TypeError, ValueError):
        return str(cat)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_by_category(
    rows: List[Dict[str, Any]],
    metric_keys: Tuple[str, ...] = ("acc", "f1", "bleu1"),
) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        name = category_name(r.get("category"))
        for k in metric_keys:
            if k in r and r[k] is not None:
                buckets[name][k].append(float(r[k]))
    out: Dict[str, Dict[str, float]] = {}
    for name, mets in sorted(buckets.items()):
        out[name] = {k: mean(v) for k, v in mets.items()}
        out[name]["n"] = float(len(next(iter(mets.values()), [])))
    return out
