"""
Basic and Compositional StateTree construction (NeurIPS 2026 StateTree paper).

Basic StateTree: complete binary tree of depth D. Edges are flat JSON {"key": "value"}
records. Leaves hold natural-language questions; internal values are child UUIDs.

Compositional StateTree: same topology, but values are nested objects
{"step": "...", "next": "uuid"} (or {"step": "..."} at leaves). The model must
aggregate step fragments along the correct path into the final question.
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class TreeEdge:
    """One edge record to be embedded in the dialogue."""

    key: str
    value: Any  # str for Basic; dict for Compositional
    is_correct_path: bool
    depth: int
    sibling_of_correct: bool = False  # True if this is the distractor sibling on a correct-path fork
    session_preference: str = "any"  # "newer" | "older" | "any"

    def to_record_string(self) -> str:
        return json.dumps({self.key: self.value}, ensure_ascii=False)


@dataclass
class BasicStateTree:
    depth: int
    root_key: str
    target_question: str
    target_answer: str
    edges: List[TreeEdge] = field(default_factory=list)
    leaves: List[str] = field(default_factory=list)
    correct_path_keys: List[str] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionalStateTree:
    depth: int
    root_key: str
    target_question: str
    target_answer: str
    edges: List[TreeEdge] = field(default_factory=list)
    path_steps: List[str] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)


def _node_depth(node_idx: int) -> int:
    """Depth of heap node index (root=0)."""
    return (node_idx + 1).bit_length() - 1


# Entity-key ablation pool (paper Appendix). Used when uuid_format="entity".
ENTITY_KEY_POOL = [
    "garden",
    "mirror",
    "temple",
    "anchor",
    "lantern",
    "ribbon",
    "candle",
    "shield",
    "bridge",
    "castle",
    "meadow",
    "feather",
    "saddle",
    "trumpet",
]


def _new_key(rng: random.Random, uuid_format: str = "default", used_entity: Optional[set] = None) -> str:
    if uuid_format == "default":
        # UUID4 with RNG-controlled bits for reproducibility of the rest of the pipeline.
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    if uuid_format.startswith("default_"):
        n = int(uuid_format.split("_", 1)[1])
        if not (1 <= n <= 32):
            raise ValueError("default_<n> requires 1<=n<=32")
        return uuid.UUID(int=rng.getrandbits(128), version=4).hex[:n]
    if uuid_format == "entity":
        used_entity = used_entity if used_entity is not None else set()
        cand = [w for w in ENTITY_KEY_POOL if w not in used_entity]
        if not cand:
            raise ValueError("entity key pool exhausted")
        w = rng.choice(cand)
        used_entity.add(w)
        return w
    raise ValueError("Unsupported uuid_format. Use 'default', 'default_<n>', or 'entity'.")


def _sample_distractors(
    pool: Sequence[str],
    target: str,
    k: int,
    rng: random.Random,
) -> List[str]:
    cand = [q for q in pool if q and q != target]
    if not cand:
        cand = [f"Distractor question {i}?" for i in range(max(k, 1))]
    out: List[str] = []
    while len(out) < k:
        need = k - len(out)
        if len(cand) >= need:
            out.extend(rng.sample(list(cand), k=need))
        else:
            out.extend(cand)
            # cycle if pool is small
            out.extend(rng.choices(list(cand), k=need - len(cand)))
    return out[:k]


def build_basic_tree(
    *,
    target_question: str,
    target_answer: str,
    distractor_questions: Sequence[str],
    depth: int = 2,
    rng: Optional[random.Random] = None,
    uuid_format: str = "default",
) -> BasicStateTree:
    """
    Build a Basic StateTree of depth D (paper Algorithm 1).

    - Internal nodes: 2^D - 1 UUID keys
    - Leaves: 2^D questions (exactly one is the target)
    - Edge records: 2 * (2^D - 1)

    At each fork on the correct path, the correct child record must later be
    placed in a *newer* session than the distractor sibling (handled by insert).
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    rng = rng or random.Random()

    n_leaves = 2**depth
    distractors = _sample_distractors(distractor_questions, target_question, n_leaves - 1, rng)

    # Choose which leaf index holds the target (any of 0..n_leaves-1).
    target_leaf_idx = rng.randrange(n_leaves)
    leaves: List[str] = []
    d_iter = iter(distractors)
    for i in range(n_leaves):
        if i == target_leaf_idx:
            leaves.append(target_question)
        else:
            leaves.append(next(d_iter))

    # Correct path as bit string from root to leaf (0=left, 1=right).
    correct_bits = []
    idx = target_leaf_idx
    for _ in range(depth):
        correct_bits.append(idx % 2)
        idx //= 2
    correct_bits.reverse()  # MSB = depth 0 choice

    # Assign UUID keys to all internal nodes. Index internal nodes in heap order:
    # node 0 = root; children of i are 2i+1, 2i+2. Leaves are conceptual only.
    n_internal = 2**depth - 1
    used_entity: set = set()
    keys = [_new_key(rng, uuid_format, used_entity) for _ in range(n_internal)]
    root_key = keys[0]

    # Map leaf index -> parent internal node index and which child (0/1).
    # Leaves attach under the last level of internal nodes (indices 2^{D-1}-1 .. 2^D-2).
    edges: List[TreeEdge] = []
    correct_path_keys = [root_key]
    fork_audit: List[Dict[str, Any]] = []

    # Walk all internal nodes.
    for node_idx in range(n_internal):
        node_depth = _node_depth(node_idx)
        # Is this node on the correct path?
        on_correct = True
        cur = 0
        for b in correct_bits[:node_depth]:
            cur = 2 * cur + 1 + b
        if cur != node_idx:
            on_correct = False

        left_is_leaf = (2 * node_idx + 1) >= n_internal
        # Child values
        child_vals: List[Any] = []
        if left_is_leaf:
            # Leaf indices under this parent: for parent at last internal level
            # parents are indices [2^{D-1}-1, 2^D-2], leaf base = 2*(node_idx+1)-n_internal ...
            # Cleaner: leaf i attaches to parent ((i + n_internal) - 1) // 2 in 0-index heap
            # For node_idx, left leaf index = node_idx*2+1 - n_internal + n_leaves? 
            # Standard: leaves numbered 0..n_leaves-1 in left-to-right order.
            # First leaf-parent index = 2**(depth-1) - 1
            first_leaf_parent = 2 ** (depth - 1) - 1
            leaf_base = (node_idx - first_leaf_parent) * 2
            child_vals = [leaves[leaf_base], leaves[leaf_base + 1]]
        else:
            child_vals = [keys[2 * node_idx + 1], keys[2 * node_idx + 2]]

        # Which child is correct (if on correct path)?
        correct_child = correct_bits[node_depth] if on_correct else None

        for child_side, val in enumerate(child_vals):
            is_corr = on_correct and (child_side == correct_child)
            is_sibling_distractor = on_correct and (child_side != correct_child)
            if is_corr:
                pref = "newer"
            elif is_sibling_distractor:
                pref = "older"
            else:
                pref = "any"
            edges.append(
                TreeEdge(
                    key=keys[node_idx],
                    value=val,
                    is_correct_path=is_corr,
                    depth=node_depth,
                    sibling_of_correct=is_sibling_distractor,
                    session_preference=pref,
                )
            )

        if on_correct:
            if not left_is_leaf and correct_child is not None:
                next_key = keys[2 * node_idx + 1 + correct_child]
                correct_path_keys.append(next_key)
            fork_audit.append(
                {
                    "depth": node_depth,
                    "key": keys[node_idx],
                    "correct_value": child_vals[correct_child] if correct_child is not None else None,
                    "distractor_value": child_vals[1 - correct_child] if correct_child is not None else None,
                }
            )

    tree = BasicStateTree(
        depth=depth,
        root_key=root_key,
        target_question=target_question,
        target_answer=target_answer,
        edges=edges,
        leaves=leaves,
        correct_path_keys=correct_path_keys,
        audit={
            "mode": f"basic_statetree_D{depth}",
            "target_leaf_idx": target_leaf_idx,
            "correct_bits": correct_bits,
            "forks": fork_audit,
            "n_edges": len(edges),
            "n_leaves": n_leaves,
        },
    )
    return tree


def _aggregate_step_question(step1: str, step2: str, step3: str) -> str:
    person_name = step1.replace("[A]", "").strip()
    event_phrase = step2.replace("[B]", "").strip()
    return step3.replace("[A]", person_name).replace("[B]", event_phrase)


def _first_leaf_under(node_idx: int, depth: int) -> int:
    """Leftmost leaf index (0 .. 2^D-1) under an internal node in a depth-D heap."""
    d = _node_depth(node_idx)
    cur = node_idx
    for _ in range(depth - d):
        cur = 2 * cur + 1
    return cur - (2**depth - 1)


def build_compositional_tree(
    *,
    decomposition: Dict[str, Any],
    rng: Optional[random.Random] = None,
    uuid_format: str = "default",
    depth: int = 3,
) -> CompositionalStateTree:
    """
    Build a Compositional StateTree from an LLM decomposition JSON
    (paper Algorithm 2 / Appendix decomposition schema).

    Expected schema::

        {
          "persons": [
            {
              "step_1": "[A] Name",
              "events": [
                {
                  "step_2": "[B] gerund",
                  "leaves": [
                    {"step_3": "When did [A] start [B]?", "answer": "...", "is_target": true},
                    {"step_3": "...", "answer": "...", "is_target": false}
                  ]
                },
                { ... }
              ]
            },
            { ... }
          ]
        }
    """
    if depth != 3:
        raise ValueError("Compositional StateTree currently supports depth=3 (2x2x2).")
    rng = rng or random.Random()
    persons = decomposition.get("persons") or []
    if len(persons) != 2:
        raise ValueError("decomposition must contain exactly 2 persons")

    by_leaf: Dict[int, Dict[str, Any]] = {}
    target_path: Optional[Tuple[int, int, int]] = None
    target_answer = ""
    target_question = ""

    for pi, person in enumerate(persons):
        events = person.get("events") or []
        if len(events) != 2:
            raise ValueError(f"person {pi} must have 2 events")
        for ei, event in enumerate(events):
            leaves = event.get("leaves") or []
            if len(leaves) != 2:
                raise ValueError(f"event ({pi},{ei}) must have 2 leaves")
            for li, leaf in enumerate(leaves):
                step1 = person["step_1"]
                step2 = event["step_2"]
                step3 = leaf["step_3"]
                q = _aggregate_step_question(step1, step2, step3)
                ans = leaf.get("answer", "")
                is_target = bool(leaf.get("is_target"))
                leaf_idx = (pi << 2) | (ei << 1) | li
                by_leaf[leaf_idx] = {
                    "steps": [step1, step2, step3],
                    "question": q,
                    "answer": ans,
                    "is_target": is_target,
                    "path": (pi, ei, li),
                }
                if is_target:
                    target_path = (pi, ei, li)
                    target_answer = ans
                    target_question = q

    if target_path is None:
        raise ValueError("decomposition must mark exactly one leaf with is_target=true")
    if len(by_leaf) != 8:
        raise ValueError("decomposition must define 8 leaves")

    n_internal = 2**depth - 1
    used_entity: set = set()
    keys = [_new_key(rng, uuid_format, used_entity) for _ in range(n_internal)]
    root_key = keys[0]
    correct_bits = list(target_path)
    path_steps = by_leaf[(target_path[0] << 2) | (target_path[1] << 1) | target_path[2]]["steps"]

    edges: List[TreeEdge] = []
    for node_idx in range(n_internal):
        node_depth = _node_depth(node_idx)
        cur = 0
        for b in correct_bits[:node_depth]:
            cur = 2 * cur + 1 + b
        on_correct = cur == node_idx
        correct_child = correct_bits[node_depth] if on_correct else None
        left_is_leaf = (2 * node_idx + 1) >= n_internal

        child_vals: List[Any] = []
        if left_is_leaf:
            first_leaf_parent = 2 ** (depth - 1) - 1
            leaf_base = (node_idx - first_leaf_parent) * 2
            for side in (0, 1):
                step = by_leaf[leaf_base + side]["steps"][node_depth]
                child_vals.append({"step": step})
        else:
            for side in (0, 1):
                child_idx = 2 * node_idx + 1 + side
                leaf_idx = _first_leaf_under(child_idx, depth)
                step = by_leaf[leaf_idx]["steps"][node_depth]
                child_vals.append({"step": step, "next": keys[child_idx]})

        for side, val in enumerate(child_vals):
            is_corr = on_correct and (side == correct_child)
            is_sibling = on_correct and (side != correct_child)
            pref = "newer" if is_corr else ("older" if is_sibling else "any")
            edges.append(
                TreeEdge(
                    key=keys[node_idx],
                    value=val,
                    is_correct_path=is_corr,
                    depth=node_depth,
                    sibling_of_correct=is_sibling,
                    session_preference=pref,
                )
            )

    return CompositionalStateTree(
        depth=depth,
        root_key=root_key,
        target_question=target_question,
        target_answer=target_answer,
        edges=edges,
        path_steps=path_steps,
        audit={
            "mode": "compositional_statetree_D3",
            "target_path": target_path,
            "path_steps": path_steps,
            "n_edges": len(edges),
        },
    )


def validate_basic_tree(tree: BasicStateTree) -> None:
    expected_edges = 2 * (2**tree.depth - 1)
    expected_leaves = 2**tree.depth
    if len(tree.edges) != expected_edges:
        raise AssertionError(f"expected {expected_edges} edges, got {len(tree.edges)}")
    if len(tree.leaves) != expected_leaves:
        raise AssertionError(f"expected {expected_leaves} leaves, got {len(tree.leaves)}")
    if tree.leaves.count(tree.target_question) != 1:
        raise AssertionError("target question must appear exactly once among leaves")
