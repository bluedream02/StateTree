"""
Embed StateTree edge records into multi-session dialogue text.

Follows the paper appendix:
- Insert inside speaker quotations at sentence boundaries
- Least-loaded session distribution
- Same-key records go to different sessions
- Correct-path edges placed in strictly newer sessions than distractor siblings
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .tree import TreeEdge

DATE_RE = re.compile(r"^\s*DATE:\s*(.+?)\s*$")

@dataclass
class SessionInfo:
    line_idx: int
    date_str: str
    dt: Optional[datetime]


def parse_locomo_timestamp(ts: str) -> Optional[datetime]:
    """Parse LoCoMo-style timestamps like '3:31 pm on 23 August, 2023'."""
    ts = (ts or "").strip()
    if not ts:
        return None
    # Common LoCoMo formats
    patterns = [
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %B %Y",
        "%H:%M on %d %B, %Y",
        "%d %B, %Y",
        "%B %d, %Y",
    ]
    # Normalize am/pm casing
    cand = ts.replace("a.m.", "am").replace("p.m.", "pm")
    cand = re.sub(r"\s+", " ", cand)
    for fmt in patterns:
        try:
            return datetime.strptime(cand, fmt)
        except ValueError:
            continue
    # Fallback: try dropping leading time
    m = re.search(r"(\d{1,2}\s+\w+,?\s+\d{4})", cand)
    if m:
        for fmt in ("%d %B, %Y", "%d %B %Y"):
            try:
                return datetime.strptime(m.group(1).replace(",", ""), "%d %B %Y")
            except ValueError:
                try:
                    return datetime.strptime(m.group(1), fmt)
                except ValueError:
                    pass
    return None


def _collect_sessions(lines: List[str]) -> List[SessionInfo]:
    sessions: List[SessionInfo] = []
    for i, ln in enumerate(lines):
        m = DATE_RE.match(ln.strip())
        if m:
            ds = m.group(1).strip()
            sessions.append(SessionInfo(line_idx=i, date_str=ds, dt=parse_locomo_timestamp(ds)))
    return sessions


def _session_index_for_line(line_idx: int, sessions: Sequence[SessionInfo]) -> int:
    best = -1
    for si, s in enumerate(sessions):
        if s.line_idx <= line_idx:
            best = si
        else:
            break
    return best


def _split_said_quote(line: str) -> Optional[Tuple[str, str, str]]:
    """Return (prefix_incl_open_quote, inner, suffix_from_close_quote) or None."""
    # Prefer explicit 'said, "' pattern; fall back to first/last quote.
    lower = line.lower()
    marker = 'said, "'
    pos = lower.find(marker)
    if pos >= 0:
        start = pos + len(marker)
        # Find matching closing quote: last " before optional " and shared..."
        # Simple heuristic: last double-quote in the line.
        end = line.rfind('"')
        if end > start:
            return line[:start], line[start:end], line[end:]
    # Generic: first and last quote
    first = line.find('"')
    last = line.rfind('"')
    if 0 <= first < last:
        return line[: first + 1], line[first + 1 : last], line[last:]
    return None


def _quote_insert_offsets(inner: str) -> List[int]:
    if not inner.strip():
        return []
    positions = set()
    for m in re.finditer(r"[.!?]\s+", inner):
        positions.add(m.end())
    stripped = inner.rstrip()
    if stripped and stripped[-1] in ".!?":
        positions.add(len(stripped))
    positions.add(len(inner))
    return sorted(positions)


def _session_time_key(sessions: Sequence[SessionInfo], sess_idx: int) -> Tuple:
    if sess_idx < 0 or sess_idx >= len(sessions):
        return (0, sess_idx)
    s = sessions[sess_idx]
    if s.dt is not None:
        return (1, s.dt.toordinal(), s.dt.hour, s.dt.minute, sess_idx)
    return (0, sess_idx)


def embed_records_in_dialogue(
    context: str,
    edges: Sequence[TreeEdge],
    rng: Optional[random.Random] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Insert edge records into dialogue. Returns (augmented_context, placement_audit).
    """
    rng = rng or random.Random()
    if not edges:
        return context, {"placements": []}

    lines = context.splitlines(True)
    sessions = _collect_sessions(lines)
    if not sessions:
        # No DATE markers: append records at the end.
        block = "\n".join(e.to_record_string() + "." for e in edges) + "\n"
        return context.rstrip() + "\n\n" + block, {"placements": [], "fallback": "append"}

    # Build insertion slots: (line_idx, offset_in_inner, session_idx)
    slots: List[Tuple[int, int, int]] = []
    for li, ln in enumerate(lines):
        parsed = _split_said_quote(ln)
        if not parsed:
            continue
        _, inner, _ = parsed
        sess_idx = _session_index_for_line(li, sessions)
        if sess_idx < 0:
            continue
        for off in _quote_insert_offsets(inner):
            slots.append((li, off, sess_idx))

    if not slots:
        block = "\n".join(e.to_record_string() + "." for e in edges) + "\n"
        return context.rstrip() + "\n\n" + block, {"placements": [], "fallback": "append_no_slots"}

    # Group edges by key for same-key separation + temporal ordering on forks.
    by_key: Dict[str, List[TreeEdge]] = {}
    for e in edges:
        by_key.setdefault(e.key, []).append(e)

    # Count load per session
    load = {i: 0 for i in range(len(sessions))}
    used_sessions_by_key: Dict[str, set] = {}
    placements: List[Tuple[TreeEdge, int, int, int]] = []  # edge, line, offset, sess

    # Snapshot of all slots for recycling when the dialogue is short.
    all_slots = list(slots)

    def pick_slot(
        edge: TreeEdge,
        *,
        forbidden_sessions: Optional[set] = None,
        prefer: str = "any",
        ref_session: Optional[int] = None,
    ) -> Optional[Tuple[int, int, int]]:
        forbidden_sessions = forbidden_sessions or set()
        # Prefer unused slots; if exhausted, recycle existing slot positions.
        pool = list(slots) if slots else list(all_slots)
        if not pool:
            return None
        candidates = [s for s in pool if s[2] not in forbidden_sessions]
        if not candidates:
            candidates = list(pool)
        if prefer == "newer" and ref_session is not None:
            newer = [
                s
                for s in candidates
                if _session_time_key(sessions, s[2]) > _session_time_key(sessions, ref_session)
            ]
            if newer:
                candidates = newer
        elif prefer == "older" and ref_session is not None:
            older = [
                s
                for s in candidates
                if _session_time_key(sessions, s[2]) < _session_time_key(sessions, ref_session)
            ]
            if older:
                candidates = older

        if not candidates:
            candidates = list(pool)
        min_load = min(load[s[2]] for s in candidates)
        least = [s for s in candidates if load[s[2]] == min_load]
        return rng.choice(least)

    # Place fork pairs first (correct newer, distractor older).
    placed_ids = set()
    for key, group in by_key.items():
        corr = [e for e in group if e.is_correct_path]
        dist = [e for e in group if e.sibling_of_correct]
        if len(corr) == 1 and len(dist) == 1:
            # Place distractor first in some session, then correct in a newer one.
            d_slot = pick_slot(dist[0], forbidden_sessions=used_sessions_by_key.get(key, set()))
            if d_slot is None:
                continue
            used_sessions_by_key.setdefault(key, set()).add(d_slot[2])
            load[d_slot[2]] += 1
            slots.remove(d_slot)
            placements.append((dist[0], d_slot[0], d_slot[1], d_slot[2]))
            placed_ids.add(id(dist[0]))

            c_slot = pick_slot(
                corr[0],
                forbidden_sessions=used_sessions_by_key.get(key, set()),
                prefer="newer",
                ref_session=d_slot[2],
            )
            if c_slot is None:
                continue
            # If we failed to get a newer session, swap preference by exchanging if needed later.
            used_sessions_by_key.setdefault(key, set()).add(c_slot[2])
            load[c_slot[2]] += 1
            if c_slot in slots:
                slots.remove(c_slot)
            placements.append((corr[0], c_slot[0], c_slot[1], c_slot[2]))
            placed_ids.add(id(corr[0]))

            # Post-check temporal order; swap if violated.
            if _session_time_key(sessions, placements[-1][3]) <= _session_time_key(
                sessions, placements[-2][3]
            ):
                # Swap session assignments by swapping placement tuples' session/line/offset
                a = placements[-2]
                b = placements[-1]
                placements[-2] = (a[0], b[1], b[2], b[3])
                placements[-1] = (b[0], a[1], a[2], a[3])

    # Place remaining edges with least-loaded + same-key separation.
    for edge in edges:
        if id(edge) in placed_ids:
            continue
        slot = pick_slot(edge, forbidden_sessions=used_sessions_by_key.get(edge.key, set()))
        if slot is None:
            continue
        used_sessions_by_key.setdefault(edge.key, set()).add(slot[2])
        load[slot[2]] += 1
        if slot in slots:
            slots.remove(slot)
        placements.append((edge, slot[0], slot[1], slot[2]))

    # Apply insertions line by line (offsets descending).
    by_line: Dict[int, List[Tuple[int, str]]] = {}
    audit_rows = []
    for edge, li, off, sess_idx in placements:
        rec = edge.to_record_string()
        by_line.setdefault(li, []).append((off, rec))
        audit_rows.append(
            {
                "record": rec,
                "session": sessions[sess_idx].date_str if 0 <= sess_idx < len(sessions) else "",
                "is_correct_path": edge.is_correct_path,
                "depth": edge.depth,
                "session_preference": edge.session_preference,
            }
        )

    out_lines = list(lines)
    for li, ops in by_line.items():
        ops.sort(key=lambda x: x[0], reverse=True)
        parsed = _split_said_quote(out_lines[li])
        if not parsed:
            continue
        prefix, inner, suffix = parsed
        for pos, rec in ops:
            if pos < len(inner):
                inner = inner[:pos] + rec + ". " + inner[pos:]
            else:
                inner = inner + rec + "."
        out_lines[li] = prefix + inner + suffix

    return "".join(out_lines), {"placements": audit_rows}
