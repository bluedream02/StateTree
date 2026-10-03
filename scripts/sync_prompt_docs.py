#!/usr/bin/env python3
"""Export appendix-style plain-text prompts from statetree.prompts (source of truth).

Writes into docs/prompts/. Re-run after editing statetree/prompts.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from statetree.prompts import (  # noqa: E402
    DECOMPOSITION_SYSTEM_PROMPT,
    LOCOMO_JUDGE_PROMPT,
    SYSTEM_PROMPT,
    basic_statetree_prompt,
    compositional_statetree_prompt,
    warmup_prompt,
)

OUT = ROOT / "docs" / "prompts"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "system.txt": SYSTEM_PROMPT.strip() + "\n",
        "warmup.txt": warmup_prompt("{question}").strip() + "\n",
        "basic_statetree.txt": basic_statetree_prompt("{root_uuid}").strip() + "\n",
        "compositional_statetree.txt": compositional_statetree_prompt("{root_uuid}").strip() + "\n",
        "decomposition_system.txt": DECOMPOSITION_SYSTEM_PROMPT.strip() + "\n",
        "locomo_judge.txt": LOCOMO_JUDGE_PROMPT.strip() + "\n",
    }
    for name, text in files.items():
        path = OUT / name
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
