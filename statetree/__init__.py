"""StateTree: data-driven RL pseudo-tasks for long-term dialogue reasoning."""

from .tree import BasicStateTree, CompositionalStateTree, TreeEdge, build_basic_tree, build_compositional_tree
from .insert import embed_records_in_dialogue
from .prompts import SYSTEM_PROMPT, basic_statetree_prompt, compositional_statetree_prompt, warmup_prompt

__version__ = "0.1.0"
__all__ = [
    "BasicStateTree",
    "CompositionalStateTree",
    "TreeEdge",
    "build_basic_tree",
    "build_compositional_tree",
    "embed_records_in_dialogue",
    "SYSTEM_PROMPT",
    "basic_statetree_prompt",
    "compositional_statetree_prompt",
    "warmup_prompt",
]
