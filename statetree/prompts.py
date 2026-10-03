"""Prompt templates from the StateTree paper (Appendix)."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant "
    "solves it. The Assistant first thinks about the reasoning process in the mind and then "
    "provides the User with the answer. The reasoning process is enclosed within <think> "
    "</think> and answer is enclosed within \\boxed{} tags, respectively, i.e., "
    "<think> reasoning process here </think> \\boxed{answer here}."
)


def warmup_prompt(question: str) -> str:
    return (
        "Based on the above conversations, write a short answer for the following question "
        "in a few words. Do not write complete and lengthy sentences. Answer with exact words "
        "from the conversations whenever possible.\n\n"
        f"Question: {question}"
    )


def basic_statetree_prompt(root_uuid: str) -> str:
    return (
        'In the conversation above, there are JSON records like {"KEY": "VALUE"} scattered '
        "throughout the dialogue text. They form UUID chains: starting from a given key, each "
        "value either points to the next key (a UUID) or contains the final question to answer.\n\n"
        f'Your task: start from key "{root_uuid}" and follow the chain to find the question, '
        "then answer it.\n\n"
        "How to follow the chain:\n"
        "1. Search the entire conversation for all records whose key matches the current UUID.\n"
        '2. Each record sits inside a specific session. Look at the nearest preceding "DATE: ..." '
        "line to determine that record's time.\n"
        "3. If the same key appears in multiple sessions, choose which record to use: among the "
        "remaining records, pick the one from the most recent session DATE.\n"
        "4. Read the chosen value:\n"
        "   - If it is a UUID, treat it as the next key and go back to step 1.\n"
        "   - If it is a natural-language question, that is the question you must answer.\n"
        "5. Once you find the question, answer it based on the conversation content."
    )


def compositional_statetree_prompt(root_uuid: str) -> str:
    return (
        'In the conversation above, there are JSON records like {"UUID": {"step": "...", "next": "UUID"}} '
        "scattered throughout the dialogue text. They form a decision tree.\n\n"
        "Each record maps a UUID key to a JSON object with:\n"
        '- "step": a piece of information to accumulate along the path\n'
        '- "next": the UUID of the next node to follow (absent at leaf nodes)\n\n'
        f'Your task: start from key "{root_uuid}" and follow the tree to a leaf, accumulating the '
        "step information at each edge. Then answer the question.\n\n"
        "How to follow the tree:\n"
        "1. Search the entire conversation for all records whose key matches the current UUID.\n"
        '2. Each record sits inside a specific session. Look at the nearest preceding "DATE: ..." '
        "line to determine that record's session time.\n"
        "3. If the same key appears in multiple sessions, pick the record from the most recent session DATE.\n"
        '4. Read the "step" field and remember it.\n'
        '5. If a "next" field exists, use it as the new key and go back to step 1.\n'
        '6. If no "next" field exists, you have reached a leaf.\n\n'
        "Once you reach a leaf, aggregate all the step information you collected along the path to "
        "form the final question. Then answer that question step by step based on the conversation content."
    )


DECOMPOSITION_SYSTEM_PROMPT = """\
You are a dataset construction assistant. Your task is to decompose a target question into a \
2x2x2 binary discrimination tree for a reading comprehension benchmark.

The tree has 3 levels:
- Level 1 (root -> 2 branches): Person selection. Each branch carries step_1 = "[A] PersonName"
- Level 2 (each person -> 2 branches): Event category. Each branch carries step_2 = "[B] gerund_phrase". \
The gerund phrase must be concrete and specific (e.g., "attending the LGBTQ support group"), NOT a vague \
category (e.g., "community activities").
- Level 3 (each event -> 2 branches): Question specificity. Each branch carries step_3 = a question \
template that MUST contain BOTH [A] and [B] as placeholders. The two questions under the same event MUST \
ask about different factual aspects.

CRITICAL: step_3 must ask about CONCRETE FACTS that have definite answers --- things like time, place, \
people involved, specific actions taken, specific results/outcomes, or specific objects.

GOOD examples (factual): "When did [A] [B]?", "What did [A] make while [B]?", "Where did [A] go for [B]?"
BAD examples (subjective/generic): "How did [A] feel about [B]?", "Why did [A] enjoy [B]?"

Exactly ONE of the 8 leaves must be marked is_target=true --- the leaf whose step_3, after substituting \
[A] and [B], yields (or is semantically equivalent to) the original target question.

Rules:
1. The target person MUST appear first in the tree.
2. The target event MUST appear first under the target person.
3. The target leaf MUST be the first leaf under the target event.
4. Each event_gerund should be a present participle phrase derived from an actual question in the pool.
5. Every leaf's "answer" field MUST be copied verbatim from a question in the pool.
6. The two leaves under the same event MUST ask about genuinely DIFFERENT ASPECTS of the event.
7. Events for the same person should be distinct and both grounded in the question pool.
8. Events for the other person should also correspond to real pool questions but thematically contrast \
with the target person's events.
9. All answers must be short (a few words or a short phrase), copied from the pool.
10. EVERY step_3 MUST contain both [A] and [B] placeholders.

Output schema (raw JSON only, no markdown fencing):
{
  "persons": [
    {
      "step_1": "[A] PersonName",
      "events": [
        {
          "step_2": "[B] gerund_phrase",
          "leaves": [
            {"step_3": "When did [A] start [B]?", "answer": "...", "is_target": true},
            {"step_3": "What did [A] do while [B]?", "answer": "...", "is_target": false}
          ]
        },
        {
          "step_2": "[B] other_gerund",
          "leaves": [
            {"step_3": "...", "answer": "...", "is_target": false},
            {"step_3": "...", "answer": "...", "is_target": false}
          ]
        }
      ]
    },
    {
      "step_1": "[A] OtherPerson",
      "events": [ ... same shape with is_target=false ... ]
    }
  ]
}
"""


def decomposition_user_prompt(
    *,
    target_question: str,
    sample_id: str,
    row_index: int,
    speaker_a: str,
    speaker_b: str,
    question_pool: str,
    num_questions: int,
) -> str:
    return (
        f"Target question (row {row_index}):\n"
        f"  Q: {target_question}\n"
        f"  sample_id: {sample_id}\n\n"
        f"Speakers in this conversation:\n"
        f"  Speaker A: {speaker_a}\n"
        f"  Speaker B: {speaker_b}\n\n"
        f"Below is the full question pool for this conversation ({num_questions} questions). "
        "Use these to find suitable distractor questions and events. Pick events and questions "
        "that are grounded in the actual conversation content.\n\n"
        "--- Question Pool ---\n"
        f"{question_pool}\n"
        "--- End of Question Pool ---\n\n"
        "Please generate the 2x2x2 decomposition tree JSON for this target question. Remember:\n"
        "- step_2 must describe the EVENT/ACTIVITY itself, NOT reveal or hint at the answer\n"
        "- EVERY step_3 MUST contain BOTH [A] and [B] placeholders\n"
        "- step_3 must ask about CONCRETE FACTS\n"
        "- The target person and target event come first in the tree\n"
        "- Exactly 1 leaf is is_target=true\n"
        "- Output raw JSON only, no markdown fencing"
    )


LOCOMO_JUDGE_PROMPT = """\
Your task is to label an answer to a question as "CORRECT" or "WRONG".

You will be given the following data: (1) a question (posed by one user to another user), \
(2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user \
based on their prior conversations. The gold answer will usually be a concise and short answer \
that includes the referenced topic, for example:

Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

The generated answer might be much longer, but you should be generous with your grading --- as \
long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The \
generated answer might be much longer or use relative time references (like 'last Tuesday' or \
'next month'), but you should be generous with your grading --- as long as it refers to the same \
date or time period as the gold answer, it should be counted as CORRECT. Even if the format \
differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.

Now it's time for the real question:

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return the label CORRECT or WRONG in a json format with the key as "label". Do NOT include both \
CORRECT and WRONG in your response, or it will break the evaluation script.
"""
