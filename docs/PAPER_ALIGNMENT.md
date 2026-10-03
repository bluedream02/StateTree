# Paper ↔ Code Alignment Checklist

Cross-checked against `NeurIPS_2026___StateTree (2)` (methods §3, experiments §4, appendix).
Re-audited 2026-09-25 against full `3_methods.tex` / `4_experiments.tex` / `7_appendix.tex`.

Legend: ✅ aligned · ⚠️ partial / needs care · ❌ missing for full table reproduction

## Data construction

| Paper claim | Code | Status |
|-------------|------|--------|
| Complete binary tree depth \(D\); \(2^D\) leaves; \(2(2^D-1)\) edges | `statetree/tree.py` `build_basic_tree` + `validate_basic_tree` | ✅ |
| Internal keys = UUID4; leaf values = questions | `uuid_format=default` (UUID4); leaves hold NL questions | ✅ |
| Correct fork edge in **newer** session than distractor | `insert.py` fork placement + temporal swap | ✅ |
| Insert **inside speaker quotes** at sentence boundaries | `embed_records_in_dialogue` / `_split_said_quote` | ✅ |
| Least-loaded session distribution + same-key ≠ same session | `pick_slot` + `used_sessions_by_key` | ✅ |
| Distractors from **same conversation** (pool size 64 Stages 1–2) | `build_statetree_curriculum.py` prefers same `sample_id`, pool cap 64 | ✅ |
| Stage 3 distractors from \(2\times2\times2\) decomp (8 leaves) | `build_compositional_tree` | ✅ |
| Compositional `{step, next}` records; GPT-4o T=0.7 + validation | `decompose.py` + appendix prompts | ✅ |
| Split 616 / 154 / 770 after filter | `locomo.split_qa_rows` | ✅ |
| Entity-key ablation pool (14 nouns) | `uuid_format=entity` + `ENTITY_KEY_POOL` | ✅ (flag only; no dedicated recipe) |

## Prompts (Appendix)

| Prompt | Code | Status |
|--------|------|--------|
| System (`<think>` + `\boxed{}`) | `statetree/prompts.py` `SYSTEM_PROMPT`; ROLL `messages` | ✅ |
| Warm-up = LoCoMo QA prompt | `warmup_prompt` | ✅ |
| Basic StateTree traversal prompt | `basic_statetree_prompt` | ✅ |
| Compositional prompt | `compositional_statetree_prompt` | ✅ |
| Decomposition system/user (GPT-4o, T=0.7) | `decompose.py` + `DECOMPOSITION_*` | ✅ |
| LoCoMo judge CORRECT/WRONG | `LOCOMO_JUDGE_PROMPT` / `StateTree-LoCoMo-judge` (+ `docs/prompts/` via sync) | ✅ |
| CoT ablation system prompt (“step by step…”) | — | ❌ (appendix text not exported as recipe) |

## RL / curriculum

| Paper claim | Code | Status |
|-------------|------|--------|
| 4 stages: Warmup → Basic \(D=2\) → Basic \(D=3\) → Compositional \(D=3\) | `build_statetree_curriculum.py` + `run_curriculum.sh` | ✅ |
| Steps 40 / 100 / 100 / 60 | `run_curriculum.sh` defaults | ✅ |
| GRPO group size 8; LR \(10^{-6}\); KL \(\beta=0.001\); T=0.6; top-p=0.95; resp 4096 | per-model yamls | ✅ |
| Prompt batch 64 (7B/8B), 32 (14B) | `grpo_qwen2.5_7b` / `grpo_qwen3_8b` =64; `grpo_qwen2.5_14b` =32 | ✅ |
| Models: Qwen2.5-7B / 14B / Qwen3-8B | dedicated yaml under `examples/roll/statetree/` | ✅ |
| Context ~10K train | `prompt_length: 10240` | ✅ |
| Reward \(r=\max(r_{\mathrm{EM}}, r_{\mathrm{LLM}})\); judge Qwen2.5-1.5B, T=0 | `StateTreeCombinedRewardWorker` | ✅ |
| Stage-3 mix 2500 DAPO-Math | `DAPO_MATH_JSONL=... run_curriculum.sh` | ⚠️ optional env (warns if unset) |
| Train on 32×H20; 3 seeds | Config template 8 GPU; no multi-seed runner | ⚠️ |

## Evaluation & ablations (reproduction gaps)

| Paper claim | Code | Status |
|-------------|------|--------|
| LoCoMo ACC (GPT-4o judge) + F1 + BLEU-1 | `scripts/evaluate.py` + `statetree/metrics.py`; `--judge_model gpt-4o` | ✅ |
| LongMemEval (GPT-4o yes/no judges) | documented in `docs/DATA.md` only | ❌ |
| PersonaMem EM (32k/128k) | same | ❌ |
| Short-context MMLU / MATH-500 / IFEval | — | ❌ |
| Indep. random keys / Single-session ablations | — | ❌ |
| Entity Keys end-to-end recipe | `--uuid_format entity` only | ⚠️ |
| Reward ablations (judge-only, F1, ROUGE-L, substr EM) | combined + `*_em_only` | ⚠️ |
| Curriculum stage-drop ablations | can omit stages manually | ⚠️ no dedicated scripts |

## Design takeaways (what to learn)

1. **Structure inside authentic dialogue** — inject navigable records into real multi-session chats, not synthetic haystacks.
2. **Temporal fork = knowledge update** — correct edge always newer than distractor; prompt encodes “most recent DATE wins”.
3. **Curriculum: navigate → deepen → compose** — Stage 1 (\(D{=}2\)) contributes most in paper ablations.
4. **Combined reward** — `max(EM, Judge)` beats dense lexical rewards and CoT-only prompting.
5. **Ablate the structure** — independent keys / single session show *why* shared keys + cross-session layout matter.
6. **Anti-forgetting mix** — DAPO-Math on Stage 3 keeps short-context flat while long-context skills transfer to 128K.

## How to re-verify quickly

```bash
python scripts/build_statetree_curriculum.py \
  --input_jsonl examples/demo_qa.jsonl \
  --output_dir outputs/demo_curriculum \
  --stages warmup,basic2
python -c "import json; d=json.loads(open('outputs/demo_curriculum/stage1_basic_D2.roll.jsonl').readline()); \
 print(json.loads(d['messages'])[0]['role'], 'boxed' in json.loads(d['messages'])[0]['content'])"
```
