# StateTree ROLL configs

**Authoritative Hydra configs** for GRPO training live in this folder (not under
a separate `configs/` directory). Paper models (Section 4):

| Config | Model | Prompt batch | Template | Notes |
|--------|-------|--------------|----------|-------|
| `grpo_qwen2.5_7b` | Qwen2.5-7B-Instruct | 64 | `statetree_qwen2_5` | default |
| `grpo_qwen2.5_14b` | Qwen2.5-14B-Instruct | 32 | `statetree_qwen2_5` | Megatron TP=2 |
| `grpo_qwen3_8b` | Qwen3-8B | 64 | `statetree_qwen3` | thinking enabled |
| `*_em_only` | same | small | same | EM reward only (smoke) |

## Shared hyper-params (full configs)

| Knob | Value |
|------|-------|
| Algorithm | GRPO (`adv_estimator: reinforce`) |
| Group size | 8 |
| LR | `1e-6` |
| Grad clip | 1.0 |
| KL β | 0.001 |
| Rollout T / top-p | 0.6 / 0.95 |
| `prompt_length` / `response_length` | 10240 / 4096 |
| Reward | `max(EM, LLM-Judge)`, judge Qwen2.5-1.5B-Instruct @ T=0 |
| Curriculum steps | Warm-up 40 → Basic D=2 100 → Basic D=3 100 → Compositional 60 |
| Stage-3 DAPO mix | optional `DAPO_MATH_JSONL` (~2500 rows) |

Default YAML `device_mapping` is 8 GPUs; paper used 32×H20 — expand mapping as needed.

```bash
# single stage
python examples/roll/start_rlvr_pipeline.py \
  --config_path examples/roll/statetree \
  --config_name grpo_qwen2.5_14b \
  max_steps=40 \
  actor_train.data_args.file_name=[data/curriculum/stage0_warmup.roll.jsonl]

# full curriculum
export DAPO_MATH_JSONL=/path/to/dapo_math.jsonl   # optional
CONFIG=grpo_qwen3_8b bash examples/roll/statetree/run_curriculum.sh
```

See also [`docs/TRAINING_ROLL.md`](../../../docs/TRAINING_ROLL.md).
