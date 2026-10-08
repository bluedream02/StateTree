# [NeurIPS 2026] StateTree: Enhancing Long-Term Dialogue Reasoning via Reinforcement Learning

Official implementation of **"StateTree: Enhancing Long-Term Dialogue Reasoning via Reinforcement Learning"** (NeurIPS 2026).

## 📑 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
  - [Demo curriculum](#demo-curriculum)
  - [Build LoCoMo curriculum](#build-locomo-curriculum)
  - [Train with GRPO](#train-with-grpo)
  - [Evaluate](#evaluate)
- [Repository Layout](#-repository-layout)
- [Acknowledgement](#-acknowledgement)
- [Citation](#-citation)

## 📌 Overview

Personalized LLM assistants must reason over long, evolving multi-session histories. **StateTree** embeds a navigable binary tree into authentic dialogues to construct challenging auxiliary tasks, then trains models with a four-stage **curriculum GRPO** to acquire:

- cross-session retrieval
- temporal discrimination and knowledge update
- compositional multi-hop reasoning

Training uses ~10K-token contexts and transfers to evaluations with contexts up to **128K** tokens (LoCoMo, LongMemEval, PersonaMem).

## 📦 Installation

```bash
cd code
conda create -n statetree python=3.10 -y
conda activate statetree
pip install -e .
pip install -r requirements.txt
```

For **GRPO training**, additionally install:

```bash
pip install -r requirements_roll_common.txt
pip install -r requirements_roll_torch280_vllm.txt
```

Set API keys when generating Stage-3 decompositions or running the GPT judge:

```bash
export OPENAI_API_KEY="your-api-key"
# export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional
```

For cluster setup and training tips, see [`docs/TRAINING_ROLL.md`](docs/TRAINING_ROLL.md).

## 🚀 Quick Start

### Demo curriculum

Build Warm-up and Basic StateTree (`D=2,3`) from the bundled dialogues:

```bash
python scripts/build_statetree_curriculum.py \
  --input_jsonl examples/demo_qa.jsonl \
  --output_dir outputs/demo_curriculum \
  --stages warmup,basic2,basic3 \
  --seed 42
```

Build Compositional StateTree with the demo decomposition file:

```bash
python scripts/build_statetree_curriculum.py \
  --input_jsonl examples/demo_qa.jsonl \
  --output_dir outputs/demo_curriculum \
  --stages compositional \
  --decompositions_json examples/demo_decompositions.json
```

### Build LoCoMo curriculum

**1) Prepare QA splits** — download [LoCoMo](https://github.com/snap-research/locomo) (`locomo10.json`):

```bash
python scripts/prepare_locomo.py \
  --input /path/to/locomo10.json \
  --output_dir data/locomo_qa
```

This writes `locomo_qa_{train,val,test,all}.jsonl` (paper split: **616 / 154 / 770** after filtering).

**2) Stages 0–2** (no LLM required):

```bash
python scripts/build_statetree_curriculum.py \
  --input_jsonl data/locomo_qa/locomo_qa_train.jsonl \
  --output_dir data/curriculum \
  --stages warmup,basic2,basic3
```

| Stage | File | Content |
|-------|------|---------|
| Warm-up | `stage0_warmup.roll.jsonl` | Direct dialogue QA |
| Stage 1 | `stage1_basic_D2.roll.jsonl` | Basic tree, D=2 (4 leaves, 6 edges) |
| Stage 2 | `stage2_basic_D3.roll.jsonl` | Basic tree, D=3 (8 leaves, 14 edges) |

**3) Stage 3 — Compositional StateTree** (requires GPT decompositions):

```bash
python scripts/generate_decompositions.py \
  --input_jsonl data/locomo_qa/locomo_qa_train.jsonl \
  --output_json data/curriculum/decompositions.json \
  --model GPT

python scripts/build_statetree_curriculum.py \
  --input_jsonl data/locomo_qa/locomo_qa_train.jsonl \
  --output_dir data/curriculum \
  --stages compositional \
  --decompositions_json data/curriculum/decompositions.json
```

### Train with GRPO

Paper configs are under [`examples/roll/statetree/`](examples/roll/statetree/).

| Config | Model | Prompt batch |
|--------|-------|--------------|
| `grpo_qwen2.5_7b` | Qwen2.5-7B-Instruct | 64 |
| `grpo_qwen2.5_14b` | Qwen2.5-14B-Instruct | 32 (TP=2) |
| `grpo_qwen3_8b` | Qwen3-8B | 64 |
| `*_em_only` | same | small (EM-only smoke test) |

Shared hyperparameters: GRPO group size 8, LR `1e-6`, KL β `0.001`, temperature 0.6, top-p 0.95, response length 4096, context ~10K; reward `max(EM, LLM-Judge)` with Qwen2.5-1.5B-Instruct. Curriculum steps: **40 → 100 → 100 → 60**.

**Single stage:**

```bash
python examples/roll/start_rlvr_pipeline.py \
  --config_path examples/roll/statetree \
  --config_name grpo_qwen2.5_7b \
  max_steps=40 \
  actor_train.data_args.file_name=[data/curriculum/stage0_warmup.roll.jsonl]
```

**Full curriculum** (optional Stage-3 DAPO-Math mix):

```bash
export DAPO_MATH_JSONL=/path/to/dapo_math.jsonl   # optional

CONFIG=grpo_qwen2.5_7b  bash examples/roll/statetree/run_curriculum.sh
CONFIG=grpo_qwen2.5_14b bash examples/roll/statetree/run_curriculum.sh
CONFIG=grpo_qwen3_8b    bash examples/roll/statetree/run_curriculum.sh
```

### Evaluate

Validation and test sets remain **clean** direct QA (no StateTree overlay). Paper LoCoMo metrics: **ACC** (GPT judge), **token-F1**, and **BLEU-1**.

```bash
python scripts/export_roll_test.py \
  --input_jsonl data/locomo_qa/locomo_qa_test.jsonl \
  --output_jsonl data/locomo_qa/locomo_qa_test.roll.jsonl

python scripts/generate_predictions.py \
  --roll_test data/locomo_qa/locomo_qa_test.roll.jsonl \
  --out_pred outputs/predictions/locomo_pred.jsonl \
  --llm_mode local --local_model_path /path/to/ckpt

# F1 + BLEU-1
python scripts/evaluate.py \
  --roll_test data/locomo_qa/locomo_qa_test.roll.jsonl \
  --pred outputs/predictions/locomo_pred.jsonl \
  --dataset locomo

# + GPT ACC
python scripts/evaluate.py \
  --roll_test data/locomo_qa/locomo_qa_test.roll.jsonl \
  --pred outputs/predictions/locomo_pred.jsonl \
  --dataset locomo --judge_model GPT \
  --out_result outputs/predictions/locomo_metrics.json
```

**Key arguments for `evaluate.py`:**
- `--dataset`: `locomo`
- `--judge_model`: e.g. `GPT-4o` for paper ACC (omit to skip ACC)
- `--out_result` / `--out_details`: save summary JSON / per-example jsonl

## 📁 Repository Layout

```text
code/
├── statetree/          # StateTree construction, rewards, and data utilities
├── scripts/            # Curriculum build, decomposition, eval entrypoints
├── examples/           # Demo data and ROLL training configs
├── roll/               # ROLL RL training framework (vendored)
├── mcore_adapter/      # Megatron-Core adapter for training backends
├── docs/               # Training notes and prompt templates
└── tests/              # Unit tests
```

## 🎁 Acknowledgement

This work builds on several excellent open-source projects and benchmarks:

- **ROLL: Reinforcement Learning Optimization for Large-Scale Learning** — [Paper](https://arxiv.org/abs/2506.06122) | [GitHub](https://github.com/alibaba/ROLL)
- **[ICLR 2026 Oral] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts** — [Paper](https://arxiv.org/abs/2510.19363) | [GitHub](https://github.com/rStar-RL/LoongRL)
- **[ACL 2024] Evaluating Very Long-Term Conversational Memory of LLM Agents** — [Paper](https://arxiv.org/abs/2402.17753) | [GitHub](https://github.com/snap-research/locomo)
- **[ICLR 2025] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory** — [Paper](https://arxiv.org/abs/2410.10813) | [GitHub](https://github.com/xiaowu0162/longmemeval)
- **[COLM 2025] Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale** — [Paper](https://arxiv.org/abs/2504.14225) | [GitHub](https://github.com/bowen-upenn/PersonaMem)

We thank the authors for their valuable contributions to the community.

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{
anonymous2026statetree,
title={StateTree: Enhancing Long-term Dialogue Reasoning via Tree-structured RL Pseudo-tasks},
author={Anonymous},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=Wtmbj6bI9C}
}
```
