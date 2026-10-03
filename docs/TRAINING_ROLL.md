"""
StateTree + ROLL training notes
================================

This repository vendors the core of Alibaba ROLL (Apache-2.0) under `roll/`
so StateTree data construction and GRPO training live in one place.

## Layout

```
code/
├── statetree/                 # paper data construction + metrics
├── roll/                      # ROLL training framework (vendored core)
├── mcore_adapter/             # Megatron-Core adapter (ROLL dependency)
├── examples/roll/
│   ├── start_rlvr_pipeline.py
│   └── statetree/             # ★ StateTree GRPO yamls + curriculum runner
├── docs/
└── requirements_roll_*.txt
```

Hydra configs under `examples/roll/statetree/` are the only training configs.
Upstream ROLL demos and local run artifacts live in sibling `code_other/`
(see that folder’s README).

## Install ROLL deps

Pick a torch / inference backend matching your cluster, e.g.:

```bash
cd code
pip install -e .   # installs roll + statetree packages (see setup.py)
pip install -r requirements_roll_common.txt
pip install -r requirements_roll_torch280_vllm.txt
# optional older stack: ../code_other/packaging/requirements_roll_torch260_vllm.txt
# mcore_adapter is pulled via requirements_common as ./mcore_adapter
```

Official ROLL docs: https://alibaba.github.io/ROLL/

## Build StateTree curriculum data

```bash
python scripts/prepare_locomo.py --input /path/to/locomo10.json --output_dir data/locomo_qa
python scripts/build_statetree_curriculum.py \
  --input_jsonl data/locomo_qa/locomo_qa_train.jsonl \
  --output_dir data/curriculum \
  --stages warmup,basic2,basic3
# Stage 3 needs decompositions.json (see README)
```

Ensure each ROLL jsonl row has `tag: locomo` (default from our exporter) so the
`statetree` reward domain routes correctly (`tag_included: [locomo, statetree]`).

## Launch one stage

```bash
python examples/roll/start_rlvr_pipeline.py \
  --config_path examples/roll/statetree \
  --config_name grpo_qwen2.5_7b \
  max_steps=40 \
  actor_train.data_args.file_name=[data/curriculum/stage0_warmup.roll.jsonl]
```

EM-only smoke (no judge GPUs):

```bash
python examples/roll/start_rlvr_pipeline.py \
  --config_path examples/roll/statetree \
  --config_name grpo_qwen2.5_7b_em_only \
  max_steps=5
```

## Full curriculum

```bash
bash examples/roll/statetree/run_curriculum.sh
```

Paper schedule: Warm-up 40 → Stage1 100 → Stage2 100 → Stage3 60 steps.
Group size 8, LR 1e-6, KL β 0.001, T=0.6, top-p=0.95, response 4096.

## Reward worker

`roll.pipeline.rlvr.rewards.statetree_combined_reward_worker.StateTreeCombinedRewardWorker`
implements `r = max(r_EM, r_LLM)` with the LoCoMo CORRECT/WRONG judge prompt
registered as `StateTree-LoCoMo-judge` in `roll/utils/prompt.py`.

## License note

- StateTree code in this repo: MIT (`LICENSE`)
- Vendored ROLL + mcore_adapter: Apache-2.0 (`LICENSE_ROLL`)
"""
