# Data & training notes

## Recommended local directories (gitignored)

```
data/
├── raw/                 # locomo10.json, etc. (you download)
├── locomo_qa/           # from scripts/prepare_locomo.py
└── curriculum/          # from scripts/build_statetree_curriculum.py
outputs/
├── predictions/
└── roll/                # checkpoints / tensorboard (training)
```

## Expected external datasets

| Dataset | Role | Where to get it |
|---------|------|-----------------|
| LoCoMo `locomo10.json` | Train / ID eval | https://github.com/snap-research/locomo |
| LongMemEval | OOD 128k eval | https://github.com/xiaowu0162/LongMemEval |
| PersonaMem | OOD eval | https://github.com/bowen-upenn/PersonaMem |
| DAPO-Math (~2500) | Stage-3 mix-in | https://huggingface.co/datasets (DAPO) |

Raw corpora are **not** redistributed in this repo.

## Curriculum sizes (paper)

- 616 train / 154 val / 770 test dialogue–QA pairs from LoCoMo after filtering
- Same 616 samples reused across Warm-up and Stages 1–3; only the tree formulation changes
- Val / test stay in clean direct-QA format (no StateTree records)

## Output formats

**ROLL jsonl** fields: `id`, `prompt`, `messages`, `ground_truth`, `question`, `category`, `source`, `tag`, …
