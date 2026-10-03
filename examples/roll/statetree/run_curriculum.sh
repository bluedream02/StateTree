#!/usr/bin/env bash
# Run StateTree four-stage curriculum with ROLL RLVR / GRPO.
#
# Prerequisites:
#   1. Build curriculum jsonl (see README)
#   2. pip install -e . && install ROLL deps (requirements_roll_*.txt)
#   3. Launch from repo root `code/`
#
# Usage:
#   bash examples/roll/statetree/run_curriculum.sh
#   CONFIG=grpo_qwen2.5_14b bash examples/roll/statetree/run_curriculum.sh
#   CONFIG=grpo_qwen3_8b bash examples/roll/statetree/run_curriculum.sh
#   DATA_DIR=data/curriculum CONFIG=grpo_qwen2.5_7b bash examples/roll/statetree/run_curriculum.sh
#
# Available CONFIG names (see README.md in this folder):
#   grpo_qwen2.5_7b | grpo_qwen2.5_14b | grpo_qwen3_8b
#   (+ optional *_em_only variants)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-data/curriculum}"
CONFIG_PATH="${CONFIG_PATH:-examples/roll/statetree}"
CONFIG="${CONFIG:-grpo_qwen2.5_7b}"
CKPT_ROOT="${CKPT_ROOT:-outputs/roll/checkpoints}"

case "${CONFIG}" in
  grpo_qwen2.5_7b|grpo_qwen2.5_14b|grpo_qwen3_8b|grpo_qwen2.5_7b_em_only|grpo_qwen2.5_14b_em_only|grpo_qwen3_8b_em_only) ;;
  *)
    echo "Unknown CONFIG=${CONFIG}. Use grpo_qwen2.5_7b | grpo_qwen2.5_14b | grpo_qwen3_8b (or *_em_only)." >&2
    exit 1
    ;;
esac

echo "Using CONFIG=${CONFIG}"

# Paper curriculum steps
STAGE0_STEPS="${STAGE0_STEPS:-40}"   # warmup
STAGE1_STEPS="${STAGE1_STEPS:-100}"  # basic D=2
STAGE2_STEPS="${STAGE2_STEPS:-100}"  # basic D=3
STAGE3_STEPS="${STAGE3_STEPS:-60}"   # compositional D=3

run_stage () {
  local name="$1"
  local data="$2"
  local steps="$3"
  local resume="$4"

  echo "========== StateTree curriculum: ${name} (steps=${steps}) =========="
  if [[ ! -f "${data}" ]]; then
    echo "ERROR: missing dataset ${data}" >&2
    exit 1
  fi

  local extra=()
  if [[ -n "${resume}" && "${resume}" != "false" ]]; then
    extra+=(resume_from_checkpoint="${resume}")
  fi

  python examples/roll/start_rlvr_pipeline.py \
    --config_path "${CONFIG_PATH}" \
    --config_name "${CONFIG}" \
    exp_name="statetree-${name}" \
    max_steps="${steps}" \
    "actor_train.data_args.file_name=[${data}]" \
    "checkpoint_config.output_dir=${CKPT_ROOT}/statetree-${name}" \
    "${extra[@]}"
}

run_stage "stage0_warmup" \
  "${DATA_DIR}/stage0_warmup.roll.jsonl" \
  "${STAGE0_STEPS}" \
  "false"

run_stage "stage1_basic_D2" \
  "${DATA_DIR}/stage1_basic_D2.roll.jsonl" \
  "${STAGE1_STEPS}" \
  "${CKPT_ROOT}/statetree-stage0_warmup"

run_stage "stage2_basic_D3" \
  "${DATA_DIR}/stage2_basic_D3.roll.jsonl" \
  "${STAGE2_STEPS}" \
  "${CKPT_ROOT}/statetree-stage1_basic_D2"

# Paper: mix ~2500 DAPO-Math samples into Stage 3 to preserve short-context reasoning.
STAGE3_DATA="${DATA_DIR}/stage3_compositional_D3.roll.jsonl"
if [[ -n "${DAPO_MATH_JSONL:-}" && -f "${DAPO_MATH_JSONL}" ]]; then
  MIXED="${DATA_DIR}/stage3_compositional_D3_with_dapo.roll.jsonl"
  python - <<PY
import json
from pathlib import Path
st = Path("${STAGE3_DATA}")
dapo = Path("${DAPO_MATH_JSONL}")
out = Path("${MIXED}")
rows = [json.loads(l) for l in st.read_text().splitlines() if l.strip()]
extra = [json.loads(l) for l in dapo.read_text().splitlines() if l.strip()][:2500]
# Ensure DAPO rows carry a tag routed by rewards (or share locomo/statetree domain).
for r in extra:
    r.setdefault("tag", "locomo")
with out.open("w") as f:
    for r in rows + extra:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"mixed stage3: {len(rows)} StateTree + {len(extra)} DAPO -> {out}")
PY
  STAGE3_DATA="${MIXED}"
else
  echo "[warn] DAPO_MATH_JSONL unset or missing; Stage 3 runs without DAPO-Math mix (paper uses 2500)."
fi

run_stage "stage3_compositional_D3" \
  "${STAGE3_DATA}" \
  "${STAGE3_STEPS}" \
  "${CKPT_ROOT}/statetree-stage2_basic_D3"

echo "Curriculum finished."
