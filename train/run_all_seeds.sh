#!/usr/bin/env bash
# DraftForge — multi-seed EAGLE-3 training orchestrator.
#
# Usage:
#   bash train/run_all_seeds.sh           # uses default seeds (42 0 1234)
#   bash train/run_all_seeds.sh 42 0      # custom seed list
#   SEEDS="42 0 1234 7" bash train/run_all_seeds.sh
#
# Frugality env (see docs/GPU_COST_OPTIMIZATION.md ladder):
#   MAX_STEPS=N / SMOKE_STEPS=N — cap training.max_steps (read by train_eagle3)
#   RESUME=1                    — resume each seed from its latest checkpoint
#                                 (community-spot preemption recovery)
#   CONFIG=train/config_smoke.yaml — 50-step smoke config (rung 3, max $2)
#
# Each seed writes to results/train/<seed>/ (loss_curve.csv, checkpoints).
# Re-runs are idempotent — overwrite previously-trained weights for that seed.

set -euo pipefail

SEEDS="${SEEDS:-${*:-42 0 1234}}"
CONFIG="${CONFIG:-train/config.yaml}"
DS_CONFIG="${DS_CONFIG:-train/ds_config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/train}"
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

mkdir -p "${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  OUT="${OUTPUT_ROOT}/${seed}"
  mkdir -p "${OUT}"
  echo "=========================================="
  echo "[DraftForge] seed=${seed}  output=${OUT}"
  echo "=========================================="

  set +e
  # Plain single-process launch: train_eagle3.py is a self-contained torch
  # script (bf16 handled in-script; it does not construct an accelerate
  # Accelerator, so `accelerate launch --config_file <deepspeed.json>` both
  # mismatched config schemas and added nothing).
  "${PYTHON:-python}" \
    -m train.train_eagle3 \
    --config "${CONFIG}" \
    --seed "${seed}" \
    --output-dir "${OUT}" \
    ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} \
    2>&1 | tee "${OUT}/train.log"
  rc=$?
  set -e

  echo "[DraftForge] seed=${seed} exit=${rc}"

  if [ "${rc}" -ne 0 ]; then
    echo "[DraftForge] seed ${seed} failed; aborting." >&2
    exit "${rc}"
  fi
done

echo "[DraftForge] all seeds done. Outputs at ${OUTPUT_ROOT}/<seed>/"
