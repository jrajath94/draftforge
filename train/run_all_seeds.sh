#!/usr/bin/env bash
# DraftForge — multi-seed EAGLE-3 training orchestrator.
#
# Usage:
#   bash train/run_all_seeds.sh           # uses default seeds (42 0 1234)
#   bash train/run_all_seeds.sh 42 0      # custom seed list
#   SEEDS="42 0 1234 7" bash train/run_all_seeds.sh
#
# Each seed writes to results/train/<seed>/ (loss_curve.csv, checkpoints).
# Re-runs are idempotent — overwrite previously-trained weights for that seed.

set -euo pipefail

SEEDS="${SEEDS:-${*:-42 0 1234}}"
CONFIG="${CONFIG:-train/config.yaml}"
DS_CONFIG="${DS_CONFIG:-train/ds_config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/train}"

mkdir -p "${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  OUT="${OUTPUT_ROOT}/${seed}"
  mkdir -p "${OUT}"
  echo "=========================================="
  echo "[DraftForge] seed=${seed}  output=${OUT}"
  echo "=========================================="

  set +e
  accelerate launch \
    --num_processes "${NUM_GPUS:-1}" \
    --mixed_precision bf16 \
    --config_file "${DS_CONFIG}" \
    -m train.train_eagle3 \
    --config "${CONFIG}" \
    --seed "${seed}" \
    --output-dir "${OUT}" \
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
