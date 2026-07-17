#!/usr/bin/env bash
# One-command reproduction for DraftForge acceptance bench.
#
# Assumes:
#   - target model downloaded (HuggingFace auth)
#   - trained draft head at $HEAD_DIR (default: ./checkpoints/head)
#   - vLLM ≥0.10.0 installed (pip install "vllm>=0.10.0")
#
# Outputs:
#   - results/eval/itl_baseline.json
#   - results/eval/itl_spec.json
#   - results/eval/acceptance_grid.csv
#   - results/itl_reduction.png
#   - results/acceptance_curves.png

set -euo pipefail

HEAD_DIR="${HEAD_DIR:-./checkpoints/head}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
RESULTS_DIR="${RESULTS_DIR:-./results/eval}"
# Rung-6 frugal default (max $4): sweep 1 4 16 first; extend to "1 4 8 16 32"
# only after the crossover region is bracketed. See docs/GPU_COST_OPTIMIZATION.md.
BATCH_SIZES="${BATCH_SIZES:-1 4 16}"
TEMPS="${TEMPS:-0.0 0.7 1.0}"
DOMAINS="${DOMAINS:-general finance}"

# --dry-run: print the exact bench plan without launching vLLM (costs $0).
if [[ "${1:-}" == "--dry-run" || "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
[bench --dry-run] no vLLM launched. Plan:
  target model : ${TARGET_MODEL}
  draft head   : ${HEAD_DIR}
  batch sizes  : ${BATCH_SIZES}
  temperatures : ${TEMPS}
  domains      : ${DOMAINS}
  results dir  : ${RESULTS_DIR}
  steps        : baseline serve :8000 → sweep → spec-decode serve :8001 → sweep
                 → plots → manifest
EOF
  exit 0
fi

mkdir -p "${RESULTS_DIR}"

echo "==> [1/4] Launch vLLM baseline (no speculation)"
vllm serve "${TARGET_MODEL}" --port 8000 &
BASELINE_PID=$!
trap 'kill "${BASELINE_PID}" 2>/dev/null || true' EXIT
sleep 90  # warmup; vLLM cold start ~60-90s on H100 with the 4B target

echo "==> [2/4] Sweep batch sizes for baseline ITL"
python -m eval.bench_client \
  --url http://127.0.0.1:8000 \
  --batch-sizes ${BATCH_SIZES} \
  --output "${RESULTS_DIR}/itl_baseline.json"

kill "${BASELINE_PID}" 2>/dev/null || true
wait "${BASELINE_PID}" 2>/dev/null || true

echo "==> [3/4] Launch vLLM with EAGLE-3 spec-decode"
SPEC_CONFIG=$(printf '{"method":"eagle3","model":"%s","num_speculative_tokens":4,"draft_model_config":{"method":"eagle3"}}' "${HEAD_DIR}")
vllm serve "${TARGET_MODEL}" --port 8001 \
  --speculative-config "${SPEC_CONFIG}" \
  --trust-remote-code &
SPEC_PID=$!
trap 'kill "${SPEC_PID}" 2>/dev/null || true' EXIT
sleep 120  # extra warmup for spec-decode

echo "==> [4/4] Sweep batch sizes for spec ITL + acceptance grid"
python -m eval.bench_client \
  --url http://127.0.0.1:8001 \
  --batch-sizes ${BATCH_SIZES} \
  --domains ${DOMAINS} \
  --temperatures ${TEMPS} \
  --output "${RESULTS_DIR}/itl_spec.json" \
  --acceptance-grid "${RESULTS_DIR}/acceptance_grid.csv"

kill "${SPEC_PID}" 2>/dev/null || true
wait "${SPEC_PID}" 2>/dev/null || true

echo "==> Rendering plots + manifest"
python -m eval.plot \
  --grid "${RESULTS_DIR}/acceptance_grid.csv" \
  --output-dir "${RESULTS_DIR}"

python -m release.aggregate \
  --results-root ./results \
  --out ./results/manifest.json

echo "==> Done. See ${RESULTS_DIR}/*.png and ./results/manifest.json"
