#!/usr/bin/env bash
# End-to-end pipeline runner for DraftForge.
#
# Chains: train → ablate → serve → analyze → release.
# Each stage emits JSON/CSV that the next stage consumes verbatim.
# Stages can be skipped via SKIP_TRAIN=1, SKIP_ABLATE=1, etc.
#
# Usage:
#   SMOKE=1 bash scripts/run_full_pipeline.sh         # rung 3: 50-step, 1 seed, max $2
#   APPROVE_GPU_SPEND=yes bash scripts/run_full_pipeline.sh        # full (rungs 5-6)
#   SKIP_TRAIN=1 SKIP_ABLATE=1 bash scripts/run_full_pipeline.sh   # analyze+release only
#
# GPU spend ladder (docs/GPU_COST_OPTIMIZATION.md) — climb in order:
#   1. CPU gate (M1 Max, $0)  2. pod boot ($0.50)  3. smoke ($2)
#   4. 1-seed ablation ($5)   5. 3-seed final ($15) 6. serve bench ($4)
#   7. optional Nsight ($1)   — optimized total target: $25
#
# Guards:
#   - Non-smoke GPU stages refuse to run without APPROVE_GPU_SPEND=yes.
#   - Final training refuses to run without a RunPod network volume cache
#     (RUNPOD_VOLUME_PATH) unless ALLOW_NO_VOLUME_CACHE=1 — redownloading the
#     target on every pod burns paid GPU-minutes.
#   - SMOKE=1 uses train/config_smoke.yaml, one seed, and skips ablate+serve.
#
# Idempotent: re-running a stage overwrites its outputs but never mutates
# committed CSVs/JSONs (results/ is gitignored).

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="${PWD}"
RESULTS="${RESULTS_ROOT:-${ROOT}/results}"

mkdir -p "${RESULTS}"

log() { echo "[pipeline $(date +%H:%M:%S)] $*"; }

# ── 0. Spend guards ───────────────────────────────────────────────────────────
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  log "SMOKE=1 — rung 3: 50-step single-seed smoke (max \$2). Ablate+serve skipped."
  CONFIG="${CONFIG:-train/config_smoke.yaml}"
  SEEDS="${SEEDS:-42}"
  SKIP_ABLATE="${SKIP_ABLATE:-1}"
  SKIP_SERVE="${SKIP_SERVE:-1}"
  export CONFIG
else
  gpu_stages=()
  [[ "${SKIP_TRAIN:-0}"  != "1" ]] && gpu_stages+=("train")
  [[ "${SKIP_ABLATE:-0}" != "1" ]] && gpu_stages+=("ablate")
  [[ "${SKIP_SERVE:-0}"  != "1" ]] && gpu_stages+=("serve")
  if [[ ${#gpu_stages[@]} -gt 0 && "${APPROVE_GPU_SPEND:-}" != "yes" ]]; then
    log "REFUSING to run GPU stages (${gpu_stages[*]}) without APPROVE_GPU_SPEND=yes."
    log "Climb the ladder first: SMOKE=1 bash scripts/run_full_pipeline.sh  (max \$2)"
    log "Then: APPROVE_GPU_SPEND=yes bash scripts/run_full_pipeline.sh     (rungs 4-6)"
    exit 3
  fi
  if [[ "${SKIP_TRAIN:-0}" != "1" && "${ALLOW_NO_VOLUME_CACHE:-0}" != "1" ]]; then
    if [[ -z "${RUNPOD_VOLUME_PATH:-}" || ! -d "${RUNPOD_VOLUME_PATH:-/nonexistent}" ]]; then
      log "REFUSING final training without a network volume cache."
      log "Set RUNPOD_VOLUME_PATH to the mounted volume (see scripts/onboard_pod.sh),"
      log "or override with ALLOW_NO_VOLUME_CACHE=1 (accepts redownload cost)."
      exit 3
    fi
  fi
fi

# ── 1. Training ───────────────────────────────────────────────────────────────
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if [[ "${SMOKE}" == "1" ]]; then
    log "stage 1: smoke training (1 seed × 50 steps)"
  else
    log "stage 1: training (≥3 seeds)"
  fi
  # SEEDS is a space-separated seed LIST (run_all_seeds.sh contract) — never
  # pass a seed COUNT positionally; it would be parsed as a single seed value.
  SEEDS="${SEEDS:-42 0 1234}" bash "${ROOT}/train/run_all_seeds.sh"
else
  log "stage 1: SKIPPED (SKIP_TRAIN=1)"
fi

# ── 2. Ablation ──────────────────────────────────────────────────────────────
if [[ "${SKIP_ABLATE:-0}" != "1" ]]; then
  log "stage 2: ablation (4 presets × ≥3 seeds)"
  bash "${ROOT}/ablate/run_ablation.sh"
  log "stage 2b: comparing variants → ${RESULTS}/ablation/comparison.json"
  python -m ablate.compare \
    --results-root "${RESULTS}/ablate" \
    --out "${RESULTS}/ablation/comparison.json"
else
  log "stage 2: SKIPPED (SKIP_ABLATE=1)"
fi

# ── 3. Serve + benchmark ─────────────────────────────────────────────────────
if [[ "${SKIP_SERVE:-0}" != "1" ]]; then
  log "stage 3: vLLM/SGLang integration + bench"
  bash "${ROOT}/serve/bench.sh"
else
  log "stage 3: SKIPPED (SKIP_SERVE=1)"
fi

# ── 4. Acceptance analysis ────────────────────────────────────────────────────
log "stage 4: acceptance analysis"
mkdir -p "${RESULTS}/eval"
python -m eval.acceptance \
  --results-root "${RESULTS}" \
  --out "${RESULTS}/eval/acceptance_grid.csv" || {
    log "WARN: eval.acceptance failed (likely no serve outputs yet); continuing"
  }

# ── 5. Release artifact + HF card ────────────────────────────────────────────
log "stage 5: aggregating release manifest"
python -m release.aggregate \
  --results-root "${RESULTS}" \
  --out "${RESULTS}/manifest.json"

log "stage 5b: rendering HF model card"
python -m release.make_card \
  --template "${ROOT}/release/hf_card.md" \
  --results "${RESULTS}" \
  --head "draftforge-eagle3-head" \
  --target "Qwen/Qwen3-4B-Instruct-2507" \
  --out "${ROOT}/HF_CARD.md"

cat <<EOF

[pipeline] DONE.

Outputs:
  ${RESULTS}/train/<variant>/<seed>/loss_curve.csv
  ${RESULTS}/ablation/comparison.{json,csv}
  ${RESULTS}/eval/acceptance_grid.csv
  ${RESULTS}/manifest.json
  ${ROOT}/HF_CARD.md

Final: huggingface-cli upload <your-org>/qwen3-4b-eagle3-finance \\
  ${ROOT}/checkpoints/<best-seed> \\
  ${ROOT}/HF_CARD.md
EOF
