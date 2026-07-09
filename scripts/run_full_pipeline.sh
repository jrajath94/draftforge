#!/usr/bin/env bash
# End-to-end pipeline runner for DraftForge.
#
# Chains: train → ablate → serve → analyze → release.
# Each stage emits JSON/CSV that the next stage consumes verbatim.
# Stages can be skipped via SKIP_TRAIN=1, SKIP_ABLATE=1, etc.
#
# Usage:
#   bash scripts/run_full_pipeline.sh                 # full
#   SKIP_TRAIN=1 SKIP_ABLATE=1 bash scripts/run_full_pipeline.sh   # analyze+release only
#
# Idempotent: re-running a stage overwrites its outputs but never mutates
# committed CSVs/JSONs (results/ is gitignored).

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="${PWD}"
RESULTS="${RESULTS_ROOT:-${ROOT}/results}"

mkdir -p "${RESULTS}"

log() { echo "[pipeline $(date +%H:%M:%S)] $*"; }

# ── 1. Training ───────────────────────────────────────────────────────────────
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  log "stage 1: training (≥3 seeds, ~24h on H100)"
  bash "${ROOT}/train/run_all_seeds.sh" "${N_SEEDS:-3}"
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
python -m eval.acceptance \
  --results-root "${RESULTS}" \
  --out "${RESULTS}/acceptance_grid.csv" || {
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
  --target "Qwen/Qwen3-4B" \
  --out "${ROOT}/HF_CARD.md"

cat <<EOF

[pipeline] DONE.

Outputs:
  ${RESULTS}/train/<variant>/<seed>/loss_curve.csv
  ${RESULTS}/ablation/comparison.{json,csv}
  ${RESULTS}/acceptance_grid.csv
  ${RESULTS}/manifest.json
  ${ROOT}/HF_CARD.md

Final: huggingface-cli upload <your-org>/qwen3-14b-eagle3-finance \\
  ${ROOT}/checkpoints/<best-seed> \\
  ${ROOT}/HF_CARD.md
EOF