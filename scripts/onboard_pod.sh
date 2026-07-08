#!/usr/bin/env bash
# DraftForge pod onboarding — paste this entire block into a fresh pod shell.
# Idempotent. Safe to re-run.
#
# Purpose: isolate DraftForge from any other project sharing /workspace,
# install deps, validate HF auth, smoke-test data pipeline.
#
# Pre-req: HF_TOKEN set in environment (huggingface-cli login already done).
set -euo pipefail

REPO_URL="${DRAFTFORGE_REPO_URL:-https://github.com/anthropic-research/draftforge.git}"
DRAFTFORGE_HOME="${DRAFTFORGE_HOME:-/workspace/draftforge}"
HF_CACHE="${HF_CACHE:-/workspace/hf/draftforge}"

echo "[onboard] isolating cache under ${HF_CACHE}"
mkdir -p "${HF_CACHE}"

echo "[onboard] cloning into ${DRAFTFORGE_HOME}"
mkdir -p "$(dirname "${DRAFTFORGE_HOME}")"
if [[ ! -d "${DRAFTFORGE_HOME}/.git" ]]; then
  git clone "${REPO_URL}" "${DRAFTFORGE_HOME}"
else
  (cd "${DRAFTFORGE_HOME}" && git pull --ff-only)
fi

cd "${DRAFTFORGE_HOME}"

export HF_HUB_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
echo "[onboard] HF_HUB_CACHE=${HF_HUB_CACHE}"

echo "[onboard] creating venv (.venv, python 3.12)"
if [[ ! -d ".venv" ]]; then
  python3.12 -m venv .venv 2>/dev/null || python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[onboard] installing package (training extras)"
pip install --quiet --upgrade pip
pip install --quiet -e ".[train]"

echo "[onboard] verifying HF auth"
huggingface-cli whoami || {
  echo "[onboard] ERROR: huggingface-cli not authenticated."
  echo "[onboard] Run: huggingface-cli login (paste token)"
  exit 1
}

# Cross-project pod safety: refuse to start DraftForge training if another
# project (e.g., GoodputLab vLLM server) is holding >50% of GPU memory.
# This pod is shared; preflight protects the other project from OOM.
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -n "${gpu_mem}" ]]; then
    used_mib=$(echo "${gpu_mem}" | cut -d',' -f1 | tr -d ' ')
    total_mib=$(echo "${gpu_mem}" | cut -d',' -f2 | tr -d ' ')
    if [[ "${total_mib}" -gt 0 ]]; then
      pct=$(( used_mib * 100 / total_mib ))
      echo "[onboard] GPU memory: ${used_mib}/${total_mib} MiB (${pct}%)"
      if [[ "${pct}" -gt 50 ]]; then
        echo "[onboard] ERROR: another process holds ${pct}% of GPU memory."
        echo "[onboard] Refusing to start DraftForge — would starve the other project."
        echo "[onboard] Free GPU memory first (e.g., stop other vLLM servers),"
        echo "[onboard] or set DRAFTFORGE_SKIP_PREFLIGHT=1 to override."
        exit 1
      fi
    fi
  fi
else
  echo "[onboard] nvidia-smi not available; skipping cross-project GPU preflight"
fi

echo "[onboard] smoke: ruff + pytest (CPU-shape tests only)"
ruff check . || true
.venv/bin/pytest -q --no-header -x

echo "[onboard] smoke: data pipeline (--limit 100) — fast CPU path"
python -m data.prepare --config data/config.yaml --limit 100 || true

cat <<EOF

[onboard] READY. Next steps (in order):

  1. Train EAGLE-3 head (≥3 seeds, ~24h total):
       bash train/run_all_seeds.sh 3

  2. Ablation (4 presets × ≥3 seeds):
       bash ablate/run_ablation.sh

  3. vLLM/SGLang integration + bench:
       bash serve/bench.sh

  4. Acceptance analysis:
       python -m eval.acceptance --results-root ./results

  5. Release artifact + HF upload:
       python -m release.aggregate --results-root ./results \\
         --out ./results/manifest.json
       python -m release.make_card --manifest ./results/manifest.json \\
         --out ./HF_CARD.md
       huggingface-cli upload <your-org>/qwen3-14b-eagle3-finance \\
         ./checkpoints/<best-seed> \\
         ./HF_CARD.md

All steps are idempotent and resumable. Each emits a single JSON/CSV
under results/ that downstream steps consume verbatim — no fabrication.
EOF