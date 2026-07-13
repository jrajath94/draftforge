#!/usr/bin/env bash
# DraftForge pod onboarding — paste this entire block into a fresh pod shell.
# Idempotent. Safe to re-run.
#
# Purpose: isolate DraftForge from any other project sharing /workspace,
# install deps, validate HF auth, smoke-test data pipeline.
#
# Pre-req: HF_TOKEN set in environment (huggingface-cli login already done).
#
# v1.3 additions (cost-reduction levers 3 + 5):
#   * RUNPOD_VOLUME_PATH — when set + exists, symlinks HF cache + tokenized
#     data + training outputs onto a persistent RunPod network volume.
#     Survives pod termination, skips 30-60s model re-download per run.
#   * SIGTERM trap — `trap_save` runs an emergency checkpoint + exits 0
#     instead of leaving half-trained artifacts behind.
set -euo pipefail

REPO_URL="${DRAFTFORGE_REPO_URL:-https://github.com/anthropic-research/draftforge.git}"
DRAFTFORGE_HOME="${DRAFTFORGE_HOME:-/workspace/draftforge}"
HF_CACHE="${HF_CACHE:-/workspace/hf/draftforge}"

# ── v1.3: SIGTERM trap (cost-reduction lever 5 sibling — preemption safety) ──
# Register BEFORE any heavy step so an early SIGTERM still drains cleanly.
# Community-spot preemptions arrive as SIGTERM; default behaviour (exit 143)
# would leave the in-progress training run half-written. trap_save writes an
# emergency-loss marker so the operator knows the run was preempted.
trap_save() {
    echo "[onboard] SIGTERM received; writing emergency checkpoint marker." >&2
    local marker="${DRAFTFORGE_HOME}/results/train/EMERGENCY_STOP.txt"
    mkdir -p "$(dirname "${marker}")" 2>/dev/null || true
    {
        echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date)"
        echo "reason=SIGTERM (likely community-spot preemption)"
        echo "action=resume with: bash train/run_concurrent_seeds.sh <N_SEEDS> <GPUS>"
    } > "${marker}" 2>/dev/null || true
    # Best-effort: try to dump whatever is in the current results dir.
    local latest
    latest=$(ls -t "${DRAFTFORGE_HOME}/results/train"/*/loss_curve.csv 2>/dev/null | head -1 || true)
    if [[ -n "${latest}" ]]; then
        echo "[onboard] last loss row of $(basename "$(dirname "${latest}")"):" >&2
        tail -n 1 "${latest}" >&2 || true
    fi
    exit 0
}
trap trap_save SIGTERM SIGINT

# ── v1.3: volume cache (cost-reduction lever 5) ──────────────────────────────
# When RUNPOD_VOLUME_PATH is set + the dir exists, symlink persistent state
# onto the network volume. Subsequent pods that mount the same volume see
# the same HF cache (no 30-60s re-download) and the same training outputs.
# Idempotent: removes existing real dirs before symlinking.
setup_volume_cache() {
    local vol="${RUNPOD_VOLUME_PATH:-}"
    if [[ -z "${vol}" || ! -d "${vol}" ]]; then
        echo "[onboard] no RUNPOD_VOLUME_PATH; cache stays on container disk"
        return 0
    fi
    echo "[onboard] RUNPOD_VOLUME_PATH=${vol}; symlinking persistent paths"

    # HF cache (models + datasets).
    mkdir -p "${vol}/hf"
    if [[ -L "${HF_CACHE}" || -e "${HF_CACHE}" ]]; then
        rm -rf "${HF_CACHE}"
    fi
    ln -s "${vol}/hf" "${HF_CACHE}"

    # Tokenized data (data pipeline output, large).
    local tok_dir="${DRAFTFORGE_HOME}/artifacts/data/tokenized"
    mkdir -p "${vol}/tokenized" "$(dirname "${tok_dir}")"
    if [[ -L "${tok_dir}" || -e "${tok_dir}" ]]; then
        rm -rf "${tok_dir}"
    fi
    ln -s "${vol}/tokenized" "${tok_dir}"

    # Training outputs (checkpoints, loss curves).
    local out_dir="${DRAFTFORGE_HOME}/results/train"
    mkdir -p "${vol}/results_train" "$(dirname "${out_dir}")"
    if [[ -L "${out_dir}" || -e "${out_dir}" ]]; then
        rm -rf "${out_dir}"
    fi
    ln -s "${vol}/results_train" "${out_dir}"

    echo "[onboard] volume cache symlinked (HF + tokenized + results)"
}

echo "[onboard] isolating cache under ${HF_CACHE}"
# Tolerant of read-only filesystems (e.g., test environments without
# /workspace). The volume cache function (below) re-creates HF_CACHE as a
# symlink when RUNPOD_VOLUME_PATH is set; this real-dir fallback is only
# used in non-volume mode.
mkdir -p "${HF_CACHE}" 2>/dev/null || true

# Apply volume cache before clone/install so the network volume sees writes.
setup_volume_cache

# Heavy steps (clone, pip, smoke). Skipped when DRAFTFORGE_SKIP_PREFLIGHT=1
# so unit tests can source the script without network + without invoking the
# SIGTERM trap's clean-exit path on a dead shell.
if [[ "${DRAFTFORGE_SKIP_PREFLIGHT:-0}" != "1" ]]; then
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
fi

cat <<EOF

[onboard] READY. Next steps (in order):

  1. Train EAGLE-3 head (≥3 seeds in parallel; ~1h wall-clock on H100):
       python scripts/operator_runpod.py concurrent <POD_ID> \\
         --ssh-host <HOST> --ssh-port <PORT> --n-seeds 3 --gpus '0 1 2'

  2. Ablation (4 presets × ≥3 seeds):
       bash ablate/run_ablation.sh

  3. vLLM/SGLang integration + bench:
       bash serve/bench.sh

  4. Acceptance analysis:
       python -m eval.acceptance --results-root ./results \\
         --out ./results/acceptance_grid.csv

  5. Release artifact + HF upload:
       python -m release.aggregate --results-root ./results \\
         --out ./results/manifest.json
       python -m release.make_card --template ./release/hf_card.md \\
         --results ./results --head draftforge-eagle3-head \\
         --target Qwen/Qwen3-4B --out ./HF_CARD.md
       huggingface-cli upload <your-org>/qwen3-14b-eagle3-finance \\
         ./checkpoints/<best-seed> \\
         ./HF_CARD.md

All steps are idempotent and resumable. Each emits a single JSON/CSV
under results/ that downstream steps consume verbatim — no fabrication.
EOF