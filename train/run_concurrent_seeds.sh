#!/usr/bin/env bash
# DraftForge — concurrent multi-seed EAGLE-3 training orchestrator (v1.3).
#
# Spawns N seeds in parallel on a single multi-GPU host. Each seed is pinned to
# a distinct GPU via CUDA_VISIBLE_DEVICES so they share the device pool
# without contention. This is the v1.3 cost-reduction lever that turns three
# sequential ~1-hour runs into one ~1-hour wall-clock — H100 idle time falls
# from ~67% to ~0% on 3-seed variance estimation.
#
# Usage:
#   bash train/run_concurrent_seeds.sh 3 "0 1 2"
#   bash train/run_concurrent_seeds.sh 1 "0"
#   LOG_DIR=/tmp/logs N_SEEDS=2 bash train/run_concurrent_seeds.sh
#
# Env overrides:
#   LOG_DIR       — where per-seed logs land (default: ./logs)
#   OUTPUT_ROOT   — where each seed's results/ live (default: results/train)
#   CONFIG        — train YAML (default: train/config.yaml)
#   DS_CONFIG     — DeepSpeed config (default: train/ds_config.json)
#   DRAFTFORGE_STUB=1 — use inline stub driver (test mode)
#
# Test-mode contract: a script named `train_eagle3` on PATH is preferred
# over the inline stub. This lets failure-injection tests put a failing
# `train_eagle3` on PATH without touching the source tree.

set -euo pipefail

N_SEEDS="${1:-3}"
GPUS="${2:-0 1 2 3}"
LOG_DIR="${LOG_DIR:-logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/train}"
CONFIG="${CONFIG:-train/config.yaml}"
DS_CONFIG="${DS_CONFIG:-train/ds_config.json}"

# Default seed list; cycle through if N_SEEDS exceeds the list length.
DEFAULT_SEEDS=(42 123 456 789 1024 2048)

mkdir -p "${LOG_DIR}"

# Build per-seed work list (cycle through defaults).
seeds=()
gpu_list=(${GPUS})
for ((i = 0; i < N_SEEDS; i++)); do
    seeds+=("${DEFAULT_SEEDS[i % ${#DEFAULT_SEEDS[@]}]}")
done

# SIGTERM / SIGINT — propagate cleanly to children.
pids=()
logs=()
trap_cleanup() {
    echo "[DraftForge] signal received; killing ${#pids[@]} children" >&2
    for pid in "${pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 0
}
trap trap_cleanup SIGTERM SIGINT

# Launch seeds concurrently.
for i in "${!seeds[@]}"; do
    seed="${seeds[$i]}"
    gpu="${gpu_list[$((i % ${#gpu_list[@]}))]}"
    log="${LOG_DIR}/seed_${seed}_gpu${gpu}.log"
    logs+=("${log}")

    (
        export DRAFTFORGE_SEED="${seed}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export DRAFTFORGE_MAX="${DRAFTFORGE_MAX:-1000}"

        # Prefer a train_eagle3 binary on PATH (production or stub).
        # Fall back to inline stub only when DRAFTFORGE_STUB=1 is explicit.
        if command -v train_eagle3 >/dev/null 2>&1; then
            exec train_eagle3
        elif [ "${DRAFTFORGE_STUB:-0}" = "1" ]; then
            sleep 0.3
            echo "[stub] seed=${DRAFTFORGE_SEED} gpu=${CUDA_VISIBLE_DEVICES} step=0..max=${DRAFTFORGE_MAX}"
            exit 0
        else
            exec accelerate launch \
                --num_processes 1 \
                --mixed_precision bf16 \
                --config_file "${DS_CONFIG}" \
                -m train.train_eagle3 \
                --config "${CONFIG}" \
                --seed "${seed}" \
                --output-dir "${OUTPUT_ROOT}/${seed}"
        fi
    ) > "${log}" 2>&1 &

    pids+=($!)
    echo "[DraftForge] launched seed=${seed} gpu=${gpu} log=${log}"
done

# Wait + aggregate failures (don't abort on first — gather all errors).
fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        fail=1
    fi
done

if [ "${fail}" -ne 0 ]; then
    echo "[DraftForge] one or more seeds failed." >&2
    for log in "${logs[@]}"; do
        if [ -f "${log}" ]; then
            echo "[DraftForge] tail of ${log}:" >&2
            tail -n 20 "${log}" >&2 || true
        fi
    done
    exit 1
fi

echo "[DraftForge] all ${N_SEEDS} seeds done. Logs at ${LOG_DIR}/"