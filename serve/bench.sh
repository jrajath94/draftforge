#!/usr/bin/env bash
set -euo pipefail

# DraftForge serve-stage contract runner.
#
# This stage is intentionally honest: it materializes runtime invocation
# scripts under results/serve/ but does NOT fabricate benchmark JSON when no
# live runtime has been executed yet. Downstream eval stages will emit an empty
# acceptance grid until real serve outputs exist.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="${RESULTS_ROOT:-${ROOT}/results}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DRAFT_HEAD="${DRAFT_HEAD:-${ROOT}/release/head.placeholder.safetensors}"
NUM_SPEC="${NUM_SPEC:-4}"

mkdir -p "${RESULTS}/serve/vllm" "${RESULTS}/serve/sglang"

python -m serve.integrate \
  --target "${TARGET_MODEL}" \
  --draft "${DRAFT_HEAD}" \
  --runtime vllm \
  --num-spec "${NUM_SPEC}" \
  --out "${RESULTS}/serve/vllm/invocation.sh"

python -m serve.integrate \
  --target "${TARGET_MODEL}" \
  --draft "${DRAFT_HEAD}" \
  --runtime sglang \
  --num-spec "${NUM_SPEC}" \
  --out "${RESULTS}/serve/sglang/invocation.sh"

echo "[serve.bench] wrote invocation scripts under ${RESULTS}/serve/"
echo "[serve.bench] no benchmark JSON emitted without a live runtime; downstream eval will stay empty until real serve outputs exist."
