#!/usr/bin/env bash
# HuggingFace upload wrapper for DraftForge trained heads.
#
# Wraps `huggingface-cli upload` with DraftForge-specific defaults and
# sanity checks. Refuses to upload placeholder content (per the integrity
# baseline: no fabricated artifacts in the public release).
#
# Usage:
#   bash scripts/upload_hf.sh \
#     --repo-id your-org/qwen3-4b-eagle3-finance \
#     --checkpoint-dir results/train/tri_layer/42/best \
#     --card-path HF_CARD.md
#
# Required env:
#   HF_TOKEN           write-scoped HF token (or already `huggingface-cli login`)
#
# Optional env:
#   HF_HUB_PRIVATE=1   create a private repo (default: public)
#   DRY_RUN=1          print the upload command but don't execute
#
# Exit codes:
#   0 — upload succeeded
#   1 — argument / preflight failure (no upload attempted)
#   2 — upload itself failed

set -euo pipefail

usage() {
  cat <<USAGE
usage: bash scripts/upload_hf.sh --repo-id ID --checkpoint-dir DIR --card-path FILE

required:
  --repo-id          HF repo id, e.g. your-org/qwen3-4b-eagle3-finance
  --checkpoint-dir   dir containing config.json + model.safetensors + training_config.yaml
  --card-path        path to the rendered HF card (markdown)

optional:
  --private          create the repo as private (HF_HUB_PRIVATE=1)
  --commit-message   git-style commit msg (default: "Upload DraftForge EAGLE-3 head")

env:
  HF_TOKEN          write-scoped HF token
  DRY_RUN=1         print commands only

USAGE
}

REPO_ID=""
CKPT_DIR=""
CARD_PATH=""
PRIVATE="${HF_HUB_PRIVATE:-0}"
COMMIT_MSG="Upload DraftForge EAGLE-3 head"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)         REPO_ID="$2"; shift 2 ;;
    --checkpoint-dir)  CKPT_DIR="$2"; shift 2 ;;
    --card-path)       CARD_PATH="$2"; shift 2 ;;
    --private)         PRIVATE=1; shift ;;
    --commit-message)  COMMIT_MSG="$2"; shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "ERROR: unknown arg '$1'" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${REPO_ID}" || -z "${CKPT_DIR}" || -z "${CARD_PATH}" ]]; then
  echo "ERROR: --repo-id, --checkpoint-dir, --card-path are all required" >&2
  usage
  exit 1
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "ERROR: checkpoint dir does not exist: ${CKPT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${CARD_PATH}" ]]; then
  echo "ERROR: card file does not exist: ${CARD_PATH}" >&2
  exit 1
fi

# Required files in checkpoint dir
for f in config.json model.safetensors training_config.yaml; do
  if [[ ! -f "${CKPT_DIR}/${f}" ]]; then
    echo "ERROR: missing required checkpoint file: ${CKPT_DIR}/${f}" >&2
    echo "ERROR: did training finish? see results/train/*/best/" >&2
    exit 1
  fi
done

# Refuse to upload a placeholder safetensors (per integrity baseline)
# The placeholder is < 1KB; real bf16 weights for a ~1B head are > 2GB.
ckpt_size_bytes=$(wc -c < "${CKPT_DIR}/model.safetensors" | tr -d ' ')
ckpt_size_mib=$(( ckpt_size_bytes / 1024 / 1024 ))
if [[ "${ckpt_size_mib}" -lt 1 ]]; then
  echo "ERROR: model.safetensors is only ${ckpt_size_mib} MiB — looks like a placeholder" >&2
  echo "ERROR: DraftForge integrity baseline refuses placeholder uploads" >&2
  echo "ERROR: train the head first (bash train/run_all_seeds.sh 3) then retry" >&2
  exit 1
fi

# Verify HF auth
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found. pip install huggingface-hub" >&2
  exit 1
fi

if ! huggingface-cli whoami >/dev/null 2>&1; then
  echo "ERROR: not authenticated. Run: huggingface-cli login" >&2
  exit 1
fi

# Build the upload command
PRIV_FLAG=""
if [[ "${PRIVATE}" == "1" ]]; then
  PRIV_FLAG="--private"
fi

UPLOAD_CMD="huggingface-cli upload ${REPO_ID} ${CKPT_DIR} --repo-type model ${PRIV_FLAG} --commit-message \"${COMMIT_MSG}\""
CARD_CMD="huggingface-cli upload ${REPO_ID} ${CARD_PATH} --repo-type model --commit-message \"Add model card: ${COMMIT_MSG}\""

echo "[upload] checkpoint dir: ${CKPT_DIR} (${ckpt_size_mib} MiB)"
echo "[upload] card file:      ${CARD_PATH}"
echo "[upload] repo id:        ${REPO_ID}"
echo "[upload] private:        ${PRIVATE}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[upload] DRY_RUN=1 — would execute:"
  echo "  ${UPLOAD_CMD}"
  echo "  ${CARD_CMD}"
  exit 0
fi

echo "[upload] step 1/2: uploading checkpoint files"
eval "${UPLOAD_CMD}"

echo "[upload] step 2/2: uploading model card"
eval "${CARD_CMD}"

echo
echo "[upload] DONE — view at: https://huggingface.co/${REPO_ID}"
