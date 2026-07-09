#!/usr/bin/env bash
# DraftForge CLI smoke verification.
#
# Walks every CLI entrypoint in the project and asserts that
# `python -m <module> --help` exits 0. Cheaper than a full subprocess
# suite; proves the argparse/typer binding for each module is intact
# and the module is importable.
#
# Modules that are pure libraries (no `__main__` block) are skipped
# with a comment. The intent is "the README's quickstart commands
# would actually run, modulo the optional --config flag".
#
# Usage:
#   bash scripts/verify.sh                 # verify all CLIs
#   QUICK=1 bash scripts/verify.sh          # skip slow ones (e.g. data.prepare)
#
# Exit codes:
#   0 — every CLI exited 0
#   1 — at least one CLI failed (stderr is printed for the offender)

set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="${PWD}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

GREEN=$'\033[32m'
RED=$'\033[31m'
YELLOW=$'\033[33m'
RESET=$'\033[0m'

pass=0
fail=0
skipped=0

check() {
  local label="$1"
  local cmd="$2"
  echo
  echo "[verify] ${label}"
  echo "[verify]   \$ ${cmd}"
  if eval "${cmd}" >/tmp/verify_out 2>/tmp/verify_err; then
    echo "${GREEN}[verify] OK${RESET} — exit 0"
    pass=$((pass + 1))
  else
    rc=$?
    echo "${RED}[verify] FAIL${RESET} — exit ${rc}"
    echo "--- stderr (last 30 lines) ---"
    tail -30 /tmp/verify_err
    echo "------------------------------"
    fail=$((fail + 1))
  fi
}

skip() {
  local label="$1"
  local reason="$2"
  echo
  echo "[verify] ${label} — ${YELLOW}SKIPPED${RESET} (${reason})"
  skipped=$((skipped + 1))
}

# ── Data pipeline (typer CLI) ──────────────────────────────────────────────
check "data.prepare (typer)" \
  "${PYTHON} -m data.prepare --help"

# ── Training driver (argparse) ─────────────────────────────────────────────
check "train.train_eagle3" \
  "${PYTHON} -m train.train_eagle3 --help"

# ── Serve invocation builders (argparse) ───────────────────────────────────
check "serve.integrate (vllm invocation builder)" \
  "${PYTHON} -m serve.integrate --help"

skip "serve.bench" "library only (no __main__); invoked via run_full_pipeline.sh"

# ── Ablation (argparse) ────────────────────────────────────────────────────
check "ablate.compare" \
  "${PYTHON} -m ablate.compare --help"

# ── Evaluation (argparse) ──────────────────────────────────────────────────
check "eval.acceptance" \
  "${PYTHON} -m eval.acceptance --help"

check "eval.crossover_analysis" \
  "${PYTHON} -m eval.crossover_analysis --help"

# ── Release (argparse + typer multi-command) ───────────────────────────────
check "release.aggregate" \
  "${PYTHON} -m release.aggregate --help"

check "release.make_card" \
  "${PYTHON} -m release.make_card --help"

check "release (typer aggregate subcommand)" \
  "${PYTHON} -m release aggregate --help"

check "release (typer make-card subcommand)" \
  "${PYTHON} -m release make-card --help"

# ── RunPod operator (argparse, scripts/operator_runpod.py) ─────────────────
check "operator_runpod (RunPod GPU operator)" \
  "${PYTHON} scripts/operator_runpod.py --help"

check "operator_runpod one-liner" \
  "${PYTHON} scripts/operator_runpod.py one-liner"

# ── Summary ────────────────────────────────────────────────────────────────
echo
echo "[verify] ============================================"
echo "[verify] passed: ${pass}, failed: ${fail}, skipped: ${skipped}"
echo "[verify] ============================================"

if [[ "${fail}" -gt 0 ]]; then
  echo "${RED}[verify] ONE OR MORE CLIs FAILED — see output above.${RESET}"
  exit 1
fi
echo "${GREEN}[verify] all CLIs parse + bind. safe to commit / tag.${RESET}"
