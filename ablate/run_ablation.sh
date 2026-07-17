#!/usr/bin/env bash
# DraftForge — ablation runner.
#
# Generates per-variant TrainConfig YAMLs, runs train/run_all_seeds.sh on
# each, then aggregates loss curves into a comparison table.
#
# Usage:
#   bash ablate/run_ablation.sh                       # all 4 presets, 3 seeds each
#   bash ablate/run_ablation.sh tri_layer final_layer # subset, default 3 seeds
#   SEEDS="42 0" bash ablate/run_ablation.sh tri_layer
#
# Frugal rung-4 invocation (one seed, two variants, max $5 — see
# docs/GPU_COST_OPTIMIZATION.md):
#   ABLATE_VARIANTS="tri_layer final_layer" SEEDS="42" bash ablate/run_ablation.sh
#
# Precedence for variant list: ABLATE_VARIANTS env > positional args > all 4.
#
# Pre-req: train/run_all_seeds.sh must work (rented GPU + HF auth).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Args / env
VARIANTS="${ABLATE_VARIANTS:-${*:-tri_layer final_layer low_only mid_only}}"
SEEDS_ENV="${SEEDS:-42 0 1234}"
BASE_CONFIG="${BASE_CONFIG:-train/config.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-results/train}"
COMPARE_OUT="${COMPARE_OUT:-results/ablation/comparison.json}"

mkdir -p "$(dirname "$COMPARE_OUT")" "$RESULTS_ROOT"

run_variant() {
  local variant="$1"
  local cfg="artifacts/ablation/${variant}.yaml"
  echo "=== ablate: variant=${variant} ==="
  .venv/bin/python -m ablate.configs \
    --base "$BASE_CONFIG" \
    --variant "$variant" \
    --out "$cfg" || true
  # NOTE: ablate.configs doesn't expose CLI yet; the script is a stub.
  # Use the Python API directly via -c.
  .venv/bin/python -c "
from pathlib import Path
from ablate.configs import PRESETS, load_yaml, write_variant_config
base = load_yaml('$BASE_CONFIG')
write_variant_config(base, PRESETS['$variant'], Path('$cfg'))
print('wrote $cfg')
"

  SEEDS="$SEEDS_ENV" \
  NUM_GPUS="${NUM_GPUS:-1}" \
    OUTPUT_ROOT="${RESULTS_ROOT}/${variant}" \
    CONFIG="$cfg" \
    bash train/run_all_seeds.sh \
    || { echo "[ablate] variant $variant failed; aborting"; return 1; }
}

for v in $VARIANTS; do
  run_variant "$v"
done

.venv/bin/python -c "
from pathlib import Path
from ablate.compare import compare_variants, write_comparison
by_v = compare_variants(Path('${RESULTS_ROOT}'))
write_comparison(by_v, Path('${COMPARE_OUT}'))
print('wrote ${COMPARE_OUT}')
"
echo "[ablate] done"
