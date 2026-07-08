# release/

HuggingFace release artifacts for the trained DraftForge draft head.

## What's here

- `hf_card.md` — HuggingFace model card template (parameterized by `$HEAD_NAME`, `$TARGET_MODEL`, `$MANIFEST_JSON`)
- `make_card.py` — render the card from a results manifest
- `bench.sh` — one-command reproduction: serves target model in vLLM, sweeps batch/temp/domain, plots acceptance
- `writeup_template.md` — 1.5K-word analysis skeleton
- `aggregate.py` — collect JSON + CSV from `results/` into a single `manifest.json`

## Usage

### Render HF card

```bash
python -m release.make_card \
  --template release/hf_card.md \
  --results ./results \
  --head Qwen3-14B-EAGLE3-Finance \
  --target Qwen/Qwen3-14B \
  --out ./release/Qwen3-14B-EAGLE3-Finance.md
```

### Run one-command bench

```bash
TARGET_MODEL=Qwen/Qwen3-14B HEAD_DIR=./checkpoints/head bash release/bench.sh
```

Outputs land in `./results/eval/`:
- `itl_baseline.json` — non-speculative ITL per batch
- `itl_spec.json` — speculative ITL per batch
- `acceptance_grid.csv` — domain × temp × batch × acceptance
- `acceptance_curves.png`, `itl_reduction.png`

### Aggregate results

```bash
python -m release.aggregate --results-root ./results --out ./results/manifest.json
```

## Acceptance

- `bench.sh` produces all plots + JSON cited in `writeup_template.md`
- `aggregate.py` produces a `manifest.json` no larger than ~100KB (per-run CSV summaries)
- HF card includes seed list, training config pointer, and bench reproduction command

## Out of scope

- Auto-upload to HuggingFace Hub (use `huggingface-cli upload` manually after review)
- Multi-model support (single base model per card render)