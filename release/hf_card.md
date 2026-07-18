---
license: mit
base_model: $TARGET_MODEL
tags:
  - eagle-3
  - speculative-decoding
  - qwen3
  - finance
  - draftforge
---

# $HEAD_NAME

EAGLE-3 speculative decoding draft head trained on $TARGET_MODEL with finance-domain emphasis.

## Intended Use

Drop-in draft head for `vllm serve $TARGET_MODEL --speculative-config '{"method":"eagle3",...}'` and SGLang's `--speculative-algorithm EAGLE3 --speculative-draft-model-path <this-repo>`.

## Training

- Recipe: EAGLE-3 (NeurIPS'25, Li et al.)
- Tri-layer fusion: hidden states from layers [7, 18, 29] of $TARGET_MODEL (low/mid/high; rescaled from Qwen3-14B's [8, 20, 32] for 36 vs 40 layers)
- Direct token prediction (not feature-level)
- Training-time-test with horizon 4
- Single-process PyTorch, bf16 (frozen 4B target + trainable head fit on one 80GB GPU without ZeRO/offload)
- Seed protocol: 42, 0, 1234 (≥3 independent runs required before release)

## Results

$RESULTS_SECTION

## Bench

```bash
bash bench.sh
```

One-command reproduction. See `results/` for committed per-seed loss curves and acceptance-grid sweeps.

## Limitations

- Draft head is calibrated to $TARGET_MODEL's hidden-state geometry; do not load against a different base model.
- Acceptance rates degrade on out-of-distribution prompts (see domain-shift analysis in writeup).
- Training targets a single hardware class (H100 bf16). Numerical behavior on other accelerators may differ.

## Citation

```bibtex
@misc{draftforge2026,
  title={DraftForge: Training EAGLE-3 Draft Heads for Domain-Targeted Speculative Decoding},
  author={Bosco, Rajath John},
  year={2026},
  url={https://github.com/rajath/draftforge}
}
```