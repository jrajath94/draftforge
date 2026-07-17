# ablate/ — EAGLE-3 architecture ablation

Varies one design choice at a time and reports the effect on training loss.

## Variants

| Variant | layer_indices | What it tests |
|---------|---------------|---------------|
| `tri_layer` | `[7, 18, 29]` | EAGLE-3 reference rescaled to the 36-layer 4B target |
| `final_layer` | `[35]` | single last-layer tap baseline |
| `low_only` | `[7]` | early-layer features only |
| `mid_only` | `[18]` | mid-layer features only |

Each variant trains with the same seed list (default `42 0 1234`),
the same data, the same hyperparameters — only `eagle3.layer_indices` differs.

## Quickstart

```bash
# Pre-req: data/ + train/ Phase 1+2 done. GPU rented.
bash ablate/run_ablation.sh                            # all 4 variants
bash ablate/run_ablation.sh tri_layer final_layer      # 2 variants
```

Writes:
- `results/train/<variant>/<seed>/loss_curve.csv`
- `results/ablation/comparison.json` & `.csv` — final-100-step mean ± std

## Acceptance

- ≥3 seeds per variant
- Same data, same hyperparameters across variants
- Final loss reported as `mean ± std` over seeds (variance, not just point estimate)
- Both `tri_layer` winner and any tie/interesting-negative-result documented
  in `docs/ablation_findings.md` (Phase 6 deliverable)
