# DraftForge

Measured EAGLE-3 speculative-decoding draft-head training and evaluation.
Trains draft heads for target model + domain pairs that lack one, measures
acceptance honestly, and says plainly where nothing is measured.

**Status:** the spend-gated training pipeline and the three-seed training run
are complete. **Serving-stack inter-token latency (ITL) is not yet measured.**
The blocker is concrete: the vLLM EAGLE-3 weight-schema adapter is not
written. Everything below is training and acceptance evidence only.

## What it is

Speculative decoding cuts inter-token latency with a draft-and-verify loop: a
small draft head proposes several tokens, the target model verifies them in
one forward pass, and accepted tokens skip full forward passes. EAGLE-3 trains
a draft head that taps hidden states at three depths of the target model
(low, mid, high layers) instead of only the final layer.

The pipeline covers, end to end:

1. **Data pipeline** - dedupe (exact + MinHash), stratified split, tokenize.
2. **Training** - frozen target model (Qwen/Qwen3-4B-Instruct-2507), draft
   head trained for 2000 steps per seed, 3 seeds, on finance-domain traces.
   Sequence packing recovers 3-7x throughput when median doc length is far
   below max_len.
3. **Ablation** - tri-layer vs final-layer fusion measured on the pod (200
   steps, 1 seed). Low-only and mid-only variants are scripted in
   `ablate/configs.py` but have not run yet.
4. **Integration** - vLLM `--speculative-config eagle3` and SGLang
   `--speculative-algorithm` invocation paths, plus a profile stage.
5. **Acceptance analysis** - code for a grid over domain x temperature x
   batch size, plus a crossover analysis that finds where speculation stops
   paying. The committed grid is still empty, so the crossover stays
   [NOT YET MEASURED].
6. **Release** - aggregates every measured number into a manifest, renders a
   HuggingFace model card from it.

## Measured results (self-reported: hardware and seeds named, not
independently reproduced)

**Training + acceptance (2026-07-18, RunPod A100-SXM4 80GB, seeds {42, 0,
1234}, ~$14 total).**

| Metric | Value | Source |
| --- | --- | --- |
| Draft/target greedy agreement (T=0, held-out val), 3 seeds | 0.687 +/- 0.010 (0.694 / 0.676 / 0.692) | `results/eval/acceptance_measured_*.json` |
| Expected acceptance length (geometric, from agreement) | ~3.2 tokens | `results/eval/acceptance_measured_*.json` |
| Ablation probe: tri-layer vs final-layer (200 steps, 1 seed) | 3.778 vs 4.104 final-mean loss (tri-layer -7.9%) | `results/ablation/comparison.json` |
| Final train loss (mean of last 100 train-tagged steps), 3 seeds | 1.727 +/- 0.044 (1.703 / 1.700 / 1.779) | `results/train/{42,0,1234}/loss_curve.csv` |

**Train-loss note.** The committed CSVs carry a `tag` column: rows tagged
`ttt` are training-time-test eval rows (`train/train_eagle3.py`), not train
loss, and the CSVs end on ttt rows (16.61 / 14.62 / 13.10 at step 2000). The
train-tagged rows give per-seed means over the last 100 train steps of 1.703
/ 1.700 / 1.779; the headline is their mean (1.727) with the sample standard
deviation across the three seeds (0.044). Read the `tag` column before
computing anything from these CSVs.

The agreement number is a direct measurement
(`python -m eval.measure_acceptance`): position-wise draft-vs-target token
agreement over teacher-forced held-out contexts. It upper-bounds
greedy-verification acceptance and feeds the geometric acceptance model.

| Metric | Status |
| --- | --- |
| ITL reduction @ batch 1 / 16 | **[NOT YET MEASURED]** - needs the vLLM EAGLE-3 weight-schema adapter |
| Batch-size crossover point | **[NOT YET MEASURED]** - depends on serving-stack ITL |
| Domain shift (general vs finance) | **[NOT YET MEASURED]** - per-domain split of the agreement eval |

No fabricated numbers. Unmeasured cells say so.

## The GPU run log

`results/gpu_run_log.md` names RunPod pod IDs, wall-clock times, and dollar
costs, including a dead pod that burned about $6.70 before the run moved. The
productive spend was about $7.70. Failed pods are part of the record.

## Quickstart

```
# fastest: local CPU demo, ~30s, no GPU, no network
make all        # setup + audit + demo + card + writeup + verify

# one artifact at a time
make demo       # local CPU pipeline; output is is_demo=true at every level
make card       # renders HF_CARD.md from the manifest
make verify     # walks every CLI, proves argparse binds
make audit      # ruff + mypy + pytest (CI gate)

# real numbers: needs a GPU pod and HF access, ~24h, ~$50-80 budget
make bench
```

The demo exercises the same code paths as the real pipeline (ablate compare,
release aggregate, model-card render, crossover analysis) but emits synthetic
shape-true artifacts watermarked `is_demo=true`. A passing demo proves the
modules accept and process data. It never stands in for a measurement.

## Serving integration (what exists, what does not)

Exists: the vLLM and SGLang invocation paths, the Nsight profiling stage, and
the crossover analysis. Does not exist: the adapter that exports this draft
head into vLLM's expected EAGLE-3 weight schema. Until that adapter is
written, the ITL row above stays `[NOT YET MEASURED]`.

## Testing

309 tests collected (verified 2026-09-21). Full CPU-only run on Python 3.12:
304 pass, 1 skip (the Qwen3 tokenizer test needs HF access), 2 fail. The 2
failures are release-tag invariants: two docs commits landed on main after
the v1.6.0 tag, so the tag no longer sits at the main tip. A new release tag
resolves them.

## License

MIT
