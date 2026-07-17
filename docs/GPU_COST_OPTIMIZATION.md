# DraftForge GPU Cost-Optimization Plan

Scope: DraftForge's expensive work is training EAGLE-3 heads, ablations,
serving benchmarks, and optional profiling. This plan preserves the quality of
the final evidence while reducing GPU time.

## Cost Thesis

DraftForge should spend GPU time only after CPU and tiny-GPU gates prove that
the data, shapes, target geometry, training loop, and release pipeline are
sound. The final 3-seed training run should be the last step, not the first
debugging surface.

The project quality target stays unchanged:

- trained head for `Qwen/Qwen3-4B-Instruct-2507`
- tri-layer vs final-layer evidence
- acceptance and ITL measurement
- honest `[NOT YET MEASURED]` markers until real artifacts exist

## Evidence Ladder

### Rung 0: CPU Shape and Pipeline

Run before renting hardware:

```bash
pytest -q tests/test_demo_pipeline.py tests/data/test_dedup.py \
  tests/train/test_config.py tests/ablate/test_configs.py \
  tests/serve/test_integration.py tests/release/test_aggregate.py
```

Acceptance:

- demo path works without heavy optional packages
- active configs point at the 4B target
- layer indices are `[7, 18, 29]`
- release manifest can be generated from synthetic artifacts

### Rung 1: Tiny GPU Smoke

Run one seed, one variant, very few steps:

- variant: `tri_layer`
- seed: 42
- steps: 20 to 50
- sequence packing: enabled
- output: isolated smoke directory

Acceptance:

- target model loads
- hidden-state taps match expected layer count
- forward/backward pass succeeds
- loss curve is finite and decreasing enough to prove plumbing
- checkpoint and config are written

### Rung 2: Short Ablation Probe

Run only:

- `tri_layer`
- `final_layer`

Use one seed and short training. This is not the final claim; it validates
that the ablation harness and result aggregation work on real GPU artifacts.

Acceptance:

- both variants complete
- `release.aggregate` reads the outputs
- comparison JSON has the expected schema
- no result is cited as final

### Rung 3: Final Training

Only after Rungs 0-2 pass, run the final training:

- at least 3 seeds
- production config
- sequence packing enabled
- network-volume cache enabled
- checkpointing enabled
- resume-safe logging enabled

Do not run a fourth or fifth seed unless the confidence interval changes the
decision. Three seeds remain the default quality/cost point.

### Rung 4: Serve Bench

Run serving measurement after the trained head exists:

- baseline target
- target + EAGLE-3 head
- paired prompt set
- batch sizes 1, 4, 8, 16, 32 only if prior points show a crossover

If batch 1, 4, and 16 already locate the crossover clearly, do not expand
unless the curve is non-monotonic.

## Concrete Changes for the Next Agent

### 1. Add Explicit Frugal Run Modes

Add or document these modes:

- `SMOKE_STEPS=50`
- `ABLATE_VARIANTS="tri_layer final_layer"`
- `N_SEEDS=1` for smoke
- `N_SEEDS=3` for final
- `SKIP_SERVE=1` until trained weights exist

The full pipeline should not default to a 24-hour GPU run when invoked by a
new agent.

### 2. Make Cost Preflight Mandatory

Before training, print:

- target model
- layer indices
- seed list
- variants
- max steps
- estimated checkpoint count
- output directory
- whether network volume cache is active
- whether this is smoke, ablation, or final

Require explicit approval for final runs.

### 3. Use Network-Volume Cache and Resume

The repo already has volume-cache logic. Treat it as mandatory for paid runs:

- model cache on persistent volume
- tokenized data on persistent volume
- checkpoints on persistent volume
- per-seed logs retained across pod restarts

Do not redownload the model on every pod.

### 4. Stop Paying for Bad Data Shapes

Before GPU training:

- compute token-length histogram on CPU
- estimate packing efficiency
- reject configs with poor packing utilization
- verify stratified split counts

Sequence packing should stay on for final runs unless a test shows data
leakage or model incompatibility.

### 5. Profile Only After Correctness

Nsight is optional and should be run after the serve bench proves a useful
speedup. Do not profile a broken or untrained head.

Minimum profiling plan:

- one baseline trace
- one spec trace
- one batch size near the crossover

## Stop Gates

Stop the paid run if:

- target hidden-size or vocab-size differs from `release/hf_config.json`
- layer indices are out of range
- first backward pass yields NaN/Inf
- loss does not write within the first checkpoint window
- sequence packing emits cross-document label leakage
- checkpoint resume cannot be verified

## Quality Guardrails

- Do not reduce below 3 final seeds for the published claim.
- Do not cite smoke/short-ablation runs as final results.
- Do not replace real ITL/acceptance measurement with synthetic demo numbers.
- Keep `[NOT YET MEASURED]` until real GPU artifacts exist.
- Do not upload placeholder weights as a production model.

## Recommended Final GPU Sequence

1. CPU validation.
2. Tiny 50-step `tri_layer` smoke.
3. One-seed short `tri_layer` vs `final_layer` ablation.
4. Three-seed final `tri_layer` training.
5. Serve bench on paired prompts.
6. Optional Nsight trace only if the serve bench exposes a meaningful speedup.

This ordering cuts debugging spend sharply while preserving the final evidence
quality expected from the project.

## Current Project State Fit

Current disk state shows DraftForge already has several cost-control levers:

- `train/packing.py` and packed-training tests exist.
- `train/run_concurrent_seeds.sh` exists for parallel seed execution.
- `scripts/onboard_pod.sh` has network-volume cache support and preemption
  handling.
- `scripts/operator_runpod.py` has community/secure tier logic.
- active configs now target `Qwen/Qwen3-4B-Instruct-2507`.
- `scripts/run_demo.py` gives a CPU pipeline shape test.

The missing piece is not another broad training plan. It is a strict
execution ladder that prevents using final GPU runs for debugging.

## Current RunPod Price Assumptions

As of 2026-07-16, public RunPod and price-aggregator data roughly show:

| GPU | Fit for DraftForge | Community estimate | Secure estimate |
|---|---|---:|---:|
| RTX 4090 24GB | too small for full 4B head training; possible tiny inference only | ~$0.34-$0.69/hr | varies |
| A100 PCIe 40GB | maybe smoke only; memory risk for full training | ~$1.19/hr | ~$1.39/hr |
| A100 80GB | frugal smoke/short ablation | ~$1.19-$1.39/hr | ~$1.39-$1.49/hr |
| H100 PCIe 80GB | best final cost/perf target | ~$1.99/hr | ~$2.39/hr |
| H100 SXM/NVL 80-94GB | fastest final target if price is acceptable | ~$2.59-$2.69/hr | ~$2.99/hr |

Always confirm current pod price in the RunPod UI/API before launch.

## Cost Model

Use:

```text
estimated_cost = hourly_rate * (setup_minutes + training_minutes + eval_minutes + teardown_minutes) / 60
```

Planning estimates:

| Phase | GPU | Wall time target | Cost at $1.99/hr | Cost at $2.69/hr |
|---|---|---:|---:|---:|
| CPU demo on M1 Max | local | local | $0 | $0 |
| 50-step smoke | A100/H100 | 20-35 min incl setup | ~$0.66-$1.16 | ~$0.90-$1.57 |
| 1-seed short ablation, 2 variants | H100 | 60-90 min | ~$1.99-$2.99 | ~$2.69-$4.04 |
| final 3-seed tri-layer | H100 | 3-5 hr if serial, lower if concurrent | ~$6-$10 | ~$8-$13.50 |
| serve bench only | H100/A100 | 30-60 min | ~$1-$2 | ~$1.35-$2.70 |
| optional Nsight | H100 | 15-25 min | ~$0.50-$0.85 | ~$0.67-$1.12 |

Set a practical total cap of `$25` for the optimized final path before any
reruns. The old `$250` ceiling should be treated as a hard emergency ceiling,
not the planned spend.

## Use the M1 Max First

The M1 Max should handle all CPU gates:

```bash
pytest -q tests/test_demo_pipeline.py tests/data/test_dedup.py \
  tests/data/test_splits.py tests/train/test_config.py \
  tests/train/test_layer_indices.py tests/train/test_packing.py \
  tests/ablate/test_configs.py tests/serve/test_integration.py \
  tests/release/test_aggregate.py
python scripts/run_demo.py --results-root results/demo
```

Use Ollama locally only for qualitative prompt sanity checks. It is not a
training substitute for the EAGLE-3 head because the head requires target
hidden states and matching geometry from `Qwen/Qwen3-4B-Instruct-2507`.

Example useful local check:

```bash
ollama serve
ollama pull qwen3:8b
ollama run qwen3:8b "Explain EBITDA margin in one sentence."
```

Use this to inspect finance prompt quality, not to produce acceptance or ITL
claims.

## RunPod Execution Ladder

### Rung 1: CPU Gate

Question: will the GPU run fail for a known static reason?

Run the M1 tests above. Stop if any fail.

Cost: `$0`.

### Rung 2: Pod Boot and Cache Verification

Question: will the pod avoid repeated setup cost?

On RunPod:

```bash
nvidia-smi
export RUNPOD_VOLUME_PATH=/workspace/cache-volume  # actual mount path
bash scripts/onboard_pod.sh
test -L ~/.cache/huggingface || true
```

Stop if:

- no persistent volume is mounted
- model cache is not symlinked
- repo install fails

Expected cost: `<$0.50` if stopped early.

### Rung 3: 50-Step Smoke

Question: does model loading, hidden-state extraction, loss, checkpointing, and
resume-safe logging work?

Recommended environment:

```bash
export CONFIG=train/config_smoke.yaml   # committed 50-step smoke config
export OUTPUT_ROOT=results/gpu_smoke
export SEEDS=42
```

`train/train_eagle3.py` honors `MAX_STEPS` (and `SMOKE_STEPS`) env overrides,
so `train/config_smoke.yaml` is the source of truth and `MAX_STEPS` can cap
any config further without editing YAML on the pod.

Run either:

```bash
bash train/run_all_seeds.sh 42
# or the gated one-liner (defaults ablate+serve to skipped):
SMOKE=1 bash scripts/run_full_pipeline.sh
```

Promote only if:

- `loss_curve.csv` exists
- checkpoint exists
- no NaN/Inf loss
- config copy exists in output

Expected cost: `$1-$2`.

### Rung 4: One-Seed Two-Variant Ablation

Question: does the real ablation loop work, and are outputs aggregatable?

Run only:

```bash
export SEEDS=42
export ABLATE_VARIANTS="tri_layer final_layer"
bash ablate/run_ablation.sh
python -m ablate.compare --results-root results/ablate \
  --out results/ablation/comparison.json
```

`ablate/run_ablation.sh` honors `ABLATE_VARIANTS` (env beats positional args).
Do not run all four variants for a plumbing check.

Expected cost: `$2-$4`.

Promote only if `comparison.json` has both variants and no schema gaps.

### Rung 5: Final Three-Seed Training

Question: can we produce publishable weights and loss curves?

Use H100 PCIe first if available near `$1.99/hr`; use H100 SXM/NVL if PCIe is
unavailable or materially slower. Use A100 80GB only if smoke showed enough
memory and the slower runtime is still cheaper.

Run:

```bash
export SEEDS="42 0 1234"
export OUTPUT_ROOT=results/train
bash train/run_all_seeds.sh
```

If using `train/run_concurrent_seeds.sh`, only do so when the pod has enough
GPU memory for concurrent jobs. For a single H100, serial is safer; for a 2x
or 4x pod, concurrent can reduce wall time but may raise hourly cost. Compare:

```text
1x H100 serial cost = 1 * hourly_rate * hours_serial
3x H100 concurrent cost = 3 * hourly_rate * hours_concurrent
```

Concurrent is cheaper only if `hours_concurrent < hours_serial / 3` after
setup overhead. Otherwise use serial on one GPU.

Expected optimized cost: `$6-$14` if smoke prevented reruns.

### Rung 6: Serve Bench

Question: does the trained head improve ITL/acceptance under paired prompts?

Run minimal first:

```bash
TARGET_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
HEAD_DIR=results/train/tri_layer/42/best \
BATCH_SIZES="1 4 16" \
TEMPS="0.7" \
DOMAINS="general finance" \
bash release/bench.sh
```

Only expand to:

```bash
BATCH_SIZES="1 4 8 16 32"
TEMPS="0.0 0.7 1.0"
```

if the minimal bench does not locate the crossover clearly.

Expected cost: `$1-$3`.

### Rung 7: Optional Nsight

Run only if serve bench shows a real speedup and you need kernel evidence.

Profile:

- one baseline batch
- one spec batch
- one batch near crossover

Expected cost: `<$1`.

## Hard Budget Caps

| Phase | Max spend |
|---|---:|
| M1/CPU validation | $0 |
| pod boot/cache verification | $0.50 |
| 50-step smoke | $2 |
| one-seed two-variant ablation | $5 |
| final 3-seed training | $15 |
| serve bench | $4 |
| optional Nsight | $1 |
| total optimized path | $25 |

Stop if a cap is exceeded. Write down why before continuing.

## Required Run Log

Every paid run must leave:

```text
date_utc:
pod_id:
gpu_type:
hourly_rate_usd:
secure_or_community:
volume_id:
target_model:
head_variant:
seeds:
max_steps:
sequence_pack:
started_at:
ended_at:
wall_minutes:
estimated_cost_usd:
actual_cost_usd:
artifacts_written:
promotion_decision:
```

## Frugality Controls (Implemented)

All pre-paid-run controls are in place:

1. `train/config_smoke.yaml` — committed 50-step, single-seed smoke config on
   the real 4B target with production `[7, 18, 29]` taps.
2. `MAX_STEPS` / `SMOKE_STEPS` env overrides in `train/train_eagle3.py`
   (`MAX_STEPS` wins when both are set).
3. `ABLATE_VARIANTS` env support in `ablate/run_ablation.sh` (beats positional
   args).
4. `release/bench.sh --dry-run` (or `DRY_RUN=1`) — prints model/head/batches/
   temps/domains plan without launching vLLM. Default batch sweep is `1 4 16`
   (rung 6); extend to `1 4 8 16 32` only after the crossover is bracketed.
5. `scripts/run_full_pipeline.sh` preflight — refuses any non-smoke GPU stage
   without `APPROVE_GPU_SPEND=yes`, refuses final training without
   `RUNPOD_VOLUME_PATH` (override: `ALLOW_NO_VOLUME_CACHE=1`), and `SMOKE=1`
   routes to the smoke config with ablate+serve skipped.
6. `RESUME=1` in `train/run_all_seeds.sh` passes `--resume` per seed
   (community-spot preemption recovery).

