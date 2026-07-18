# GPU Run Log — DraftForge evidence ladder

Required by docs/GPU_COST_OPTIMIZATION.md. One block per paid run.

## Run 1 — evidence ladder (rungs 2-6, single pod)

```text
date_utc: 2026-07-17/18
pod_id: pxw3dvpn5ep4ci (dud machine — container never started; deleted) then lo4tt4ddxn7vz7
gpu_type: NVIDIA A100-SXM4-80GB (dud was A100 80GB PCIe)
hourly_rate_usd: 1.39 (dud: 1.19)
secure_or_community: community
volume_id: none (pod-local /workspace 80GB; ALLOW_NO_VOLUME_CACHE=1 — single-pod run, full teardown after)
target_model: Qwen/Qwen3-4B-Instruct-2507
head_variant: tri_layer (+ final_layer in rung-4 probe)
seeds: 42 (smoke/probe) -> 42 0 1234 (final)
max_steps: 50 (smoke) -> 200 (probe) -> 2000 (final)
sequence_pack: on
started_at: 2026-07-17T21:42:13Z (dud) / 2026-07-18T03:22:03Z (real pod)
ended_at: 2026-07-18 (~08:30Z, after acceptance measurement + artifact pull)
wall_minutes: ~330 productive on lo4tt4ddxn7vz7 (+ ~340 dud-pod idle before self-stop design)
estimated_cost_usd: 6-10 (ladder at $1.39/hr)
actual_cost_usd: ~7.70 productive (lo4tt4ddxn7vz7) + ~6.70 wasted (dud pod pxw3dvpn5ep4ci, billed while container never started) ≈ 14.40 total
artifacts_written:
  - results/train/{42,0,1234}/loss_curve.{csv,json} — 3 seeds × 2000 steps, production config
  - results/ablate/{tri_layer,final_layer}/42/loss_curve.csv — 200-step probe
  - results/ablation/comparison.{json,csv} — tri 3.778 vs final 4.104 (last-100 train-step mean)
  - results/eval/acceptance_measured_{42,0,1234}.json — greedy agreement 0.694/0.676/0.692
  - results/figures/loss_curves_measured.png
  - best checkpoints (target-excluded, ~2.8 GB each) backed up off-pod
promotion_decision:
  - rung 3 smoke: PROMOTED (loss 14.46 -> 6.41, finite)
  - rung 4 probe: PROMOTED (both variants complete, schema OK; comparison
    initially aggregated zeros due to results-root mismatch — fixed in
    ablate/run_ablation.sh, recomputed from curves)
  - rung 5 final: PROMOTED (3/3 seeds exit 0; first attempt of seed 42 died
    at step 1000 on checkpoint bloat — fixed in v1.5.10, retrained)
  - rung 6 serve bench: NOT RUN — vLLM absent from train extras AND the
    checkpoint uses DraftForge module names, not vLLM's EAGLE-3 weight
    schema; adapter required (documented in README Limitations). Direct
    acceptance measurement (eval/measure_acceptance.py) run instead.
findings_fixed_during_run: 12 releases v1.5.2-v1.5.10 + v1.6.0 (launcher,
  data stage, dataset id, label-mask dtype, 4-D attention mask, fp32 head,
  RoPE position_embeddings, venv path, checkpoint bloat, ablation root,
  ttt-row contamination of loss statistics)
```
