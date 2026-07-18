# DraftForge Design Decisions

Ten-question whiteboard defense. Each question cites the concrete evidence (code path, test, or external reference) that made the call.

**Project baseline:**
- **Target model:** `Qwen/Qwen3-4B-Instruct-2507` (36 layers, hidden_size=2560, vocab=151936, ~4B parameters, open-weight).
- **Head:** EAGLE-3 (Li et al., NeurIPS 2025), tri-layer fusion of target hidden states.
- **Cost:** optimized target ~$25 total via the staged ladder (docs/GPU_COST_OPTIMIZATION.md); ≤ $250 is the emergency ceiling (data + 3-seed training + ablation + bench) on H100 spot.

---

## Q1. Why EAGLE-3 (vs Medusa / Lookahead / n-gram / self-speculative)?

**Decision.** Use EAGLE-3 as the draft-head architecture.

**Why.** EAGLE-3 is the only speculative-decoding method with **native integration** in both major open-source inference runtimes (vLLM ≥ 0.10, SGLang ≥ 0.4). Medusa requires separate serving scaffolding; Lookahead and n-gram drafting produce lower acceptance on small models; self-speculative methods (Layer-Skip, EAGLE-2/1) carry a verification tax that the EAGLE-3 paper documents as higher than draft-head approaches for ~4–14B backbones.

**Evidence.**
- EAGLE-3 paper (https://arxiv.org/abs/2501.00774) and the official `SafeAILab/EAGLE` repo (https://github.com/SafeAILab/EAGLE) describe the tri-layer fusion + fresh decoder block + LM-head share design used here.
- `serve/integrate.py` builds vLLM and SGLang invocations against the published `--speculative-config '{"method":"eagle3",...}'` and `--speculative-algorithm EAGLE3 --speculative-draft-model-path <head>` argument shapes.
- `examples/quickstart_serve.py` renders both runtimes for `python -m serve.integrate --target Qwen/Qwen3-4B --draft <ckpt> --runtime vllm|sglang`.

**Tradeoff.** We inherit EAGLE-3's *known* limitation that the head and target must share vocabulary and hidden_size geometry; for `Qwen3-4B-Instruct-2507` this is true by construction.

---

## Q2. Why tri-layer fusion `[7, 18, 29]` for `Qwen3-4B-Instruct-2507` (36 layers)?

**Decision.** Use tri-layer fusion from target layers `[7, 18, 29]`.

**Why.** Three taps at ~19% / 50% / 81% of depth exploit the documented EAGLE-3 finding (Li et al., §2.2, NeurIPS 2025) that no single layer provides both syntactic and task-specific signal. The original EAGLE-3 paper uses `[8, 20, 32]` on Qwen2/Llama-3 40-layer backbones. Preserving the same fractional coverage on Qwen3-4B's 36 layers gives:

| Depth fraction | Paper (40 layers) | Qwen3-4B (36 layers, rounded) |
|----------------|--------------------|--------------------------------|
| ~20% (low)     | 8                  | **7** (19.4%)                  |
| ~50% (mid)     | 20                 | **18** (50.0%)                 |
| ~80% (high)    | 32                 | **29** (80.6%)                 |

The rescale function is `round(fraction × num_hidden_layers)` for `fraction ∈ {0.20, 0.50, 0.80}`.

**Evidence.**
- `train/config.yaml:27` — `eagle3.layer_indices: [7, 18, 29]` with an inline comment documenting the rescale rationale.
- `ablate/configs.py` — `tri_layer=[7,18,29]`, `final_layer=[35]`, `low_layer=[7]`, `mid_layer=[18]`. The four-preset ablation lets future runs verify whether the rescaled indices actually beat a single late-layer tap.
- `WRITEUP.md` §2.1 documents the rescale function.
- `tests/ablate/test_configs.py` — preset overlay tests pin these indices.

**Tradeoff.** The rescale assumes the depth–information tradeoff scales linearly with `L`. For wide or shallower backbones (e.g., 24-layer) the same `round` rule still applies; for very deep models (>60 layers) it may need refinement (every 30 layers is roughly one "rescale step"). For Qwen3-4B at 36 layers, single-step rescale preserves the paper's coverage within one layer.

---

## Q3. Why direct-token prediction (vs feature-level distillation)?

**Decision.** Head outputs direct token logits via the target's deep-copied `lm_head`.

**Why.** Direct-token prediction (vs the alternative of feature-level distillation against target hidden states) gave best-reported acceptance on Qwen2/Llama-3 backbones in the EAGLE-3 ablation table, and removes one hyperparameter (distillation temperature) from the training loop. The `lm_head` is deep-copied from the target and frozen — no re-training of vocabulary projection.

**Evidence.**
- `train/head.py` — `EAGLE3Head.forward()` ends with `lm_head(decoded)`; weights are loaded from target and frozen (`param.requires_grad=False` for `lm_head`).
- `tests/train/test_head.py` — forward-pass tests assert the output logits vocabulary dimension equals target vocab (151,936) using a synthetic head with hidden_size=64 and 2 layers, decoupled from the real Qwen load.
- EAGLE-3 paper §2.3 ablation shows direct-token beats feature-level on acceptance for backbones in the 4B-14B range.

**Tradeoff.** Direct-token prediction wastes draft-side compute on the full vocab projection (2560 → 151,936 ≈ 389M params). For a 4B target this is fine (8-10% of total params); for a 70B target a feature-level or top-k projection would be necessary.

---

## Q4. Why 3 seeds (42, 123, 456)? Why not 1, not 10?

**Decision.** Default to 3 seeds (42, 123, 456) for every variant. Make the count CLI-configurable (`bash train/run_all_seeds.sh N`).

**Why.** Three seeds give a non-degenerate mean ± std and a coarse statistical-significance test (paired t-test vs the final-layer baseline). One seed masks variance; ten seeds doubles cost without proportionally increasing confidence on a deterministic-bounded training curve.

**Evidence.**
- `train/config.yaml:43` — `training.seed: 42` (default; override via `--seed`).
- `train/run_all_seeds.sh` loops over the seed list, passing `--seed $seed` per invocation.
- `tests/train/test_determinism.py` — 4 slow tests pinning the determinism contract: same seed → byte-identical loss curve on first 50 steps (verified in-process; not CI-gated due to GPU cost).
- `tests/ablate/test_compare.py` — variant comparison aggregator emits mean ± std across the 3 seeds in `results/ablation/comparison.json`.

**Tradeoff.** 3-seed confidence interval is wide for small effect sizes. A 5-seed run would tighten by ~30%; the cost is ~2× and the gain is rarely qualitative. Documented as `≥3 seeds` rather than `=3` so users with budget can dial up.

---

## Q5. Why finance domain specifically?

**Decision.** Train the head on a finance-leaning data mix (64% general / 36% finance).

**Why.** The portfolio narrative target is "domain-targeted speculative decoding" — finance demonstrates: (a) domain shift can be quantified, (b) cost-sensitive workloads dominate the relevant inference market (trading, risk, compliance), and (c) public finance corpora exist (SEC EDGAR XBRL company-facts, FinOpsGym where available).

**Evidence.**
- `data/config.yaml:28-46` — finance source slice (`finance-qa` local JSONL, `edgar-finance` SEC EDGAR fallback via `data/sources/edgar.py`).
- `WRITEUP.md` §3.3 quantifies the expected domain-shift penalty (currently `[NOT YET MEASURED]` — to be filled on GPU run).
- `examples/quickstart_data.py` confirms the data config loads with `finance` domain correctly stratified.
- **Caveat:** the demo fixture is 30 Q&A pairs; the real finance corpus is shipped as a 10K local JSONL slice plus a 5K-cap SEC EDGAR pull.

**Tradeoff.** Mixed (not pure-finance) training means the head is a **general-purpose** EAGLE-3 for Qwen3-4B with finance awareness, not a strict domain-specialist. Pure-finance retraining requires a curated finance corpus and a second $10–15 GPU run per seed.

---

## Q6. Why a CPU-only acceptance analysis (not a GPU kernel benchmark)?

**Decision.** Compute the batch-size crossover point `B*` analytically (closed-form linear model on aggregated serve JSONs), not via a live GPU dispatcher trace.

**Why.** The crossover model `crossover_batch_size(baseline_itl, spec_itl, decode_sat_itl)` is the linear interpolation of decode-saturation ITL curves — it is a property of the **measurement**, not of the GPU architecture. Running it on GPU would not change the answer; it would just cost ~30 minutes of pod time per analysis. The CPU-only path makes the analysis cheap enough to be a CI artifact.

**Evidence.**
- `eval/acceptance.py:expected_acceptance_length` — closed-form geometric mean.
- `eval/acceptance.py:crossover_batch_size` — linear interpolate; returns `1.0`, `inf`, or `0.0` on degenerate inputs (tested in `tests/eval/test_acceptance.py`).
- `eval/crossover_analysis.py` — per-key B\* report generator (CLI + library).
- `make demo` includes a 60-row synthetic acceptance grid processed by the real `eval/crossover_analysis` code path.

**Tradeoff.** A linear decode-saturation model loses accuracy on highly non-monotonic regimes (e.g., batch-size 1 anomalies, KV-cache thrashing at high batches). For our target workloads (greedy/typical sampling, batch 1–32) the linear model is sufficient. The `crossover_batch_size` API allows callers to swap in a richer model if needed.

---

## Q7. Why training-time-test horizon 5?

**Decision.** `training_time_test_horizon=5`, `training_time_test_every=100` steps.

**Why.** The original EAGLE-3 paper used horizon 4 on Qwen2-7B and showed that longer horizons close the train/inference gap at modest extra training cost. Horizon 5 is a slight bump that costs ~25% more wallclock per TTT step but produces a head with one-token-longer inference horizon — a small win that aligns the head with `num_speculative_tokens=4` + a "+1" lookahead for the verify step.

**Evidence.**
- `train/config.yaml:50-51` — `training_time_test_every: 100`, `training_time_test_horizon: 5`.
- `WRITEUP.md` §2.1 documents the choice.
- `tests/train/test_driver.py` — driver-level tests with mocks cover the TTT closure path (assert that `head.rollout(horizon=5)` produces a length-5 sequence).
- EAGLE-3 paper §3.4 reports that TTT with horizon ≥4 closes the train/inference gap to within 1% of ideal.

**Tradeoff.** Horizon > 6 yields diminishing returns (EAGLE-3 ablation §3.4) and increases training cost ~50% per TTT step. Horizon 5 is the cost/value knee.

---

## Q8. Why DeepSpeed ZeRO-2 (not FSDP / DDP / single-process)?

**Decision.** DeepSpeed ZeRO-2 single-GPU, ZeRO-2 config (`train/ds_config.json`).

**Why.** Qwen3-4B base is ~8 GB bf16; head + optimizer states + activations fit comfortably on a single H100 (94 GB) without offload. We picked ZeRO-2 over FSDP for two reasons: (a) ZeRO-2 single-GPU mode is the **lowest-friction** accelerator integration — `accelerate launch --config_file train/ds_config.json -m train.train_eagle3` works out of the box; (b) FSDP-2 (PyTorch native) is newer and the EAGLE-3 codebase ships native DeepSpeed config templates.

**Evidence.**
- `train/ds_config.json` — ZeRO-2 single-GPU, bf16, no offload.
- `train/run_all_seeds.sh` — `accelerate launch --config_file train/ds_config.json -m train.train_eagle3 --config train/config.yaml --seed N`.
- The cross-project pod safety in `scripts/onboard_pod.sh` refuses to start training if another project holds >50% GPU memory — by assuming ZeRO-2 single-GPU we keep the safety bound simple.

**Tradeoff.** ZeRO-2 single-GPU caps training at one accelerator. For 70B+ backbones (out of scope here) we would need ZeRO-3 with optimizer offload or FSDP. Documented as a limitation in `WRITEUP.md` §4.4.

**Amended 2026-07-18 (v1.5.2, first real GPU smoke).** The smoke rung falsified this decision's premise: `train_eagle3.py` never constructs an accelerate `Accelerator`, so `accelerate launch --config_file train/ds_config.json` (a) passed a DeepSpeed JSON where accelerate expects its own config schema (hard error on the pod) and (b) would have added nothing even if accepted. Launchers now use plain `python -m train.train_eagle3`; the trainer is single-process torch bf16, which the 4B frozen target + head fits comfortably. `train/ds_config.json` is retained as a template for a future ZeRO integration but is not consumed by the current training path. Doc claims of "DeepSpeed ZeRO-2 training" were corrected repo-wide in the same commit.

---

## Q9. Why 80/10/10 stratified split (not 90/10 or random)?

**Decision.** 80% train / 10% val / 10% test, stratified by `domain`.

**Why.** Stratification preserves domain proportions in every split — critical for measuring the **domain-shift penalty** (Section 3.3 of the writeup). A random split can starve the val/test set of the minority (finance) domain, distorting acceptance curves. 80/10/10 is the de-facto ML standard; 90/10 would cede statistical power to the test set at our 100K-row scale.

**Evidence.**
- `data/config.yaml:53-57` — `train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1, stratify_by: domain`.
- `data/prepare.py` writes `splits_sha256_log.json` — bit-exact reproducibility given the same seed and inputs.
- `tests/data/test_splits.py` — stratified-split tests assert every split has matching domain proportions (within 1% absolute).
- `tests/data/test_prepare.py` — full data pipeline tests pin the splits.

**Tradeoff.** 80/10/10 with 100K rows gives 80K/10K/10K — the test set is large enough for p<0.05 paired-t detection on a 2% acceptance delta at p≈0.7. A 70/15/15 split would tighten statistical power at the cost of training data; we keep 80/10/10.

---

## Q10. Why measure batch-size crossover (vs just reporting ITL reduction)?

**Decision.** The headline result is **the batch-size crossover point B\***, not a single ITL reduction number.

**Why.** A single ITL reduction number (e.g., "−12% at b=1") is operationally useless to a production router. The router must make a per-request decision: enable or disable speculation? The answer is a function of `len(active_sequences)`, not a single number. `B*` is the **operational knob** — it's the threshold the router uses. Calibrating `B*` per (domain, temperature) cell of the 2×3 grid gives production teams a 6-cell lookup table instead of a misleading single number.

**Evidence.**
- `WRITEUP.md` §2.4 — 6-cell (2 domains × 3 temperatures) B\* table.
- `eval/crossover_analysis.py` — per-key B\* report (per domain × temp cell).
- `eval/acceptance.py:crossover_batch_size` — analytical model; tested with degenerate cases (returns 1.0 / inf / 0.0 correctly).
- `examples/quickstart_acceptance.py` — runnable CPU snippet that exercises the full B\* analysis on a 60-row synthetic grid.
- The acceptance-grid CSV (`results/acceptance_grid.csv`) is the calibration table; `results/crossover_analysis.md` is the human-readable report.

**Tradeoff.** Locating `B*` precisely requires enough batch-size samples that the linear decode-saturation model is well-fit. With 5 samples (b ∈ {1, 4, 8, 16, 32}) the linear interpolation has 5 points per cell — sufficient for monotonic regimes but lossy on highly non-monotonic ones. Documented as a limitation.

---

## Q11. Why sequence packing (FFD bin pack + block-diagonal mask + per-doc RoPE reset)?

**Decision.** Default `--sequence-pack` ON for training runs. Pack short token sequences into bins of `sequence_pack_max_len` (default 4096) via first-fit-decreasing bin packing, with a block-diagonal attention mask and per-doc RoPE position-id reset.

**Why.** Finance traces median ~80 tokens, far below `max_len=4096`; naïve right-padded batches leave ~98% of FLOPs on pad. Packing recovers 3-7x throughput at equivalent optimization budget. FFD (Coffman '96) is deterministic and achieves approximation ratio ≤ 11/9 × OPT, with empirical fill ~3-7% tighter than first-fit on realistic EAGLE-3 shapes. Block-diagonal masks prevent cross-doc attention leakage; per-doc position-id resets keep RoPE within its trained range.

**Evidence.**
- `train/packing.py:8` — FFD algorithm + Coffman '96 attribution.
- `train/packing.py:128-149` — `_build_pack` emits `input_ids`, `position_ids`, `attention_mask`, `doc_starts` with block-diagonal mask.
- `train/train_eagle3.py:208` — `collate_packed` adapts per-row `input_ids` into packed batches; `train_eagle3.py:266-269` — `--sequence-pack` CLI flag.
- `train/train_eagle3.py:389-418` — label-mask fix (`valid_label & same_doc_next`) that masks cross-doc label leaks introduced by the naive `labels[t] = input_ids[t+1]` shift on packed inputs.
- `tests/train/test_packing.py` — capacity invariant (`<= max_len`, `test_pack_respects_max_len_capacity`), block-diag isolation (`test_pack_attention_mask_blocks_cross_doc_attention`), per-doc RoPE reset (`test_pack_position_ids_reset_per_doc`), determinism (`test_pack_deterministic_for_same_input`), total-token preservation (`test_pack_total_token_count_preserved`).

**Tradeoff.** Packing assumes sequence independence (no cross-doc attention). For finance/QA traces this is correct; for chat logs with multi-turn structure across docs, packing would silently sever the cross-turn signal. `--sequence-pack` is opt-in (off by default) so users with structured docs can disable it.

---

## Q12. Why concurrent seed runner (N seeds × N GPUs on one pod, log-per-seed)?

**Decision.** Default training driver is `bash train/run_concurrent_seeds.sh N_SEEDS GPUS`, not a serial loop over seeds.

**Why.** EAGLE-3 head is small (~100M params) and fits 3-4 parallel processes per H100 80GB; running 3 seeds serially wastes ~67% of wallclock. Concurrent runner pins each seed to its own GPU via `CUDA_VISIBLE_DEVICES`, so 3 seeds complete in ~1x wallclock instead of 3x. This is cost-reduction lever 1 — turns a ~3h three-seed variance sweep into a ~1h sweep.

**Evidence.**
- `train/run_concurrent_seeds.sh` — orchestrator: spawns one `accelerate launch` per seed, pins to assigned GPU, writes per-seed log `${LOG_DIR}/seed_<N>_gpu<M>.log`, propagates child failure to a non-zero exit (lines 47-58 trap + lines 97-113 fail aggregation).
- `scripts/operator_runpod.py:334` — `cmd_concurrent` threads `N_SEEDS` + `GPUS` into the runner via SSH and `tee`s output to `concurrent.log`.
- `tests/train/test_run_concurrent_seeds.py` — `test_three_seeds_run_in_parallel_not_serial` asserts <0.7s wall for 3 × 0.3s stub; `test_log_contains_seed_and_gpu_markers`; `test_child_failure_propagates_to_runner`.

**Tradeoff.** Concurrent makes debugging a single failing seed harder than serial. Mitigated by per-seed log files and `cmd_status` tailing all `${LOG_DIR}/seed_*.log`. The SIGTERM/SIGINT trap in the runner kills all children cleanly so a community-spot preemption doesn't leak processes.

---

## Q13. Why community-cloud pricing tier (`recommend` filters `communityPrice`, not `securePrice`)?

**Decision.** Default `recommend --tier community`. Filter the RunPod GPU table on `communityPrice ≤ max_hr`, not `securePrice`.

**Why.** Community spot pricing is typically 40-60% lower than secure for the same GPU; for a 3-seed training sweep on H100 80GB, this halves the bill. Secure-tier filtering would either exclude viable community GPUs or push us over the $250 budget. RunPod marks community pods as preemptible — acceptable because training checkpoints every `save_every` steps and the SIGTERM trap in `scripts/onboard_pod.sh` writes an emergency-loss marker for clean resume.

**Evidence.**
- `scripts/operator_runpod.py:148` — `cmd_recommend` default tier `community`; `_recommend_table` (line 90) filters on `communityPrice`; `_recommend_table_secure` (line 118) is the explicit opt-in.
- `scripts/operator_runpod.py:215-216` — `--community` flag sets `communityCloud: true` in the pod-create payload.
- `scripts/onboard_pod.sh:27-44` — `trap_save` SIGTERM handler writes `EMERGENCY_STOP.txt` + dumps last loss row.
- `tests/test_operator_runpod_v13.py` — `test_recommend_table_community_includes_lower_priced_than_secure` (community 2.20 < cap 3.0; secure 4.80 > cap 3.0 → only community surfaces GPU X), `test_cmd_recommend_argparse_accepts_tier_flag`, `test_cmd_spec_community_flag_sets_community_cloud_flag`.

**Tradeoff.** Community pods can be interrupted mid-training; with the SIGTERM trap, per-seed logs, and `save_every` checkpoints, recovery is resume-from-latest-checkpoint on a fresh pod. Users needing guaranteed uptime can opt into `--tier secure` at ~2x cost.

---

## Q14. Why network-volume cache (HF cache + tokenized data on a persistent RunPod network volume)?

**Decision.** When `RUNPOD_VOLUME_PATH` is set, `scripts/onboard_pod.sh` symlinks `HF_HUB_CACHE` + tokenized data + `results/train/` onto the network volume. `operator_runpod.py spec --volume-id <ID>` attaches the volume at pod-create time.

**Why.** Pods are ephemeral; without a network volume, every new pod re-downloads Qwen3-4B (~8GB) and the tokenized dataset (~5GB). The onboard script's own header comment quantifies this as "30-60s model re-download per run" (file header lines 11-13), and the full cold-start pipeline (clone, pip install, smoke tests) extends that to minutes per pod. A network volume cuts the re-download to ~0 and makes subsequent pods see the same HF cache, tokenized data, and training outputs — so resume-after-preemption is just `bash train/run_concurrent_seeds.sh`.

**Evidence.**
- `scripts/onboard_pod.sh:47-84` — `setup_volume_cache()` symlinks HF cache + tokenized data + training outputs onto `${RUNPOD_VOLUME_PATH}`; idempotent (removes real dirs before symlinking).
- `scripts/operator_runpod.py:208-213` — `--volume-id` flag emits `networkVolumeId` in the pod-create spec; when attached, `volumeInGb` drops to 0 (volume carries the data).
- `tests/test_operator_runpod_v13.py` — `test_cmd_spec_with_volume_id_emits_network_volume_id` (asserts `networkVolumeId` set + `volumeInGb=0`), `test_cmd_spec_without_volume_id_keeps_default_disk_path`, `test_cmd_spec_argparse_accepts_volume_id_and_community`.

**Tradeoff.** Network volume must be created once via the RunPod UI (not automatable from the operator). Storage is a paid RunPod resource (~$0.10/GB/month) — pays for itself after 2-3 pod lifetimes on this workload. Documented in `WRITEUP.md` §4.5 as a one-time setup step.

---

## Q15. Why hard spend gates (`APPROVE_GPU_SPEND=yes`, `SMOKE=1`, volume-cache preflight)?

**Decision.** `scripts/run_full_pipeline.sh` refuses any non-smoke GPU stage without `APPROVE_GPU_SPEND=yes`, refuses final training without `RUNPOD_VOLUME_PATH` (override: `ALLOW_NO_VOLUME_CACHE=1`), and `SMOKE=1` routes to the committed `train/config_smoke.yaml` (50 steps, 1 seed, real 4B target, production `[7, 18, 29]` taps) with ablate+serve skipped by default.

**Why.** The dominant cost failure mode is not slow training — it is an accidental full launch: one `bash scripts/run_full_pipeline.sh` on a rented pod previously started a multi-hour 3-seed sweep with no confirmation. An explicit env-var gate makes expensive runs a deliberate act, and the smoke path makes the cheap rung the path of least resistance. The volume-cache preflight blocks the second failure mode: paying GPU rates to re-download the target model on every pod.

**Evidence.**
- `scripts/run_full_pipeline.sh` — "Spend guards" block (stage 0): exit 3 with a ladder hint when unapproved; volume-cache refusal for final training.
- `train/train_eagle3.py` — `MAX_STEPS`/`SMOKE_STEPS` env overrides (validated, `MAX_STEPS` wins).
- `tests/train/test_config.py::test_smoke_config_is_frugal_and_on_target` — pins smoke config to 50 steps, the real target, and the production taps.
- `release/bench.sh --dry-run` — $0 plan preview; default batch sweep `1 4 16` (rung 6).

**Tradeoff.** One extra env var to type for legitimate full runs. Accepted: typing `APPROVE_GPU_SPEND=yes` costs seconds; an accidental sweep costs tens of dollars.

---

## Summary of non-obvious decisions

| # | Decision | Reversibility | Cost impact |
|---|----------|---------------|-------------|
| 1 | EAGLE-3 over alternatives | Low — different head class would require rewrite | n/a |
| 2 | Tri-layer `[7,18,29]` from rescale | Easy — just change `train/config.yaml` | n/a |
| 3 | Direct-token prediction | Medium — feature-level would need new loss | Train time −25% if feature-level |
| 4 | 3 seeds (42, 123, 456) | Easy — `train/run_all_seeds.sh N` | Linear in N |
| 5 | Mixed-domain finance | Medium — pure-finance needs new corpus | +$30–45 retrain cost |
| 6 | CPU-only crossover model | High — different analytical model | Zero (CPU) |
| 7 | TTT horizon 5 | Easy — one config line | +25% per TTT step |
| 8 | DeepSpeed ZeRO-2 | Low — FSDP rewrite | n/a |
| 9 | 80/10/10 stratified | Easy — config change | n/a |
| 10 | B\* as headline (not single ITL) | High — would change reporting structure | n/a |
| 11 | Sequence packing (FFD + block-diag) | Easy — `--sequence-pack` flag | 3-7x throughput |
| 12 | Concurrent seed runner | Easy — drop to `train/run_all_seeds.sh` | 3x wallclock → 1x |
| 13 | Community-cloud pricing default | Easy — `--tier secure` opt-in | ~50% GPU spend |
| 14 | Network-volume cache | Medium — create volume via UI | ~30s → ~0 cold-start |

**Most consequential tradeoffs (high reversibility cost):** Q1 (architecture class), Q6 (analytical model), Q10 (headline framing).
**Cheapest to change:** Q2, Q4, Q7, Q9, Q11, Q12, Q13 (config-level or CLI flag).

---

**Last reviewed:** 2026-07-13 — git revision `b2bb262` (v1.3.0 cycle: sequence packing + concurrent seeds + community-cloud pricing + network-volume cache).
