# DraftForge Design Decisions

Ten-question whiteboard defense. Each question cites the concrete evidence (code path, test, or external reference) that made the call.

**Project baseline:**
- **Target model:** `Qwen/Qwen3-4B-Instruct-2507` (36 layers, hidden_size=2560, vocab=151936, ~4B parameters, open-weight).
- **Head:** EAGLE-3 (Li et al., NeurIPS 2025), tri-layer fusion of target hidden states.
- **Cost ceiling:** ≤ $250 total (data + 3-seed training + ablation + bench) on H100 spot.

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

**Most consequential tradeoffs (high reversibility cost):** Q1 (architecture class), Q6 (analytical model), Q10 (headline framing).
**Cheapest to change:** Q2, Q4, Q7, Q9 (config-level).

---

**Last reviewed:** 2026-07-09 — git revision `80a0261` (post CODE/MEASURED split in `.planning/PROJECT.md`).
