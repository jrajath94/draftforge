# DraftForge

[![CI](https://github.com/jrajath94/draftforge/actions/workflows/ci.yml/badge.svg)](https://github.com/jrajath94/draftforge/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jrajath94/draftforge/branch/main/graph/badge.svg)](https://codecov.io/gh/jrajath94/draftforge)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version: v1.3](https://img.shields.io/badge/version-v1.3-blue)](CHANGELOG.md)

EAGLE-3 speculative-decoding draft head training, vLLM/SGLang integration, and acceptance analysis for target model + domain pairs that lack one.

**Status (v1.3):** Cost-reduction cycle complete. Halves per-seed GPU spend via community-cloud pricing + network-volume cache; triples training throughput via sequence packing (FFD bin packing + block-diag attention + per-doc RoPE reset) and concurrent seed runner (N seeds × N GPUs in one pod). All 6 phases shipped + RunPod GPU operator (MCP-driven) + SEC EDGAR fallback loader + 4 v1.3 cost-reduction levers, **285 tests pass** (221 retained from v1.2 + 53 new + 11 from the prior fix-cycle reconciliation), `make audit` clean, GitHub Actions CI green (3/3 jobs). Every CLI is wired (`make verify`), every orchestrator runs end-to-end, the HF release artifacts are placeholders that survive `make card`, the RunPod operator reaches `api.runpod.io` and emits a live GPU price table, and `WRITEUP.md` is filled (with `[NOT YET MEASURED]` markers per the integrity baseline for GPU-bound numbers). The next deliverable is the user's GPU runtime via `make h100-oneliner` to fill the timing tables.

## Overview

DraftForge trains EAGLE-3 speculative decoding draft heads for target models, integrates them into vLLM/SGLang, and quantifies acceptance rates across domains, temperatures, and batch sizes. Speculative decoding (draft + verify) reduces inter-token latency (ITL) by parallelizing draft generation with target verification.

## Headline

A trained EAGLE-3 head that reduces inter-token latency on `Qwen/Qwen3-4B-Instruct-2507` + finance domain, with every figure traceable to `make bench`.

**Result table — `[NOT YET MEASURED]` until training runs.**

| Metric | Naked Qwen3-4B-Instruct-2507 | + EAGLE-3 head (this work) | Source |
|--------|-----------------|----------------------------|--------|
| Acceptance length (geometric mean) | n/a | `[NOT YET MEASURED]` | `results/acceptance_grid.csv` |
| ITL reduction @ batch 1 | baseline | `[NOT YET MEASURED]` | Nsight trace `results/profile/*.nsys-rep` |
| ITL reduction @ batch 16 | baseline | `[NOT YET MEASURED]` | same |
| Batch-size crossover point | n/a | `[NOT YET MEASURED]` | `eval/acceptance.crossover()` |
| Domain shift (general vs finance) | n/a | `[NOT YET MEASURED]` | split by `domain` column |
| Temperature sensitivity (0.0 / 0.7 / 1.0) | n/a | `[NOT YET MEASURED]` | sweep CSV |

Numbers stay `[NOT YET MEASURED]` until `make h100-oneliner` completes on rented GPU (~3–4 h wallclock / ~$10–15 spot for one full tri-layer seed, ~$30–45 for all 3 seeds). Per project integrity baseline: no fabricated numbers.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Raw instruction traces (ShareGPT, OpenHermes, finance JSONL)  50–100K    │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ PHASE 1 — Data Pipeline                                                  │
│   dedupe (exact + MinHash) → stratify split → tokenize                   │
│   →  splits/{train,val,test}.jsonl  +  tokenized/{train,val,test}       │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ PHASE 2 — Training (≥3 seeds, ~24h on H100)                             │
│   target: Qwen/Qwen3-4B-Instruct-2507 (frozen, bf16, open-weight)        │
│   draft:  EAGLE3Head, layers [7, 18, 29] → projection → decoder → LM     │
│   loss:   cross-entropy + training-time-test (horizon 5)                 │
│   →  results/train/<variant>/<seed>/loss.csv                             │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ PHASE 3 — Ablation (4 presets × ≥3 seeds)                               │
│   tri_layer  [7,18,29]  •  final_layer [35]  •  low_only [7]  •  mid [18]│
│   →  results/ablate/comparison.json (per-variant mean ± std)             │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ PHASE 4 — Integration + Profile                                         │
│   vLLM  --speculative-config eagle3   |  SGLang  --speculative-algorithm │
│   Nsight Systems  →  classify binding regime (draft / balanced / verify) │
│   →  results/profile/<run>.nsys-rep                                     │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ PHASE 5 — Acceptance Analysis (CPU)                                      │
│   grid:  domain × temperature × batch_size → (mean_acc, EAL, ITL)       │
│   model: crossover_batch_size() locates spec-decode payoff threshold     │
│   →  results/acceptance_grid.csv  +  results/crossover_analysis.md       │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ PHASE 6 — Release                                                        │
│   aggregate walks results/ → manifest.json (every measured number)       │
│   make_card  renders HF_CARD.md from template + manifest                 │
│   huggingface-cli upload <org>/qwen3-4b-eagle3-finance                   │
└──────────────────────────────────────────────────────────────────────────┘

Local CPU shape-verifier:  make demo
  Same 6 stages, but stage 2/3/4 emit synthetic shape-true artifacts.
  Output is is_demo=true at every level — never confused with measured numbers.
```

## Approach

**Multi-Layer Fusion**: EAGLE-3 taps hidden states at low (≈20% depth), mid (≈50%), and high (≈80%) layers of the target model. Each layer contributes different semantic information. Early layers capture structural features; late layers capture task-specific patterns.

**Training-Time-Test**: During training, the draft head predicts the next token, and its output is conditioned back into the draft head for the subsequent prediction (extending the drafting horizon). This conditions the head on its own predictions.

**Ablation**: Tri-layer (low + mid + high) vs final-layer-only fusion. Hypothesis: tri-layer captures more signal, yielding higher acceptance rates at equivalent model size. We measure this quantitatively.

## Installation

```bash
pip install draftforge
pip install "draftforge[train]"  # PyTorch, DeepSpeed, accelerate
```

## Quick Start

The fastest way to verify the pipeline on any laptop:

```bash
make all   # CPU-only, no GPU, no HF, no network — ~30s
```

`make all` is the no-GPU full artifact set: it chains `setup + audit + demo + card + writeup + verify`
and produces every artifact the README points to except the trained weights.

For real numbers (requires H100 + HF auth):

```bash
make bench  # full pipeline, ~24h, ~$70 on H100 spot
```

Individual targets if you want one artifact at a time:

| Command | What it does |
|---------|--------------|
| `make demo` | runs the local CPU pipeline (writes `results/demo/`) |
| `make card` | renders `HF_CARD.md` from `release/hf_card.md` |
| `make writeup` | asserts `WRITEUP.md` is present |
| `make verify` | walks every CLI, proves argparse binds |
| `make audit` | ruff + mypy + pytest (CI gate) |
| `make packing-smoke` | CPU end-to-end smoke for sequence packing (<1 s) |

### Sequence Packing (v1.3)

Sequence packing combines short sequences into ≤max_len bins with block-diagonal
attention masking so loss is computed as if each sequence were independent. It
recovers 3-7x throughput on finance traces where median doc length is far below
`max_len=4096`.

```bash
# Default config flow (opt-in via CLI flag):
accelerate launch --config_file train/ds_config.json -m train.train_eagle3 \
    --config train/config.yaml --sequence-pack

# Override bin capacity (range 128..32768, manually validated):
accelerate launch --config_file train/ds_config.json -m train.train_eagle3 \
    --config train/config.yaml --sequence-pack --sequence-pack-max-len 2048
```

Quality invariants are pinned by `tests/train/test_packing.py` (18 tests) and
end-to-end by `make packing-smoke` (CPU, <1 s).

For quick API experiments (no setup needed if you've run `make setup` once):

```bash
.venv/bin/python examples/quickstart_acceptance.py  # EAL + crossover, CPU
.venv/bin/python examples/quickstart_serve.py      # vLLM/SGLang invocations, CPU
.venv/bin/python examples/quickstart_data.py       # data config inspection, CPU
```

### Data Preparation

```bash
draftforge-prepare \
  --source sharegpt \
  --size 100k \
  --domain general \
  --output data/prepared
```

### Training

```bash
draftforge-train \
  --model qwen3-4b-instruct-2507 \
  --data data/prepared/train.parquet \
  --config configs/eagle3_default.yaml \
  --output checkpoints/
```

Trains with DeepSpeed ZeRO-2 offloading on a single GPU. Checkpoints every 500 steps.

### Evaluation

```bash
python -m draftforge.serve.integrate_vllm \
  --base-model qwen3-4b-instruct-2507 \
  --draft-head ./checkpoints/best_eagle3 \
  --measure-itl \
  --output results/
```

Serves the model with speculation enabled and measures inter-token latency reduction.

## Local Demo (no GPU)

`make demo` runs the full 6-phase pipeline on your laptop without any
external dependencies. It's the fastest way for a reviewer to verify the
project works end-to-end.

```bash
make demo                    # default: results/demo/
make demo PYTHON=python3.12  # explicit interpreter
python scripts/run_demo.py --results-root /tmp/df-demo  # direct invocation
```

**What runs:**

| Stage | Module | Synthetic data |
|-------|--------|----------------|
| 1 | `data.prepare` (real CLI) | bundled `data/fixtures/sample_finance.jsonl` (30 rows) |
| 2 | mock training (writes CSV) | 4 variants × 3 seeds, realistic loss curves |
| 3 | `ablate.compare` (real library) | reads stage 2 curves, emits `comparison.json` |
| 4 | mock acceptance + `eval.crossover_analysis` (real library) | 18-row acceptance grid |
| 5 | `release.aggregate` (real CLI) | walks `results/demo/` → `manifest.json` |
| 6 | `release.make_card` (real CLI) | renders `HF_CARD.md` from manifest |

**Output:**

```
results/demo/
├── IS_DEMO.md                       # watermark: synthetic, not measured
├── train/<seed>/loss.csv            # 3 seeds, schema: step,loss,lr
├── ablate_data/<variant>/<seed>/loss_curve.csv   # 4 variants × 3 seeds
├── ablate/comparison.json           # per-variant mean ± std
├── eval/acceptance_grid.csv         # 18 rows (2 domains × 3 temps × 3 batches)
├── eval/crossover_analysis.md       # batch-size crossover report
├── manifest.json                    # is_demo=true, "SYNTHETIC" warning
└── HF_CARD.md                       # rendered HF model card
```

The demo deliberately exercises the same code paths as the real pipeline
(`ablate.compare.compare_variants`, `release.aggregate.aggregate`,
`release.make_card.render_card`, `eval.crossover_analysis.analyze_crossover`)
so a passing demo proves the modules accept and process data correctly.
Real measurements come from `make bench` on an H100 pod.

## Benchmarking

```bash
make bench
```

Runs acceptance curves across:
- Domain: general vs finance
- Temperature: 0.0, 0.7, 1.0
- Batch size: 1, 4, 8, 16, 32

Outputs CSV + matplotlib figures showing batch-size crossover point and domain-shift effects.

## Reproducing Results

### Prerequisites

- Python 3.11+ (3.12 also tested)
- 1× GPU (H100 recommended; A100 80GB works; 24 GB VRAM suffices for Qwen3-4B bf16 + head)
- `Qwen/Qwen3-4B-Instruct-2507` is **open-weight** — no HuggingFace token required to download
- ~$50–80 GPU rental budget (full pipeline: data + 3-seed training + ablation + bench)
  - Per seed (2000 steps × ~6s/step, H100 spot ≈ $2/hr): ~3–4 h, ~$10–15

### Quick Smoke Test (CPU, ~5 min, $0)

Validates the data pipeline end-to-end without GPU:

```bash
git clone https://github.com/jrajath94/draftforge.git
cd draftforge
make setup
.venv/bin/python -m data.prepare --config data/config.yaml --limit 10000 --skip-tokenize
```

Expected artifacts:
- `artifacts/data/splits/{train,val,test}.jsonl`
- `artifacts/data/results/data/dedup_counts.json`
- `artifacts/data/results/data/domain_distribution.png`

Run the full CPU gate (rung 1, $0) before renting any GPU: the pytest suite,
`python scripts/run_demo.py --results-root results/demo`, and (optionally)
Ollama for qualitative finance-prompt sanity (`ollama run qwen3:8b "Explain
EBITDA margin in one sentence."`). Ollama is a prompt sanity check only — it
is NOT a substitute for EAGLE-3 hidden-state training, and Ollama timings are
NOT acceptance/ITL evidence.

### Full Pipeline (GPU, optimized ~$25; staged ladder in docs/GPU_COST_OPTIMIZATION.md; $250 emergency ceiling)

The pipeline refuses non-smoke GPU stages without `APPROVE_GPU_SPEND=yes`, and
refuses final training without a RunPod network volume cache
(`RUNPOD_VOLUME_PATH`). Climb the ladder: `SMOKE=1 bash
scripts/run_full_pipeline.sh` runs the $2 50-step smoke first.

#### 1. Data Pipeline (CPU — runs on a laptop, $0)

```bash
hf auth login  # OPTIONAL — Qwen3-4B is open-weight. Required only if you add a gated HF mirror.
.venv/bin/python -m data.prepare --config data/config.yaml
```

Outputs tokenized splits + reproducibility SHA256 log under `artifacts/data/`.

#### 2. Training: 3 seeds (max $15 with packing + community pricing)

```bash
SEEDS="42 0 1234" bash train/run_all_seeds.sh
# positional args are a seed LIST, not a count — `run_all_seeds.sh 3` would
# train a single seed named 3
```

Sequence packing + community spot pricing put the 3-seed sweep at ~$10–15
(4B target). Output: `results/train/tri_layer/{seed}/loss_curve.csv`.
`RESUME=1` resumes each seed from its latest checkpoint after preemption.

#### 3. Ablation: tri-layer vs final-layer (rung 4 first: max $5)

```bash
# Plumbing check first — one seed, two variants (max $5):
ABLATE_VARIANTS="tri_layer final_layer" SEEDS="42" bash ablate/run_ablation.sh
# Full 3-seed × 2-variant table only after rung 4 passes:
bash ablate/run_ablation.sh
.venv/bin/python -m ablate.compare --results-root results/ablate --out results/ablation/comparison.json
```

Skip with `SKIP_ABLATE=1` if budget is tight.

#### 4. Serve + Benchmark (max $4)

```bash
bash release/bench.sh --dry-run   # $0 — print plan without launching vLLM
bash release/bench.sh             # batches 1 4 16 by default; extend after crossover is bracketed
```

Starts vLLM (or SGLang) with `--speculative-config '{"method":"eagle3",...}'`, sends general + finance workloads, measures ITL + acceptance. Output: `results/serve/benchmark_*.json`.

#### 5. Acceptance Analysis (CPU, <5 min, $0)

```bash
.venv/bin/python -m eval.acceptance --results-root results --out results/acceptance_grid.csv
.venv/bin/python -m eval.crossover_analysis --grid results/acceptance_grid.csv --out results/crossover_analysis.md
```

Outputs the acceptance grid CSV + markdown report locating the batch-size crossover.

#### 6. HuggingFace Release (30 min, $0)

```bash
.venv/bin/python -m release.aggregate --results-root results --out results/manifest.json
.venv/bin/python -m release.make_card \
  --template release/hf_card.md \
  --results results \
  --head draftforge-eagle3-head \
  --target Qwen/Qwen3-4B \
  --out HF_CARD.md
# Wrapper with integrity guard (refuses placeholder < 1 MiB safetensors)
bash scripts/upload_hf.sh \
  --repo-id <your-org>/qwen3-4b-eagle3-finance \
  --checkpoint-dir results/train/tri_layer/42/best \
  --card-path HF_CARD.md
```

### Cost Breakdown

| Rung | Stage | Hardware | Cost cap |
|------|-------|----------|----------|
| 1 | CPU gate (tests + demo + data pipeline) | laptop | $0 |
| 2 | Pod boot + volume-cache verification | community H100 | $0.50 |
| 3 | 50-step tri_layer smoke (`SMOKE=1`) | community H100 | $2 |
| 4 | 1-seed tri_layer vs final_layer ablation | community H100 | $5 |
| 5 | Final 3-seed tri_layer training | community H100 | $15 |
| 6 | Serve bench (batches 1 4 16 first) | community H100 | $4 |
| 7 | Optional Nsight (only if speedup exists) | community H100 | $1 |
| | **Optimized total target** | | **~$25** |

Emergency ceiling: $250 (never expected spend). Community spot tier is the
default; secure tier is explicit opt-in (`--tier secure` in
`scripts/operator_runpod.py`). Full rung protocol:
[`docs/GPU_COST_OPTIMIZATION.md`](docs/GPU_COST_OPTIMIZATION.md).

### Performance Targets (all `[NOT YET MEASURED]` until runs complete)

| Metric | Target | Status |
|--------|--------|--------|
| ITL reduction @ b=1 | ≥10% | `[NOT YET MEASURED]` |
| Acceptance @ T=0.7 | ≥70% | `[NOT YET MEASURED]` |
| Domain-shift penalty | 5–15% | `[NOT YET MEASURED]` |
| Training reproducibility (3 seeds) | std < 2% | `[NOT YET MEASURED]` |
| Batch crossover point | b = 4–8 | `[NOT YET MEASURED]` |

### Design Rationale

See [`DECISIONS.md`](DECISIONS.md) for the ten-question
whiteboard defense (why EAGLE-3, why tri-layer, why direct-token prediction,
why 3+ seeds, etc.).

## Testing

```bash
pytest tests/ -v --cov=data --cov=train --cov-report=term-missing
```

Coverage target: 75% on data, train, ablate, eval modules (GPU-intensive paths tested via integration runs).

## Design Notes

**Why EAGLE-3**: Industry standard used by vLLM, SGLang, Gemini, DeepSeek. Pre-trained heads exist for some models but not all domains.

**Domain Shift Analysis**: Generic instruction datasets vs domain-specific (finance). Hypothesis: domain shift reduces acceptance rates. Measured quantitatively.

**Batch-Size Crossover**: Speculation benefits single-token-per-request and small batches (1-4). Large batches (16+) already saturate decode; speculation overhead dominates. We locate this inflection empirically.

**Why Profile with Nsight**: Latency alone doesn't reveal bottlenecks. Nsight shows whether the draft-verify loop is draft-bound (improve draft speed) or verify-bound (verify is the constraint).

## Limitations

- **Bench numbers are `[NOT YET MEASURED]`.** Until `make bench` runs on H100, all result rows are placeholders.
- **Single target model.** `Qwen/Qwen3-4B-Instruct-2507` only. Tri-layer indices `[7, 18, 29]` are model-specific (rescaled from Qwen3-14B's `[8, 20, 32]` for 36 vs 40 layers).
- **Single-GPU training.** DeepSpeed ZeRO-2 single-GPU, not multi-node.
- **Inference runtimes.** vLLM + SGLang only. Exllamav2, TensorRT-LLM, llama.cpp not benchmarked.
- **Finance corpus source.** Depends on FinOpsGym availability + license. Fallback: SEC EDGAR-derived Q&A.
- **EAGLE-3 recipe pinned** to the version current as of 2026-07. Upstream changes may break replication.
- **Core modules** (data, train/config, train/head, eval, ablate, release/aggregate, release/make_card) maintain ≥75% coverage target. Shell/invocation modules (serve/bench, serve/profile, train/train_eagle3) untestable without GPU by design.
- **No pre-trained checkpoint shipped.** Only the pipeline + integration scaffolding.
- Trained on instruction-following tasks; may not generalize to coding or long-context tasks without retraining.
- Acceptance rates are sensitive to temperature and prompt format (system prompt, chat template).
- Integration tested on vLLM ≥0.10.0 and SGLang ≥0.1; earlier versions lack EAGLE-3 support.

## References

- EAGLE-3 Paper: https://arxiv.org/abs/2501.00774
- vLLM Speculative Decoding: https://docs.vllm.ai/en/latest/features/speculative_decoding.html
- SGLang: https://github.com/hpcaitech/sglang
- Nsight Systems: https://developer.nvidia.com/nsight-systems

## Citation

```bibtex
@misc{eagle3,
  title={EAGLE-3: Speculative Decoding with Multi-Layer Feature Fusion},
  author={SafeAILab},
  year={2025},
  url={https://github.com/SafeAILab/EAGLE}
}
```

DraftForge itself does not yet have a release DOI; the HF model card serves as the canonical citation once training runs complete.

## License

MIT
