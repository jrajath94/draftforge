# train/ — EAGLE-3 draft head training

Trains an EAGLE-3 draft head for `Qwen/Qwen3-4B-Instruct-2507` with
multi-layer feature fusion, training-time-test, and direct token
prediction. The head is the only thing trained; the 4B target stays
frozen.

## Architecture (per EAGLE-3 paper, Li et al. 2025)

```
Qwen3-4B hidden states at layers [7, 18, 29]
            ↓ concat along channel
Linear(3·hidden → hidden)         # fusion_proj, fresh init
            ↓
N decoder blocks (default 1)        # fresh init copies of target layer
            ↓
lm_head                            # copy of target's lm_head
            ↓
token logits  →  CE loss on next token
```

The draft head is ~`3·hidden² · (1 + N_decoder_blocks) + vocab·hidden`
parameters — hundreds of millions, not billions. Backprop flows only
through the head; the target is a frozen teacher.

Training-time-test (every 100 steps): the head samples its own drafts for a
short horizon, conditions on those drafts, and recomputes CE loss. This
extends the effective context past teacher forcing and reduces
training/inference distribution shift. Direct token prediction means we
predict the same token ids the teacher would — no hidden-state regression.

## Quickstart

```bash
# 1. Install training deps
pip install -e '.[train]'

# 2. Hugging Face auth is optional for the default target
#    (`Qwen/Qwen3-4B-Instruct-2507` is open-weight).

# 3. Rent a single H100 (RunPod, Vast, OCI — ~$2-3/hr)
#    Then on the rented box, run:
bash train/run_all_seeds.sh 3        # trains 3 seeds: 42 0 1234
# or with custom seeds:
SEEDS="42 7 99" bash train/run_all_seeds.sh
```

Each seed writes to `results/train/<seed>/`:
- `loss_curve.csv` — per-step loss (commit and plot this)
- `loss_curve.json` — same data, JSON
- `checkpoint-<step>/trainer.pt` — periodic snapshots (latest + best)
- `config.yaml` — frozen config used for that run
- `train.log` — full stdout/stderr

## Files

| File | Purpose |
|------|---------|
| `config.py` | `TrainConfig` pydantic model; `load_config` |
| `head.py` | `EAGLE3Head` nn.Module (tri-layer fusion + decoder + lm_head) |
| `train_eagle3.py` | training driver (CLI); invoked via `accelerate launch` |
| `ds_config.json` | DeepSpeed ZeRO-2 single-GPU config |
| `run_all_seeds.sh` | multi-seed wrapper |
| `config.yaml` (TODO) | sample training config — fill in dataset.train_dir path |

## Limitations

- Single-GPU only (multi-node out of scope per project budget/spec).
- bf16 only — fp16 risks numerical overflow on Qwen, per official EAGLE warning.
- Tri-layer `[7, 18, 29]` is the default for the 36-layer 4B target; empirical validation in Phase 3 (ablation).
- No curriculum, no dynamic top-k, no custom CUDA — minimal EAGLE-3 baseline.

## Citations

- EAGLE-3: Li, Wei, et al. *EAGLE-3: Scaling up Inference Acceleration with
  Training-Time-Test and Feature Fusion.* Mar 2025. (Spec reference.)
- SpecForge: https://github.com/SafeAILab/EAGLE (recommended training pipeline for EAGLE-3).
- Qwen3-4B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- DeepSpeed: Rasley et al., 2020.
