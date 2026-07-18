"""Measure draft-head acceptance directly (no serving stack).

Loads the frozen target + a trained EAGLE3Head checkpoint, runs both over
held-out tokenized sequences, and reports position-wise draft/target
agreement — the acceptance probability `p` that eval/acceptance.py's
geometric model consumes (E[accepted] = 1/(1-p) for horizon → ∞).

This is a direct measurement of the head's token-level fidelity on real
data, independent of vLLM/SGLang. The serving-stack ITL delta additionally
depends on scheduler/kernel overheads and is measured separately by
release/bench.sh; see WRITEUP.md §"Limitations" for the head-format adapter
required before vLLM can load this checkpoint.

Usage (GPU strongly recommended):
    python -m eval.measure_acceptance \
        --config train/config.yaml \
        --checkpoint results/train/42/best/checkpoint-2000/trainer.pt \
        --val-dir artifacts/data/tokenized/val \
        --max-rows 200 \
        --out results/eval/acceptance_measured.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk

from train.config import load_config
from train.head import EAGLE3Head
from train.train_eagle3 import build_target_model


@torch.no_grad()
def measure_agreement(
    head: EAGLE3Head,
    input_ids: torch.Tensor,
    temperature: float = 0.0,
) -> tuple[int, int]:
    """Count positions where the draft head's next-token prediction matches
    the target model's own prediction (greedy at T=0; sampled agreement is
    not modeled here — T=0 matches vLLM's rejection-sampling acceptance
    upper bound for greedy verification)."""
    logits = head(input_ids=input_ids)  # (B, L, V) — draft prediction per position
    target_out = head.target_model(input_ids=input_ids)
    target_logits = target_out.logits  # (B, L, V)

    # Predict token t+1 from position t; compare on non-pad positions.
    draft_next = logits[:, :-1].argmax(dim=-1)
    target_next = target_logits[:, :-1].argmax(dim=-1)
    valid = (input_ids[:, 1:] != 0) & (input_ids[:, :-1] != 0)
    agree = ((draft_next == target_next) & valid).sum().item()
    total = valid.sum().item()
    return int(agree), int(total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--val-dir", required=True, type=Path)
    ap.add_argument("--max-rows", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[measure] loading target {cfg.model.name_or_path} ...", flush=True)
    target_model, target_config = build_target_model(cfg)
    head = EAGLE3Head(
        target_model=target_model,
        target_config=target_config,
        layer_indices=cfg.eagle3.layer_indices,
        num_decoder_layers=cfg.eagle3.num_decoder_layers,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = head.load_state_dict(state["head_state"], strict=False)
    # target_model.* keys are intentionally absent from checkpoints
    # (v1.5.10); anything else missing is a real load failure.
    real_missing = [k for k in missing if not k.startswith("target_model.")]
    if real_missing or unexpected:
        print(f"[measure] BAD LOAD missing={real_missing} unexpected={unexpected}")
        return 2
    if torch.cuda.is_available():
        head = head.cuda()
    head.eval()

    ds = load_from_disk(str(args.val_dir))
    n = min(args.max_rows, len(ds))
    print(f"[measure] {n} val rows (of {len(ds)}), ckpt step {state['step']}", flush=True)

    agree_total = 0
    count_total = 0
    per_row: list[dict[str, Any]] = []
    for i in range(n):
        ids = ds[i]["input_ids"][: args.max_len]
        if len(ids) < 8:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        agree, total = measure_agreement(head, input_ids)
        agree_total += agree
        count_total += total
        per_row.append({"row": i, "agree": agree, "total": total})

    p = agree_total / count_total if count_total else 0.0
    expected_accept_len = 1.0 / (1.0 - p) if p < 1.0 else float("inf")
    result = {
        "checkpoint": str(args.checkpoint),
        "step": state["step"],
        "rows_evaluated": len(per_row),
        "positions": count_total,
        "agreement_rate_greedy": round(p, 6),
        "expected_acceptance_length_geometric": round(expected_accept_len, 4),
        "note": (
            "Greedy (T=0) position-wise draft/target agreement over teacher-forced "
            "contexts. Upper-bounds greedy-verification acceptance; serving-stack "
            "acceptance additionally reflects multi-token drafting error compounding."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
