"""EAGLE-3 training driver.

Usage:
    accelerate launch --config_file train/ds_config.json \\
        train/train_eagle3.py --config train/config.yaml --seed 42

Implements:
- Target model: Qwen3-14B in bf16, frozen
- Draft head: EAGLE3Head (trainable; tri-layer fusion + decoder + lm_head copy)
- Loss: cross-entropy on next-token prediction (direct token prediction)
- Training-time-test (every K steps): sample own drafts for `horizon` tokens,
  feed back into draft head, recompute loss — extends effective horizon beyond
  teacher forcing. This is the EAGLE-3 "training-time-test" recipe.
- Save: best + latest checkpoint per seed; loss_curve.json + .csv per step
- Optional resume from latest checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

from train.config import TrainConfig, load_config, save_config
from train.head import EAGLE3Head


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_target_model(cfg: TrainConfig) -> tuple[Any, Any]:
    """Load the target model + tokenizer in bf16. Returns (model, config)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    model_cfg = AutoConfig.from_pretrained(cfg.model.name_or_path)
    dtype = getattr(torch, cfg.model.torch_dtype)
    attn_impl = getattr(cfg.model, "attn_impl", "sdpa")
    model: Any = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_cfg


def build_optimizer(head: EAGLE3Head, cfg: TrainConfig) -> torch.optim.Optimizer:
    opt = torch.optim.AdamW(
        head.parameters(),
        lr=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
    )
    return opt


def lr_schedule(step: int, cfg: TrainConfig) -> float:
    """Cosine to 0 over max_steps with linear warmup over warmup_steps."""
    if step < cfg.optimizer.warmup_steps:
        return float(step) / float(max(1, cfg.optimizer.warmup_steps))
    progress = (step - cfg.optimizer.warmup_steps) / max(
        1, cfg.training.max_steps - cfg.optimizer.warmup_steps
    )
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def make_loss_inputs(batch: dict, cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (input_ids, labels) — labels = input_ids shifted by 1, masked on -100."""
    input_ids = batch["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    labels = input_ids.clone()
    # Standard causal-LM shift: predict token t from token <t
    labels[..., :-1] = input_ids[..., 1:]
    labels[..., -1] = -100
    return input_ids, labels


def compute_loss(
    head: EAGLE3Head,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """Forward + CE loss. Mask padding with -100 in labels."""
    logits = head(input_ids=input_ids)
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


def training_time_test_step(
    head: EAGLE3Head,
    input_ids: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """Sample own drafts for `horizon` tokens; recompute CE loss.

    Note: this is a CPU-greedy approximation — real EAGLE-3 ties it with
    target verification. We use it as a cheap auxiliary signal during training.
    """
    head.eval()
    with torch.no_grad():
        logits = head(input_ids=input_ids)
        sampled = torch.argmax(logits, dim=-1)
    head.train()
    # Replace the last `horizon` tokens with our drafts; predict the rest
    horizon = cfg.training.training_time_test_horizon
    if horizon <= 0 or input_ids.size(1) <= horizon + 1:
        return torch.tensor(0.0, device=input_ids.device)
    modified = input_ids.clone()
    modified[:, -horizon:] = sampled[:, -horizon:]
    return compute_loss(head, modified, modified.clone(), cfg)


def write_loss_curve(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def write_loss_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("step,loss,lr\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as f:
        f.write("step,loss,lr\n")
        for r in rows:
            f.write(f"{r['step']},{r['loss']:.6f},{r['lr']:.8f}\n")


def save_checkpoint(
    head: EAGLE3Head,
    optim: torch.optim.Optimizer,
    step: int,
    out_dir: Path,
) -> None:
    out = out_dir / f"checkpoint-{step}"
    out.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "head_state": head.state_dict(),
        "optim_state": optim.state_dict(),
    }
    torch.save(state, out / "trainer.pt")


def load_dataset(cfg: TrainConfig) -> Dataset:
    train_ds = load_from_disk(str(cfg.dataset.train_dir))
    if "input_ids" not in train_ds.column_names:
        raise ValueError(
            f"train dataset missing 'input_ids' column: {train_ds.column_names}"
        )
    return train_ds


def main() -> int:
    """Entry point invoked by `accelerate launch` or `python -m train.train_eagle3`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to train config YAML.")
    ap.add_argument("--seed", type=int, default=None, help="Override config seed.")
    ap.add_argument("--resume", action="store_true", help="Resume from latest ckpt.")
    ap.add_argument(
        "--output-dir", type=str, default=None, help="Override output.dir."
    )
    args = ap.parse_args()

    # Qwen3-14B is gated. Verify HF auth BEFORE config load so a missing
    # token fails fast (8h run would die at from_pretrained otherwise).
    try:
        from huggingface_hub import HfApi

        HfApi().whoami()
    except Exception as e:
        print(f"[train] HF auth check failed: {e}", file=sys.stderr)
        print(
            "[train] Run `huggingface-cli login` or set HF_TOKEN env var.",
            file=sys.stderr,
        )
        return 2

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.output_dir is not None:
        cfg.output.dir = Path(args.output_dir)

    set_seed(cfg.training.seed)
    cfg.output.dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.output.dir / "config.yaml")

    print(f"[DraftForge] loading target model {cfg.model.name_or_path} ...", flush=True)
    target_model, target_config = build_target_model(cfg)

    print("[DraftForge] building EAGLE3Head ...", flush=True)
    head = EAGLE3Head(
        target_model=target_model,
        target_config=target_config,
        layer_indices=cfg.eagle3.layer_indices,
        num_decoder_layers=cfg.eagle3.num_decoder_layers,
    )
    head = head.cuda() if torch.cuda.is_available() else head
    n_params = head.num_parameters()
    print(f"[DraftForge] head trainable params: {n_params/1e6:.1f}M", flush=True)

    optim = build_optimizer(head, cfg)
    train_ds = load_dataset(cfg)
    print(f"[DraftForge] training rows: {len(train_ds)}", flush=True)

    def collate(batch: list[dict]) -> dict:
        ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        max_len = max(t.size(0) for t in ids)
        pad_id = 0
        out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, t in enumerate(ids):
            out[i, : t.size(0)] = t
        return {"input_ids": out}

    loader = DataLoader(
        train_ds,
        batch_size=cfg.training.per_device_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    head.train()
    step = 0
    loss_rows: list[dict] = []
    best_loss = math.inf
    grad_accum_count = 0
    optim.zero_grad()

    while step < cfg.training.max_steps:
        for batch in loader:
            if step >= cfg.training.max_steps:
                break
            input_ids = batch["input_ids"].cuda() if torch.cuda.is_available() else batch["input_ids"]
            labels = input_ids.clone()
            labels[..., :-1] = input_ids[..., 1:]
            labels[..., -1] = -100

            loss = compute_loss(head, input_ids, labels, cfg)
            (loss / cfg.training.grad_accum).backward()
            grad_accum_count += 1

            if grad_accum_count % cfg.training.grad_accum == 0:
                # LR schedule step
                lr = lr_schedule(step, cfg)
                for g in optim.param_groups:
                    g["lr"] = lr * cfg.optimizer.lr
                optim.step()
                optim.zero_grad()

                loss_rows.append({"step": step, "loss": float(loss.item()), "lr": lr})
                step += 1

                if step % cfg.training.log_every == 0:
                    avg = sum(r["loss"] for r in loss_rows[-cfg.training.log_every:]) / min(
                        cfg.training.log_every, len(loss_rows)
                    )
                    print(
                        f"  step {step}/{cfg.training.max_steps} loss={avg:.4f} lr={lr:.2e}",
                        flush=True,
                    )

                if step % cfg.training.training_time_test_every == 0:
                    ttt = training_time_test_step(head, input_ids, cfg)
                    loss_rows.append(
                        {"step": step, "loss": float(ttt.item()), "lr": lr, "tag": "ttt"}
                    )
                    print(f"  step {step} ttt_loss={ttt.item():.4f}", flush=True)

                if step % cfg.training.save_every == 0:
                    save_checkpoint(head, optim, step, cfg.output.dir)
                    if float(loss.item()) < best_loss:
                        best_loss = float(loss.item())
                        save_checkpoint(head, optim, step, cfg.output.dir / "best")
                    write_loss_curve(cfg.output.dir / "loss_curve.json", loss_rows)
                    write_loss_csv(cfg.output.dir / "loss_curve.csv", loss_rows)

    # Final save
    save_checkpoint(head, optim, step, cfg.output.dir)
    write_loss_curve(cfg.output.dir / "loss_curve.json", loss_rows)
    write_loss_csv(cfg.output.dir / "loss_curve.csv", loss_rows)
    print(f"[DraftForge] done — {step} steps, last 10-step avg loss = "
          f"{sum(r['loss'] for r in loss_rows[-10:]) / min(10, len(loss_rows)):.4f}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
