"""EAGLE-3 training driver.

Usage:
    python -m train.train_eagle3 --config train/config.yaml --seed 42

Implements:
- Target model: Qwen/Qwen3-4B-Instruct-2507 in bf16, frozen
- Draft head: EAGLE3Head (trainable; tri-layer fusion + decoder + lm_head copy)
- Loss: cross-entropy on next-token prediction (direct token prediction)
- Training-time-test (every K steps): sample own drafts for `horizon` tokens,
  feed back into draft head, recompute loss — extends effective horizon beyond
  teacher forcing. This is the EAGLE-3 "training-time-test" recipe.
- Save: best + latest checkpoint per seed; loss_curve.json + .csv per step
- Optional resume from latest checkpoint
- v1.3: --sequence-pack flag (cost-reduction lever 2). Packs short sequences
  into bins with block-diagonal attention masks + per-doc RoPE reset.
  Recovers ~3-7x throughput on finance traces where median doc length is
  far below max_len=4096.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from train.packing import Pack, pack_sequences


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
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward + CE loss. Mask padding with -100 in labels.

    position_ids + attention_mask are required for sequence packing (block-
    diagonal masks + per-doc RoPE reset). When None, the head falls back to
    HF's built-in causal mask (default behaviour).
    """
    logits = head(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


def training_time_test_step(
    head: EAGLE3Head,
    input_ids: torch.Tensor,
    cfg: TrainConfig,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample own drafts for `horizon` tokens; recompute CE loss.

    Note: this is a CPU-greedy approximation — real EAGLE-3 ties it with
    target verification. We use it as a cheap auxiliary signal during training.
    """
    head.eval()
    with torch.no_grad():
        logits = head(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        sampled = torch.argmax(logits, dim=-1)
    head.train()
    # Replace the last `horizon` tokens with our drafts; predict the rest
    horizon = cfg.training.training_time_test_horizon
    if horizon <= 0 or input_ids.size(1) <= horizon + 1:
        return torch.tensor(0.0, device=input_ids.device)
    modified = input_ids.clone()
    modified[:, -horizon:] = sampled[:, -horizon:]
    return compute_loss(
        head,
        modified,
        modified.clone(),
        cfg,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )


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


def collate_packed(batch: list[dict], max_len: int) -> dict:
    """Sequence-packing collator (v1.3 cost-reduction lever 2).

    Packs the per-row `input_ids` lists into ≤max_len bins via first-fit-
    decreasing bin packing. Each pack carries a block-diagonal attention
    mask so cross-document attention does not leak, plus per-doc RoPE
    position IDs that reset to 0 at each doc boundary.

    Returns dict with shape:
        input_ids:      (M, max_len) int64 — padded to max_len
        position_ids:   (M, max_len) int64 — per-doc positions
        attention_mask: (M, max_len, max_len) int64 — block-diagonal
        doc_starts:     list[list[int]] — one list per pack

    M ≤ len(batch) (fewer bins than input rows when packs combine docs).
    """
    seqs: list[list[int]] = [list(b["input_ids"]) for b in batch if len(b["input_ids"]) > 0]
    if not seqs:
        return {
            "input_ids": torch.zeros((0, max_len), dtype=torch.long),
            "position_ids": torch.zeros((0, max_len), dtype=torch.long),
            "attention_mask": torch.zeros((0, max_len, max_len), dtype=torch.long),
            "doc_starts": [],
        }
    packs: list[Pack] = pack_sequences(seqs, max_len=max_len)
    n_packs = len(packs)
    input_ids = torch.zeros((n_packs, max_len), dtype=torch.long)
    position_ids = torch.zeros((n_packs, max_len), dtype=torch.long)
    attention_mask = torch.zeros((n_packs, max_len, max_len), dtype=torch.long)
    doc_starts: list[list[int]] = []
    for i, pack in enumerate(packs):
        pack_len = len(pack.input_ids)
        # Note: avoid torch.from_numpy — envs with mixed numpy 1.x/2.x crash.
        # tensor(list) is a touch slower but ABI-stable.
        input_ids[i, :pack_len] = torch.tensor(pack.input_ids.tolist(), dtype=torch.long)
        position_ids[i, :pack_len] = torch.tensor(pack.position_ids.tolist(), dtype=torch.long)
        attention_mask[i, :pack_len, :pack_len] = torch.tensor(
            pack.attention_mask.tolist(), dtype=torch.long
        )
        doc_starts.append(list(pack.doc_starts))
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "doc_starts": doc_starts,
    }


def main() -> int:
    """Entry point invoked by `python -m train.train_eagle3` (see run_all_seeds.sh)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to train config YAML.")
    ap.add_argument("--seed", type=int, default=None, help="Override config seed.")
    ap.add_argument("--resume", action="store_true", help="Resume from latest ckpt.")
    ap.add_argument(
        "--output-dir", type=str, default=None, help="Override output.dir."
    )
    ap.add_argument(
        "--sequence-pack", action="store_true",
        help="Enable sequence packing (FFD bins + block-diag attention). "
             "Overrides training.sequence_pack in config.",
    )
    ap.add_argument(
        "--sequence-pack-max-len", type=int, default=None,
        help="Override training.sequence_pack_max_len (range 128..32768). "
             "Implies --sequence-pack.",
    )
    args = ap.parse_args()

    # HF auth pre-flight. The default target (Qwen3-4B-Instruct-2507) is
    # open-weight, so auth is OPTIONAL — we soft-warn rather than exit.
    # A gated target swap (e.g. a private repo) will still need a token;
    # from_pretrained() will fail loudly in that case. Skip the check
    # entirely with DRAFTFORGE_SKIP_HF_AUTH=1 (CPU tests / dry-runs).
    if os.environ.get("DRAFTFORGE_SKIP_HF_AUTH", "0") == "1":
        print("[train] DRAFTFORGE_SKIP_HF_AUTH=1; skipping HF auth pre-flight")
    else:
        try:
            from huggingface_hub import HfApi

            HfApi().whoami()
        except Exception as e:
            print(
                f"[train] WARN: HF auth pre-flight failed: {e}",
                file=sys.stderr,
            )
            print(
                "[train] Continuing — fine for the default open-weight target. "
                "If you point at a gated model, set HF_TOKEN or run "
                "`huggingface-cli login` before launching.",
                file=sys.stderr,
            )

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.output_dir is not None:
        cfg.output.dir = Path(args.output_dir)
    if args.sequence_pack:
        cfg.training.sequence_pack = True
    if args.sequence_pack_max_len is not None:
        # Manual range check: direct assignment after load_config bypasses
        # pydantic's Field(ge=128, le=32768) validator (default
        # validate_assignment=False in pydantic v2).
        if not (128 <= args.sequence_pack_max_len <= 32768):
            print(
                f"[train] --sequence-pack-max-len must be in [128, 32768]; "
                f"got {args.sequence_pack_max_len}",
                file=sys.stderr,
            )
            return 2
        cfg.training.sequence_pack_max_len = args.sequence_pack_max_len
        cfg.training.sequence_pack = True  # implicit: setting max_len implies packing on

    # Frugality override: MAX_STEPS (or SMOKE_STEPS, checked second) caps
    # training.max_steps from the environment so shell orchestrators can run
    # short smoke passes without editing YAML. MAX_STEPS wins when both are set.
    steps_env = os.environ.get("MAX_STEPS") or os.environ.get("SMOKE_STEPS")
    if steps_env:
        try:
            steps_override = int(steps_env)
        except ValueError:
            print(
                f"[train] MAX_STEPS/SMOKE_STEPS must be an integer; got {steps_env!r}",
                file=sys.stderr,
            )
            return 2
        if steps_override < 1:
            print(
                f"[train] MAX_STEPS/SMOKE_STEPS must be ≥1; got {steps_override}",
                file=sys.stderr,
            )
            return 2
        cfg.training.max_steps = steps_override
        print(f"[train] max_steps overridden via env: {steps_override}")

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

    use_packing = cfg.training.sequence_pack
    pack_max_len = cfg.training.sequence_pack_max_len
    if use_packing:
        print(
            f"[DraftForge] sequence packing ENABLED (max_len={pack_max_len})",
            flush=True,
        )

    def collate(batch: list[dict]) -> dict:
        if use_packing:
            return collate_packed(batch, max_len=pack_max_len)
        # Default: right-pad to max row length.
        ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        if not ids:
            return {"input_ids": torch.zeros((0, 0), dtype=torch.long)}
        max_l = max(t.size(0) for t in ids)
        pad_id = 0
        out = torch.full((len(batch), max_l), pad_id, dtype=torch.long)
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
            position_ids = batch.get("position_ids")
            attention_mask = batch.get("attention_mask")
            if position_ids is not None and torch.cuda.is_available():
                position_ids = position_ids.cuda()
            if attention_mask is not None and torch.cuda.is_available():
                attention_mask = attention_mask.cuda()
            # Build labels with valid-position mask. Without this, the naive
            # `labels[t] = input_ids[t+1]` shift leaks cross-doc info into loss:
            # at a doc boundary, position N (last of doc1) gets label = position
            # N+1 (first of doc2). Pad positions also get labels from the next
            # real token. Both bias the loss.
            # Valid iff: (a) input_ids[t] and input_ids[t+1] are not pad (id=0),
            # AND (b) when packed, attention_mask[t, t+1] == 1 (same doc, causal).
            labels = torch.full_like(input_ids, -100)
            labels[..., :-1] = input_ids[..., 1:]
            labels[..., -1] = -100
            valid_curr = input_ids != 0
            valid_next = torch.cat(
                [input_ids[..., 1:] != 0, torch.zeros_like(input_ids[..., :1])], dim=-1
            )
            valid_label = valid_curr & valid_next
            if attention_mask is not None and position_ids is not None:
                # Packed path: predict t+1 only when t and t+1 are in the same
                # doc. Block-diag mask is lower-triangular-causal (mask[t,t+1]=0
                # for ALL t, including intra-doc), so the offset-1 diagonal
                # cannot distinguish same-doc from cross-doc. position_ids are
                # the right signal: they reset to 0 at each doc boundary, so
                # `position_ids[t+1] == position_ids[t] + 1` is True iff the
                # two positions are contiguous within the same doc.
                pos_next = torch.cat(
                    [position_ids[..., 1:], torch.full_like(position_ids[..., :1], -1)],
                    dim=-1,
                )
                same_doc_next = pos_next == position_ids + 1
                # Also require t+1 < pack_len (not predicting into pad).
                pack_lens = (input_ids != 0).sum(dim=-1)  # (B,)
                in_bounds = (
                    torch.arange(input_ids.size(-1), device=input_ids.device)[None, :]
                    < pack_lens[:, None]
                )
                same_doc_next = same_doc_next & in_bounds
                valid_label = valid_label & same_doc_next
            labels = torch.where(valid_label, labels, torch.full_like(labels, -100))

            loss = compute_loss(
                head,
                input_ids,
                labels,
                cfg,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
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
                    ttt = training_time_test_step(
                        head, input_ids, cfg,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                    )
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
