"""Unit tests for pure helper functions in train/train_eagle3.py.

The driver itself depends on a multi-billion-parameter HF model (Qwen3-4B-Instruct-2507) — exercised on rented
GPU. These tests focus on the deterministic helper functions that don't need
GPU or HF model loading.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
import torch

from train.train_eagle3 import (
    build_masked_labels,
    lr_schedule,
    set_seed,
    write_loss_csv,
    write_loss_curve,
)

# ---- set_seed ----------------------------------------------------------------


def test_set_seed_python_random_deterministic() -> None:
    set_seed(7)
    a = random.random()
    set_seed(7)
    b = random.random()
    assert a == b


def test_set_seed_torch_deterministic() -> None:
    set_seed(123)
    a = torch.rand(1).item()
    set_seed(123)
    b = torch.rand(1).item()
    assert a == b


# ---- lr_schedule -------------------------------------------------------------


def _sched_cfg(warmup_steps: int, max_steps: int) -> Any:
    """Build a minimal object with the attrs lr_schedule reads."""
    o = type("O", (), {"warmup_steps": warmup_steps})
    t = type("T", (), {"max_steps": max_steps})
    c = type("C", (), {"optimizer": o(), "training": t()})
    return c()


def test_lr_schedule_warmup_monotone() -> None:
    cfg: Any = _sched_cfg(warmup_steps=10, max_steps=100)
    # Warmup
    assert lr_schedule(0, cfg) == pytest.approx(0.0)
    assert lr_schedule(5, cfg) == pytest.approx(0.5)
    assert lr_schedule(10, cfg) == pytest.approx(1.0)
    # Decay
    assert 0.0 < lr_schedule(50, cfg) < 1.0


def test_lr_schedule_at_max_steps_is_zero() -> None:
    cfg: Any = _sched_cfg(warmup_steps=10, max_steps=100)
    assert lr_schedule(100, cfg) == pytest.approx(0.0, abs=1e-6)


def test_lr_schedule_past_max_steps_clamped() -> None:
    cfg: Any = _sched_cfg(warmup_steps=10, max_steps=100)
    assert lr_schedule(500, cfg) == pytest.approx(0.0, abs=1e-6)


# ---- write_loss_curve / write_loss_csv --------------------------------------


def test_write_loss_curve_writes_json(tmp_path: Path) -> None:
    rows = [{"step": 0, "loss": 1.5, "lr": 1e-4}, {"step": 1, "loss": 1.2, "lr": 9e-5}]
    p = tmp_path / "lc.json"
    write_loss_curve(p, rows)
    loaded = json.loads(p.read_text())
    assert loaded[0]["step"] == 0
    assert loaded[1]["loss"] == 1.2


def test_write_loss_csv_writes_header_and_rows(tmp_path: Path) -> None:
    rows = [{"step": 0, "loss": 1.5, "lr": 1e-4}]
    p = tmp_path / "lc.csv"
    write_loss_csv(p, rows)
    text = p.read_text()
    assert "step,loss,lr" in text
    assert "0,1.500000,0.00010000" in text


def test_write_loss_csv_handles_empty(tmp_path: Path) -> None:
    p = tmp_path / "lc_empty.csv"
    write_loss_csv(p, [])
    assert p.read_text() == "step,loss,lr\n"


# ---- compute_loss smoke ------------------------------------------------------


def test_compute_loss_returns_scalar_on_stub() -> None:
    from train.train_eagle3 import compute_loss

    class _Head:
        def __call__(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return torch.randn(2, 4, 5)

    head: Any = _Head()
    input_ids = torch.randint(0, 5, (2, 4))
    labels = input_ids.clone()
    loss = compute_loss(head, input_ids, labels, cfg=None)  # type: ignore[arg-type]  # type: ignore[arg-type]
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_compute_loss_passes_position_ids_and_attention_mask_to_head() -> None:
    """The packed-training path must propagate position_ids + attention_mask
    to the head call. Without this, block-diagonal masking is computed but
    never reaches the model.
    """
    from train.train_eagle3 import compute_loss

    captured: dict[str, Any] = {}

    class _RecordingHead:
        def __call__(self, **kwargs: Any) -> torch.Tensor:
            captured.update(kwargs)
            # Return logits matching input shape: (B=1, L=5, V=7).
            return torch.randn(kwargs["input_ids"].shape[0], kwargs["input_ids"].shape[1], 7)

    head: Any = _RecordingHead()
    input_ids = torch.randint(0, 7, (1, 5))
    labels = input_ids.clone()
    position_ids = torch.arange(5).unsqueeze(0)
    attention_mask = torch.tril(torch.ones(5, 5, dtype=torch.long)).unsqueeze(0)

    loss = compute_loss(
        head, input_ids, labels, cfg=None,  # type: ignore[arg-type]
        position_ids=position_ids, attention_mask=attention_mask,
    )
    # The head must receive all three inputs the driver promises to forward.
    assert "input_ids" in captured
    assert "position_ids" in captured
    assert "attention_mask" in captured
    torch.testing.assert_close(captured["position_ids"], position_ids)
    torch.testing.assert_close(captured["attention_mask"], attention_mask)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


# ---- build_masked_labels -----------------------------------------------------


def test_build_masked_labels_returns_long_and_no_dtype_error() -> None:
    """Regression: torch.cat([bool, long]) promoted the mask to long, and
    torch.where raised 'expected condition to be a boolean tensor' on the
    first real GPU batch (rung-3 smoke). The call must simply not raise and
    must return integer labels."""
    input_ids = torch.tensor([[5, 6, 7, 0]])
    labels = build_masked_labels(input_ids)
    assert labels.dtype == input_ids.dtype
    # last real token predicts pad -> masked; pad position masked; final -100
    assert labels.tolist() == [[6, 7, -100, -100]]


def test_build_masked_labels_masks_doc_boundary_when_packed() -> None:
    """Packed path: position_ids reset at doc boundaries; the last token of
    doc1 must NOT get doc2's first token as its label."""
    # Two docs packed: [10, 11, 12] + [20, 21], then pad.
    input_ids = torch.tensor([[10, 11, 12, 20, 21, 0]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 0]])
    labels = build_masked_labels(input_ids, position_ids)
    # t=2 (last of doc1) crosses into doc2 -> -100. t=4 predicts pad -> -100.
    assert labels.tolist() == [[11, 12, -100, 21, -100, -100]]


def test_build_masked_labels_unpacked_keeps_intra_sequence_shift() -> None:
    input_ids = torch.tensor([[3, 4, 5, 6]])
    labels = build_masked_labels(input_ids)
    assert labels.tolist() == [[4, 5, 6, -100]]
