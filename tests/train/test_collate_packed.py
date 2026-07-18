"""Tests for collate_packed — the v1.3 sequence-packing collator.

These exercise the production wiring (not just the packing primitives in
train/packing.py which have their own test suite).

Test-data note: synthetic token ranges start from 1, not 0. The collate
uses pad_id=0 (inherited from v1.2 baseline), so token 0 would be
indistinguishable from pad. Using `range(1, N+1)` makes occupancy checks
robust without changing the production code. The pad-id-vs-token-0
collision is a pre-existing v1.2 issue, out of scope for v1.3 packing.
"""
from __future__ import annotations

import torch

from train.train_eagle3 import collate_packed


def test_collate_packed_short_docs_combine_into_one_bin() -> None:
    """3 short docs (50, 60, 80 tokens) → 1 bin of 190 tokens (≤ max_len=200).

    FFD sorts descending before packing: [80, 60, 50] all fit into bin 1,
    so doc_starts is [0, 80, 140] (offsets into the packed sequence),
    NOT [0, 50, 110] (which would assume insertion order).
    """
    batch = [
        {"input_ids": list(range(1, 51))},          # 50 tokens
        {"input_ids": list(range(100, 160))},      # 60 tokens
        {"input_ids": list(range(200, 280))},      # 80 tokens
    ]
    out = collate_packed(batch, max_len=200)
    assert out["input_ids"].shape == (1, 200)
    # Total non-pad tokens = 50 + 60 + 80 = 190 (FFD preserves total).
    assert (out["input_ids"] != 0).sum().item() == 190
    assert len(out["doc_starts"]) == 1
    assert out["doc_starts"][0] == [0, 80, 140]


def test_collate_packed_emits_block_diagonal_attention_mask() -> None:
    """The 2-D mask zeros out cross-doc cells. No attention leak across docs.

    FFD order [60, 50] → doc1 occupies rows 0..59, doc2 occupies rows 60..109.
    Cross-doc cells (doc1-row x doc2-col, doc2-row x doc1-col) must be 0.
    """
    batch = [
        {"input_ids": list(range(1, 51))},         # 50 tokens
        {"input_ids": list(range(100, 160))},      # 60 tokens
    ]
    out = collate_packed(batch, max_len=200)
    mask = out["attention_mask"][0, 0]
    # doc1 = rows [0:60] (60 tokens, placed first by FFD), doc2 = rows [60:110].
    cross_top_right = mask[0:60, 60:110].sum()
    cross_bottom_left = mask[60:110, 0:60].sum()
    assert cross_top_right.item() == 0, (
        f"cross-doc leak top-right: {cross_top_right.item()} non-zero cells"
    )
    assert cross_bottom_left.item() == 0, (
        f"cross-doc leak bottom-left: {cross_bottom_left.item()} non-zero cells"
    )


def test_collate_packed_resets_position_ids_per_doc() -> None:
    """Per-doc RoPE reset: position_ids[i] == 0 at each doc_starts[i].

    FFD order [60, 50] → doc_starts = [0, 60]. position_ids must be 0 at
    row 0 (start of doc1) and at row 60 (start of doc2).
    """
    batch = [
        {"input_ids": list(range(1, 51))},         # 50 tokens
        {"input_ids": list(range(100, 160))},      # 60 tokens
    ]
    out = collate_packed(batch, max_len=200)
    pos = out["position_ids"][0]
    starts = out["doc_starts"][0]
    assert starts == [0, 60], f"expected FFD-ordered starts, got {starts}"
    for s in starts:
        assert pos[s].item() == 0, f"position reset invariant violated at {s}"


def test_collate_packed_empty_batch_returns_empty_tensors() -> None:
    """Empty input → empty (0, max_len) tensors, no crash."""
    out = collate_packed([], max_len=128)
    assert out["input_ids"].shape == (0, 128)
    assert out["doc_starts"] == []


def test_collate_packed_skips_empty_rows() -> None:
    """Rows with empty input_ids are dropped silently (no wasted pack budget)."""
    batch = [
        {"input_ids": []},
        {"input_ids": list(range(1, 11))},         # 10 tokens, no zero
        {"input_ids": []},
    ]
    out = collate_packed(batch, max_len=128)
    assert out["input_ids"].shape == (1, 128)
    assert (out["input_ids"] != 0).sum().item() == 10


def test_collate_packed_capacity_invariant() -> None:
    """No pack exceeds max_len — would OOM the activation buffer otherwise."""
    batch = [{"input_ids": list(range(1, 101))} for _ in range(20)]
    out = collate_packed(batch, max_len=256)
    for _i, pack_ids in enumerate(out["input_ids"]):
        assert (pack_ids != 0).sum().item() <= 256


def test_collate_packed_returns_torch_long_tensors() -> None:
    """input_ids/position_ids are torch.long; the mask is a 4-D bool custom
    mask (B, 1, L, L) — the layout HF transformers honors as-is (a 3-D long
    mask was mistaken for a padding mask and unsqueezed to 5-D on GPU)."""
    batch = [{"input_ids": list(range(1, 21))}]
    out = collate_packed(batch, max_len=64)
    assert out["input_ids"].dtype == torch.long
    assert out["position_ids"].dtype == torch.long
    assert out["attention_mask"].dtype == torch.bool
    assert out["attention_mask"].shape == (1, 1, 64, 64)
