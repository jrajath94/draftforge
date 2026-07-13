"""Tests for train/packing.py — sequence bin packing for EAGLE-3 training.

Packing combines short token sequences into ≤max_len bins with block-diagonal
attention masking so loss is computed as if each sequence were independent.
This is first-fit-decreasing bin packing (Coffman '96).

These tests pin the invariants that prevent a pack-bug regression from
silently biasing training loss. They run in <1s on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from train.packing import BinPacker, Pack, pack_sequences


# ---- empty / degenerate ------------------------------------------------------


def test_pack_empty_input_returns_empty_list() -> None:
    """No sequences → no packs (nothing to pack)."""
    assert pack_sequences([], max_len=128) == []


def test_pack_skips_empty_sequences() -> None:
    """A zero-length sequence is silently dropped (otherwise pad-to-max wastes budget)."""
    out = pack_sequences([[], [1, 2, 3], []], max_len=128)
    assert len(out) == 1
    assert out[0].input_ids.tolist() == [1, 2, 3]


def test_pack_max_len_too_small_raises() -> None:
    """max_len < 1 → ValueError. No silent degenerate pack."""
    with pytest.raises(ValueError, match=r"max_len"):
        pack_sequences([[1, 2, 3]], max_len=0)


# ---- perfect fit (single sequence fits exactly) ------------------------------


def test_pack_single_sequence_of_exact_max_len() -> None:
    """1 seq, len==max_len → 1 pack, exact capacity."""
    seq = list(range(100))
    out = pack_sequences([seq], max_len=100)
    assert len(out) == 1
    assert len(out[0].input_ids) == 100
    assert out[0].input_ids.tolist() == seq


def test_pack_single_sequence_smaller_than_max_len() -> None:
    """1 seq shorter than max_len → 1 pack, < max_len."""
    seq = [42, 7, 99]
    out = pack_sequences([seq], max_len=4096)
    assert len(out) == 1
    assert out[0].input_ids.tolist() == seq
    # Position IDs: reset to 0 at start of (single) doc
    assert out[0].position_ids.tolist() == [0, 1, 2]


# ---- capacity invariant (the bug-killer) ------------------------------------


def test_pack_respects_max_len_capacity() -> None:
    """Every pack has token count ≤ max_len. This is THE invariant.

    A regressed packing that exceeds max_len would OOM the activation buffer
    on real training and either crash or silently truncate.
    """
    seqs = [[1] * 100 for _ in range(20)]  # 20 × 100-token seqs
    out = pack_sequences(seqs, max_len=256)
    for i, pack in enumerate(out):
        assert len(pack.input_ids) <= 256, (
            f"pack {i} has {len(pack.input_ids)} tokens, exceeds max_len=256"
        )


def test_pack_does_not_truncate_sequences() -> None:
    """Every input seq's tokens appear in some pack (no silent dropping)."""
    seqs = [
        [1, 2, 3, 4, 5],
        [100, 200],
        [99, 88, 77, 66, 55, 44],
    ]
    out = pack_sequences(seqs, max_len=8)
    # Concatenate to reconstruct (order may be rearranged by sorting)
    seen: list[int] = []
    for pack in out:
        seen.extend(pack.input_ids.tolist())
    assert sorted(seen) == sorted([t for s in seqs for t in s])


# ---- block-diagonal attention mask ------------------------------------------


def test_pack_attention_mask_is_block_diagonal_single_doc() -> None:
    """1-doc pack → mask is fully 1 (no cross-doc concern)."""
    out = pack_sequences([[1, 2, 3]], max_len=8)
    mask = out[0].attention_mask
    assert mask.shape == (3, 3)
    # Causal: lower-triangular with 1s (post-causal convention; the head may
    # mask upper triangle separately, but the pack-mask is in here).
    for i in range(3):
        for j in range(3):
            if j <= i:
                assert mask[i, j] == 1
            else:
                assert mask[i, j] == 0


def test_pack_attention_mask_blocks_cross_doc_attention() -> None:
    """2 docs in 1 pack → mask zeros out cross-doc cells.

    The killer invariant: if cross-doc attention leaks, packed training loss
    is biased by token ordering across docs and benchmarks become untrustworthy.
    """
    doc1 = list(range(10))       # tokens 0..9
    doc2 = list(range(100, 110)) # tokens 100..109
    out = pack_sequences([doc1, doc2], max_len=30)
    assert len(out) == 1
    mask = out[0].attention_mask
    assert mask.shape == (20, 20)
    # doc1 occupies rows 0..9, doc2 occupies rows 10..19
    # Cross-doc cells (doc1 row, doc2 col OR doc2 row, doc1 col) MUST be 0.
    for i in range(10):
        for j in range(10, 20):
            assert mask[i, j] == 0, (
                f"doc1 row {i} attends to doc2 col {j} — block-diag invariant violated"
            )
    for i in range(10, 20):
        for j in range(10):
            assert mask[i, j] == 0, (
                f"doc2 row {i} attends to doc1 col {j} — block-diag invariant violated"
            )


# ---- position IDs ------------------------------------------------------------


def test_pack_position_ids_reset_per_doc() -> None:
    """Multi-doc pack: position_ids start at 0 at each doc boundary.

    Without per-doc reset, the rotary embedding (RoPE) extrapolates beyond the
    model's trained max-position length, hurting attention quality.

    NOTE: FFD reorders by descending length, so doc boundaries in the pack
    correspond to PACK ORDER (longest first), not input order.
    """
    # doc_A is shorter (50 tokens); doc_B is longer (60 tokens).
    # FFD puts doc_B first; doc_A second.
    doc_A = list(range(50))
    doc_B = list(range(100, 160))  # 60 tokens
    out = pack_sequences([doc_A, doc_B], max_len=200)
    assert len(out) == 1
    starts = out[0].doc_starts
    pos = out[0].position_ids
    assert len(starts) == 2
    assert starts[0] == 0
    # Each doc boundary must reset position to 0.
    for i, s in enumerate(starts):
        assert pos[s] == 0, f"doc {i} start {s} should reset pos to 0; got {pos[s]}"
    # And each doc has monotonic positions ending at len-1.
    for i in range(len(starts)):
        end = starts[i + 1] if i + 1 < len(starts) else len(pos)
        assert pos[end - 1] == end - starts[i] - 1


# ---- first-fit-decreasing ordering ------------------------------------------


def test_pack_first_fit_decreasing_sorts_descending_by_length() -> None:
    """The packer sorts seqs by descending length before placing.

    This is the standard FFD heuristic (better bin-fill than unsorted first-fit).
    Longest-first guarantees a tighter pack on most inputs.
    """
    packer = BinPacker(max_len=100)
    seqs = [[1] * 30, [2] * 80, [3] * 50]
    packer._sort_descending(seqs)  # type: ignore[attr-defined]
    # After sort: 80, 50, 30
    assert len(seqs[0]) == 80
    assert len(seqs[1]) == 50
    assert len(seqs[2]) == 30


# ---- doc_starts / first-doc-starts invariants -------------------------------


def test_pack_doc_starts_are_zero_indexed_in_order() -> None:
    """doc_starts[0] == 0; each starts[i+1] - starts[i] equals the i-th doc's length IN PACK ORDER.

    FFD sorts by descending length, so the in-pack ordering is not the same as
    input ordering. Lengths must be measured against the pack, not the input.
    """
    out = pack_sequences([[1, 2], [3, 4, 5], [6]], max_len=10)
    assert len(out) == 1
    starts = out[0].doc_starts
    # FFD sort by length desc → pack order = [3,4,5]=3 tokens, [1,2]=2, [6]=1.
    lens_in_pack_order = [3, 2, 1]
    assert starts[0] == 0
    assert starts == sorted(starts)
    assert len(starts) == 3
    for i, s in enumerate(starts):
        if i + 1 < len(starts):
            assert (starts[i + 1] - s) == lens_in_pack_order[i]
        else:
            assert (len(out[0].input_ids) - s) == lens_in_pack_order[i]


# ---- multiple sequences: bin count ------------------------------------------


def test_pack_multiple_bins_when_capacity_exhausted() -> None:
    """Many short seqs that can't all fit → multiple bins."""
    seqs = [[1] * 100 for _ in range(10)]  # 10 × 100-token seqs, max=256
    out = pack_sequences(seqs, max_len=256)
    # Each pack holds ≤ 2 seqs (256/100 floor = 2). 10 seqs → ≥ 5 packs.
    assert len(out) >= 5
    # Every pack capacity check
    for p in out:
        assert len(p.input_ids) <= 256


# ---- total-tokens preserved -------------------------------------------------


def test_pack_total_token_count_preserved() -> None:
    """Sum of pack token counts == sum of input sequence token counts."""
    seqs = [[1] * 7, [2] * 13, [3] * 5, [4] * 22, [5] * 9]
    out = pack_sequences(seqs, max_len=20)
    packed = sum(len(p.input_ids) for p in out)
    assert packed == sum(len(s) for s in seqs)


# ---- determinism (same input → same output) --------------------------------


def test_pack_deterministic_for_same_input() -> None:
    """Same input seqs (same order) → same output packs. No randomness.

    First-fit-decreasing is fully deterministic; this test pins that contract
    so downstream test fixtures can rely on reproducible packing artifacts.
    """
    seqs = [[1] * 30, [2] * 90, [3] * 25, [4] * 60, [5] * 40]
    a = pack_sequences(seqs, max_len=100)
    b = pack_sequences(seqs, max_len=100)
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        assert pa.input_ids.tolist() == pb.input_ids.tolist()
        assert pa.doc_starts == pb.doc_starts


# ---- Pack dataclass surface -------------------------------------------------


def test_pack_dataclass_exposes_required_arrays() -> None:
    """Pack must carry input_ids / position_ids / attention_mask / doc_starts.

    These are the four artifacts the training driver consumes. Missing any
    breaks the contract.
    """
    p = Pack(
        input_ids=np.array([1, 2, 3], dtype=np.int64),
        position_ids=np.array([0, 1, 2], dtype=np.int64),
        attention_mask=np.ones((3, 3), dtype=np.int64),
        doc_starts=[0],
    )
    assert p.input_ids.dtype == np.int64
    assert p.position_ids.dtype == np.int64
    assert p.attention_mask.dtype == np.int64
    assert isinstance(p.doc_starts, list)
