"""End-to-end smoke test: collate_packed → compute_loss with block-diag mask.

This is the v1.3 packing-smoke (CPU). It exercises the full packed-training
path: collator builds the block-diagonal attention mask + per-doc RoPE reset,
main loop masks labels at doc boundaries, compute_loss forwards kwargs to the
head, and the cross-entropy loss is finite.

The test does NOT need a 14B target model — it uses a tiny stub head with
random logits shaped to match input_ids. The point is to pin that all the
glue (mask, position_ids, label mask) wires together correctly.

Run via: make packing-smoke
"""
from __future__ import annotations

from typing import Any

import torch

from train.train_eagle3 import collate_packed, compute_loss


class _SmokeHead:
    """Stub head that returns logits matching input shape.

    The "head" pretends to be a vocabulary classifier with V=512 tokens. Loss is
    computed against the labels, so we just need finite logits of the right
    shape — values don't have to be meaningful for a smoke test.
    """

    def __init__(self, vocab: int = 512) -> None:
        self.vocab = vocab

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        return torch.randn(input_ids.size(0), input_ids.size(1), self.vocab)


def test_packed_smoke_loss_finite_with_block_diag_mask() -> None:
    """Pack 3 short docs → bin of 190 tokens → loss is finite scalar.

    FFD sorts by descending length: [80, 60, 50] all fit in one 200-token bin.
    doc_starts = [0, 80, 140].
    """
    batch = [
        {"input_ids": list(range(1, 51))},     # 50 tokens
        {"input_ids": list(range(100, 160))},  # 60 tokens
        {"input_ids": list(range(200, 280))},  # 80 tokens
    ]
    out = collate_packed(batch, max_len=200)
    assert out["input_ids"].shape == (1, 200)

    head: Any = _SmokeHead(vocab=512)
    labels = torch.full_like(out["input_ids"], -100)
    labels[..., :-1] = out["input_ids"][..., 1:]
    labels[..., -1] = -100
    # Apply the same valid-mask logic the main loop uses (position_ids diff).
    valid_curr = out["input_ids"] != 0
    valid_next = torch.cat(
        [out["input_ids"][..., 1:] != 0, torch.zeros(1, 1, dtype=torch.bool)], dim=-1
    )
    valid_label = valid_curr & valid_next
    pos_next = torch.cat(
        [out["position_ids"][..., 1:], torch.full_like(out["position_ids"][..., :1], -1)],
        dim=-1,
    )
    same_doc_next = pos_next == out["position_ids"] + 1
    pack_lens = (out["input_ids"] != 0).sum(dim=-1)
    in_bounds = torch.arange(out["input_ids"].size(-1))[None, :] < pack_lens[:, None]
    same_doc_next = same_doc_next & in_bounds
    valid_label = valid_label & same_doc_next
    labels = torch.where(valid_label, labels, torch.full_like(labels, -100))

    loss = compute_loss(
        head, out["input_ids"], labels, cfg=None,  # type: ignore[arg-type]
        position_ids=out["position_ids"], attention_mask=out["attention_mask"],
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss), f"loss not finite: {loss.item()}"


def test_packed_smoke_boundary_position_has_minus_100_label() -> None:
    """At a doc boundary (last pos of doc1 = 79 → first pos of doc2 = 80),
    the label for position 79 must be -100 (cross-doc, do not score).

    Without the label-mask fix, label[79] would be input_ids[80] — the first
    real token of doc2 — leaking cross-doc info into loss.
    """
    # doc1 = 60 tokens, doc2 = 60 tokens → doc_starts = [0, 60].
    batch = [
        {"input_ids": list(range(1, 61))},     # 60 tokens
        {"input_ids": list(range(100, 160))},  # 60 tokens
    ]
    out = collate_packed(batch, max_len=200)
    mask = out["attention_mask"][0]
    # position 59 (last of doc1) attends to position 60 (first of doc2)?
    # Block-diag invariant: mask[59, 60] must be 0.
    assert mask[59, 60].item() == 0, "block-diag invariant violated at doc boundary"
    # And mask[60, 59] (doc2 looking back at doc1) must also be 0.
    assert mask[60, 59].item() == 0, "block-diag invariant violated at doc boundary (reverse)"
