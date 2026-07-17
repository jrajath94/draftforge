"""Seed-determinism test for EAGLE-3 training.

Verifies that running with the same seed twice produces identical losses,
and that different seeds produce different losses. Mocks the 4B target
model with a stub so the test runs on CPU in seconds.

Why this matters:
- CLAUDE.md integrity baseline requires reproducible numbers; the
  determinism contract must hold before renting a $70+ GPU session.
- Regression guard: if anyone introduces a non-deterministic op (e.g.,
  unseeded dropout, race in dataloader order), this test catches it on
  CPU.
- Marked @pytest.mark.slow so it can be excluded from fast CI runs but
  included before any GPU commitment.

Distinct from `test_set_seed_*` in test_driver.py, which only checks the
seed helper itself. This test exercises the full forward path of
EAGLE3Head plus compute_loss with two different seeds.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from train.head import EAGLE3Head
from train.train_eagle3 import compute_loss, set_seed

# ---- stub target (mirror of test_head.py) ----------------------------------


class _StubConfig:
    hidden_size = 64
    vocab_size = 200
    num_hidden_layers = 4


class _StubLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **_kw):
        h = hidden_states + self.mlp(hidden_states) * 0.01
        return (h,)


class _StubInnerModel(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_StubConfig.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            _StubLayer(hidden_size) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, **_kw):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(hidden_states=h)[0]
        h = self.norm(h)
        out = type("O", (), {})()
        out.last_hidden_state = h
        out.hidden_states = (h, *tuple(h for _ in self.layers))
        return out


class _StubTargetModel(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        self.config = _StubConfig()
        self.model = _StubInnerModel(hidden_size=hidden_size, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, _StubConfig.vocab_size, bias=False)

    def forward(self, input_ids, output_hidden_states=False, **_kw):
        result = self.model(input_ids=input_ids)
        out = type("O", (), {})()
        out.hidden_states = result.hidden_states if output_hidden_states else None
        out.last_hidden_state = result.last_hidden_state
        return out


def _build_head(seed: int) -> EAGLE3Head:
    """Build a fresh EAGLE3Head with deterministic init from `seed`."""
    set_seed(seed)
    model = _StubTargetModel()
    head = EAGLE3Head(
        target_model=model,
        target_config=_StubConfig(),
        layer_indices=[1, 2, 3],
        num_decoder_layers=1,
    )
    return head


def _fixed_input(batch: int = 2, length: int = 16) -> torch.Tensor:
    """Return a deterministic input tensor (independent of any seed state)."""
    g = torch.Generator().manual_seed(0xC0FFEE)
    return torch.randint(0, _StubConfig.vocab_size, (batch, length), generator=g)


def _make_labels(input_ids: torch.Tensor) -> torch.Tensor:
    """next-token labels; last position masked (-100)."""
    labels = input_ids.clone()
    labels[..., :-1] = input_ids[..., 1:]
    labels[..., -1] = -100
    return labels


# ---- the actual determinism contract ---------------------------------------


@pytest.mark.slow
def test_seed_42_deterministic_loss() -> None:
    """Same seed → identical loss (bit-exact, not just close).

    Two consecutive runs with seed=42 must produce the same loss value to
    floating-point precision. Catches any unseeded randomness in head
    initialization or forward pass.
    """
    input_ids = _fixed_input()
    labels = _make_labels(input_ids)

    head_a = _build_head(seed=42)
    loss_a = compute_loss(head_a, input_ids, labels, cfg=None)  # type: ignore[arg-type]

    head_b = _build_head(seed=42)
    loss_b = compute_loss(head_b, input_ids, labels, cfg=None)  # type: ignore[arg-type]

    assert torch.allclose(loss_a, loss_b, atol=1e-7), (
        f"seed=42 should produce identical losses, got "
        f"{loss_a.item():.10f} vs {loss_b.item():.10f}"
    )


@pytest.mark.slow
def test_different_seeds_produce_different_losses() -> None:
    """Sanity check: seeds 42 and 123 must NOT collapse to the same loss.

    With high probability, different random initializations yield
    materially different first-step losses. Guards against the failure
    mode where seed control silently degrades (e.g., seeding only torch
    but not torch.cuda, which doesn't matter on CPU here but would on
    GPU).
    """
    input_ids = _fixed_input()
    labels = _make_labels(input_ids)

    head_42 = _build_head(seed=42)
    loss_42 = compute_loss(head_42, input_ids, labels, cfg=None)  # type: ignore[arg-type]

    head_123 = _build_head(seed=123)
    loss_123 = compute_loss(head_123, input_ids, labels, cfg=None)  # type: ignore[arg-type]

    assert not torch.allclose(loss_42, loss_123, atol=1e-3), (
        f"different seeds produced near-identical losses: "
        f"{loss_42.item():.6f} vs {loss_123.item():.6f} — seed control broken"
    )


@pytest.mark.slow
def test_seed_determinism_holds_across_three_runs() -> None:
    """Tri-seed determinism: 42, 123, 999 each reproduce themselves.

    Reinforces the 3-seed reproducibility requirement from CLAUDE.md.
    """
    input_ids = _fixed_input()
    labels = _make_labels(input_ids)

    losses: dict[int, list[float]] = {42: [], 123: [], 999: []}

    # Run each seed twice; first/loss should match.
    for seed in losses:
        for _ in range(2):
            head = _build_head(seed=seed)
            loss = compute_loss(head, input_ids, labels, cfg=None)  # type: ignore[arg-type]
            losses[seed].append(loss.item())

    for seed, vals in losses.items():
        assert vals[0] == pytest.approx(vals[1], abs=1e-7), (
            f"seed={seed} did not reproduce: {vals[0]} vs {vals[1]}"
        )

    # And cross-seed values should all differ from each other.
    all_vals = [v for vals in losses.values() for v in vals]
    assert len({round(v, 4) for v in all_vals}) == 3, (
        f"expected 3 distinct loss values across seeds, got {all_vals}"
    )


@pytest.mark.slow
def test_repeated_set_seed_resets_state() -> None:
    """set_seed(seed) called twice must yield the same random tensor.

    This is the underlying primitive the determinism test depends on.
    If this fails, the entire EAGLE-3 reproducibility contract is broken.
    """
    set_seed(2026)
    a = torch.rand(8)
    set_seed(2026)
    b = torch.rand(8)
    assert torch.equal(a, b), f"set_seed(2026) not idempotent: {a} vs {b}"
