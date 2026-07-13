"""Tests for `train/layer_indices.py` — EAGLE-3 tap rescaling helper.

These tests pin the rescale behavior that lets DraftForge migrate to a
new target model (Qwen 3.5 / 3.6, Llama 3.x, etc.) by editing one config
field rather than rewriting the head.
"""

from __future__ import annotations

import pytest

from train.layer_indices import DEFAULT_TAPS, layer_indices_for_depth

# ---- reference: published EAGLE-3 recipe ----------------------------------


def test_qwen3_4b_36_layers_matches_recipe() -> None:
    """Qwen3-4B-Instruct-2507 (36 layers) → [7, 18, 29].

    Matches the canonical config in `train/config.yaml`. If this drifts
    the ablation will compare the wrong variant against the baseline.
    """
    assert layer_indices_for_depth(36) == [7, 18, 29]


def test_qwen3_14b_40_layers_matches_paper() -> None:
    """Qwen3-14B (40 layers) → [8, 20, 32]. Original paper recipe."""
    assert layer_indices_for_depth(40) == [8, 20, 32]


# ---- regression: future target model sizes -------------------------------


@pytest.mark.parametrize(
    ("num_layers", "expected"),
    [
        (48, [10, 24, 38]),
        (32, [6, 16, 26]),
        (28, [6, 14, 22]),
        (24, [5, 12, 19]),
        (80, [16, 40, 64]),
        (64, [13, 32, 51]),
    ],
)
def test_rescale_arbitrary_depths(num_layers: int, expected: list[int]) -> None:
    """Rescale must hold for any depth in the typical LLM range (24-80).

    Spot-check that the fractional coverage stays close to 20%/50%/80%
    across the range, never drifting by more than one layer.
    """
    result = layer_indices_for_depth(num_layers)
    assert result == expected
    # The high tap may share an index with a lower tap on tiny targets,
    # so check coverage against the unique sorted list.
    for i, idx in enumerate(result):
        depth = idx / num_layers
        tap = DEFAULT_TAPS[i]
        assert abs(depth - tap) <= 1 / num_layers + 1e-9


# ---- edge cases ----------------------------------------------------------


def test_single_layer_returns_zero() -> None:
    """One layer is degenerate but must not crash; tap collapses to 0."""
    assert layer_indices_for_depth(1) == [0]


def test_two_layers_uses_both() -> None:
    """Two layers: round(0.20*2)=0, round(0.50*2)=1, round(0.80*2)=2 → clamp to 1."""
    # 0.80 * 2 = 1.6 → 2; clamp to L-1=1. Unique: [0, 1].
    assert layer_indices_for_depth(2) == [0, 1]


def test_three_layers_covers_all() -> None:
    """Three layers: 0.20→1, 0.50→2, 0.80→2 → unique [1, 2]."""
    # 0.20 * 3 = 0.6 → 1; 0.50 * 3 = 1.5 → 2; 0.80 * 3 = 2.4 → 2.
    assert layer_indices_for_depth(3) == [1, 2]


def test_duplicate_taps_collapse() -> None:
    """Two fractional taps that round to the same layer collapse."""
    # 0.20 * 10 = 2.0 → 2; 0.25 * 10 = 2.5 → 2 (banker's even). Collapses.
    assert layer_indices_for_depth(10, taps=(0.20, 0.25)) == [2]


def test_taps_sorted_and_unique() -> None:
    """Return is sorted ascending, deduplicated, even for unsorted taps."""
    result = layer_indices_for_depth(40, taps=(0.80, 0.20, 0.50))
    assert result == sorted(set(result))


def test_clamp_at_layer_zero() -> None:
    """A tap at 0.0 must return 0, not -1, even though 0.0 * (L-1) = 0."""
    assert layer_indices_for_depth(36, taps=(0.0,)) == [0]


def test_clamp_below_top_layer() -> None:
    """A tap near 1.0 must clamp to L-1, not L."""
    assert layer_indices_for_depth(36, taps=(0.999,)) == [35]


# ---- validation errors ---------------------------------------------------


def test_zero_layers_raises() -> None:
    with pytest.raises(ValueError, match="num_hidden_layers must be >= 1"):
        layer_indices_for_depth(0)


def test_negative_layers_raises() -> None:
    with pytest.raises(ValueError, match="num_hidden_layers must be >= 1"):
        layer_indices_for_depth(-1)


def test_empty_taps_raises() -> None:
    with pytest.raises(ValueError, match="at least one fractional depth"):
        layer_indices_for_depth(36, taps=())


def test_tap_below_zero_raises() -> None:
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\)"):
        layer_indices_for_depth(36, taps=(-0.1, 0.5))


def test_tap_at_one_raises() -> None:
    """Tap = 1.0 is invalid (excludes the final layer, which is usually
    the LM-head input projection layer and not a hidden state)."""
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\)"):
        layer_indices_for_depth(36, taps=(0.5, 1.0))


# ---- defaults exposed ---------------------------------------------------


def test_default_taps_match_paper() -> None:
    """Sanity: the helper's default taps are the three EAGLE-3 taps."""
    assert DEFAULT_TAPS == (0.20, 0.50, 0.80)
