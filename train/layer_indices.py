"""Compute EAGLE-3 layer tap indices for a target of any depth.

EAGLE-3 (Li et al., NeurIPS 2025) taps hidden states at three layers of the
target: about 20% (syntactic), 50% (semantic), 80% (task-specific) of the
target's depth. The published recipe uses [8, 20, 32] for a 40-layer
backbone. To migrate to a target with a different layer count, rescale
each tap to the nearest layer at the same fractional depth.

Use this helper when adding a new target model (for example, a Qwen 3.5
or Qwen 3.6 release). The function lives here, not in `train/config.py`,
so other modules (ablate configs, the HF card renderer, the demo
orchestrator) can call it without pulling in the full TrainConfig.
"""

from __future__ import annotations

from collections.abc import Sequence

# Fractional depths of the three taps. 20% / 50% / 80% match the EAGLE-3
# paper for ~36-40 layer backbones.
DEFAULT_TAPS: tuple[float, ...] = (0.20, 0.50, 0.80)


def layer_indices_for_depth(
    num_hidden_layers: int,
    taps: Sequence[float] = DEFAULT_TAPS,
) -> list[int]:
    """Map a fractional tap list onto an integer layer-index list.

    Args:
        num_hidden_layers: Total layer count of the target model
            (callers may read this from `config.num_hidden_layers`,
            the model's `config.json`, or HF's `num_hidden_layers` attr).
        taps: Fractional depths in [0.0, 1.0). Default is the three EAGLE-3
            recipe taps at 20%, 50%, 80%.

    Returns:
        Sorted, deduplicated integer layer indices, each in
        `[0, num_hidden_layers - 1]`. Indices are clamped and rounded to
        the nearest integer. Duplicate taps collapse to one index.

    Raises:
        ValueError: If `num_hidden_layers < 1`, if any tap is outside
            `[0.0, 1.0)`, or if `taps` is empty.

    Examples:
        >>> layer_indices_for_depth(36)
        [7, 18, 29]
        >>> layer_indices_for_depth(40)
        [8, 20, 32]
        >>> layer_indices_for_depth(48)
        [10, 24, 38]
        >>> layer_indices_for_depth(3)
        [1, 2, 2]
    """
    if num_hidden_layers < 1:
        raise ValueError(
            f"num_hidden_layers must be >= 1, got {num_hidden_layers}"
        )
    if not taps:
        raise ValueError("taps must contain at least one fractional depth")
    if any(t < 0.0 or t >= 1.0 for t in taps):
        raise ValueError(
            f"each tap must be in [0.0, 1.0); got {list(taps)}"
        )

    # The EAGLE-3 recipe encodes each tap as a fraction of L (the total
    # layer count), not L-1 (the last index). For Qwen3-4B-Instruct-2507:
    #   round(0.20 * 36) = 7,  round(0.50 * 36) = 18,  round(0.80 * 36) = 29.
    # Using L-1 instead would give [7, 18, 28] and shift the high tap by
    # one layer — a silent recipe drift that the ablation would inherit.
    indices: set[int] = set()
    for t in taps:
        raw = t * num_hidden_layers
        idx = round(raw)
        idx = max(0, min(num_hidden_layers - 1, idx))
        indices.add(idx)
    return sorted(indices)
