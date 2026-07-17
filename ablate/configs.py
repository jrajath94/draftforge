"""Ablation configuration presets.

Each preset is a self-contained TrainConfig override that differs from the
baseline in exactly one design dimension. The ablation experiment varies
EAGLE3.layer_indices to test whether tri-layer fusion beats single-layer.

Hypothesis (per Phase 3 ROADMAP):
- tri-layer [7, 18, 29] (low + mid + high) > final-layer-only [35]
- low-layer only [7] < mid [18] (low captures structural; mid captures task)

Variance across ≥3 seeds reported in the writeup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class AblationConfig(BaseModel):
    name: str        # "tri_layer" | "final_layer" | "low_only" | "mid_only"
    layer_indices: list[int]
    rationale: str


PRESETS: dict[str, AblationConfig] = {
    "tri_layer": AblationConfig(
        name="tri_layer",
        layer_indices=[7, 18, 29],
        rationale="low + mid + high feature fusion (EAGLE-3 reference design, rescaled for 36 layers)",
    ),
    "final_layer": AblationConfig(
        name="final_layer",
        layer_indices=[35],
        rationale="single final-layer tap; ablation baseline",
    ),
    "low_only": AblationConfig(
        name="low_only",
        layer_indices=[7],
        rationale="early-layer features only",
    ),
    "mid_only": AblationConfig(
        name="mid_only",
        layer_indices=[18],
        rationale="mid-layer features only",
    ),
}


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load base TrainConfig YAML; not validated (we only override eagle3)."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def overlay_eagle3(
    base: dict[str, Any], preset: AblationConfig
) -> dict[str, Any]:
    """Return a new dict with eagle3.layer_indices + name replaced."""
    out = {**base}
    out["eagle3"] = {
        **base.get("eagle3", {}),
        "layer_indices": list(preset.layer_indices),
    }
    out["ablation_name"] = preset.name
    return out


def write_variant_config(
    base: dict[str, Any], preset: AblationConfig, out_path: Path
) -> None:
    """Materialize a per-variant config YAML for the training driver."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = overlay_eagle3(base, preset)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
