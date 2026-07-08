"""Tests for ablate/configs.py — preset ablation configurations."""

from __future__ import annotations

from pathlib import Path

import yaml

from ablate.configs import (
    PRESETS,
    overlay_eagle3,
    write_variant_config,
)


def test_presets_defined() -> None:
    assert "tri_layer" in PRESETS
    assert "final_layer" in PRESETS
    assert "low_only" in PRESETS
    assert "mid_only" in PRESETS


def test_tri_layer_has_three_indices() -> None:
    assert len(PRESETS["tri_layer"].layer_indices) == 3


def test_final_layer_has_one_index() -> None:
    assert PRESETS["final_layer"].layer_indices == [32]


def test_overlay_preserves_base() -> None:
    base = {"model": {"name_or_path": "Qwen/Qwen3-14B"}, "training": {"max_steps": 1000}}
    out = overlay_eagle3(base, PRESETS["tri_layer"])
    assert out["model"]["name_or_path"] == "Qwen/Qwen3-14B"
    assert out["training"]["max_steps"] == 1000
    assert out["eagle3"]["layer_indices"] == [8, 20, 32]


def test_write_variant_config(tmp_path: Path) -> None:
    base = {"dataset": {"train_dir": "/tmp/x"}, "output": {"dir": "/tmp/out"}}
    p = tmp_path / "ablate.yaml"
    write_variant_config(base, PRESETS["final_layer"], p)
    assert p.exists()
    loaded = yaml.safe_load(p.read_text())
    assert loaded["eagle3"]["layer_indices"] == [32]
    assert loaded["ablation_name"] == "final_layer"
