"""Tests for train/config.py pydantic schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from train.config import (
    TrainConfig,
    load_config,
)


def _minimal_cfg_dict() -> dict:
    return {
        "dataset": {"train_dir": "/tmp/train_ds"},
        "output": {"dir": "/tmp/out"},
    }


def test_defaults_populate() -> None:
    cfg = TrainConfig.model_validate(_minimal_cfg_dict())
    assert cfg.model.name_or_path == "Qwen/Qwen3-14B"
    assert cfg.eagle3.layer_indices == [8, 20, 32]
    assert cfg.eagle3.num_decoder_layers == 1
    assert cfg.optimizer.lr == 1e-4
    assert cfg.training.bf16 is True


def test_custom_eagle3_layers() -> None:
    raw = _minimal_cfg_dict()
    raw["eagle3"] = {"layer_indices": [4, 12], "num_decoder_layers": 2}
    cfg = TrainConfig.model_validate(raw)
    assert cfg.eagle3.layer_indices == [4, 12]
    assert cfg.eagle3.num_decoder_layers == 2


def test_empty_layer_indices_rejected() -> None:
    raw = _minimal_cfg_dict()
    raw["eagle3"] = {"layer_indices": [], "num_decoder_layers": 1}
    with pytest.raises(Exception, match=r"layer_indices"):
        TrainConfig.model_validate(raw)


def test_negative_layer_index_rejected() -> None:
    raw = _minimal_cfg_dict()
    raw["eagle3"] = {"layer_indices": [-1, 5]}
    with pytest.raises(Exception, match=r"non-negative"):
        TrainConfig.model_validate(raw)


def test_load_yaml_round_trip(tmp_path: Path) -> None:
    raw = _minimal_cfg_dict()
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    cfg = load_config(p)
    assert isinstance(cfg, TrainConfig)
    assert Path(cfg.dataset.train_dir) == Path("/tmp/train_ds")


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_max_step_one_allowed() -> None:
    raw = _minimal_cfg_dict()
    raw["training"] = {"max_steps": 1}
    cfg = TrainConfig.model_validate(raw)
    assert cfg.training.max_steps == 1


def test_max_step_zero_rejected() -> None:
    raw = _minimal_cfg_dict()
    raw["training"] = {"max_steps": 0}
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(raw)
