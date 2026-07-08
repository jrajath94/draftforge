"""Tests for data/config.py pydantic schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from data.config import (
    DataConfig,
    DedupMethod,
    SourceType,
    StratifyBy,
    load_config,
)


def _base_cfg_dict() -> dict:
    return {
        "seed": 42,
        "output_dir": "/tmp/draftforge-test",
        "sources": [
            {
                "name": "sharegpt",
                "type": "sharegpt",
                "hf_dataset_id": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                "max_examples": 1000,
                "domain": "general",
            }
        ],
        "dedup": {"method": "exact", "minhash_threshold": 0.9, "num_perm": 64},
        "split": {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "stratify_by": "domain",
        },
        "tokenizer": {
            "name_or_path": "Qwen/Qwen3-14B",
            "max_length": 4096,
        },
    }


def test_valid_config_parses() -> None:
    cfg = DataConfig.model_validate(_base_cfg_dict())
    assert cfg.seed == 42
    assert cfg.sources[0].type == SourceType.SHAREGPT
    assert cfg.dedup.method == DedupMethod.EXACT
    assert cfg.split.stratify_by == StratifyBy.DOMAIN


def test_ratios_must_sum_to_one() -> None:
    bad = _base_cfg_dict()
    bad["split"]["test_ratio"] = 0.2  # 0.8 + 0.1 + 0.2 = 1.1
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        DataConfig.model_validate(bad)


def test_source_requires_path_or_id() -> None:
    cfg = _base_cfg_dict()
    cfg["sources"][0].pop("hf_dataset_id")
    cfg["sources"][0]["path"] = "/tmp/x.jsonl"
    ok = DataConfig.model_validate(cfg)  # path is fine
    assert ok.sources[0].path == Path("/tmp/x.jsonl")
    # Now remove both
    bad = _base_cfg_dict()
    bad["sources"][0].pop("hf_dataset_id")
    with pytest.raises(ValidationError, match="hf_dataset_id or path"):
        DataConfig.model_validate(bad)


def test_minhash_threshold_range() -> None:
    bad = _base_cfg_dict()
    bad["dedup"]["minhash_threshold"] = 1.5
    with pytest.raises(ValidationError):
        DataConfig.model_validate(bad)


def test_round_trip_yaml(tmp_path: Path) -> None:
    raw = _base_cfg_dict()
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    cfg = load_config(p)
    assert cfg.seed == 42
    # Write back
    p2 = tmp_path / "cfg2.yaml"
    p2.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    cfg2 = load_config(p2)
    assert cfg2 == cfg


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_at_least_one_source() -> None:
    bad = _base_cfg_dict()
    bad["sources"] = []
    with pytest.raises(ValidationError, match="at least one source"):
        DataConfig.model_validate(bad)
