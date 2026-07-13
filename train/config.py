"""Pydantic training-config schema for DraftForge EAGLE-3 training.

Loaded from `train/config.yaml`. Validates tri-layer indices and
optimizer hyperparams at startup so failures are caught before GPU
allocation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    name_or_path: str = "Qwen/Qwen3-14B"
    torch_dtype: str = "bfloat16"
    attn_impl: str = "sdpa"  # sdpa | flash_attention_2


class DatasetConfig(BaseModel):
    train_dir: Path  # output of Phase 1 (tokenized/ train split)
    val_dir: Path | None = None
    max_length: int = Field(default=4096, ge=64, le=32768)
    text_field: str = "text"  # HF Dataset column name


class Eagle3Config(BaseModel):
    layer_indices: list[int] = Field(default_factory=lambda: [8, 20, 32])
    num_decoder_layers: int = Field(default=1, ge=1, le=12)
    head_dim: int | None = None  # None = target hidden_size

    @field_validator("layer_indices")
    @classmethod
    def _at_least_one(cls, v: list[int]) -> list[int]:
        if len(v) < 1:
            raise ValueError("layer_indices must contain at least one layer")
        if any(i < 0 for i in v):
            raise ValueError("layer_indices must be non-negative")
        return v

    @property
    def fusion_size(self) -> int:
        """EAGLE-3 concatenates features along the hidden dim."""
        return len(self.layer_indices)


class OptimizerConfig(BaseModel):
    name: str = "adamw"
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_steps: int = 100


class TrainingConfig(BaseModel):
    seed: int = 42
    max_steps: int = Field(default=2000, ge=1)
    per_device_batch_size: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=8, ge=1)
    log_every: int = 10
    eval_every: int = 200
    save_every: int = 500
    bf16: bool = True
    gradient_checkpointing: bool = True
    training_time_test_every: int = 100  # K steps between conditional rollouts
    training_time_test_horizon: int = 5
    # v1.3: sequence packing (cost-reduction lever 2).
    # When True, the collator packs short sequences into <=max_len bins with
    # block-diagonal attention masks. ~3-7x throughput on finance traces
    # where median doc length is 50-150 tokens vs max_len=4096.
    sequence_pack: bool = False
    sequence_pack_max_len: int = Field(default=4096, ge=128, le=32768)


class DeepSpeedConfig(BaseModel):
    config_path: Path = Path("train/ds_config.json")
    num_gpus: int = 1
    mixed_precision: str = "bf16"


class OutputConfig(BaseModel):
    dir: Path
    save_every: int = 500
    keep_last_n_checkpoints: int = 2


class TrainConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig
    eagle3: Eagle3Config = Field(default_factory=Eagle3Config)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    ds: DeepSpeedConfig = Field(default_factory=DeepSpeedConfig)
    output: OutputConfig


def load_config(path: Path | str) -> TrainConfig:
    """Load and validate TrainConfig from YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"train config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    return TrainConfig.model_validate(raw)


def save_config(cfg: TrainConfig, path: Path | str) -> None:
    """Dump TrainConfig to YAML (helpers for traceable runs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)
