"""Pydantic config schema for DraftForge data pipeline.

Loaded from `data/config.yaml` at startup. Validates:
  - split ratios sum to 1.0
  - source type ∈ {sharegpt, openhermes, finance, edgar}
  - dedup method ∈ {exact, minhash, exact+minhash}
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class SourceType(StrEnum):
    SHAREGPT = "sharegpt"
    OPENHERMES = "openhermes"
    FINANCE = "finance"
    EDGAR = "edgar"


class DedupMethod(StrEnum):
    EXACT = "exact"
    MINHASH = "minhash"
    EXACT_PLUS_MINHASH = "exact+minhash"


class StratifyBy(StrEnum):
    DOMAIN = "domain"
    SOURCE = "source"


class SourceConfig(BaseModel):
    name: str
    type: SourceType
    hf_dataset_id: str | None = None
    path: Path | None = None
    max_examples: int = Field(default=100_000, ge=1)
    domain: str = "general"
    # EDGAR-specific: list of SEC CIKs (10-digit zero-padded). Ignored for non-EDGAR.
    ciks: list[str] | None = None
    # EDGAR-specific: User-Agent contact string. SEC fair-access policy requires one.
    user_agent: str = "DraftForge/0.1 (research; contact@example.com)"

    @model_validator(mode="after")
    def _check_path_or_id(self) -> SourceConfig:
        # EDGAR sources don't need hf_dataset_id or path (use DEFAULT_CIKS or
        # override via the `ciks` field; pass `path` to read from a JSONL cache).
        if self.type == SourceType.EDGAR:
            return self
        if self.hf_dataset_id is None and self.path is None:
            raise ValueError(
                f"source {self.name!r}: must specify hf_dataset_id or path"
            )
        return self


class DedupConfig(BaseModel):
    method: DedupMethod = DedupMethod.EXACT_PLUS_MINHASH
    minhash_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    num_perm: int = Field(default=128, ge=16, le=2048)


class SplitConfig(BaseModel):
    train_ratio: float = Field(ge=0.0, le=1.0)
    val_ratio: float = Field(ge=0.0, le=1.0)
    test_ratio: float = Field(ge=0.0, le=1.0)
    stratify_by: StratifyBy = StratifyBy.DOMAIN

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> SplitConfig:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split ratios must sum to 1.0 (got {total:.4f})"
            )
        return self


class TokenizerConfig(BaseModel):
    name_or_path: str = "Qwen/Qwen3-14B"
    max_length: int = Field(default=4096, ge=64, le=32768)
    chat_template: str | None = None  # None = tokenizer default
    trust_remote_code: bool = False


class DataConfig(BaseModel):
    seed: int = 42
    output_dir: Path
    sources: list[SourceConfig]
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    split: SplitConfig
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)

    @field_validator("sources")
    @classmethod
    def _at_least_one_source(cls, v: list[SourceConfig]) -> list[SourceConfig]:
        if len(v) < 1:
            raise ValueError("at least one source required")
        return v


def load_config(path: Path | str) -> DataConfig:
    """Load and validate DataConfig from YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    return DataConfig.model_validate(raw)
