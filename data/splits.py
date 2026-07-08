"""Stratified train/val/test splits with SHA256 reproducibility log.

The same seed + same input data = the same SHA256 of each split JSONL.
A second run yields identical hashes — verifier for SC4 of Phase 1.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from data.config import SplitConfig, StratifyBy
from data.types import Example


@dataclass
class Split:
    name: str  # "train" | "val" | "test"
    examples: list[Example]


def stratified_split(
    examples: list[Example], config: SplitConfig, seed: int
) -> tuple[list[Example], list[Example], list[Example]]:
    """80/10/10 stratified split (or configured ratios)."""
    stratify_key = (lambda ex: ex.domain) if config.stratify_by == StratifyBy.DOMAIN \
        else (lambda ex: ex.source)
    # First split: train vs (val + test)
    train, vt = train_test_split(
        examples,
        test_size=(1.0 - config.train_ratio),
        stratify=[stratify_key(ex) for ex in examples],
        random_state=seed,
    )
    # Second split: val vs test, proportional to remaining ratios
    vt_ratio_total = config.val_ratio + config.test_ratio
    val_share_of_vt = config.val_ratio / vt_ratio_total
    val, test = train_test_split(
        vt,
        test_size=(1.0 - val_share_of_vt),
        stratify=[stratify_key(ex) for ex in vt],
        random_state=seed,
    )
    return train, val, test


def write_split_jsonl(examples: list[Example], path: Path) -> None:
    """Write split as JSONL (one Example per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            row = {
                "id": ex.id,
                "domain": ex.domain,
                "messages": ex.messages,
                "source": ex.source,
                "meta": ex.meta,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_splits_sha256_log(
    path: Path,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    seed: int,
) -> dict[str, object]:
    """Hash each split JSONL; same seed + same input = identical hashes."""
    hashes = {
        "seed": seed,
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "test_sha256": sha256_file(test_path),
        "train_size": sum(1 for _ in train_path.open()),
        "val_size": sum(1 for _ in val_path.open()),
        "test_size": sum(1 for _ in test_path.open()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)
    return hashes


def domain_counts(examples: list[Example]) -> dict[str, int]:
    return dict(Counter(ex.domain for ex in examples))
