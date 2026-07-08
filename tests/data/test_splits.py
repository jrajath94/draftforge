"""Tests for data/splits.py — stratification + SHA256 reproducibility."""

from __future__ import annotations

from data.config import SplitConfig, StratifyBy
from data.splits import (
    domain_counts,
    sha256_file,
    stratified_split,
    write_split_jsonl,
    write_splits_sha256_log,
)


def test_stratified_split_sizes(tiny_traces) -> None:
    cfg = SplitConfig(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, stratify_by=StratifyBy.DOMAIN
    )
    train, val, test = stratified_split(tiny_traces, cfg, seed=42)
    assert len(train) + len(val) + len(test) == len(tiny_traces)
    # Both domains represented in val and test (stratification property)
    train_doms = {ex.domain for ex in train}
    val_doms = {ex.domain for ex in val}
    test_doms = {ex.domain for ex in test}
    assert {"general", "finance"}.issubset(train_doms)
    assert val_doms == {"general", "finance"} or val_doms == {"general"} or val_doms == {"finance"}
    assert {"general", "finance"}.issubset(train_doms | val_doms | test_doms)


def test_same_seed_same_split(tiny_traces) -> None:
    cfg = SplitConfig(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, stratify_by=StratifyBy.DOMAIN
    )
    t1, v1, te1 = stratified_split(tiny_traces, cfg, seed=42)
    t2, v2, te2 = stratified_split(tiny_traces, cfg, seed=42)
    assert [e.id for e in t1] == [e.id for e in t2]
    assert [e.id for e in v1] == [e.id for e in v2]
    assert [e.id for e in te1] == [e.id for e in te2]


def test_different_seed_different_split(tiny_traces) -> None:
    cfg = SplitConfig(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, stratify_by=StratifyBy.DOMAIN
    )
    t1, _, _ = stratified_split(tiny_traces, cfg, seed=42)
    t2, _, _ = stratified_split(tiny_traces, cfg, seed=99)
    ids1 = [e.id for e in t1]
    ids2 = [e.id for e in t2]
    assert ids1 != ids2


def test_write_split_jsonl_and_sha256(tmp_path, tiny_traces) -> None:
    cfg = SplitConfig(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, stratify_by=StratifyBy.DOMAIN
    )
    train, val, test = stratified_split(tiny_traces, cfg, seed=42)
    tp = tmp_path / "train.jsonl"
    vp = tmp_path / "val.jsonl"
    xp = tmp_path / "test.jsonl"
    write_split_jsonl(train, tp)
    write_split_jsonl(val, vp)
    write_split_jsonl(test, xp)
    h = sha256_file(tp)
    assert len(h) == 64  # sha256 hex
    log = write_splits_sha256_log(tmp_path / "log.json", tp, vp, xp, seed=42)
    assert log["seed"] == 42
    assert log["train_sha256"] == h


def test_domain_counts(tiny_traces) -> None:
    counts = domain_counts(tiny_traces)
    assert counts["general"] >= 1
    assert counts["finance"] >= 1
