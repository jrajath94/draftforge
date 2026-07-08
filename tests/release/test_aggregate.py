"""Tests for release/aggregate.py — manifest collection across phases."""

from __future__ import annotations

import json
from pathlib import Path

from release.aggregate import aggregate, write_manifest


def _seed_train(root: Path, seed: str, rows: list[tuple[int, float]]) -> None:
    seed_dir = root / "train" / seed
    seed_dir.mkdir(parents=True, exist_ok=True)
    with (seed_dir / "loss.csv").open("w", encoding="utf-8") as f:
        f.write("step,loss,lr\n")
        for step, loss in rows:
            f.write(f"{step},{loss},1e-4\n")


def _seed_eval(root: Path, rows: list[dict[str, str]]) -> None:
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "acceptance_grid.csv").open("w", encoding="utf-8") as f:
        f.write("domain,temperature,batch_size,mean_acceptance,eal,itl_ms\n")
        for r in rows:
            f.write(
                f"{r['domain']},{r['temperature']},{r['batch_size']},"
                f"{r['mean_acceptance']},{r['eal']},{r['itl_ms']}\n"
            )


def test_aggregate_empty_root(tmp_path: Path) -> None:
    manifest = aggregate(tmp_path)
    assert manifest["root"] == str(tmp_path)
    assert "training" not in manifest
    assert "ablation" not in manifest
    assert "eval" not in manifest


def test_aggregate_collects_train_seeds(tmp_path: Path) -> None:
    _seed_train(tmp_path, "seed_42", [(0, 1.5), (1, 1.4), (2, 1.3)])
    _seed_train(tmp_path, "seed_0", [(0, 1.6), (1, 1.5)])
    manifest = aggregate(tmp_path)
    assert "training" in manifest
    assert set(manifest["training"]["seeds"]) == {"seed_42", "seed_0"}
    assert len(manifest["training"]["seeds"]["seed_42"]) == 3
    assert manifest["training"]["seeds"]["seed_42"][0]["loss"] == 1.5


def test_aggregate_collects_eval_grid(tmp_path: Path) -> None:
    _seed_eval(
        tmp_path,
        [
            {
                "domain": "finance",
                "temperature": "0.7",
                "batch_size": "4",
                "mean_acceptance": "0.62",
                "eal": "2.63",
                "itl_ms": "35.2",
            },
        ],
    )
    manifest = aggregate(tmp_path)
    assert len(manifest["eval"]["grid"]) == 1
    assert manifest["eval"]["grid"][0]["domain"] == "finance"


def test_write_manifest_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "manifest.json"
    write_manifest({"a": 1, "b": [1, 2, 3]}, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == {"a": 1, "b": [1, 2, 3]}
