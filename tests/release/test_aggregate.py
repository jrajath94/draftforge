"""Tests for release/aggregate.py — manifest collection across phases."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from release.aggregate import aggregate, write_manifest


def _seed_train(root: Path, seed: str, rows: list[tuple[int, float]]) -> None:
    seed_dir = root / "train" / seed
    seed_dir.mkdir(parents=True, exist_ok=True)
    with (seed_dir / "loss.csv").open("w", encoding="utf-8") as f:
        f.write("step,loss,lr\n")
        for step, loss in rows:
            f.write(f"{step},{loss},1e-4\n")


def _seed_train_canonical(
    root: Path, seed: str, rows: list[tuple[int, float]]
) -> None:
    """Seed using the canonical `loss_curve.csv` schema that the training
    driver writes (per addendum P0)."""
    seed_dir = root / "train" / seed
    seed_dir.mkdir(parents=True, exist_ok=True)
    with (seed_dir / "loss_curve.csv").open("w", encoding="utf-8") as f:
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


def test_aggregate_reads_legacy_ablation_and_flat_eval_paths(tmp_path: Path) -> None:
    ablation_dir = tmp_path / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "comparison.json").write_text(
        json.dumps({"winner": "tri_layer"}), encoding="utf-8"
    )
    (tmp_path / "acceptance_grid.csv").write_text(
        "domain,temperature,batch_size,mean_acceptance,eal,itl_ms\n"
        "finance,0.7,4,0.62,2.63,35.2\n",
        encoding="utf-8",
    )

    manifest = aggregate(tmp_path)
    assert manifest["ablation"]["comparison"]["winner"] == "tri_layer"
    assert manifest["eval"]["grid"][0]["domain"] == "finance"


def test_write_manifest_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "manifest.json"
    write_manifest({"a": 1, "b": [1, 2, 3]}, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == {"a": 1, "b": [1, 2, 3]}


def test_aggregate_reads_canonical_loss_curve_csv(tmp_path: Path) -> None:
    """Canonical `loss_curve.csv` (written by train driver) -> non-empty seed data.

    P0 regression: previous version read only legacy `loss.csv` and produced
    a 0-row manifest on real runs that use the canonical schema.
    """
    _seed_train_canonical(tmp_path, "seed_42", [(0, 1.5), (1, 1.4), (2, 1.3)])
    manifest = aggregate(tmp_path)
    seeds = manifest["training"]["seeds"]
    assert "seed_42" in seeds
    assert len(seeds["seed_42"]) == 3
    assert seeds["seed_42"][0]["loss"] == 1.5


def test_aggregate_prefers_loss_curve_over_loss_csv(tmp_path: Path) -> None:
    """When both schemas coexist, the canonical `loss_curve.csv` wins."""
    seed_dir = tmp_path / "train" / "seed_x"
    seed_dir.mkdir(parents=True)
    # Legacy with sentinel values
    (seed_dir / "loss.csv").write_text(
        "step,loss,lr\n999,9.99,0\n", encoding="utf-8"
    )
    # Canonical with real values
    (seed_dir / "loss_curve.csv").write_text(
        "step,loss,lr\n0,1.5,1e-4\n1,1.4,1e-4\n", encoding="utf-8"
    )
    manifest = aggregate(tmp_path)
    rows = manifest["training"]["seeds"]["seed_x"]
    # First row's loss must be from the canonical file (1.5), not the legacy 9.99.
    assert rows[0]["loss"] == 1.5


# ---- CLI entrypoint -------------------------------------------------------
#
# `python -m release.aggregate --results-root X --out Y` — arg shape used
# by scripts/run_full_pipeline.sh and scripts/onboard_pod.sh.
#
# main() is tested in-process for coverage; the `__main__` argparse binding
# has ONE subprocess smoke test.


def test_cli_writes_manifest(tmp_path: Path) -> None:
    """main(): writes manifest.json with seed data."""
    from release.aggregate import main as cli_main

    _seed_train(tmp_path, "seed_42", [(0, 1.5), (1, 1.4)])
    out = tmp_path / "manifest.json"
    rc = cli_main(tmp_path, out)
    assert rc == 0
    assert out.exists()
    manifest = json.loads(out.read_text())
    assert manifest["root"] == str(tmp_path)
    assert "training" in manifest
    assert "seed_42" in manifest["training"]["seeds"]


def test_cli_argparse_binding_smoke(tmp_path: Path) -> None:
    """Subprocess: actual `python -m release.aggregate ...` works."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "release.aggregate",
            "--results-root", str(tmp_path),
            "--out", str(tmp_path / "m.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "m.json").exists()


def test_cli_missing_results_root_exits_nonzero(tmp_path: Path) -> None:
    """Subprocess: Missing --results-root -> argparse exits 2."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "release.aggregate",
            "--out", str(tmp_path / "m.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
