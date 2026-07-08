"""Aggregate results across train/ ablate/ serve/ eval/ into a single manifest.

Reads JSON + CSV outputs from prior phases and emits a single `results.json`
suitable for embedding in the HuggingFace model card or writeup.

Pure file I/O. No GPU, no model loading.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_loss(path: Path) -> list[dict[str, float]]:
    """Read a loss-curve CSV written by train/train_eagle3.py."""
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items() if v})
    return rows


def _read_acceptance_grid(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def aggregate(results_root: Path) -> dict[str, Any]:
    """Walk results_root, collect loss curves + ablations + accept grid."""
    out: dict[str, Any] = {"root": str(results_root)}

    # Training: per-seed loss curves
    train_root = results_root / "train"
    if train_root.exists():
        seeds: dict[str, list[dict[str, float]]] = {}
        for seed_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
            csv_path = seed_dir / "loss.csv"
            seeds[seed_dir.name] = _read_csv_loss(csv_path)
        out["training"] = {"seeds": seeds}

    # Ablation: comparison JSON
    ablate_root = results_root / "ablate"
    if ablate_root.exists():
        out["ablation"] = {
            "comparison": _read_json(ablate_root / "comparison.json"),
            "variants": {
                p.name: _read_json(p / "summary.json")
                for p in ablate_root.iterdir()
                if p.is_dir()
            },
        }

    # Acceptance grid from eval
    eval_root = results_root / "eval"
    if eval_root.exists():
        out["eval"] = {
            "grid": _read_acceptance_grid(eval_root / "acceptance_grid.csv"),
            "summary": _read_json(eval_root / "summary.json"),
        }

    return out


def write_manifest(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)


def main(results_root: Path, out_path: Path) -> int:
    manifest = aggregate(results_root)
    write_manifest(manifest, out_path)
    print(f"wrote {out_path}")
    return 0
