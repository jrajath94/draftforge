"""End-to-end test for `make demo` — the local CPU pipeline runner.

Invokes `scripts/run_demo.py` against a tmp results dir and asserts every
expected artifact exists + carries the `is_demo: true` watermark so it can
never be confused with measured results.

Slow (~10s) because it spawns subprocesses. Marked `@pytest.mark.slow`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = ROOT / "scripts" / "run_demo.py"
# data.prepare reads `output_dir: ./artifacts/demo` from data/demo_config.yaml
# (relative to cwd). The demo runs with cwd=ROOT, so artifacts land under ROOT/artifacts.
DEMO_ARTIFACTS = ROOT / "artifacts" / "demo"


@pytest.fixture
def clean_demo_artifacts() -> None:
    """Remove any prior `artifacts/demo` left over from a previous demo run."""
    if DEMO_ARTIFACTS.exists():
        shutil.rmtree(DEMO_ARTIFACTS)


@pytest.mark.slow
def test_demo_pipeline_runs(tmp_path: Path, clean_demo_artifacts: None) -> None:
    """`make demo` end-to-end: every artifact exists and is_demo flag propagates."""
    result = subprocess.run(
        [sys.executable, str(DEMO_SCRIPT), "--results-root", str(tmp_path / "demo")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"demo failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    demo = tmp_path / "demo"
    assert (demo / "IS_DEMO.md").exists(), "missing IS_DEMO.md marker"

    # data.prepare orchestrator output (committed fixture)
    assert (DEMO_ARTIFACTS / "splits" / "train.jsonl").exists(), "data.prepare did not emit splits"
    assert (DEMO_ARTIFACTS / "results" / "data" / "pipeline_summary.json").exists()

    # Stage 2 — mock training: 3 flat seeds under train/, 4×3 under ablate_data/
    seeds = [d.name for d in (demo / "train").iterdir() if d.is_dir()]
    assert sorted(seeds) == ["0", "1234", "42"], seeds
    for seed in seeds:
        loss_csv = demo / "train" / seed / "loss.csv"
        assert loss_csv.exists()
        header = loss_csv.read_text(encoding="utf-8").splitlines()[0]
        assert header.strip() == "step,loss,lr", header
        summary = json.loads((demo / "train" / seed / "summary.json").read_text(encoding="utf-8"))
        assert summary["is_demo"] is True

    # Stage 3 — ablation comparison
    comparison = json.loads((demo / "ablate" / "comparison.json").read_text(encoding="utf-8"))
    assert set(comparison.keys()) == {"tri_layer", "final_layer", "low_only", "mid_only"}
    for variant, stats in comparison.items():
        assert stats["is_demo"] is True
        assert stats["n_seeds"] == 3
        assert len(stats["per_seed"]) == 3

    # Stage 4 — acceptance grid + crossover report
    grid_path = demo / "eval" / "acceptance_grid.csv"
    assert grid_path.exists()
    header = grid_path.read_text(encoding="utf-8").splitlines()[0]
    assert "domain,temperature,batch_size,mean_acceptance,eal,itl_ms" in header
    assert (demo / "eval" / "crossover_analysis.md").exists()

    # Stage 5 — manifest + HF card with is_demo watermark
    manifest = json.loads((demo / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["is_demo"] is True
    assert "SYNTHETIC demo output" in manifest["warning"]
    assert manifest["ablation"]["comparison"]  # non-empty
    assert len(manifest["eval"]["grid"]) == 18  # 2 domains × 3 temps × 3 batches

    card_path = demo / "HF_CARD.md"
    assert card_path.exists()
    card_text = card_path.read_text(encoding="utf-8")
    assert "demo-eagle3-head" in card_text
    assert "Qwen3-14B" in card_text


@pytest.mark.slow
def test_demo_is_idempotent(tmp_path: Path, clean_demo_artifacts: None) -> None:
    """Re-running the demo over the same dir should not crash and should
    overwrite the manifest cleanly."""
    out = tmp_path / "demo"
    first = subprocess.run(
        [sys.executable, str(DEMO_SCRIPT), "--results-root", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert first.returncode == 0
    second = subprocess.run(
        [sys.executable, str(DEMO_SCRIPT), "--results-root", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert second.returncode == 0, f"second run failed:\n{second.stdout}\n{second.stderr}"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["is_demo"] is True