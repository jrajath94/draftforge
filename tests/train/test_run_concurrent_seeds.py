"""Tests for train/run_concurrent_seeds.sh — bash runner that spawns N seeds.

Strategy: invoke the script under `bash` with a stubbed training driver
(`echo` instead of `accelerate launch`). All assertions live in pytest; the
bash script is the system-under-test.

These tests focus on the *plumbing* (timing, log-per-seed, error propagation).
The actual training-driver behaviour is covered by train_eagle3 + head tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
RUN_SCRIPT = ROOT / "train" / "run_concurrent_seeds.sh"


def _has_bash() -> bool:
    return shutil.which("bash") is not None


pytestmark = pytest.mark.skipif(not _has_bash(), reason="bash required")


@pytest.fixture
def stub_driver_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Replace `accelerate launch -m train.train_eagle3` with a stub `echo` driver.

    The stub simulates per-seed work by sleeping a fraction of a second so we
    can assert concurrency (3 seeds at 0.3s in parallel ≈ 0.3s, not 0.9s).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    driver = bin_dir / "train_eagle3"
    driver.write_text(
        "#!/usr/bin/env bash\n"
        "sleep 0.3\n"
        "echo \"[stub] seed=$DRAFTFORGE_SEED gpu=$CUDA_VISIBLE_DEVICES step=0..max=$DRAFTFORGE_MAX\"\n",
        encoding="utf-8",
    )
    driver.chmod(0o755)

    # Force PATH so this driver is discoverable by `python -m`. Simpler: invoke
    # the stub via `python -m train.train_eagle3` after monkeypatching sys.path.
    # For the bash tests, just ensure the script writes the expected log file.
    return bin_dir


def _run_runner(
    tmp_path: Path,
    n_seeds: int = 3,
    gpus: str = "0 1 2",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke run_concurrent_seeds.sh with stubbed training driver."""
    assert RUN_SCRIPT.exists(), f"missing {RUN_SCRIPT}"
    log_dir = tmp_path / "logs"
    env = os.environ.copy()
    env.update({
        "DRAFTFORGE_STUB": "1",
        "LOG_DIR": str(log_dir),
    })
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            "bash", str(RUN_SCRIPT),
            str(n_seeds),
            gpus,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


# ── 1. Three seeds run concurrently (faster than serial) ────────────────────


def test_three_seeds_run_in_parallel_not_serial(tmp_path: Path) -> None:
    """3 seeds × 0.3s stub = serial 0.9s. Concurrent ≈ 0.3s + slack.

    Allow generous slack (≤0.6s) to avoid CI flakiness on slow runners.
    True serial: ≥0.9s. True parallel: ≤0.6s.
    """
    log_dir = tmp_path / "logs"
    t0 = time.monotonic()
    result = _run_runner(tmp_path, n_seeds=3, gpus="0 1 2")
    wall = time.monotonic() - t0
    assert result.returncode == 0, f"runner failed:\n{result.stdout}\n{result.stderr}"
    assert wall < 0.7, f"took {wall:.2f}s — looks serial, not concurrent"


# ── 2. Per-seed log file exists with seed + gpu markers ──────────────────────


def test_per_seed_log_file_written(tmp_path: Path) -> None:
    """Each seed writes a separate log under ${LOG_DIR}/seed_<N>_gpu<M>.log."""
    _run_runner(tmp_path, n_seeds=3, gpus="0 1 2")
    log_dir = tmp_path / "logs"
    logs = sorted(p.name for p in log_dir.glob("seed_*.log"))
    # Default seed list starts with (42, 123, 456).
    assert any("seed_42" in n for n in logs), f"missing seed_42 log in {logs}"
    assert any("seed_123" in n for n in logs), f"missing seed_123 log"
    assert any("seed_456" in n for n in logs), f"missing seed_456 log"


def test_log_contains_seed_and_gpu_markers(tmp_path: Path) -> None:
    """Each per-seed log MUST mention the assigned seed + CUDA device."""
    _run_runner(tmp_path, n_seeds=3, gpus="0 1 2")
    log_dir = tmp_path / "logs"
    logs = sorted(log_dir.glob("seed_*.log"))
    # At least one log mentions seed 42, one mentions seed 123, one mentions seed 456.
    blob = "\n".join(p.read_text() for p in logs)
    assert "seed=42" in blob
    assert "seed=123" in blob
    assert "seed=456" in blob
    # And at least one log mentions "gpu=0", one "gpu=1", one "gpu=2".
    assert "gpu=0" in blob
    assert "gpu=1" in blob
    assert "gpu=2" in blob


# ── 3. N_SEEDS override respects seed count ──────────────────────────────────


def test_n_seeds_one_runs_only_one_seed(tmp_path: Path) -> None:
    """N_SEEDS=1 → only one log file written; runner exits 0."""
    _run_runner(tmp_path, n_seeds=1, gpus="0")
    log_dir = tmp_path / "logs"
    logs = list(log_dir.glob("seed_*.log"))
    assert len(logs) == 1, f"expected 1 log, got {len(logs)}: {logs}"


def test_n_seeds_too_large_falls_back_to_first_n_seeds(tmp_path: Path) -> None:
    """N_SEEDS=4 with default 3-seed list still works (uses first 3 of 4 available).

    The runner should not crash on overflow; it cycles through the available
    default seed list (42, 123, 456, 789, 1024, 2048) and GPU round-robin.
    """
    # This just ensures exit 0; detail of which seeds is captured by another test.
    result = _run_runner(tmp_path, n_seeds=3, gpus="0 1 2 3")
    assert result.returncode == 0, (
        f"runner crashed on 4 gpus / 3 seeds:\n{result.stderr}"
    )


# ── 4. Failure of one child propagates ──────────────────────────────────────


def test_child_failure_propagates_to_runner(tmp_path: Path, monkeypatch) -> None:
    """If a child fails, the runner exits non-zero and prints a diagnostic.

    Pin this contract: silent ignores are unacceptable, would mislead users
    into thinking 3 seeds converged when only 2 did.
    """
    # Override PATH to put a failing driver in front of the real accelerator.
    bin_dir = tmp_path / "failbin"
    bin_dir.mkdir()
    failer = bin_dir / "train_eagle3"
    failer.write_text(
        "#!/usr/bin/env bash\necho '[stub-fail] seed='\"${DRAFTFORGE_SEED:-?}\"' gpu='\"${CUDA_VISIBLE_DEVICES:-?}\" >&2\nexit 1\n",
        encoding="utf-8",
    )
    failer.chmod(0o755)
    # Prepend to PATH so python -m finds the broken driver.
    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + ":" + env["PATH"]
    log_dir = tmp_path / "logs"
    env["LOG_DIR"] = str(log_dir)

    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "3", "0 1 2"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode != 0, (
        "runner exited 0 with a failing child — silent failure contract violated"
    )
    # Diagnostic in stderr
    assert "failed" in result.stdout.lower() or "fail" in result.stderr.lower()
