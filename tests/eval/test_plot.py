"""Tests for eval/plot.py — matplotlib smoke tests (no figure display)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no-display backend for CI

from pathlib import Path

from eval.plot import plot_acceptance_by_batch, plot_itl_reduction


def test_plot_acceptance_by_batch_writes_png(tmp_path: Path) -> None:
    rows = [
        {"domain": "finance", "temperature": 0.0, "batch_size": 1, "eal": 2.5},
        {"domain": "finance", "temperature": 0.0, "batch_size": 8, "eal": 2.1},
        {"domain": "general", "temperature": 0.0, "batch_size": 1, "eal": 3.1},
        {"domain": "general", "temperature": 0.0, "batch_size": 8, "eal": 2.8},
        {"domain": "finance", "temperature": 0.7, "batch_size": 1, "eal": 2.3},
        {"domain": "finance", "temperature": 0.7, "batch_size": 8, "eal": 1.9},
        {"domain": "general", "temperature": 0.7, "batch_size": 1, "eal": 2.9},
        {"domain": "general", "temperature": 0.7, "batch_size": 8, "eal": 2.6},
    ]
    out = tmp_path / "acceptance.png"
    plot_acceptance_by_batch(rows, out, title="acceptance vs batch")
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG


def test_plot_itl_reduction_writes_png(tmp_path: Path) -> None:
    base = {("general", "0.7", 1): 50.0, ("general", "0.7", 8): 80.0}
    spec = {("general", "0.7", 1): 30.0, ("general", "0.7", 8): 70.0}
    out = tmp_path / "itl.png"
    plot_itl_reduction(base, spec, out)
    assert out.exists()


def test_plot_itl_reduction_handles_empty_keys(tmp_path: Path) -> None:
    out = tmp_path / "itl.png"
    plot_itl_reduction({}, {("a", "b", 1): 1.0}, out)
    assert not out.exists()
