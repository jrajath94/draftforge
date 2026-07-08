"""Tests for eval/crossover_analysis.py — pure-Python CSV→MD report.

These tests pin the contract that the writeup's "Batch-Size Crossover"
section depends on: feed in an acceptance grid, get back a markdown
report whose numbers are traceable to the CSV. CPU-only, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.acceptance import write_acceptance_grid
from eval.crossover_analysis import _co_str, analyze_crossover

# ---- _co_str formatting helper --------------------------------------------


def test_co_str_infinity_and_zero() -> None:
    """Infinity and zero have their own glyphs; do not regress to 'inf'."""
    assert _co_str(float("inf")) == "∞"
    assert _co_str(0.0) == "N/A"


def test_co_str_rounds_to_one_decimal() -> None:
    assert _co_str(3.14159) == "3.1"
    assert _co_str(16.0) == "16.0"


# ---- analyze_crossover happy paths ----------------------------------------


def _grid(rows: list[dict]) -> list[dict]:
    return rows


def test_analyze_crossover_single_row(tmp_path: Path) -> None:
    """One (domain, temp) row → one crossover entry in the report."""
    csv_path = tmp_path / "grid.csv"
    write_acceptance_grid(
        [
            {
                "domain": "finance",
                "temperature": 0.7,
                "batch_size": 4,
                "mean_acceptance": 0.62,
                "eal": 2.63,
                "itl_ms": 35.2,
            },
        ],
        csv_path,
    )
    out = tmp_path / "report.md"
    result = analyze_crossover(csv_path, out)
    assert out.exists()
    assert "Batch-Size Crossover Analysis" in out.read_text()
    # Keys are encoded as "{domain}|{temperature}" (see analyze_crossover return).
    assert "finance|0.7" in result["crossovers"]
    assert "| finance | 0.7 |" in out.read_text()


def test_analyze_crossover_multi_group(tmp_path: Path) -> None:
    """Multiple (domain, temp) groups → sorted table entries."""
    csv_path = tmp_path / "grid.csv"
    rows: list[dict] = []
    for domain in ("finance", "general"):
        for temp in (0.0, 0.7, 1.0):
            for batch in (1, 4, 16):
                rows.append(
                    {
                        "domain": domain,
                        "temperature": temp,
                        "batch_size": batch,
                        "mean_acceptance": 0.5,
                        "eal": 2.0,
                        "itl_ms": 50.0 - batch * 2.0,  # decreases with batch
                    }
                )
    write_acceptance_grid(rows, csv_path)
    out = tmp_path / "report.md"
    result = analyze_crossover(csv_path, out)
    # 2 domains x 3 temps = 6 groups
    assert len(result["crossovers"]) == 6
    text = out.read_text()
    assert text.count("| finance |") == 3
    assert text.count("| general |") == 3


# ---- analyze_crossover edge cases -----------------------------------------


def test_analyze_crossover_empty_grid_raises(tmp_path: Path) -> None:
    """Empty / missing grid must raise — the caller (CLI) maps to exit 1."""
    with pytest.raises(ValueError, match="empty or not found"):
        analyze_crossover(tmp_path / "missing.csv", tmp_path / "out.md")


def test_analyze_crossover_malformed_rows_silently_skipped(tmp_path: Path) -> None:
    """Rows missing required keys or with non-numeric batch_size are dropped
    but the rest of the grid still produces a report (best-effort)."""
    csv_path = tmp_path / "grid.csv"
    # Write a CSV with one good row + one header line + one bogus line.
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("domain,temperature,batch_size,mean_acceptance,eal,itl_ms\n")
        f.write("finance,0.7,4,0.62,2.63,35.2\n")  # good
        f.write(",,not_a_number,0.62,2.63,35.2\n")  # malformed
    out = tmp_path / "report.md"
    result = analyze_crossover(csv_path, out)
    assert "finance|0.7" in result["crossovers"]
    # Malformed row dropped — only the one good row group remains.
    assert len(result["crossovers"]) == 1


# ---- CLI entrypoint -------------------------------------------------------


def test_cli_main_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI: --grid valid.csv → exit 0 + markdown written + stdout message."""
    csv_path = tmp_path / "grid.csv"
    write_acceptance_grid(
        [
            {
                "domain": "general",
                "temperature": 0.0,
                "batch_size": 4,
                "mean_acceptance": 0.71,
                "eal": 3.45,
                "itl_ms": 28.1,
            },
        ],
        csv_path,
    )
    out = tmp_path / "report.md"
    # main() reads sys.argv — invoke via subprocess for full CLI path coverage.
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.crossover_analysis",
            "--grid",
            str(csv_path),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "[crossover] report written to" in proc.stdout


def test_cli_main_error_exit_code(tmp_path: Path) -> None:
    """CLI: --grid <missing> → exit 1 + stderr message."""
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.crossover_analysis",
            "--grid",
            str(tmp_path / "nope.csv"),
            "--out",
            str(tmp_path / "out.md"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "error" in proc.stderr
