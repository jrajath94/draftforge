"""Tests for eval/acceptance.py — pure analytics (CPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.acceptance import (
    crossover_batch_size,
    expected_acceptance_length,
    json_dump,
    load_grid,
    write_acceptance_grid,
)


def test_expected_acceptance_length_geometric() -> None:
    # p=0.5 → E[c] = 2
    assert expected_acceptance_length(0.5, horizon=8) == pytest.approx(2.0)
    # p=0.0 → E[c] = 1 (no acceptance)
    assert expected_acceptance_length(0.0, horizon=8) == pytest.approx(1.0)
    # p=0.75 → E[c] = 4
    assert expected_acceptance_length(0.75, horizon=8) == pytest.approx(4.0)


def test_expected_acceptance_length_boundary_conditions() -> None:
    # p=1.0 capped at horizon
    assert expected_acceptance_length(1.0, horizon=8) == 8.0
    # p negative treated as 1.0 floor
    assert expected_acceptance_length(-0.1, horizon=4) == 1.0


def test_all_drafts_rejected_geometric_mean_stability() -> None:
    """Addendum 4: every draft rejected -> accept_len = 1 per token, regardless of horizon.

    Pathological case: when the verifier rejects every draft (mean_acceptance = 0),
    the expected acceptance length is exactly 1.0 for any horizon value. This is
    the geometric-mean stability invariant: E[c] = 1/(1-p) at p=0 evaluates to
    1 (no acceptance, no speedup). Guards against division-by-zero / horizon-
    pollution regressions.

    Distinct from the cap-at-horizon case: mean_acceptance >= 1.0 (all accepted)
    is undefined geometrically; the implementation caps at horizon there.
    """
    # mean_acceptance = 0 (all drafts rejected) -> exactly 1.0 for every horizon.
    for h in (1, 4, 8, 16, 32, 128):
        assert expected_acceptance_length(0.0, horizon=h) == 1.0, (
            f"all-rejected must return 1.0 at horizon={h}, got "
            f"{expected_acceptance_length(0.0, horizon=h)}"
        )
    # Negative mean_acceptance (numerically invalid input) floors at 1.0.
    assert expected_acceptance_length(-0.5, horizon=8) == 1.0
    # Cap-at-horizon: when acceptance is total, E[c] is bounded by horizon.
    assert expected_acceptance_length(1.0, horizon=8) == 8.0
    # Sanity: monotonic in p over the stable range (where E[c] <= horizon).
    assert (
        expected_acceptance_length(0.1, horizon=8)
        < expected_acceptance_length(0.5, horizon=8)
    )


def test_crossover_batch_size_when_spec_better() -> None:
    # baseline 50ms, spec 25ms, decode-saturation 50ms → benefit 25, saturation 0
    # → returns 1.0 (fallback)
    bs = crossover_batch_size(50.0, 25.0, 50.0)
    assert bs == pytest.approx(1.0)


def test_crossover_batch_size_normal_case() -> None:
    # baseline 50ms, spec 25ms, decode-sat 25ms → benefit 25, saturation 25 → 1.0
    bs = crossover_batch_size(50.0, 25.0, 25.0)
    assert bs == pytest.approx(1.0)
    # baseline 50ms, spec 30ms, decode-sat 20ms → benefit 20, saturation 30 → 0.667
    bs2 = crossover_batch_size(50.0, 30.0, 20.0)
    assert bs2 == pytest.approx(20.0 / 30.0)


def test_crossover_batch_size_infinity_when_spec_worse() -> None:
    # spec slower than baseline → no crossover
    bs = crossover_batch_size(50.0, 60.0, 40.0)
    assert bs == float("inf")


def test_crossover_batch_size_zero_on_bad_inputs() -> None:
    assert crossover_batch_size(0.0, 10.0, 5.0) == 0.0
    assert crossover_batch_size(10.0, 0.0, 5.0) == 0.0


def test_write_and_load_grid_roundtrip(tmp_path: Path) -> None:
    rows = [
        {
            "domain": "finance",
            "temperature": 0.7,
            "batch_size": 4,
            "mean_acceptance": 0.62,
            "eal": 2.63,
            "itl_ms": 35.2,
        },
        {
            "domain": "general",
            "temperature": 0.7,
            "batch_size": 4,
            "mean_acceptance": 0.71,
            "eal": 3.45,
            "itl_ms": 28.1,
        },
    ]
    out = tmp_path / "grid.csv"
    write_acceptance_grid(rows, out)
    loaded = load_grid(out)
    assert len(loaded) == 2
    assert loaded[0]["domain"] == "finance"
    assert loaded[0]["batch_size"] == "4"
    assert loaded[1]["eal"] == "3.4500"  # writer uses :.4f


def test_load_grid_missing_file(tmp_path: Path) -> None:
    assert load_grid(tmp_path / "nope.csv") == []


def test_json_dump_creates_sorted_json(tmp_path: Path) -> None:
    payload = {"b": 2, "a": 1, "c": 3}
    out = tmp_path / "x.json"
    json_dump(payload, out)
    loaded = json.loads(out.read_text())
    assert loaded == payload
    # sorted keys
    assert out.read_text().index('"a"') < out.read_text().index('"b"')


# ---- CLI entrypoint -------------------------------------------------------
#
# `python -m eval.acceptance --results-root X --out Y` — arg shape used by
# scripts/run_full_pipeline.sh and scripts/onboard_pod.sh.
#
# Tests invoke main() in-process so coverage measures the code path; the
# __main__ arg-parser wiring is verified by ONE subprocess smoke test below.


def test_cli_writes_empty_grid_when_no_serve_outputs(tmp_path: Path) -> None:
    """main(): No serve JSONs -> empty grid still emitted; exit 0."""
    from eval.acceptance import main as cli_main

    results_root = tmp_path / "results"
    results_root.mkdir()
    out = tmp_path / "acceptance_grid.csv"
    rc = cli_main(results_root, out)
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith(
        "domain,temperature,batch_size,mean_acceptance,eal,itl_ms"
    )
    assert text.strip().count("\n") == 0  # header only


def test_cli_writes_rows_from_serve_jsons(tmp_path: Path) -> None:
    """main(): Seeded serve JSONs -> grid populates with one row per JSON."""
    from eval.acceptance import main as cli_main

    results_root = tmp_path / "results"
    serve_dir = (
        results_root / "serve" / "vllm" / "finance" / "b4" / "t0.7"
    )
    serve_dir.mkdir(parents=True)
    (serve_dir / "out.json").write_text(
        json.dumps(
            {
                "domain": "finance",
                "temperature": 0.7,
                "batch_size": 4,
                "mean_acceptance": 0.71,
                "itl_ms": 35.2,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "acceptance_grid.csv"
    rc = cli_main(results_root, out)
    assert rc == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # header + 1 row
    row = dict(
        zip(lines[0].split(","), lines[1].split(","), strict=True)
    )
    assert row["domain"] == "finance"
    assert row["batch_size"] == "4"


def test_cli_infers_domain_temperature_batch_from_path(tmp_path: Path) -> None:
    """main(): Path shapes like .../general/.../b8/.../t0.0/... -> row parsed."""
    from eval.acceptance import main as cli_main

    results_root = tmp_path / "results"
    nested = results_root / "serve" / "vllm" / "general" / "b8" / "t0.0"
    nested.mkdir(parents=True)
    (nested / "out.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "acceptance_grid.csv"
    rc = cli_main(results_root, out)
    assert rc == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    row = dict(
        zip(lines[0].split(","), lines[1].split(","), strict=True)
    )
    assert row["domain"] == "general"
    assert row["batch_size"] == "8"
    assert float(row["temperature"]) == 0.0


def test_cli_argparse_binding_smoke(tmp_path: Path) -> None:
    """Subprocess: actual `python -m eval.acceptance ...` argparse path
    works end-to-end (cheaper than full subprocess suite, but covers the
    `if __name__ == "__main__":` block)."""
    import eval.acceptance as cli  # noqa: F401 (ensures importable)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.acceptance",
            "--results-root", str(tmp_path),
            "--out", str(tmp_path / "g.csv"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "g.csv").exists()


def test_cli_missing_results_root_exits_nonzero(tmp_path: Path) -> None:
    """Subprocess: Missing --results-root -> argparse exits 2."""
    proc = subprocess.run(
        [sys.executable, "-m", "eval.acceptance", "--out", str(tmp_path / "g.csv")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "--results-root" in proc.stderr or "required" in proc.stderr
