"""Tests for ablate/compare.py — variant comparison aggregation."""

from __future__ import annotations

from pathlib import Path

from ablate.compare import (
    _final_mean,
    _read_loss_csv,
    compare_variants,
    write_comparison,
)


def test_read_loss_csv_parses(tmp_path: Path) -> None:
    p = tmp_path / "lc.csv"
    p.write_text("step,loss,lr\n0,2.5,1e-4\n1,1.8,9e-5\n2,1.5,8e-5\n")
    rows = _read_loss_csv(p)
    assert rows == [(0, 2.5), (1, 1.8), (2, 1.5)]


def test_read_loss_csv_missing(tmp_path: Path) -> None:
    assert _read_loss_csv(tmp_path / "nope.csv") == []


def test_final_mean_with_window(tmp_path: Path) -> None:
    p = tmp_path / "lc.csv"
    p.write_text("step,loss,lr\n" + "\n".join(f"{i},{1.0 + i*0.01},1e-4" for i in range(200)))
    rows = _read_loss_csv(p)
    mean, _std = _final_mean(rows, window=50)
    assert 1.0 < mean < 3.0


def test_compare_variants_with_seeded_data(tmp_path: Path) -> None:
    """Build fake training output dirs with realistic CSVs; verify aggregation."""
    import json

    for variant in ("tri_layer", "final_layer"):
        for seed in ("42", "0"):
            seed_dir = tmp_path / variant / seed
            seed_dir.mkdir(parents=True, exist_ok=True)
            csv_p = seed_dir / "loss_curve.csv"
            csv_p.write_text(
                "step,loss,lr\n"
                + "\n".join(f"{i},{1.0 + i * 0.001},1e-4" for i in range(50))
            )

    by_v = compare_variants(tmp_path)
    assert set(by_v.keys()) >= {"tri_layer", "final_layer"}
    for v in ("tri_layer", "final_layer"):
        assert by_v[v]["n_seeds"] == 2
        assert by_v[v]["mean_of_means"] > 0.0

    summary_p = tmp_path / "comparison.json"
    write_comparison(by_v, summary_p)
    payload = json.loads(summary_p.read_text())
    assert "tri_layer" in payload
    csv_p2 = tmp_path / "comparison.csv"
    assert csv_p2.exists()


# ---- Edge branches: defensive paths and zero-data cases -------------------


def test_read_loss_csv_raises_on_missing_header_columns(tmp_path: Path) -> None:
    """Header missing 'step' or 'loss' column → ValueError naming the path."""
    import pytest

    p = tmp_path / "bad.csv"
    p.write_text("foo,bar,baz\n1,2,3\n")
    with pytest.raises(ValueError, match="missing required 'step' or 'loss'"):
        _read_loss_csv(p)


def test_read_loss_csv_raises_on_malformed_row(tmp_path: Path) -> None:
    """Row with wrong number of columns → ValueError naming path:lineno."""
    import pytest

    p = tmp_path / "bad.csv"
    # Header OK, but row 2 has only 1 column.
    p.write_text("step,loss,lr\n0,1.5,1e-4\nbroken_row\n")
    with pytest.raises(ValueError, match=r"malformed row"):
        _read_loss_csv(p)


def test_final_mean_empty_rows_returns_zeros() -> None:
    """_final_mean([]) → (0.0, 0.0) (no windowing possible)."""
    mean, std = _final_mean([])
    assert mean == 0.0
    assert std == 0.0


def test_final_mean_single_row_zero_std() -> None:
    """n=1 → std = 0.0 (no variance possible with one sample)."""
    mean, std = _final_mean([(0, 1.5)])
    assert mean == 1.5
    assert std == 0.0


def test_compare_variants_skips_non_directory_seed_entries(tmp_path: Path) -> None:
    """Files inside a variant dir (e.g. stray logs) are skipped, not crashed on."""
    variant_dir = tmp_path / "tri_layer"
    variant_dir.mkdir()
    seed_dir = variant_dir / "seed_42"
    seed_dir.mkdir()
    (seed_dir / "loss_curve.csv").write_text(
        "step,loss,lr\n0,1.5,1e-4\n1,1.4,1e-4\n"
    )
    # A stray file at the variant root — should be skipped silently.
    (variant_dir / "README.md").write_text("# scratch notes")

    by_v = compare_variants(tmp_path)
    assert by_v["tri_layer"]["n_seeds"] == 1
    assert "seed_42" in by_v["tri_layer"]["per_seed"]


def test_compare_variants_empty_seed_dirs_yield_zero_stats(tmp_path: Path) -> None:
    """Variant dir exists but contains no seed subdirs → 0 seeds, mean=0.0."""
    variant_dir = tmp_path / "tri_layer"
    variant_dir.mkdir()  # empty variant dir

    by_v = compare_variants(tmp_path)
    assert by_v["tri_layer"]["n_seeds"] == 0
    assert by_v["tri_layer"]["mean_of_means"] == 0.0
    assert by_v["tri_layer"]["std_of_means"] == 0.0


def test_compare_variants_missing_variant_dir_yields_zero_stats(tmp_path: Path) -> None:
    """Variant dir doesn't exist → zeroed stats (no crash, no exception)."""
    # tmp_path has no variant subdirs.
    by_v = compare_variants(tmp_path)
    assert by_v["tri_layer"]["n_seeds"] == 0
    assert by_v["tri_layer"]["mean_of_means"] == 0.0


def test_default_results_root() -> None:
    """default_results_root() returns the canonical results/train path."""
    from ablate.compare import default_results_root

    assert default_results_root() == Path("results/train")
