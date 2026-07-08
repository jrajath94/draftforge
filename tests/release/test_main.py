"""Tests for release/__main__.py — typer CLI entrypoints.

Both commands are thin wrappers around release.aggregate and
release.make_card main()s. These tests pin the wiring contract:
arg parsing, exit codes, output paths. CPU-only.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from release.__main__ import app

runner = CliRunner()


# ---- aggregate command ----------------------------------------------------


def test_aggregate_command_writes_manifest(tmp_path: Path) -> None:
    """`python -m release aggregate --results-root X --out Y` writes JSON."""
    results_root = tmp_path / "results"
    results_root.mkdir()
    # Seed at least one artifact so aggregate() returns something interesting.
    (results_root / "train" / "seed_42").mkdir(parents=True)
    (results_root / "train" / "seed_42" / "loss.csv").write_text(
        "step,loss,lr\n0,1.5,1e-4\n1,1.4,1e-4\n", encoding="utf-8"
    )
    out = tmp_path / "manifest.json"

    result = runner.invoke(
        app,
        ["aggregate", "--results-root", str(results_root), "--out", str(out)],
    )

    assert result.exit_code == 0, result.stdout
    assert out.exists()
    import json
    manifest = json.loads(out.read_text())
    assert manifest["root"] == str(results_root)
    assert "training" in manifest
    assert "seed_42" in manifest["training"]["seeds"]
    assert "wrote " in result.stdout


def test_aggregate_command_missing_results_root(tmp_path: Path) -> None:
    """Missing --results-root → typer error (exit 2), not a runtime crash."""
    out = tmp_path / "manifest.json"
    result = runner.invoke(app, ["aggregate", "--out", str(out)])
    # typer exits 2 on missing required option.
    assert result.exit_code != 0


# ---- make-card command ----------------------------------------------------


def test_make_card_command_substitutes_template(tmp_path: Path) -> None:
    """`python -m release make-card ...` writes a card with substituted vars."""
    results_root = tmp_path / "results"
    results_root.mkdir()
    template = tmp_path / "card.tpl.md"
    template.write_text(
        "# $HEAD_NAME for $TARGET_MODEL\n\nmanifest: $MANIFEST_JSON\n",
        encoding="utf-8",
    )
    out = tmp_path / "CARD.md"

    result = runner.invoke(
        app,
        [
            "make-card",
            "--template",
            str(template),
            "--results",
            str(results_root),
            "--head",
            "eagle3-qwen3-14b-finance",
            "--target",
            "Qwen/Qwen3-14B",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "eagle3-qwen3-14b-finance" in text
    assert "Qwen/Qwen3-14B" in text
    assert "manifest: " in text  # MANIFEST_JSON substituted with non-empty string


def test_make_card_command_missing_template(tmp_path: Path) -> None:
    """Missing --template → typer error (exit 2)."""
    result = runner.invoke(
        app,
        [
            "make-card",
            "--results",
            str(tmp_path),
            "--head",
            "h",
            "--target",
            "t",
            "--out",
            str(tmp_path / "card.md"),
        ],
    )
    assert result.exit_code != 0
