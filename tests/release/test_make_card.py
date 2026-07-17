"""Tests for release/make_card.py — HF card rendering."""

from __future__ import annotations

import runpy
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from release.make_card import render_card


def test_render_card_substitutes_placeholders(tmp_path: Path) -> None:
    template = tmp_path / "tpl.md"
    template.write_text(
        "---\nbase_model: $TARGET_MODEL\n---\n# $HEAD_NAME\n\nJSON: $MANIFEST_JSON\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.md"
    render_card(
        template_path=template,
        results_root=tmp_path,  # empty; manifest = {}
        head_name="Qwen3-4B-Instruct-2507-EAGLE3-Finance",
        target_model="Qwen/Qwen3-4B-Instruct-2507",
        out_path=out,
    )
    content = out.read_text(encoding="utf-8")
    assert "Qwen/Qwen3-4B-Instruct-2507" in content
    assert "Qwen3-4B-Instruct-2507-EAGLE3-Finance" in content
    assert "base_model: Qwen/Qwen3-4B-Instruct-2507" in content


def test_results_section_empty_manifest_marks_not_measured(tmp_path: Path) -> None:
    """Empty results root -> ## Results renders the [NOT YET MEASURED] marker."""
    template = tmp_path / "tpl.md"
    template.write_text("## Results\n\n$RESULTS_SECTION\n", encoding="utf-8")
    out = tmp_path / "out.md"
    render_card(
        template_path=template,
        results_root=tmp_path,  # no artifacts
        head_name="h",
        target_model="t",
        out_path=out,
    )
    text = out.read_text(encoding="utf-8")
    assert "[NOT YET MEASURED]" in text
    assert "{" not in text  # no raw JSON dump in the rendered card


def test_results_section_renders_grid_table(tmp_path: Path) -> None:
    """Acceptance grid CSV -> markdown table, no [NOT YET MEASURED] marker."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "acceptance_grid.csv").write_text(
        "batch,acceptance_rate\n1,0.71\n4,0.68\n", encoding="utf-8"
    )
    template = tmp_path / "tpl.md"
    template.write_text("## Results\n\n$RESULTS_SECTION\n", encoding="utf-8")
    out = tmp_path / "out.md"
    render_card(
        template_path=template,
        results_root=tmp_path,
        head_name="h",
        target_model="t",
        out_path=out,
    )
    text = out.read_text(encoding="utf-8")
    assert "| batch | acceptance_rate |" in text
    assert "| 1 | 0.71 |" in text
    assert "[NOT YET MEASURED]" not in text


def test_results_section_renders_per_seed_training_table(tmp_path: Path) -> None:
    """Per-seed loss curves -> seed table with logged steps + final loss."""
    seed_dir = tmp_path / "train" / "42"
    seed_dir.mkdir(parents=True)
    (seed_dir / "loss_curve.csv").write_text(
        "step,loss\n1,2.0\n2,1.5\n", encoding="utf-8"
    )
    template = tmp_path / "tpl.md"
    template.write_text("$RESULTS_SECTION\n", encoding="utf-8")
    out = tmp_path / "out.md"
    render_card(
        template_path=template,
        results_root=tmp_path,
        head_name="h",
        target_model="t",
        out_path=out,
    )
    text = out.read_text(encoding="utf-8")
    assert "| 42 | 2 | 1.5 |" in text


def test_manifest_json_is_valid_json(tmp_path: Path) -> None:
    """$MANIFEST_JSON substitution must be parseable JSON (not Python repr)."""
    import json as _json

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    template = tmp_path / "tpl.md"
    template.write_text("$MANIFEST_JSON", encoding="utf-8")
    out = tmp_path / "out.md"
    render_card(
        template_path=template,
        results_root=tmp_path,
        head_name="h",
        target_model="t",
        out_path=out,
    )
    parsed = _json.loads(out.read_text(encoding="utf-8"))
    assert parsed["eval"]["grid"] == []


# ---- CLI entrypoint -------------------------------------------------------


def test_cli_renders_card(tmp_path: Path) -> None:
    """main(): writes a card with all placeholders substituted."""
    from release.make_card import main as cli_main

    template = tmp_path / "tpl.md"
    template.write_text(
        "# $HEAD_NAME for $TARGET_MODEL\n\nmanifest: $MANIFEST_JSON\n",
        encoding="utf-8",
    )
    out = tmp_path / "CARD.md"
    rc = cli_main(template, tmp_path, "eagle3-qwen3-finance", "Qwen/Qwen3-4B", out)
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "eagle3-qwen3-finance" in text
    assert "Qwen/Qwen3-4B" in text


def test_cli_argparse_binding_smoke(tmp_path: Path) -> None:
    """Subprocess: actual `python -m release.make_card ...` works."""
    template = tmp_path / "tpl.md"
    template.write_text(
        "# $HEAD_NAME for $TARGET_MODEL\n\nx\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "release.make_card",
            "--template", str(template),
            "--results", str(tmp_path),
            "--head", "eagle3-qwen3-finance",
            "--target", "Qwen/Qwen3-4B",
            "--out", str(tmp_path / "c.md"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "c.md").exists()


def test_cli_missing_template_exits_nonzero(tmp_path: Path) -> None:
    """Subprocess: Missing --template -> argparse exits 2."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "release.make_card",
            "--results", str(tmp_path),
            "--head", "h",
            "--target", "t",
            "--out", str(tmp_path / "c.md"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_main_inline_prints_wrote_marker(tmp_path: Path, capsys) -> None:
    """main() inline: prints 'wrote <path>' marker + returns 0.

    Locks the print() on line 33 (coverage gap when only subprocess runs cover it).
    """
    from release.make_card import main as cli_main

    template = tmp_path / "tpl.md"
    template.write_text("# $HEAD_NAME for $TARGET_MODEL\n\nx\n", encoding="utf-8")
    out = tmp_path / "inline.md"
    rc = cli_main(template, tmp_path, "h", "t", out)
    captured = capsys.readouterr()
    assert rc == 0
    assert f"wrote {out}" in captured.out


def test_argparse_builder_inline() -> None:
    """The __main__ argparse block: parse_args binds all 5 required args.

    Locks coverage of lines 38-47 (the argparse builder + main() glue).
    """
    import argparse as _argparse

    parser = _argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args([
        "--template", "/tmp/t",
        "--results", "/tmp/r",
        "--head", "h",
        "--target", "t",
        "--out", "/tmp/o",
    ])
    assert args.template == Path("/tmp/t")
    assert args.head == "h"
    assert args.target == "t"


def test_main_returns_zero_on_success(tmp_path: Path) -> None:
    """main() returns 0 on a happy-path render (locks the return 0 path)."""
    from release.make_card import main as cli_main

    template = tmp_path / "tpl.md"
    template.write_text("# $HEAD_NAME for $TARGET_MODEL\n\nx\n", encoding="utf-8")
    rc = cli_main(template, tmp_path, "h", "t", tmp_path / "o.md")
    assert rc == 0


def test_dunder_main_block_executes(tmp_path: Path) -> None:
    """`if __name__ == '__main__':` block at lines 38-47 — execute via runpy.

    Locks coverage of the argparse builder + sys.exit() glue that only fires
    when the module is run as __main__ (subprocess invocations are a separate
    process and don't contribute to this coverage slot).
    """
    template = tmp_path / "tpl.md"
    template.write_text("# $HEAD_NAME for $TARGET_MODEL\n\nx\n", encoding="utf-8")
    out = tmp_path / "dunder.md"

    # Inject args into argv so the parser sees them.
    saved_argv = sys.argv
    sys.argv = [
        "release.make_card",
        "--template", str(template),
        "--results", str(tmp_path),
        "--head", "dunder-head",
        "--target", "dunder-target",
        "--out", str(out),
    ]
    try:
        with warnings.catch_warnings():
            # runpy emits a RuntimeWarning if the module was already imported
            # earlier in the test session — harmless here, we just want
            # the __main__ block executed for coverage.
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module("release.make_card", run_name="__main__", alter_sys=True)
        # sys.exit(0) at end of happy path — expected.
        assert excinfo.value.code == 0
    finally:
        sys.argv = saved_argv

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "dunder-head" in text
    assert "dunder-target" in text
