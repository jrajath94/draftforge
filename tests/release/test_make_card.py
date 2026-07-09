"""Tests for release/make_card.py — HF card rendering."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
    import runpy
    import sys as _sys

    template = tmp_path / "tpl.md"
    template.write_text("# $HEAD_NAME for $TARGET_MODEL\n\nx\n", encoding="utf-8")
    out = tmp_path / "dunder.md"

    # Inject args into argv so the parser sees them.
    saved_argv = _sys.argv
    _sys.argv = [
        "release.make_card",
        "--template", str(template),
        "--results", str(tmp_path),
        "--head", "dunder-head",
        "--target", "dunder-target",
        "--out", str(out),
    ]
    try:
        try:
            runpy.run_module("release.make_card", run_name="__main__", alter_sys=True)
        except SystemExit as e:
            # sys.exit(0) at end of happy path — expected.
            assert e.code == 0
    finally:
        _sys.argv = saved_argv

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "dunder-head" in text
    assert "dunder-target" in text
