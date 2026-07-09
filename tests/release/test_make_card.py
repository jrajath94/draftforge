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
