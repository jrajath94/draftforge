"""Tests for release/make_card.py — HF card rendering."""

from __future__ import annotations

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
        head_name="Qwen3-14B-EAGLE3-Finance",
        target_model="Qwen/Qwen3-14B",
        out_path=out,
    )
    content = out.read_text(encoding="utf-8")
    assert "Qwen/Qwen3-14B" in content
    assert "Qwen3-14B-EAGLE3-Finance" in content
    assert "base_model: Qwen/Qwen3-14B" in content
