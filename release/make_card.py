"""Render the HuggingFace model card from results + template."""

from __future__ import annotations

from pathlib import Path
from string import Template

from release.aggregate import aggregate


def render_card(
    template_path: Path,
    results_root: Path,
    head_name: str,
    target_model: str,
    out_path: Path,
) -> None:
    manifest = aggregate(results_root)
    tpl = template_path.read_text(encoding="utf-8")
    rendered = Template(tpl).substitute(
        HEAD_NAME=head_name,
        TARGET_MODEL=target_model,
        MANIFEST_JSON=str(manifest).replace("'", '"'),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")


def main(template: Path, results: Path, head: str, target: str, out: Path) -> int:
    render_card(template, results, head, target, out)
    print(f"wrote {out}")
    return 0
