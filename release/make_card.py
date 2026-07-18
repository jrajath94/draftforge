"""Render the HuggingFace model card from results + template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from string import Template
from typing import Any

from release.aggregate import aggregate

NOT_MEASURED = (
    "**[NOT YET MEASURED]** — no benchmark artifacts found under `results/`. "
    "Rendered numbers appear here only after the GPU evidence ladder "
    "(`docs/GPU_COST_OPTIMIZATION.md`) produces real acceptance/ITL grids."
)


def _render_results(manifest: dict[str, Any]) -> str:
    """Render the ## Results body: tables when evidence exists, honest marker when not."""
    sections: list[str] = []

    eval_block = manifest.get("eval", {})
    measured = eval_block.get("measured", {})
    measured_rows = [
        f"| {seed} | {m.get('step', '?')} | {m.get('agreement_rate_greedy', '?')} "
        f"| {m.get('expected_acceptance_length_geometric', '?')} |"
        for seed, m in sorted(measured.items())
        if isinstance(m, dict) and m
    ]
    if measured_rows:
        sections.append(
            "### Measured acceptance (greedy draft/target agreement, held-out val)\n\n"
            "| seed | ckpt step | agreement p | E[accept len] (geometric) |\n"
            "|---|---|---|---|\n" + "\n".join(measured_rows)
        )

    grid = eval_block.get("grid", [])
    if grid:
        cols = list(grid[0].keys())
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join("---" for _ in cols) + "|"
        rows = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in grid]
        sections.append("### Acceptance grid\n\n" + "\n".join([header, sep, *rows]))

    seeds = manifest.get("training", {}).get("seeds", {})
    seed_rows = []
    for seed, curve in sorted(seeds.items()):
        if not curve:
            continue
        tail = [r["loss"] for r in curve[-100:] if "loss" in r]
        final = sum(tail) / len(tail) if tail else float("nan")
        seed_rows.append(f"| {seed} | {len(curve)} | {final:.4f} |")
    if seed_rows:
        sections.append(
            "### Training (per seed)\n\n"
            "| seed | logged steps | final loss (mean of last ≤100 train steps) |\n"
            "|---|---|---|\n" + "\n".join(seed_rows)
        )

    if not sections:
        return NOT_MEASURED
    return "\n\n".join(sections)


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
        MANIFEST_JSON=json.dumps(manifest, indent=2, sort_keys=True, default=str),
        RESULTS_SECTION=_render_results(manifest),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")


def main(template: Path, results: Path, head: str, target: str, out: Path) -> int:
    render_card(template, results, head, target, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render the HuggingFace model card from a template + manifest."
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(
        main(
            args.template, args.results, args.head, args.target, args.out
        )
    )
