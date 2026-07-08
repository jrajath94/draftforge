"""CLI entry points for release package.

Usage:
    python -m release.aggregate --results-root ./results --out ./results/manifest.json
    python -m release.make_card --template release/hf_card.md --head NAME --target MODEL \
        --results ./results --out ./release/CARD.md
"""

from __future__ import annotations

from pathlib import Path

import typer

from release.aggregate import main as aggregate_main
from release.make_card import main as card_main

app = typer.Typer(help="DraftForge release utilities")

_RESULTS_ROOT_OPT = typer.Option(..., "--results-root", help="results/ dir")
_OUT_OPT = typer.Option(..., "--out", help="output manifest.json")
_TEMPLATE_OPT = typer.Option(..., "--template")
_RESULTS_OPT = typer.Option(..., "--results")
_HEAD_OPT = typer.Option(..., "--head")
_TARGET_OPT = typer.Option(..., "--target")
_OUT_OPT_2 = typer.Option(..., "--out")


@app.command("aggregate")
def aggregate_cmd(
    results_root: Path = _RESULTS_ROOT_OPT,
    out: Path = _OUT_OPT,
) -> None:
    raise SystemExit(aggregate_main(results_root, out))


@app.command("make-card")
def make_card_cmd(
    template: Path = _TEMPLATE_OPT,
    results: Path = _RESULTS_OPT,
    head: str = _HEAD_OPT,
    target: str = _TARGET_OPT,
    out: Path = _OUT_OPT_2,
) -> None:
    raise SystemExit(card_main(template, results, head, target, out))


if __name__ == "__main__":
    app()
