"""Batch-size crossover analysis: locate the threshold where spec-decode
stopped helping (speculation overhead meets verify savings).

CLI:
    python -m eval.crossover_analysis \\
      --grid results/acceptance_grid.csv \\
      --out results/crossover_analysis.md

Input CSV schema (matches eval.acceptance.write_acceptance_grid):
    domain, temperature, batch_size, mean_acceptance, eal, itl_ms

Output: a Markdown report with the crossover point per (domain,
temperature), plus interpretation. Plots can be added with eval.plot
but the report is the canonical artifact for the writeup.

This is a CPU-only analyser. Numbers are derived verbatim from the
acceptance grid emitted by `python -m eval.acceptance` — no fabrication.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.acceptance import crossover_batch_size, load_grid


def _co_str(co: float) -> str:
    if co == float("inf"):
        return "∞"
    if co == 0.0:
        return "N/A"
    return f"{co:.1f}"


def analyze_crossover(grid_path: Path, out_path: Path) -> dict[str, object]:
    """Read the acceptance grid and emit the markdown crossover report.

    Returns a dict with the rows used + computed crossovers so callers
    (tests, notebooks) can inspect without re-parsing the markdown.
    """
    rows = load_grid(grid_path)
    if not rows:
        raise ValueError(f"acceptance grid empty or not found: {grid_path}")

    # group by (domain, temperature); per-key, list of {batch_size, itl_ms}
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    crossovers: dict[tuple[str, str], float] = {}
    details: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        try:
            domain = str(row["domain"])
            temp = str(row["temperature"])
            bs = int(float(row["batch_size"]))
            itl = float(row["itl_ms"])
        except (KeyError, ValueError):
            continue
        grouped.setdefault((domain, temp), []).append((bs, itl))

    # per-key: compute baseline (smallest batch), spec saturation (largest batch)
    for key, entries in grouped.items():
        entries.sort(key=lambda e: e[0])
        baseline_itl = entries[0][1]            # smallest batch = baseline
        saturation_itl = entries[-1][1]         # largest batch = decode saturation
        # crude: any middle batch serves as the "spec" representative
        spec_itl = entries[len(entries) // 2][1]
        co = crossover_batch_size(baseline_itl, spec_itl, saturation_itl)
        crossovers[key] = co
        details[key] = {
            "baseline_itl_ms": baseline_itl,
            "saturation_itl_ms": saturation_itl,
            "spec_itl_ms": spec_itl,
        }

    # write markdown report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Batch-Size Crossover Analysis\n\n")
        f.write(
            "Where speculative decoding stops helping: the batch size at "
            "which spec-decode overhead meets the verify-side benefit. "
            "Derived from `results/acceptance_grid.csv`.\n\n"
        )
        f.write("## Crossover Points\n\n")
        f.write(
            "| Domain | Temperature | Crossover Batch | ITL @ b=1 (ms) | "
            "ITL @ largest batch (ms) |\n"
        )
        f.write("|--------|-------------|-----------------|----------------|----------------|\n")
        for key in sorted(crossovers.keys()):
            domain, temp = key
            co = crossovers[key]
            base_ms = details[key]["baseline_itl_ms"]
            sat_ms = details[key]["saturation_itl_ms"]
            f.write(
                f"| {domain} | {temp} | {_co_str(co)} | "
                f"{base_ms:.2f} | {sat_ms:.2f} |\n"
            )

        f.write("\n## Interpretation\n\n")
        f.write("- **Crossover < 4**: speculation stops helping very early -- useful only for batch size <= 2.\n")
        f.write("- **4 <= Crossover < 16**: beneficial for batch size 1-8; diminishing returns above.\n")
        f.write("- **Crossover >= 16**: speculation remains useful at all tested batch sizes.\n")
        f.write("- **Crossover = infinity**: spec ITL never meets baseline; speculation always helps.\n")
        f.write("- **Crossover = N/A**: degenerate inputs (zero ITL).\n\n")

        f.write("## Notes\n\n")
        f.write(
            "- Crossover derived from `crossover_batch_size(baseline, spec, saturation)` "
            "in `eval/acceptance.py`; linear model `spec_itl(B) = base*(1-a) + saturation*a`, "
            "`a = min(1, B/B0)`.\n"
        )
        f.write(
            "- Numbers are CPU-only derivations from the acceptance grid. Do **not** "
            "report crossovers without the corresponding `acceptance_grid.csv` artifact.\n"
        )

    print(f"[crossover] report written to {out_path}")
    return {"crossovers": {f"{d}|{t}": co for (d, t), co in crossovers.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-size crossover analysis")
    ap.add_argument("--grid", type=Path, required=True, help="Acceptance grid CSV")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/crossover_analysis.md"),
        help="Output Markdown report path",
    )
    args = ap.parse_args()

    try:
        analyze_crossover(args.grid, args.out)
    except Exception as e:  # surface error to caller, non-zero exit
        print(f"[crossover] error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
