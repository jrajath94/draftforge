"""Quickstart: geometric EAL + batch-size crossover from a synthetic grid.

Run: .venv/bin/python examples/quickstart_acceptance.py

No GPU, no HF, no network. Demonstrates the public API of eval/acceptance.py
and eval/crossover_analysis.py.

Adjust MEAN_ACCEPTANCE / BASELINE_ITL / SPEC_ITL / DECODE_SAT to explore
how the EAL and crossover model respond.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.acceptance import (
    crossover_batch_size,
    expected_acceptance_length,
    write_acceptance_grid,
)
from eval.crossover_analysis import analyze_crossover


def main() -> int:
    print("=" * 60)
    print("DraftForge quickstart: geometric EAL + crossover")
    print("=" * 60)

    # 1. Expected acceptance length at varying per-token acceptance rates.
    print("\n[1] Geometric EAL E[c] = 1 / (1 - p), capped at horizon H=8")
    print("    p        E[c]    interpretation")
    for p in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        eal = expected_acceptance_length(p, horizon=8)
        interp = {
            0.0: "no acceptance, no speedup",
            0.25: "1.33x speedup ceiling",
            0.5: "2x speedup ceiling",
            0.75: "4x speedup ceiling",
            0.9: "capped at horizon (8)",
            1.0: "every draft accepted, capped at horizon (8)",
        }[p]
        print(f"    {p:.2f}     {eal:.2f}    {interp}")

    # 2. Build a synthetic acceptance grid (2 domains x 3 temperatures x 5 batches).
    print("\n[2] Synthetic acceptance grid")
    rows: list[dict[str, float | int | str]] = []
    for domain in ("general", "finance"):
        for temp in (0.0, 0.7, 1.0):
            # Finance + high-temp drags acceptance down (the domain-shift hypothesis).
            base_p = 0.75 if domain == "general" else 0.65
            temp_penalty = 0.05 * abs(temp - 0.7) / 0.7
            for batch in (1, 4, 8, 16, 32):
                p = max(0.0, base_p - temp_penalty - 0.005 * batch)
                itl_baseline = 50.0 + 0.5 * batch  # ms, baseline (no spec)
                itl_spec = 30.0 + 0.4 * batch + (5.0 if domain == "finance" else 0.0)
                rows.append(
                    {
                        "domain": domain,
                        "temperature": temp,
                        "batch_size": batch,
                        "mean_acceptance": round(p, 4),
                        "eal": round(expected_acceptance_length(p, horizon=8), 4),
                        "itl_ms": round(itl_spec, 2),
                    }
                )
                # Baseline row (itl_ms = baseline, eal=1) — for the plotter to overlay.
                rows.append(
                    {
                        "domain": domain,
                        "temperature": temp,
                        "batch_size": batch,
                        "mean_acceptance": 0.0,
                        "eal": 1.0,
                        "itl_ms": round(itl_baseline, 2),
                        "_baseline": True,  # type: ignore[dict-item]
                    }
                )

    out_dir = Path("examples/_out")
    out_dir.mkdir(exist_ok=True)
    grid_path = out_dir / "acceptance_grid.csv"
    write_acceptance_grid(rows, grid_path)
    print(f"    wrote {grid_path} ({len(rows)} rows)")

    # 3. Compute batch-size crossover for one (domain, temperature) cell.
    print("\n[3] Batch-size crossover for general/T=0.7")
    baseline_itl = 50.0  # ms at b=1
    spec_itl = 30.0  # ms at b=1
    decode_sat = 50.0  # ms when decode saturates
    b_star = crossover_batch_size(baseline_itl, spec_itl, decode_sat)
    print(f"    baseline={baseline_itl}ms  spec={spec_itl}ms  decode_sat={decode_sat}ms")
    print(f"    crossover B* = {b_star:.2f}")
    if b_star == float("inf"):
        print("    → speculation loses at all batch sizes; do not enable")
    elif b_star == 1.0:
        print("    → speculation wins unconditionally; enable always")
    else:
        print(f"    → enable speculation for batch_size ≤ {b_star:.0f}")

    # 4. Run the full per-key crossover analyzer.
    print("\n[4] Per-key crossover report")
    report_path = out_dir / "crossover_analysis.md"
    report = analyze_crossover(grid_path, report_path)
    print(f"    wrote {report_path}")
    summary = {
        k: v for k, v in report["crossovers"].items() if isinstance(v, (int, float))
    }
    print(f"    {len(summary)} (domain, temperature) cells analyzed")
    print(f"    sample: {json.dumps(dict(list(summary.items())[:2]), indent=2)}")

    print("\n" + "=" * 60)
    print("done. Inspect examples/_out/ for the generated artifacts.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
