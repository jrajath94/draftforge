#!/usr/bin/env python3
"""DraftForge end-to-end demo orchestrator.

Runs the full pipeline shape on a laptop with NO GPU, NO HF auth, NO network.
Every emitted artifact carries an `is_demo: true` watermark so it can never be
confused with measured results.

Pipeline stages (mirrors `scripts/run_full_pipeline.sh` but with synthetic data):
  1. data.prepare  --config data/demo_config.yaml --skip-tokenize
  2. mock training: 4 variants x 3 seeds -> loss.csv per (variant, seed)
  3. ablate.compare: combine variant summaries -> comparison.json
  4. mock acceptance: synthetic acceptance_grid.csv
  5. eval.crossover_analysis: derive batch-size crossover (mock)
  6. release.aggregate: walk results/demo/ → manifest.json (is_demo: true)
  7. release.make_card: render HF card from manifest

All outputs land under `./results/demo/`. Re-running is idempotent.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --results-root ./results/demo
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from pathlib import Path

# Repo root: parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Match ablate.configs.PRESETS keys exactly so ablate.compare picks them up.
VARIANTS = ("tri_layer", "final_layer", "low_only", "mid_only")
SEEDS = (42, 0, 1234)
TEMPERATURES = (0.0, 0.7, 1.0)
BATCH_SIZES = (1, 4, 16)
DOMAINS = ("general", "finance")


def log(msg: str) -> None:
    print(f"[demo] {msg}", flush=True)


def write_demo_marker(results_root: Path) -> None:
    """Drop a clearly-marked file so anyone landing in the dir knows it's demo."""
    marker = results_root / "IS_DEMO.md"
    marker.write_text(
        "# Demo output — synthetic, not measured\n\n"
        "All artifacts under this directory were produced by `scripts/run_demo.py`.\n"
        "Numbers are shape-true synthetic data, NOT benchmark results.\n\n"
        "For real measurements, run `bash scripts/run_full_pipeline.sh` on an H100 pod.\n",
        encoding="utf-8",
    )


def step_data_prepare(results_root: Path) -> None:
    """Run the real data.prepare orchestrator on the bundled fixture."""
    log("stage 1: data.prepare on sample finance fixture")
    cfg = ROOT / "data" / "demo_config.yaml"
    # Override output_dir to live under the demo results root
    cmd = [
        sys.executable,
        "-m",
        "data.prepare",
        "--config",
        str(cfg),
        "--skip-tokenize",
        "--seed",
        "7",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        log(f"data.prepare failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(result.returncode)
    log(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "data.prepare OK")


def synth_loss_curve(variant: str, seed: int, n_steps: int = 100) -> list[tuple[int, float]]:
    """Generate a realistic-looking loss curve for one (variant, seed).

    Pattern: warmup (steep drop), cosine decay to a floor, mild noise.
    Tri-layer (our recommended) converges fastest and lowest.
    """
    rng = random.Random(hash((variant, seed)) & 0xFFFFFFFF)
    floors = {"tri_layer": 1.85, "final_layer": 1.95, "low_only": 2.10, "mid_only": 2.00}
    starts = {"tri_layer": 3.6, "final_layer": 3.7, "low_only": 3.8, "mid_only": 3.7}
    start = starts[variant] + rng.uniform(-0.1, 0.1)
    floor = floors[variant] + rng.uniform(-0.05, 0.05)
    curve: list[tuple[int, float]] = []
    for step in range(1, n_steps + 1):
        # Warmup over 5 steps, then cosine to floor
        if step <= 5:
            v = start - (start - start * 0.85) * (step / 5.0)
        else:
            progress = (step - 5) / max(1, n_steps - 5)
            v = start * 0.85 + (floor - start * 0.85) * (1 - math.cos(math.pi * progress)) / 2
        v += rng.gauss(0, 0.02)
        curve.append((step, round(v, 4)))
    return curve


def _write_loss_csv(path: Path, curve: list[tuple[int, float]]) -> None:
    """Write a loss curve CSV in the schema ablate.compare._read_loss_csv expects:
    `step,loss,lr` (lr column may be empty — ablate.compare ignores it).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "lr"])
        for step, loss in curve:
            w.writerow([step, loss, ""])


def step_mock_training(results_root: Path) -> None:
    """Emit synthetic loss curves.

    Two layouts (matching the real pipeline):
      - results/demo/train/<seed>/loss.csv            (aggregator expects this — flat)
      - results/demo/ablate/<variant>/<seed>/loss_curve.csv  (ablate.compare expects this)
    """
    log("stage 2: mock training - 4 variants x 3 seeds + flat train/ for aggregator")
    # Flat train/ - only the recommended variant goes here (aggregator assumes one)
    train_root = results_root / "train"
    for seed in SEEDS:
        seed_dir = train_root / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        _write_loss_csv(seed_dir / "loss.csv", synth_loss_curve("tri_layer", seed))
        summary = {
            "variant": "tri_layer",
            "seed": seed,
            "is_demo": True,
            "final_loss": synth_loss_curve("tri_layer", seed)[-1][1],
            "n_steps": 100,
        }
        (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Full variant grid under ablate/ — ablate.compare iterates PRESETS
    ablate_data_root = results_root / "ablate_data"
    for variant in VARIANTS:
        for seed in SEEDS:
            seed_dir = ablate_data_root / variant / str(seed)
            seed_dir.mkdir(parents=True, exist_ok=True)
            _write_loss_csv(seed_dir / "loss_curve.csv", synth_loss_curve(variant, seed))
    log(f"wrote {len(SEEDS)} train/ seeds + {len(VARIANTS) * len(SEEDS)} ablate curves")


def step_ablation_compare(results_root: Path) -> None:
    """Aggregate variant summaries into comparison.json via the real ablate.compare API.

    ablate.compare is a library, not a CLI — invoke compare_variants + write_comparison
    directly so the demo exercises the same code path as the real pipeline.
    """
    log("stage 3: ablation — aggregating variants → comparison.json")
    from ablate.compare import compare_variants, write_comparison

    source_root = results_root / "ablate_data"
    ablate_out = results_root / "ablate"
    ablate_out.mkdir(parents=True, exist_ok=True)
    by_variant = compare_variants(source_root)
    write_comparison(by_variant, ablate_out / "comparison.json")
    # Stamp is_demo on every variant entry so aggregator-level consumers know
    for v in by_variant.values():
        v["is_demo"] = True
    (ablate_out / "comparison.json").write_text(
        json.dumps(by_variant, indent=2, sort_keys=True), encoding="utf-8"
    )
    log(f"wrote {ablate_out / 'comparison.json'} ({len(by_variant)} variants)")


def synth_acceptance_grid(domains: tuple[str, ...], temps: tuple[float, ...], batches: tuple[int, ...]) -> list[dict]:
    """Synthetic geometric-acceptance grid — pattern matches real EAGLE-3 behaviour."""
    rng = random.Random(20260708)
    rows: list[dict] = []
    for domain in domains:
        # Tri-layer head (our best) vs non-spec baseline: eal ratio ~ 1.6-2.4x
        base_p = {"general": 0.62, "finance": 0.71}[domain]
        for temp in temps:
            # Higher temperature slightly reduces acceptance
            p = base_p - 0.03 * abs(temp - 0.7)
            for b in batches:
                p_jitter = p + rng.gauss(0, 0.01)
                p_jitter = max(0.05, min(0.99, p_jitter))
                eal = 1.0 / (1.0 - p_jitter)
                # ITL model: spec helps more at high batch (decode-bound)
                base_itl = 35.0 + b * 2.0
                spec_itl = base_itl * (1.0 - min(0.5, 0.05 + 0.1 * math.log2(max(1, b))))
                rows.append(
                    {
                        "domain": domain,
                        "temperature": temp,
                        "batch_size": b,
                        "mean_acceptance": round(p_jitter, 4),
                        "eal": round(eal, 4),
                        "itl_ms": round(spec_itl, 3),
                    }
                )
    return rows


def step_mock_acceptance(results_root: Path) -> None:
    log("stage 4: mock acceptance grid + crossover analysis")
    eval_root = results_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    rows = synth_acceptance_grid(DOMAINS, TEMPERATURES, BATCH_SIZES)
    grid_path = eval_root / "acceptance_grid.csv"
    with grid_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["domain", "temperature", "batch_size", "mean_acceptance", "eal", "itl_ms"],
        )
        w.writeheader()
        w.writerows(rows)
    # Run eval.crossover_analysis for shape — takes --grid and --out
    cmd = [
        sys.executable,
        "-m",
        "eval.crossover_analysis",
        "--grid",
        str(grid_path),
        "--out",
        str(eval_root / "crossover_analysis.md"),
    ]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    # Materialize the acceptance-by-batch plot so `make figures` passes.
    # The plot module is real; only the demo was missing the call.
    from eval.acceptance import load_grid
    from eval.plot import plot_acceptance_by_batch

    grid_rows = load_grid(grid_path)
    if grid_rows:
        plot_acceptance_by_batch(
            grid_rows,
            eval_root / "acceptance_by_batch.png",
            title="EAGLE-3 Acceptance Length vs Batch Size (demo)",
        )
    log(f"wrote {len(rows)} acceptance rows + crossover report + plot")


def step_release(results_root: Path) -> None:
    log("stage 5: aggregate + render HF card")
    # Aggregate
    aggregate_cmd = [
        sys.executable,
        "-m",
        "release.__main__",
        "aggregate",
        "--results-root",
        str(results_root),
        "--out",
        str(results_root / "manifest.json"),
    ]
    result = subprocess.run(aggregate_cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        log(f"release.aggregate failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(result.returncode)
    # Inject is_demo into manifest root
    manifest_path = results_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["is_demo"] = True
        manifest["warning"] = (
            "SYNTHETIC demo output from scripts/run_demo.py. "
            "Numbers are shape-true mock data, NOT benchmark results."
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    # Render HF card
    card_cmd = [
        sys.executable,
        "-m",
        "release.__main__",
        "make-card",
        "--template",
        str(ROOT / "release" / "hf_card.md"),
        "--results",
        str(results_root),
        "--head",
        "demo-eagle3-head",
        "--target",
        "Qwen/Qwen3-14B",
        "--out",
        str(results_root / "HF_CARD.md"),
    ]
    subprocess.run(card_cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    log("aggregated + card rendered")


def main() -> int:
    parser = argparse.ArgumentParser(description="DraftForge end-to-end demo runner")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "demo")
    args = parser.parse_args()

    results_root: Path = args.results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    log(f"results root: {results_root}")
    write_demo_marker(results_root)

    step_data_prepare(results_root)
    step_mock_training(results_root)
    step_ablation_compare(results_root)
    step_mock_acceptance(results_root)
    step_release(results_root)

    log("DONE. Outputs:")
    log(f"  {results_root}/train/<seed>/loss.csv  (3 seeds, flat)")
    log(f"  {results_root}/ablate_data/<variant>/<seed>/loss_curve.csv  (4 variants x 3 seeds)")
    log(f"  {results_root}/ablate/comparison.json")
    log(f"  {results_root}/eval/acceptance_grid.csv")
    log(f"  {results_root}/eval/crossover_analysis.md")
    log(f"  {results_root}/manifest.json  (with is_demo=true warning)")
    log(f"  {results_root}/HF_CARD.md")
    log("For real measurements, run: bash scripts/run_full_pipeline.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
