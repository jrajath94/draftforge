"""Compare per-seed loss curves across ablation variants.

Reads results/train/<variant>/<seed>/loss_curve.csv, computes final-100-step
mean ± std for each (variant, seed), emits a comparison table (JSON + CSV).

Important: this is CPU-only analysis. Numbers are committed verbatim from
each seed's loss_curve.csv — no interpolation, no fabrication.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ablate.configs import PRESETS


def _read_loss_csv(path: Path) -> list[tuple[int, float]]:
    """Return [(step, loss), ...] parsed from a CSV. Skip header.

    Expected schema: header `step,loss,lr` (extra columns ignored).
    Malformed rows raise ValueError naming the source path so the caller
    knows which seed/variant produced the bad CSV.
    """
    if not path.exists():
        return []
    rows: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()  # skip
        if "step" not in header or "loss" not in header:
            raise ValueError(
                f"{path}: header missing required 'step' or 'loss' column; got: {header.strip()!r}"
            )
        for lineno, line in enumerate(f, start=2):  # header is line 1
            try:
                step_s, loss_s, _ = line.strip().split(",")
                rows.append((int(step_s), float(loss_s)))
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"{path}:{lineno}: malformed row {line.strip()!r}: {e}"
                ) from e
    return rows


def _final_mean(
    rows: list[tuple[int, float]], window: int = 100
) -> tuple[float, float]:
    if not rows:
        return (0.0, 0.0)
    last = rows[-window:]
    losses = [r[1] for r in last]
    n = len(losses)
    mean = sum(losses) / n
    if n < 2:
        return (mean, 0.0)
    var = sum((x - mean) ** 2 for x in losses) / (n - 1)
    return (mean, var ** 0.5)


def compare_variants(results_root: Path) -> dict[str, dict]:
    """Aggregate per-variant final loss across all seed subdirs.

    Returns:
        {variant_name: {"per_seed": {seed: mean}, "mean_of_means": float,
                        "std_of_means": float, "n_seeds": int}}
    """
    by_variant: dict[str, dict] = {}
    for variant in PRESETS:
        variant_dir = results_root / variant
        per_seed: dict[str, float] = {}
        if not variant_dir.is_dir():
            by_variant[variant] = {"per_seed": {}, "mean_of_means": 0.0,
                                   "std_of_means": 0.0, "n_seeds": 0}
            continue
        for seed_dir in sorted(variant_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            csv_p = seed_dir / "loss_curve.csv"
            rows = _read_loss_csv(csv_p)
            mean, _ = _final_mean(rows)
            if rows:
                per_seed[seed_dir.name] = mean
        means = list(per_seed.values())
        if means:
            n = len(means)
            mean_of_means = sum(means) / n
            var = (
                sum((x - mean_of_means) ** 2 for x in means) / (n - 1)
                if n > 1
                else 0.0
            )
        else:
            mean_of_means = 0.0
            var = 0.0
        by_variant[variant] = {
            "per_seed": per_seed,
            "mean_of_means": mean_of_means,
            "std_of_means": var ** 0.5,
            "n_seeds": len(means),
        }
    return by_variant


def write_comparison(by_variant: dict[str, dict], out_path: Path) -> None:
    """Write JSON comparison table. CSV for human reading too."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(by_variant, f, indent=2, sort_keys=True)
    # Pretty table
    csv_p = out_path.with_suffix(".csv")
    with csv_p.open("w", encoding="utf-8") as f:
        f.write("variant,n_seeds,mean_final_loss,std_final_loss\n")
        for variant, stats in sorted(by_variant.items()):
            f.write(
                f"{variant},{stats['n_seeds']},"
                f"{stats['mean_of_means']:.6f},"
                f"{stats['std_of_means']:.6f}\n"
            )


def default_results_root() -> Path:
    return Path("results/train")


# ---- CLI -----------------------------------------------------------------


def main(results_root: Path, out: Path) -> int:
    """Walk variant/seed/loss_curve.csv under `results_root`, write comparison.

    Always writes `out` (presets keyed with n_seeds=0 if missing) so the
    downstream release.aggregate step never crashes on a missing file.
    """
    by_variant = compare_variants(results_root)
    n_seeds_total = sum(v.get("n_seeds", 0) for v in by_variant.values())
    write_comparison(by_variant, out)
    if n_seeds_total == 0:
        print(
            f"[ablate.compare] no loss_curve.csv under {results_root}; "
            f"wrote empty comparison to {out}",
            file=sys.stderr,
        )
    else:
        print(
            f"[ablate.compare] aggregated {n_seeds_total} seeds across "
            f"{len(by_variant)} variants -> {out}"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare per-seed loss curves across ablation variants."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="directory containing <variant>/<seed>/loss_curve.csv subdirs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="comparison JSON output path (a .csv sidecar is also written)",
    )
    args = parser.parse_args()
    sys.exit(main(args.results_root, args.out))
