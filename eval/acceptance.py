"""Acceptance-length sweep — combined dataset, GPU-rendered; analytics here."""

from __future__ import annotations

import json
from pathlib import Path


def expected_acceptance_length(mean_acceptance: float, horizon: int) -> float:
    """Geometric mean acceptance length given per-token acceptance probability."""
    if 0.0 <= mean_acceptance < 1.0:
        # E[c] = 1 / (1 - p)
        return 1.0 / (1.0 - mean_acceptance)
    return float(horizon) if mean_acceptance >= 1.0 else 1.0


def crossover_batch_size(
    base_itl_ms: float, spec_itl_ms: float, decode_saturation_itl_ms: float
) -> float:
    """Estimate batch size where speculative ITL meets baseline (crossover).

    Linear model:
        spec_itl(B) = base_itl * (1 - alpha) + decode_saturation * alpha
        where alpha = min(1, B / B0).
    """
    if base_itl_ms <= 0 or spec_itl_ms <= 0:
        return 0.0
    if spec_itl_ms >= base_itl_ms:
        return float("inf")
    benefit = base_itl_ms - spec_itl_ms
    saturation = base_itl_ms - decode_saturation_itl_ms
    if saturation <= 0:
        return 1.0
    return benefit / max(1e-6, saturation)


def write_acceptance_grid(
    rows: list[dict], out_path: Path
) -> None:
    """Write acceptance-grid CSV (domain x temperature x batch x acceptance).

    Row keys: domain, temperature, batch_size, mean_acceptance, eal, itl_ms
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("domain,temperature,batch_size,mean_acceptance,eal,itl_ms\n")
        for r in rows:
            f.write(
                f"{r['domain']},{r['temperature']},{r['batch_size']},"
                f"{r['mean_acceptance']:.4f},{r['eal']:.4f},{r['itl_ms']:.3f}\n"
            )


def load_grid(path: Path) -> list[dict]:
    """Reverse of write_acceptance_grid (lightweight CSV reader)."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            rows.append({k: parts[i] for i, k in enumerate(header)})
    return rows


def json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
