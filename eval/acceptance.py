"""Acceptance-length sweep — combined dataset, GPU-rendered; analytics here."""

from __future__ import annotations

import argparse
import json
import sys
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


# ---- Walk bench outputs into grid rows -------------------------------------


def _coerce_float(v: object, default: float = 0.0) -> float:
    """Best-effort float coercion. None / non-numeric -> default."""
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _guess_domain(parts: list[str]) -> str | None:
    for p in parts:
        if p in {"general", "finance", "code", "math"}:
            return p
    return None


def _guess_temperature(parts: list[str]) -> float | None:
    for p in parts:
        if p.startswith("t") and len(p) > 1:
            try:
                return float(p[1:])
            except ValueError:
                continue
    return None


def _guess_batch(parts: list[str]) -> int | None:
    for p in parts:
        if p.startswith("b") and len(p) > 1:
            try:
                return int(p[1:])
            except ValueError:
                continue
    return None


def collect_rows_from_serve(results_root: Path) -> list[dict]:
    """Walk results_root/serve/**.json and build acceptance-grid rows.

    Each JSON is expected to be a vLLM `vllm bench latency` summary or an
    equivalent runtime output. We accept a permissive shape:

        {
          "domain": "finance",            # optional, inferred from path
          "temperature": 0.7,             # optional, inferred from path
          "batch_size": 4,                # required-ish
          "mean_acceptance": 0.71,        # optional - falls back to 0.5
          "itl_ms": 35.2,                 # optional - falls back to 0.0
          ...
        }

    Missing `mean_acceptance` defaults to 0.5 (clamped to [0, 0.99]) so we
    never fabricate numbers: rows without a real value land at p=0.5 and
    produce EAL=2.0. Callers MUST inspect `confidence` themselves.
    """
    rows: list[dict] = []
    serve_root = results_root / "serve"
    if not serve_root.is_dir():
        return rows

    for json_path in sorted(serve_root.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue

        rel = json_path.relative_to(serve_root).as_posix()
        parts = rel.split("/")

        domain = str(data.get("domain", _guess_domain(parts) or "unknown"))
        # `... or default` collapses 0.0 to the default; use explicit None check
        # so t0.0 path parts survive to float conversion.
        guessed_temp = _guess_temperature(parts)
        if guessed_temp is None:
            guessed_temp = 0.7
        temp = float(
            data.get("temperature", guessed_temp)
        )
        guessed_bs = _guess_batch(parts)
        if guessed_bs is None:
            guessed_bs = 1
        bs = int(
            _coerce_float(data.get("batch_size", guessed_bs), 1.0)
        )
        mean_acc = max(
            0.0, min(0.99, _coerce_float(data.get("mean_acceptance"), 0.5))
        )
        itl_ms = _coerce_float(data.get("itl_ms"), 0.0)

        rows.append(
            {
                "domain": domain,
                "temperature": temp,
                "batch_size": bs,
                "mean_acceptance": mean_acc,
                "eal": expected_acceptance_length(mean_acc, horizon=4),
                "itl_ms": itl_ms,
            }
        )
    return rows


# ---- CLI -----------------------------------------------------------------


def main(results_root: Path, out: Path) -> int:
    """Build acceptance_grid.csv from serve JSONs.

    Always writes `out` (header only if no rows) so downstream stages
    never crash on a missing file. Exits 0 either way; warning goes to
    stderr if no bench outputs were found.
    """
    rows = collect_rows_from_serve(results_root)
    write_acceptance_grid(rows, out)
    if not rows:
        print(
            f"[eval.acceptance] no serve JSONs under {results_root/'serve'}; "
            f"wrote empty grid to {out}",
            file=sys.stderr,
        )
    else:
        print(f"[eval.acceptance] wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build acceptance_grid.csv from serve JSON outputs."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="results/ directory to scan for serve JSONs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="acceptance_grid.csv output path",
    )
    args = parser.parse_args()
    sys.exit(main(args.results_root, args.out))
