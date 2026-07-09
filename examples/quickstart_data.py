"""Quickstart: inspect the DraftForge data config.

Run: .venv/bin/python examples/quickstart_data.py

No GPU, no HF, no network. Loads data/config.yaml via the pydantic
schema and prints the resolved view. Useful for verifying YAML edits
before kicking off a long-running data preparation job.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.config import load_config


def main() -> int:
    print("=" * 60)
    print("DraftForge quickstart: data config inspection")
    print("=" * 60)

    cfg_path = Path("data/config.yaml")
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found. Run from the repo root.")
        return 1

    cfg = load_config(cfg_path)

    # Render a human-friendly view of the resolved config.
    print(f"\n[1] Loaded {cfg_path}")
    print(f"    seed:        {cfg.seed}")
    print(f"    output_dir:  {cfg.output_dir}")
    print(f"    sources:     {len(cfg.sources)}")
    for i, src in enumerate(cfg.sources, 1):
        print(f"      [{i}] {src.name:<24} type={src.type:<10} domain={src.domain}")
        if src.hf_dataset_id:
            print(f"           hf_dataset_id: {src.hf_dataset_id}")
        if src.path:
            print(f"           path:          {src.path}")
        if src.max_examples:
            print(f"           max_examples:  {src.max_examples}")
    print(f"    dedup:       {cfg.dedup.method} (minhash_thr={cfg.dedup.minhash_threshold})")
    print(
        f"    split:       {int(cfg.split.train_ratio * 100)}/"
        f"{int(cfg.split.val_ratio * 100)}/{int(cfg.split.test_ratio * 100)}"
        f"  stratify_by={cfg.split.stratify_by}"
    )
    print(
        f"    tokenizer:   {cfg.tokenizer.name_or_path} (max_length={cfg.tokenizer.max_length})"
    )

    # 2. Print the JSON-serializable form (the same shape that
    # data/prepare.py writes to artifacts/data/results/pipeline_summary.json).
    print("\n[2] JSON-serializable form (ready for pipeline_summary.json):")
    print(json.dumps(cfg.model_dump(mode="json"), indent=2))

    print("\n" + "=" * 60)
    print("done. To run the full data pipeline (requires HF auth + tokenizer):")
    print("    $ python -m data.prepare --config data/config.yaml")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
