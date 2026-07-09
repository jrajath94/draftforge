"""Orchestrator CLI: ingest → dedup → split → tokenize → plot.

Single command:

    python -m data.prepare --config data/config.yaml [--seed N] [--skip-tokenize]

Writes to <output_dir>:
    splits/{train,val,test}.jsonl       — raw Example JSONL
    tokenized/{train,val,test}          — HF Dataset (parquet)
    results/data/dedup_counts.json      — before/after breakdown
    results/data/splits_sha256.json     — verifier for reproducibility
    results/data/tokenization_meta.json — tokenizer stats
    results/data/domain_distribution.png — visualization
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from data.config import DataConfig, SourceConfig, SourceType, load_config
from data.dedup import deduplicate, write_counts_log
from data.plot import plot_domain_distribution
from data.sources.edgar import load_edgar_finance
from data.sources.finance import load_finance
from data.sources.openhermes import load_openhermes
from data.sources.sharegpt import load_sharegpt
from data.splits import (
    domain_counts,
    stratified_split,
    write_split_jsonl,
    write_splits_sha256_log,
)
from data.tokenize import (
    tokenize_split,
    write_tokenization_meta,
    write_tokenized_split,
)
from data.types import Example

app = typer.Typer(help="DraftForge data pipeline orchestrator")

# Module-level typer.Options satisfy ruff B008 (no function-call defaults).
_CONFIG_OPTION = typer.Option(..., "--config", help="Path to data config YAML.")
_SEED_OPTION: int | None = typer.Option(None, "--seed", help="Override config seed.")
_SKIP_TOKENIZE_OPTION: bool = typer.Option(
    False, "--skip-tokenize", help="Skip Qwen3 tokenization (faster smoke test)."
)


def _load_source(src_cfg: SourceConfig) -> list[Example]:
    """Dispatch by source type."""
    if src_cfg.type == SourceType.SHAREGPT:
        assert src_cfg.hf_dataset_id is not None
        return load_sharegpt(
            hf_dataset_id=src_cfg.hf_dataset_id,
            max_examples=src_cfg.max_examples,
        )
    if src_cfg.type == SourceType.OPENHERMES:
        assert src_cfg.hf_dataset_id is not None
        return load_openhermes(
            hf_dataset_id=src_cfg.hf_dataset_id,
            max_examples=src_cfg.max_examples,
        )
    if src_cfg.type == SourceType.FINANCE:
        return load_finance(
            hf_dataset_id=src_cfg.hf_dataset_id,
            path=src_cfg.path,
            max_examples=src_cfg.max_examples,
        )
    if src_cfg.type == SourceType.EDGAR:
        return load_edgar_finance(
            ciks=src_cfg.ciks,
            path=src_cfg.path,
            max_examples=src_cfg.max_examples,
            user_agent=src_cfg.user_agent,
            offline=src_cfg.path is not None and src_cfg.hf_dataset_id is None,
        )
    raise ValueError(f"unknown source type: {src_cfg.type}")


def run(cfg: DataConfig, skip_tokenize: bool = False) -> dict:
    """End-to-end pipeline. Returns summary stats."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = cfg.output_dir / "results" / "data"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest
    print("[1/5] ingesting sources...", flush=True)
    all_examples: list[Example] = []
    for src in cfg.sources:
        print(f"  - {src.name} ({src.type.value}, max={src.max_examples})", flush=True)
        all_examples.extend(_load_source(src))
    print(f"  total raw examples: {len(all_examples)}", flush=True)

    # 2. Dedup
    print(f"[2/5] deduplicating ({cfg.dedup.method.value})...", flush=True)
    deduped = deduplicate(
        all_examples,
        cfg.dedup.method,
        threshold=cfg.dedup.minhash_threshold,
        num_perm=cfg.dedup.num_perm,
    )
    write_counts_log(
        results_dir / "dedup_counts.json",
        before=all_examples,
        after=deduped,
        method=cfg.dedup.method.value,
    )
    print(f"  after dedup: {len(deduped)} (removed {len(all_examples) - len(deduped)})", flush=True)

    # 3. Split
    print(f"[3/5] stratified split (ratios={cfg.split.train_ratio}/{cfg.split.val_ratio}/{cfg.split.test_ratio})...", flush=True)
    train, val, test = stratified_split(deduped, cfg.split, cfg.seed)
    splits_dir = cfg.output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_p = splits_dir / "train.jsonl"
    val_p = splits_dir / "val.jsonl"
    test_p = splits_dir / "test.jsonl"
    write_split_jsonl(train, train_p)
    write_split_jsonl(val, val_p)
    write_split_jsonl(test, test_p)
    hashes = write_splits_sha256_log(
        results_dir / "splits_sha256.json",
        train_p,
        val_p,
        test_p,
        cfg.seed,
    )
    print(f"  sizes: train={len(train)} val={len(val)} test={len(test)}", flush=True)

    # 4. Tokenize (may be skipped for quick smoke tests)
    splits_meta: dict[str, dict] = {}
    if not skip_tokenize:
        print(f"[4/5] tokenizing ({cfg.tokenizer.name_or_path}, max_len={cfg.tokenizer.max_length})...", flush=True)
        tok_dir = cfg.output_dir / "tokenized"
        for name, examples in [("train", train), ("val", val), ("test", test)]:
            ds = tokenize_split(examples, cfg.tokenizer, name)
            write_tokenized_split(ds, tok_dir / name)
            splits_meta[name] = {
                "num_rows": len(ds),
                "avg_len": float(sum(ds["_len"]) / max(len(ds), 1)),
                "max_len": int(max(ds["_len"]) if len(ds) else 0),
            }
        write_tokenization_meta(results_dir / "tokenization_meta.json", cfg.tokenizer, splits_meta)
        print(f"  tokenized splits saved to {tok_dir}", flush=True)
    else:
        print("[4/5] tokenization SKIPPED (--skip-tokenize)", flush=True)

    # 5. Plot
    print("[5/5] plotting domain distribution...", flush=True)
    plot_domain_distribution(
        domain_counts(train),
        domain_counts(val),
        domain_counts(test),
        results_dir / "domain_distribution.png",
    )
    print(f"  saved {results_dir / 'domain_distribution.png'}", flush=True)

    summary = {
        "seed": cfg.seed,
        "raw_examples": len(all_examples),
        "after_dedup": len(deduped),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "splits_sha256": hashes,
        "tokenized": not skip_tokenize,
    }
    with (results_dir / "pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("done.", flush=True)
    return summary


@app.command()
def main(
    config: Path = _CONFIG_OPTION,
    seed: int | None = _SEED_OPTION,
    skip_tokenize: bool = _SKIP_TOKENIZE_OPTION,
) -> None:
    """Run the DraftForge data pipeline end-to-end."""
    cfg = load_config(config)
    if seed is not None:
        cfg = cfg.model_copy(update={"seed": seed})
    run(cfg, skip_tokenize=skip_tokenize)


if __name__ == "__main__":
    sys.exit(app())
