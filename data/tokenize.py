"""Qwen3 tokenization via AutoTokenizer. Saves tokenized splits to parquet.

Trainer can `load_from_disk()` and skip all preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from data.config import TokenizerConfig
from data.types import Example


def _load_tokenizer(cfg: TokenizerConfig) -> Any:
    """Return a tokenizer. Typed as Any because transformers' stubs don't
    include dynamic attributes (apply_chat_template, __call__)."""
    return AutoTokenizer.from_pretrained(
        cfg.name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )


def tokenize_split(
    examples: list[Example],
    cfg: TokenizerConfig,
    name: str,
) -> Dataset:
    """Tokenize one split to HF Dataset (input_ids, attention_mask, labels)."""
    tok = _load_tokenizer(cfg)
    # Build raw text per row by concatenating message roles + content
    rows: list[dict[str, Any]] = []
    for ex in examples:
        # Use tokenizer's apply_chat_template if available (Qwen3 has one)
        try:
            text = tok.apply_chat_template(
                ex.messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # Fallback: roll our own simple concat
            text = "\n".join(f"{m['role']}: {m['content']}" for m in ex.messages)
        rows.append(
            {
                "id": ex.id,
                "domain": ex.domain,
                "source": ex.source,
                "text": text,
            }
        )
    ds = Dataset.from_list(rows)

    def _tok_fn(batch: dict[str, Any]) -> dict[str, Any]:
        out = tok(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )
        return out

    ds_tok = ds.map(_tok_fn, batched=True, remove_columns=["text"])
    # Add a length column (debugging + acceptance analysis)
    ds_tok = ds_tok.map(lambda b: {"_len": [len(x) for x in b["input_ids"]]}, batched=True)
    return ds_tok


def write_tokenized_split(ds: Dataset, path: Path) -> None:
    """Save tokenized split as HF Dataset (parquet under the hood)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))


def write_tokenization_meta(
    path: Path,
    cfg: TokenizerConfig,
    splits_meta: dict[str, dict],
) -> None:
    """Sidecar JSON with tokenizer config + per-split stats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokenizer": {
            "name_or_path": cfg.name_or_path,
            "max_length": cfg.max_length,
            "trust_remote_code": cfg.trust_remote_code,
        },
        "splits": splits_meta,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
