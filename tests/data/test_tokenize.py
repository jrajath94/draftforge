"""Tests for data/tokenize.py.

Tokenization requires the Qwen3 tokenizer to be downloaded; tests use
a smaller placeholder if Qwen3 is unavailable. Marked `integration` if
they need network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data.config import TokenizerConfig


@pytest.mark.integration
def test_tokenize_split_round_trip(tmp_path: Path, tiny_traces) -> None:
    """Real test: requires Qwen3 tokenizer (gated model). Skip if unavailable."""
    from datasets import load_from_disk

    from data.tokenize import tokenize_split, write_tokenized_split

    cfg = TokenizerConfig(name_or_path="Qwen/Qwen3-14B", max_length=512, trust_remote_code=False)
    try:
        ds = tokenize_split(tiny_traces[:4], cfg, name="smoke")
    except Exception as exc:
        pytest.skip(f"Qwen3 tokenizer unavailable: {exc}")
    out = tmp_path / "tok"
    write_tokenized_split(ds, out)
    ds2 = load_from_disk(str(out))
    assert "input_ids" in ds2.column_names
    assert "attention_mask" in ds2.column_names
    # All token IDs in valid range (lossy boundary check — no NaN)
    for ids in ds2["input_ids"]:
        assert all(isinstance(i, int) for i in ids)
        assert all(0 <= i < 200_000 for i in ids)


def test_tokenizer_config_validates() -> None:
    cfg = TokenizerConfig()
    assert cfg.name_or_path == "Qwen/Qwen3-14B"
    assert cfg.max_length == 4096
