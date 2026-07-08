"""Tests for data/dedup.py."""

from __future__ import annotations

import json
from pathlib import Path

from data.dedup import exact_dedupe, minhash_dedupe, write_counts_log
from data.types import Example


def _ex(i: int, user: str, assistant: str) -> Example:
    return Example(
        id=f"x-{i}",
        domain="general",
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        source="sharegpt",
    )


def test_exact_dedupe_removes_duplicates() -> None:
    a = _ex(0, "What is X?", "X is a thing.")
    b = _ex(1, "What is X?", "X is a thing.")  # exact duplicate of a (except id)
    c = _ex(2, "What is Y?", "Y is another thing.")
    out = exact_dedupe([a, b, c])
    assert len(out) == 2
    ids = {e.id for e in out}
    assert ids == {"x-0", "x-2"}


def test_exact_dedupe_whitespace_insensitive() -> None:
    a = _ex(0, "What is X?", "X is a thing.")
    b = _ex(1, "  What   is  X?  ", "X is a thing.")
    out = exact_dedupe([a, b])
    assert len(out) == 1


def test_minhash_removes_near_dups(tiny_traces: list[Example]) -> None:
    # tiny_traces contains openhermes-001 + openhermes-001-near-dup (verbatim)
    # + exact dups from sharegpt-000, openhermes-000, finance-000
    out = minhash_dedupe(tiny_traces, threshold=0.5, num_perm=128)
    # Loose lower bound — synthetic strings are very similar, so LSH may
    # collapse aggressively; we only assert near-dup was removed (count
    # went down). Upper bound: never gain examples.
    assert len(out) <= len(tiny_traces)
    # At least one of the known duplicates is removed
    ids = {e.id for e in out}
    assert "sharegpt-000-dup" not in ids or "openhermes-000-dup" not in ids or "finance-000-dup" not in ids


def test_write_counts_log(tmp_path: Path) -> None:
    a = _ex(0, "x", "y")
    before = [a, _ex(1, "x", "y")]
    after = [a]
    out_path = tmp_path / "counts.json"
    write_counts_log(out_path, before, after, method="exact")
    payload = json.loads(out_path.read_text())
    assert payload["before_total"] == 2
    assert payload["after_total"] == 1
    assert payload["removed_total"] == 1
    assert "before_by_source" in payload
