"""Deduplication: exact (SHA256 of normalized text) + MinHash near-dup.

Outputs a count log (before/after, per source) so the README can cite
exact dedup numbers — no fabrication.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from datasketch import MinHash, MinHashLSH
except ModuleNotFoundError:  # pragma: no cover - exercised via fallback path
    MinHash = None  # type: ignore[assignment, unused-ignore]
    MinHashLSH = None  # type: ignore[assignment, unused-ignore]

from data.config import DedupMethod
from data.types import Example


def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def _text_hash(ex: Example) -> str:
    """SHA256 of whitespace-normalized concatenation of messages."""
    return hashlib.sha256(_normalize(ex.render()).encode("utf-8")).hexdigest()


def exact_dedupe(examples: list[Example]) -> list[Example]:
    """Remove exact duplicates by SHA256 of message text."""
    seen: set[str] = set()
    out: list[Example] = []
    for ex in examples:
        h = _text_hash(ex)
        if h in seen:
            continue
        seen.add(h)
        out.append(ex)
    return out


def minhash_dedupe(
    examples: list[Example], threshold: float = 0.85, num_perm: int = 128
) -> list[Example]:
    """Remove near-duplicate messages using datasketch LSH."""
    if MinHash is None or MinHashLSH is None:
        return _fallback_near_dedupe(examples, threshold=threshold)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[Example] = []
    for i, ex in enumerate(examples):
        mh = MinHash(num_perm=num_perm)
        for tok in _normalize(ex.render()).split():
            mh.update(tok.encode("utf-8"))
        # query() returns neighbors; if any neighbor already kept, this is dup
        if lsh.query(mh):
            continue
        lsh.insert(str(i), mh)
        kept.append(ex)
    return kept


def deduplicate(
    examples: list[Example],
    config: DedupMethod,
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
) -> list[Example]:
    """Dispatch by DedupMethod and return unique list."""
    if config == DedupMethod.EXACT:
        return exact_dedupe(examples)
    if config == DedupMethod.MINHASH:
        return minhash_dedupe(examples, threshold=threshold, num_perm=num_perm)
    # EXACT_PLUS_MINHASH: run both, in order
    after_exact = exact_dedupe(examples)
    return minhash_dedupe(after_exact, threshold=threshold, num_perm=num_perm)


def _fallback_near_dedupe(examples: list[Example], threshold: float) -> list[Example]:
    """Stdlib-only near-dedupe fallback for environments without datasketch.

    This keeps the CPU demo path and basic data pipeline functional on a fresh
    laptop install. It uses Jaccard overlap on normalized token sets; weaker
    than MinHash-LSH, but good enough for small fixtures and deterministic
    tests.
    """
    kept: list[Example] = []
    kept_tokens: list[set[str]] = []
    for ex in examples:
        toks = set(_normalize(ex.render()).split())
        is_dup = False
        for prev in kept_tokens:
            denom = len(toks | prev) or 1
            jaccard = len(toks & prev) / denom
            if jaccard >= threshold:
                is_dup = True
                break
        if is_dup:
            continue
        kept.append(ex)
        kept_tokens.append(toks)
    return kept


def write_counts_log(
    path: Path,
    before: list[Example],
    after: list[Example],
    method: str,
) -> None:
    """Write {before, after, per-source breakdown} as JSON for traceability."""
    before_by_source = Counter(ex.source for ex in before)
    after_by_source = Counter(ex.source for ex in after)
    removed_by_source = {
        s: before_by_source.get(s, 0) - after_by_source.get(s, 0) for s in before_by_source
    }
    payload = {
        "method": method,
        "before_total": len(before),
        "after_total": len(after),
        "removed_total": len(before) - len(after),
        "before_by_source": dict(before_by_source),
        "after_by_source": dict(after_by_source),
        "removed_by_source": removed_by_source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
