"""Shared fixtures for DraftForge tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.types import Example

FIXTURE_PATH = Path(__file__).parent / "data" / "fixtures" / "tiny_traces.jsonl"


@pytest.fixture
def tiny_traces() -> list[Example]:
    """Load 54 synthetic traces (50 unique + 3 exact dups + 1 near-dup)."""
    out: list[Example] = []
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue
            d = json.loads(row)
            out.append(
                Example(
                    id=d["id"],
                    domain=d["domain"],
                    messages=d["messages"],
                    source=d["source"],
                    meta=d.get("meta", {}),
                )
            )
    return out


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    """Output directory for orchestrator smoke tests."""
    d = tmp_path / "artifacts" / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d
