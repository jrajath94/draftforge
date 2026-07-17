"""Finance-domain loader.

Primary target: FinOpsGym (numerical-finance QA, when available on HF).
Fallback: local JSONL file (one row per Q&A) — user supplies via config.
Schema per row:
    {"messages": [...], "domain": "finance", "meta": {...}}

Domain tag is always "finance" so the orchestrator can stratify the
domain-shift study for EAGLE-3 acceptance analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from datasets import load_dataset
except ModuleNotFoundError:  # pragma: no cover - depends on optional extra
    load_dataset = None  # type: ignore[assignment, unused-ignore]

from data.types import Example

# Finance fixture guard — see test fixture generator path.
# Addendum 7: synthetic test fixtures must never reach `results/`.
_FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "tests" / "data" / "fixtures" / "tiny_traces.jsonl"
)
_RESULTS_DIR_NAMES = {"results", "results_dummy", "results_smoke"}


def _results_path_check(path: Path | None) -> None:
    """Refuse to load fixture data into results/. Addendum 7.

    Synthetic finance test fixtures live under tests/data/fixtures/. If a
    downstream caller (data.prepare, benchmark post-processing) ever
    reads fixture-derived data and writes a result into a results/
    directory, the domain-shift measurement would be biased. Block
    explicitly when the requested path is the fixture path OR lives
    under a results/ directory.
    """
    if path is None:
        return
    if path == _FIXTURE_PATH or _FIXTURE_PATH.parent in path.parents:
        if any(name in path.parts for name in _RESULTS_DIR_NAMES):
            raise ValueError(
                f"finance loader: refusing fixture data into results dir: {path}. "
                f"Fixture {_FIXTURE_PATH} is synthetic test data only."
            )


def load_finance(
    hf_dataset_id: str | None = None,
    path: Path | None = None,
    max_examples: int = 50_000,
) -> list[Example]:
    """Load finance-domain instruction/response traces.

    Tries HF dataset first; if absent or no id, reads local JSONL.
    Addendum 7: blocks fixture data reaching `results/`.
    """
    if hf_dataset_id is None and path is None:
        raise ValueError("finance source requires hf_dataset_id or path")
    if hf_dataset_id is not None:
        return _load_from_hf(hf_dataset_id, max_examples)
    assert path is not None
    _results_path_check(path)
    return _load_from_jsonl(path, max_examples)


def _load_from_hf(hf_dataset_id: str, max_examples: int) -> list[Example]:
    if load_dataset is None:
        raise ModuleNotFoundError(
            "datasets is required for hf_dataset_id-based finance loading; "
            "install the data extras or use a local JSONL path"
        )
    ds = load_dataset(hf_dataset_id, split="train", streaming=True)
    out: list[Example] = []
    for i, row in enumerate(ds):
        if i >= max_examples:
            break
        messages = row.get("messages") or []
        if not messages or len(messages) < 2:
            continue
        norm = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
        out.append(
            Example(
                id=f"finance-{i:06d}",
                domain="finance",
                messages=norm,
                source="finance",
                meta={"hf_dataset_id": hf_dataset_id},
            )
        )
    return out


def _load_from_jsonl(path: Path, max_examples: int) -> list[Example]:
    out: list[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            if not messages or len(messages) < 2:
                continue
            norm = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
            out.append(
                Example(
                    id=f"finance-{i:06d}",
                    domain="finance",
                    messages=norm,
                    source="finance",
                    meta=row.get("meta", {}),
                )
            )
    return out
