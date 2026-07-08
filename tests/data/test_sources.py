"""Tests for source loaders — fixture-based, no network calls.

Verifies each loader emits a uniform Example list with correct domain tag.
For network-bound loaders (sharegpt, openhermes, finance HF), we test
the pure `_extract_messages` / dispatch helpers directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.sources.finance import _load_from_jsonl, load_finance
from data.sources.openhermes import _extract_messages as openhermes_extract
from data.sources.sharegpt import _extract_messages as sharegpt_extract
from data.types import Example


def _row(user: str, asst: str, source: str, domain: str) -> dict:
    return {
        "id": f"{source}-x",
        "domain": domain,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": asst},
        ],
        "source": source,
        "meta": {},
    }


def test_finance_jsonl_loader(tmp_path: Path) -> None:
    p = tmp_path / "finance.jsonl"
    rows = [_row(f"Q{i}", f"A{i}", "finance", "finance") for i in range(5)]
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    examples = _load_from_jsonl(p, max_examples=10)
    assert len(examples) == 5
    assert all(e.domain == "finance" for e in examples)
    assert all(e.source == "finance" for e in examples)
    assert all(len(e.messages) == 2 for e in examples)


def test_finance_jsonl_skips_short_rows(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    with p.open("w") as f:
        f.write(json.dumps({"id": "x", "domain": "finance", "messages": [{"role": "user", "content": "?"}], "source": "finance"}) + "\n")
        f.write(json.dumps(_row("Q", "A", "finance", "finance")) + "\n")
    examples = _load_from_jsonl(p, max_examples=10)
    assert len(examples) == 1


def test_finance_dispatch_requires_id_or_path() -> None:
    import pytest

    with pytest.raises(ValueError, match="hf_dataset_id or path"):
        load_finance()


def test_sharegpt_extract_messages_normalizes_sharegpt_format() -> None:
    row = {
        "conversations": [
            {"from": "human", "value": "What is X?"},
            {"from": "gpt", "value": "X is a thing."},
        ]
    }
    out = sharegpt_extract(row)
    assert len(out) == 2
    assert out[0]["role"] == "user"
    assert out[1]["role"] == "assistant"


def test_sharegpt_extract_messages_handles_openai_format() -> None:
    row = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
    out = sharegpt_extract(row)
    assert out[0]["role"] == "user"


def test_sharegpt_extract_messages_handles_text_format() -> None:
    row = {"text": "Just text"}
    out = sharegpt_extract(row)
    assert out == [{"role": "user", "content": "Just text"}]


def test_sharegpt_extract_messages_empty() -> None:
    assert sharegpt_extract({}) == []


def test_openhermes_extract_messages_conversations() -> None:
    row = {"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]}
    out = openhermes_extract(row)
    assert out[0]["role"] == "user"
    assert out[1]["role"] == "assistant"


def test_openhermes_extract_messages_instruction_output() -> None:
    row = {"instruction": "Do X", "output": "Did X"}
    out = openhermes_extract(row)
    assert out == [
        {"role": "user", "content": "Do X"},
        {"role": "assistant", "content": "Did X"},
    ]


def test_openhermes_extract_messages_empty() -> None:
    assert openhermes_extract({}) == []


def test_example_render_deterministic() -> None:
    e1 = Example(
        id="x",
        domain="general",
        messages=[
            {"role": "user", "content": "  Hello   world  "},
            {"role": "assistant", "content": "Hi!"},
        ],
        source="sharegpt",
    )
    e2 = Example(
        id="x",
        domain="general",
        messages=[
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi!"},
        ],
        source="sharegpt",
    )
    # Whitespace-normalized render is identical → exact dedupe catches them
    assert e1.render() == e2.render()


# ---- Main loader iteration via mocked datasets.load_dataset --------------


class _FakeDataset:
    """Minimal stand-in for datasets.Dataset — exposes iter over a row list."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __iter__(self) -> _FakeDataset:
        return self

    def __next__(self) -> dict:
        if not self._rows:
            raise StopIteration
        return self._rows.pop(0)


def test_load_sharegpt_iterates_and_skips_short(monkeypatch) -> None:
    """load_sharegpt iterates the dataset, normalizes via _extract_messages,
    skips rows with < 2 messages, and emits domain=general/source=sharegpt."""
    from data.sources import sharegpt

    rows = [
        {"messages": [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"}]},
        {"messages": [{"role": "user", "content": "alone"}]},  # too short — skipped
        {"conversations": [{"from": "human", "value": "q2"}, {"from": "gpt", "value": "a2"}]},
        {"text": "just text"},  # too short — only 1 message — skipped
        {"messages": [{"role": "user", "content": "q4"}, {"role": "assistant", "content": "a4"}]},
    ]
    monkeypatch.setattr(
        sharegpt, "load_dataset", lambda *_a, **_kw: _FakeDataset(list(rows))
    )

    out = sharegpt.load_sharegpt(max_examples=100)
    assert len(out) == 3
    assert all(e.domain == "general" for e in out)
    assert all(e.source == "sharegpt" for e in out)
    assert [e.id for e in out] == ["sharegpt-000000", "sharegpt-000002", "sharegpt-000004"]


def test_load_sharegpt_respects_max_examples(monkeypatch) -> None:
    """max_examples caps the iteration at N (drops rows beyond the cap)."""
    from data.sources import sharegpt

    rows = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(20)
    ]
    monkeypatch.setattr(
        sharegpt, "load_dataset", lambda *_a, **_kw: _FakeDataset(list(rows))
    )

    out = sharegpt.load_sharegpt(max_examples=3)
    assert len(out) == 3
    assert [e.id for e in out] == ["sharegpt-000000", "sharegpt-000001", "sharegpt-000002"]


def test_load_openhermes_iterates_and_skips_short(monkeypatch) -> None:
    """load_openhermes iterates, normalizes via _extract_messages, skips short."""
    from data.sources import openhermes

    rows = [
        {"conversations": [{"from": "human", "value": "q0"}, {"from": "gpt", "value": "a0"}]},
        {"instruction": "Do X", "output": "Did X"},
        {"instruction": "alone"},  # missing output → empty extraction → skipped
        {},  # nothing extractable → skipped
        {"conversations": [{"from": "user", "value": "q4"}, {"from": "assistant", "value": "a4"}]},
    ]
    monkeypatch.setattr(
        openhermes, "load_dataset", lambda *_a, **_kw: _FakeDataset(list(rows))
    )

    out = openhermes.load_openhermes(max_examples=100)
    assert len(out) == 3
    assert all(e.domain == "general" for e in out)
    assert all(e.source == "openhermes" for e in out)
    assert out[1].messages == [
        {"role": "user", "content": "Do X"},
        {"role": "assistant", "content": "Did X"},
    ]


def test_load_openhermes_respects_max_examples(monkeypatch) -> None:
    """max_examples caps iteration in load_openhermes."""
    from data.sources import openhermes

    rows = [
        {"instruction": f"q{i}", "output": f"a{i}"} for i in range(10)
    ]
    monkeypatch.setattr(
        openhermes, "load_dataset", lambda *_a, **_kw: _FakeDataset(list(rows))
    )

    out = openhermes.load_openhermes(max_examples=2)
    assert len(out) == 2


def test_load_from_hf_finance_iterates_and_tags(monkeypatch) -> None:
    """_load_from_hf (finance) iterates HF dataset, normalizes, tags domain=finance."""
    from data.sources import finance

    rows = [
        {"messages": [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"}]},
        {"messages": [{"role": "user", "content": "alone"}]},  # too short → skipped
        {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]},
    ]
    monkeypatch.setattr(
        finance, "load_dataset", lambda *_a, **_kw: _FakeDataset(list(rows))
    )

    out = finance._load_from_hf("acme/finance-data", max_examples=100)
    assert len(out) == 2
    assert all(e.domain == "finance" for e in out)
    assert all(e.source == "finance" for e in out)
    assert out[0].meta == {"hf_dataset_id": "acme/finance-data"}
    assert out[0].id == "finance-000000"


def test_load_from_hf_finance_respects_max_examples(monkeypatch) -> None:
    """_load_from_hf stops iteration at max_examples."""
    from data.sources import finance

    rows = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(10)
    ]
    monkeypatch.setattr(
        finance, "load_dataset", lambda *_a, **_kw: _FakeDataset(list(rows))
    )

    out = finance._load_from_hf("acme/finance-data", max_examples=4)
    assert len(out) == 4


def test_load_finance_dispatches_to_hf_when_id_given(monkeypatch) -> None:
    """load_finance with hf_dataset_id → _load_from_hf, not the JSONL path."""
    from data.sources import finance

    called = {"hf": 0, "jsonl": 0}

    def fake_hf(_id, _max):
        called["hf"] += 1
        return []

    def fake_jsonl(_p, _max):
        called["jsonl"] += 1
        return []

    monkeypatch.setattr(finance, "_load_from_hf", fake_hf)
    monkeypatch.setattr(finance, "_load_from_jsonl", fake_jsonl)

    out = finance.load_finance(hf_dataset_id="acme/finance", path=None, max_examples=10)
    assert out == []
    assert called == {"hf": 1, "jsonl": 0}


def test_load_finance_dispatches_to_jsonl_when_only_path(monkeypatch, tmp_path) -> None:
    """load_finance with only path → _load_from_jsonl (with the path-guard check)."""
    from data.sources import finance

    jsonl = tmp_path / "fin.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")
    called = {"hf": 0, "jsonl": 0}

    monkeypatch.setattr(finance, "_load_from_hf", lambda *_a, **_kw: (called.__setitem__("hf", called["hf"] + 1) or []))
    monkeypatch.setattr(
        finance,
        "_load_from_jsonl",
        lambda p, m: (called.__setitem__("jsonl", called["jsonl"] + 1) or [Example(id="x", domain="finance", messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], source="finance")]),
    )

    out = finance.load_finance(hf_dataset_id=None, path=jsonl, max_examples=10)
    assert len(out) == 1
    assert called["hf"] == 0
    assert called["jsonl"] == 1


# ---- _results_path_check (addendum 7 fixture guard) ----------------------


def test_results_path_check_none_passes() -> None:
    """None path → guard passes (no-op)."""
    from data.sources.finance import _results_path_check

    _results_path_check(None)  # must not raise


def test_results_path_check_normal_path_passes(tmp_path) -> None:
    """A normal user-supplied JSONL path → guard passes."""
    from data.sources.finance import _results_path_check

    _results_path_check(tmp_path / "user_finance.jsonl")  # must not raise


def test_results_path_check_blocks_fixture_into_results(tmp_path) -> None:
    """Fixture data being written into a results/ dir → ValueError."""
    import pytest

    from data.sources.finance import _FIXTURE_PATH, _results_path_check

    # Construct a path that lives both under the fixture path AND under a results/ dir.
    bad = _FIXTURE_PATH.parent / "results" / "tiny_traces.jsonl"
    bad.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="refusing fixture data into results dir"):
        _results_path_check(bad)


def test_results_path_check_allows_fixture_outside_results() -> None:
    """Fixture path itself (NOT under results/) → guard passes (test fixtures may be read)."""
    from data.sources.finance import _FIXTURE_PATH, _results_path_check

    # Reading the fixture file directly is fine — only writing it into results/ is blocked.
    _results_path_check(_FIXTURE_PATH)  # must not raise (no results/ in path.parts)
