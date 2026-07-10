"""Tests for source loaders — fixture-based, no network calls.

Verifies each loader emits a uniform Example list with correct domain tag.
For network-bound loaders (sharegpt, openhermes, finance HF), we test
the pure `_extract_messages` / dispatch helpers directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ---- EDGAR loader (data/sources/edgar.py) ----------------------------------


def _fake_edgar_payload(entity_name: str = "Apple Inc.", cik: str = "0000320193") -> dict:
    """Minimal XBRL company-facts payload — exercises the Q&A emission path."""
    return {
        "cik": cik,
        "entityName": entity_name,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"form": "10-K", "fp": "FY", "end": "2023-09-30",
                             "val": 383_285_000_000},
                            {"form": "10-K", "fp": "FY", "end": "2022-09-24",
                             "val": 394_328_000_000},
                            {"form": "10-Q", "fp": "Q3", "end": "2024-06-29",
                             "val": 100_000_000_000},  # non-10-K → skipped
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"form": "10-K", "fp": "FY", "end": "2023-09-30",
                             "val": 96_995_000_000},
                        ]
                    }
                },
                "Assets": {"units": {}},  # empty units → no Q&A emitted
            }
        },
    }


def _mock_urlopen(payloads: list[dict]):
    """Build a urlopen side_effect that returns the next payload on each call."""
    queue = list(payloads)
    cm = MagicMock()
    cm.read.return_value = json.dumps(queue.pop(0)).encode("utf-8")
    cm.__enter__ = lambda s: s
    cm.__exit__ = lambda s, *a: False

    def side_effect(*_args, **_kwargs):
        if not queue:
            raise StopIteration("no more payloads")
        cm.read.return_value = json.dumps(queue.pop(0)).encode("utf-8")
        return cm

    return side_effect


def test_load_edgar_offline_requires_path() -> None:
    """offline=True without path → ValueError (no implicit network fallback)."""
    import pytest

    from data.sources.edgar import load_edgar_finance
    with pytest.raises(ValueError, match="offline mode requires path"):
        load_edgar_finance(offline=True)


def test_load_edgar_jsonl_cache_replay(tmp_path: Path) -> None:
    """offline path reads JSONL cache, tags domain=finance, source=edgar."""
    from data.sources.edgar import load_edgar_finance

    cache = tmp_path / "edgar_cache.jsonl"
    rows = [
        {
            "id": "edgar-test-2023",
            "domain": "finance",
            "messages": [
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "A."},
            ],
            "source": "edgar",
            "meta": {"cik": "0000320193"},
        }
    ]
    with cache.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = load_edgar_finance(path=cache, offline=True)
    assert len(out) == 1
    assert out[0].domain == "finance"
    assert out[0].source == "edgar"
    assert out[0].meta["cik"] == "0000320193"


def test_load_edgar_finance_emits_qa_per_concept_year(tmp_path: Path) -> None:
    """Live fetch (mocked) emits one Q&A per (concept, FY), skipping 10-Q."""
    from data.sources import edgar

    payloads = [_fake_edgar_payload("Apple Inc.", "0000320193")]
    with patch.object(edgar, "_http_get_json", side_effect=payloads):
        out = edgar.load_edgar_finance(
            ciks=["0000320193"],
            rate_limit_sec=0.0,  # speed up test
        )
    # 2 fiscal years of Revenues + 1 of NetIncomeLoss = 3 examples
    # Assets has no units → 0 examples
    assert len(out) == 3
    assert all(e.domain == "finance" for e in out)
    assert all(e.source == "edgar" for e in out)
    # All Q&A are 2-message exchanges
    assert all(len(e.messages) == 2 for e in out)
    # Verify content sanity: numeric value rendered as billions
    rev_2023 = next(e for e in out if "2023" in e.id and "revenues" in e.id)
    assert "383.29 billion" in rev_2023.messages[1]["content"]
    assert "10-K" in rev_2023.messages[1]["content"]


def test_load_edgar_skips_cik_on_network_error() -> None:
    """Network errors per CIK are non-fatal — loader continues to next."""
    import urllib.error

    from data.sources import edgar

    def flaky(_url, _ua, **_kw):
        if "0000789019" in _url:
            raise urllib.error.URLError("simulated timeout")
        return _fake_edgar_payload("Apple Inc.", "0000320193")

    with patch.object(edgar, "_http_get_json", side_effect=flaky):
        out = edgar.load_edgar_finance(
            ciks=["0000789019", "0000320193"],
            rate_limit_sec=0.0,
        )
    # Microsoft failed, Apple succeeded with 3 Q&A
    assert len(out) == 3
    assert all(e.meta["cik"] == "0000320193" for e in out)


def test_load_edgar_raises_runtimeerror_when_all_ciks_blocked_by_waf() -> None:
    """SEC WAF 403 on every CIK must fail loudly, not silently return [].

    Addendum (post-data-pre-flight): if a User-Agent contains '+' or other
    non-conforming chars, the SEC WAF returns 403 for every CIK. Returning an
    empty list would propagate garbage through the pipeline (0 training rows
    after dedup), so the loader raises RuntimeError with a fixable hint.

    Without this guard, operators hit the failure 24 h into a $70 training run
    when the model trains on 0 finance examples — debug nightmare.
    """
    import urllib.error

    import pytest

    from data.sources import edgar

    def forbidden(_url, _ua, **_kw):
        raise urllib.error.HTTPError(
            url=_url, code=403, msg="Forbidden", hdrs=None, fp=None  # type: ignore[arg-type]
        )

    with patch.object(edgar, "_http_get_json", side_effect=forbidden):
        with pytest.raises(RuntimeError, match="SEC EDGAR blocked all"):
            edgar.load_edgar_finance(
                ciks=["0000320193", "0000789019"],
                rate_limit_sec=0.0,
                user_agent="DraftForge/test (+bad@example.com)",
            )


def test_load_edgar_succeeds_when_partial_ciks_blocked() -> None:
    """WAF blocks some CIKs but others succeed → return what we can.

    If at least one CIK returns data, the loader emits it without raising.
    Only when EVERY CIK is blocked (out is empty AND blocked_count > 0) does
    it raise. This avoids false alarms when the WAF blocks only a subset
    of issuers.
    """
    import urllib.error

    from data.sources import edgar

    def partial(_url, _ua, **_kw):
        if "0000320193" in _url:
            raise urllib.error.HTTPError(
                url=_url, code=403, msg="Forbidden", hdrs=None, fp=None  # type: ignore[arg-type]
            )
        return _fake_edgar_payload("Microsoft", "0000789019")

    with patch.object(edgar, "_http_get_json", side_effect=partial):
        out = edgar.load_edgar_finance(
            ciks=["0000320193", "0000789019"],
            rate_limit_sec=0.0,
        )
    # Microsoft (CIK 0000789019) yielded 3 Q&A; Apple (0000320193) was 403'd
    assert len(out) == 3
    assert all(e.meta["cik"] == "0000789019" for e in out)


def test_load_edgar_respects_max_examples() -> None:
    """max_examples caps the output size."""
    from data.sources import edgar

    with patch.object(edgar, "_http_get_json", return_value=_fake_edgar_payload()):
        out = edgar.load_edgar_finance(
            ciks=["0000320193"],
            max_examples=1,
            rate_limit_sec=0.0,
        )
    assert len(out) == 1


def test_write_edgar_cache_roundtrip(tmp_path: Path) -> None:
    """write_edgar_cache → load_edgar_finance(offline=True) round-trips losslessly."""
    from data.sources import edgar
    from data.types import Example

    src = [
        Example(
            id="edgar-rt-2024",
            domain="finance",
            messages=[
                {"role": "user", "content": "What was X?"},
                {"role": "assistant", "content": "X was Y."},
            ],
            source="edgar",
            meta={"cik": "0000320193", "concept": "Revenues"},
        )
    ]
    cache = tmp_path / "rt.jsonl"
    edgar.write_edgar_cache(src, cache)
    out = edgar.load_edgar_finance(path=cache, offline=True)
    assert len(out) == 1
    assert out[0].id == "edgar-rt-2024"
    assert out[0].meta["concept"] == "Revenues"


def test_facts_to_qa_formats_usd_smartly() -> None:
    """USD formatting: billions / millions / raw thresholds."""
    from data.sources.edgar import _format_usd

    assert _format_usd(383_285_000_000) == "$383.29 billion"
    assert _format_usd(96_995_000_000) == "$97.00 billion"
    assert _format_usd(2_500_000) == "$2.50 million"
    assert _format_usd(123_456) == "$123,456"
    assert _format_usd(-500_000_000) == "$-500.00 million"


def test_facts_to_qa_skips_non_annual() -> None:
    """Only form=10-K + fp=FY rows become Q&A; 10-Q rows ignored."""
    from data.sources.edgar import _facts_to_qa

    concept_data = {
        "units": {
            "USD": [
                {"form": "10-K", "fp": "FY", "end": "2023-12-31", "val": 100_000_000_000},
                {"form": "10-Q", "fp": "Q1", "end": "2024-03-31", "val": 25_000_000_000},
                {"form": "10-Q", "fp": "Q2", "end": "2024-06-30", "val": 30_000_000_000},
            ]
        }
    }
    out = _facts_to_qa(
        entity_name="Test Co",
        cik="0000999999",
        concept_data=concept_data,
        concept_tag="Revenues",
        concept_label="revenues",
    )
    assert len(out) == 1
    assert "2023" in out[0].id
