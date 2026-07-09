"""Smoke test for data/prepare.py orchestrator.

Builds an in-memory config + source list, runs the orchestrator end-to-end
with --skip-tokenize, verifies all expected output artifacts exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from typer.testing import CliRunner

from data.config import (
    DataConfig,
    DedupConfig,
    DedupMethod,
    SourceConfig,
    SourceType,
    SplitConfig,
    StratifyBy,
    TokenizerConfig,
)

runner = CliRunner()


@pytest.fixture
def finance_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "finance.jsonl"
    rows = [
        {
            "id": f"finance-{i:03d}",
            "domain": "finance",
            "messages": [
                {"role": "user", "content": f"Finance question {i}?"},
                {"role": "assistant", "content": f"Finance answer {i}."},
            ],
            "source": "finance",
            "meta": {},
        }
        for i in range(15)
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


@pytest.fixture
def orchestrator_cfg(tmp_path: Path, finance_jsonl: Path) -> DataConfig:
    return DataConfig(
        seed=42,
        output_dir=tmp_path / "artifacts",
        sources=[
            SourceConfig(
                name="finance-local",
                type=SourceType.FINANCE,
                path=finance_jsonl,
                max_examples=100,
                domain="finance",
            )
        ],
        dedup=DedupConfig(method=DedupMethod.EXACT, minhash_threshold=0.85, num_perm=64),
        split=SplitConfig(
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, stratify_by=StratifyBy.DOMAIN
        ),
        tokenizer=TokenizerConfig(name_or_path="placeholder", max_length=128),
    )


def test_orchestrator_smoke(orchestrator_cfg: DataConfig) -> None:
    from data.prepare import run

    summary = run(orchestrator_cfg, skip_tokenize=True)
    out = orchestrator_cfg.output_dir
    results = out / "results" / "data"

    # Files exist
    assert (out / "splits" / "train.jsonl").exists()
    assert (out / "splits" / "val.jsonl").exists()
    assert (out / "splits" / "test.jsonl").exists()
    assert (results / "dedup_counts.json").exists()
    assert (results / "splits_sha256.json").exists()
    assert (results / "domain_distribution.png").exists()
    assert (results / "pipeline_summary.json").exists()

    # Summary sane
    assert summary["raw_examples"] == 15
    assert summary["after_dedup"] == 15
    assert summary["train"] + summary["val"] + summary["test"] == 15
    assert summary["tokenized"] is False


def test_orchestrator_reproducible(orchestrator_cfg: DataConfig) -> None:
    """Run twice with same seed → identical SHA256 hashes."""
    from data.prepare import run

    s1 = run(orchestrator_cfg, skip_tokenize=True)
    s2 = run(orchestrator_cfg, skip_tokenize=True)
    for k in ("train_sha256", "val_sha256", "test_sha256"):
        assert s1["splits_sha256"][k] == s2["splits_sha256"][k]


# ---- _load_source dispatcher branches ------------------------------------


def test_load_source_sharegpt_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """SHAREGPT source → load_sharegpt is called with the right kwargs."""
    from data import prepare
    from data.types import Example

    fake_examples = [
        Example(
            id="s-0",
            domain="general",
            messages=[{"role": "user", "content": "hi"}],
            source="sharegpt",
        )
    ]

    def fake_loader(*, hf_dataset_id: str, max_examples: int) -> list[Example]:
        assert hf_dataset_id == "anon-neverlord/sharegpt-clean"
        assert max_examples == 50
        return fake_examples

    monkeypatch.setattr(prepare, "load_sharegpt", fake_loader)
    cfg = SourceConfig(
        name="sg",
        type=SourceType.SHAREGPT,
        hf_dataset_id="anon-neverlord/sharegpt-clean",
        max_examples=50,
    )
    out = prepare._load_source(cfg)
    assert out == fake_examples


def test_load_source_openhermes_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENHERMES source → load_openhermes called with right kwargs."""
    from data import prepare
    from data.types import Example

    fake_examples = [
        Example(
            id="o-0",
            domain="general",
            messages=[{"role": "user", "content": "hi"}],
            source="openhermes",
        )
    ]

    def fake_loader(*, hf_dataset_id: str, max_examples: int) -> list[Example]:
        assert hf_dataset_id == "teknium/OpenHermes-2.5"
        assert max_examples == 25
        return fake_examples

    monkeypatch.setattr(prepare, "load_openhermes", fake_loader)
    cfg = SourceConfig(
        name="oh",
        type=SourceType.OPENHERMES,
        hf_dataset_id="teknium/OpenHermes-2.5",
        max_examples=25,
    )
    out = prepare._load_source(cfg)
    assert out == fake_examples


def test_load_source_finance_dispatches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """FINANCE source → load_finance called with both hf_dataset_id + path."""
    from data import prepare
    from data.types import Example

    fake_examples = [
        Example(
            id="f-0",
            domain="finance",
            messages=[{"role": "user", "content": "q"}],
            source="finance",
        )
    ]

    finance_jsonl = tmp_path / "finance.jsonl"
    finance_jsonl.write_text("{}\n", encoding="utf-8")

    def fake_loader(
        *, hf_dataset_id: str | None, path: Path | None, max_examples: int
    ) -> list[Example]:
        assert hf_dataset_id is None
        assert path == finance_jsonl
        assert max_examples == 10
        return fake_examples

    monkeypatch.setattr(prepare, "load_finance", fake_loader)
    cfg = SourceConfig(
        name="fin",
        type=SourceType.FINANCE,
        path=finance_jsonl,
        max_examples=10,
        domain="finance",
    )
    out = prepare._load_source(cfg)
    assert out == fake_examples


def test_load_source_unknown_type_raises() -> None:
    """Source type not in {SHAREGPT, OPENHERMES, FINANCE, EDGAR} → ValueError."""
    from data import prepare

    # SimpleNamespace bypasses pydantic validation — simulate a stray enum value
    # that the dispatcher doesn't know about.
    fake_src = SimpleNamespace(
        type="MYSTERY_SOURCE",
        hf_dataset_id="x",
        path=None,
        max_examples=1,
    )
    with pytest.raises(ValueError, match="unknown source type"):
        prepare._load_source(fake_src)  # type: ignore[arg-type]


def test_load_source_edgar_dispatches_with_ciks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EDGAR source with ciks override → load_edgar_finance called with those ciks."""
    from data import prepare
    from data.types import Example

    fake_examples = [
        Example(
            id="edgar-apple-revenues-2023",
            domain="finance",
            messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            source="edgar",
        )
    ]
    cache_path = tmp_path / "edgar_cache.jsonl"
    captured: dict = {}

    def fake_loader(
        *,
        ciks: list[str] | None,
        path: Path | None,
        max_examples: int,
        user_agent: str,
        offline: bool,
    ) -> list[Example]:
        captured["ciks"] = ciks
        captured["path"] = path
        captured["max_examples"] = max_examples
        captured["user_agent"] = user_agent
        captured["offline"] = offline
        return fake_examples

    monkeypatch.setattr(prepare, "load_edgar_finance", fake_loader)
    cfg = SourceConfig(
        name="edgar-finance",
        type=SourceType.EDGAR,
        max_examples=200,
        domain="finance",
        ciks=["0000320193", "0000789019"],
        user_agent="DraftForge/test (test@example.com)",
    )
    out = prepare._load_source(cfg)
    assert out == fake_examples
    assert captured["ciks"] == ["0000320193", "0000789019"]
    assert captured["max_examples"] == 200
    assert captured["user_agent"] == "DraftForge/test (test@example.com)"
    # Without path or hf_dataset_id, loader runs in network mode (offline=False)
    assert captured["offline"] is False


def test_load_source_edgar_offline_when_path_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EDGAR source with path only (no ciks) → offline=True, replay from JSONL."""
    from data import prepare
    from data.types import Example

    fake_examples: list[Example] = []
    captured: dict = {}
    cache = tmp_path / "edgar.jsonl"
    cache.write_text("{}\n", encoding="utf-8")

    def fake_loader(
        *,
        ciks: list[str] | None,
        path: Path | None,
        max_examples: int,
        user_agent: str,
        offline: bool,
    ) -> list[Example]:
        captured["offline"] = offline
        captured["path"] = path
        return fake_examples

    monkeypatch.setattr(prepare, "load_edgar_finance", fake_loader)
    cfg = SourceConfig(
        name="edgar-offline",
        type=SourceType.EDGAR,
        path=cache,
        max_examples=100,
        domain="finance",
    )
    out = prepare._load_source(cfg)
    assert out == fake_examples
    assert captured["offline"] is True
    assert captured["path"] == cache


def test_source_config_edgar_does_not_require_id_or_path() -> None:
    """EDGAR source may omit hf_dataset_id and path (uses DEFAULT_CIKS)."""
    from data.config import SourceConfig, SourceType

    cfg = SourceConfig(
        name="edgar-default",
        type=SourceType.EDGAR,
        max_examples=10,
        domain="finance",
    )
    assert cfg.ciks is None  # loader will use DEFAULT_CIKS


def test_source_config_non_edgar_still_requires_id_or_path() -> None:
    """Non-EDGAR sources must still specify hf_dataset_id or path."""
    from data.config import SourceConfig, SourceType

    with pytest.raises(ValueError, match="hf_dataset_id or path"):
        SourceConfig(
            name="sg-bad",
            type=SourceType.SHAREGPT,
            max_examples=10,
        )


# ---- CLI main() -----------------------------------------------------------


def test_cli_main_runs_orchestrator(
    tmp_path: Path, finance_jsonl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`python -m data.prepare --config X --skip-tokenize` runs end-to-end."""
    from data import prepare

    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text(
        f"""
seed: 7
output_dir: {tmp_path / "out"}
sources:
  - name: finance-local
    type: finance
    path: {finance_jsonl}
    max_examples: 100
    domain: finance
dedup:
  method: exact
  minhash_threshold: 0.85
  num_perm: 64
split:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  stratify_by: domain
tokenizer:
  name_or_path: placeholder
  max_length: 128
""",
        encoding="utf-8",
    )
    # Stub tokenize_split so the call doesn't try to load a real tokenizer
    # (skip_tokenize=True should already skip it, but be defensive).
    monkeypatch.setattr(prepare, "tokenize_split", mock.MagicMock())

    result = runner.invoke(
        prepare.app,
        ["--config", str(cfg_yaml), "--skip-tokenize"],
    )
    assert result.exit_code == 0, result.stdout
    out = tmp_path / "out"
    assert (out / "splits" / "train.jsonl").exists()
    assert (out / "results" / "data" / "pipeline_summary.json").exists()


def test_cli_main_seed_override(
    tmp_path: Path, finance_jsonl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--seed N` overrides the seed in the loaded config."""
    from data import prepare

    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text(
        f"""
seed: 42
output_dir: {tmp_path / "out"}
sources:
  - name: finance-local
    type: finance
    path: {finance_jsonl}
    max_examples: 100
    domain: finance
dedup:
  method: exact
  minhash_threshold: 0.85
  num_perm: 64
split:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  stratify_by: domain
tokenizer:
  name_or_path: placeholder
  max_length: 128
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare, "tokenize_split", mock.MagicMock())

    result = runner.invoke(
        prepare.app,
        ["--config", str(cfg_yaml), "--skip-tokenize", "--seed", "999"],
    )
    assert result.exit_code == 0, result.stdout
    summary = json.loads(
        (tmp_path / "out" / "results" / "data" / "pipeline_summary.json").read_text()
    )
    # --seed 999 should have overridden the config seed of 42.
    assert summary["seed"] == 999
