"""Tests for serve/integrate.py — vLLM + SGLang invocation builders."""

from __future__ import annotations

import json
import re
import stat
import sys
from pathlib import Path

import pytest

from serve.integrate import build_sglang_invocation, build_vllm_invocation, main


def test_vllm_invocation_contains_eagle3_method() -> None:
    cmd = build_vllm_invocation(
        target_model="Qwen/Qwen3-4B-Instruct-2507",
        draft_head_path=Path("/tmp/head"),
        num_speculative_tokens=4,
    )
    assert "vllm serve" in cmd
    assert "Qwen/Qwen3-4B-Instruct-2507" in cmd
    assert '"method": "eagle3"' in cmd
    assert '"num_speculative_tokens": 4' in cmd


def test_vllm_spec_config_is_valid_json() -> None:
    cmd = build_vllm_invocation(
        target_model="Qwen/Qwen3-4B-Instruct-2507",
        draft_head_path=Path("/tmp/head"),
    )
    matches = re.findall(r"\{[^{}]*method[^{}]*eagle3[^{}]*\}", cmd)
    assert matches, f"no JSON found in: {cmd}"
    payload = json.loads(matches[0])
    assert payload["method"] == "eagle3"


def test_sglang_invocation_uses_uppercase_eagle3() -> None:
    cmd = build_sglang_invocation(
        target_model="Qwen/Qwen3-4B-Instruct-2507",
        draft_head_path=Path("/tmp/head"),
        num_speculative_tokens=4,
    )
    assert "sglang" in cmd.lower()
    assert "--speculative-algorithm EAGLE3" in cmd  # uppercase
    assert "--speculative-draft-model-path /tmp/head" in cmd
    assert "--speculative-num-steps 4" in cmd


def test_num_speculative_tokens_propagates() -> None:
    cmd = build_sglang_invocation(
        target_model="M",
        draft_head_path=Path("/d"),
        num_speculative_tokens=8,
    )
    assert "num-steps 8" in cmd


# ---- CLI main() entrypoint -----------------------------------------------


def _invoke_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    """Patch sys.argv so serve.integrate.main() reads our args, then call it."""
    monkeypatch.setattr(sys, "argv", ["serve.integrate", *args])
    main()


def test_cli_main_vllm_writes_invocation_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--runtime vllm ...` writes a bash script with the vllm invocation,
    chmod 0o755, stdout echoes the command."""
    out = tmp_path / "invocation.sh"
    _invoke_main(
        monkeypatch,
        "--target",
        "Qwen/Qwen3-4B-Instruct-2507",
        "--draft",
        "/tmp/head",
        "--runtime",
        "vllm",
        "--out",
        str(out),
    )
    assert out.exists()
    script = out.read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "vllm serve" in script
    assert '"method": "eagle3"' in script
    # chmod 0o755
    mode = out.stat().st_mode
    assert mode & stat.S_IXUSR, "owner-execute bit not set"
    assert mode & stat.S_IXGRP, "group-execute bit not set"
    assert mode & stat.S_IXOTH, "other-execute bit not set"


def test_cli_main_sglang_writes_invocation_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--runtime sglang` → bash script with sglang.launch_server form."""
    out = tmp_path / "invocation.sh"
    _invoke_main(
        monkeypatch,
        "--target",
        "Qwen/Qwen3-4B-Instruct-2507",
        "--draft",
        "/tmp/head",
        "--runtime",
        "sglang",
        "--out",
        str(out),
    )
    script = out.read_text(encoding="utf-8")
    assert "python -m sglang.launch_server" in script
    assert "--speculative-algorithm EAGLE3" in script


def test_cli_main_num_spec_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--num-spec N` propagates into the vLLM invocation."""
    out = tmp_path / "invocation.sh"
    _invoke_main(
        monkeypatch,
        "--target",
        "M",
        "--draft",
        "/d",
        "--runtime",
        "vllm",
        "--num-spec",
        "8",
        "--out",
        str(out),
    )
    script = out.read_text(encoding="utf-8")
    assert '"num_speculative_tokens": 8' in script


def test_cli_main_missing_runtime_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown --runtime → argparse error → SystemExit(2)."""
    with pytest.raises(SystemExit) as exc:
        _invoke_main(
            monkeypatch,
            "--target",
            "M",
            "--draft",
            "/d",
            "--runtime",
            "tensorflow",  # not in choices
        )
    assert exc.value.code == 2


def test_cli_main_missing_required_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing --target → argparse error → SystemExit(2)."""
    with pytest.raises(SystemExit) as exc:
        _invoke_main(
            monkeypatch,
            "--draft",
            "/d",
            "--runtime",
            "vllm",
        )
    assert exc.value.code == 2


def test_cli_main_creates_parent_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--out` under a non-existent parent dir → main() mkdir-p's it."""
    out = tmp_path / "deep" / "nested" / "invocation.sh"
    _invoke_main(
        monkeypatch,
        "--target",
        "M",
        "--draft",
        "/d",
        "--runtime",
        "vllm",
        "--out",
        str(out),
    )
    assert out.exists()
