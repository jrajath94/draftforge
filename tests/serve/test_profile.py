"""Tests for serve/profile.py — Nsight wrapper + kernel-attribution regime classifier.

The Nsight wrapper itself shells out to `nsys`, which is GPU-only and not
available in CI. We mock subprocess.call to verify the arg list without
running nsys. The classifier (`classify_binding`) and report-hint builder
(`nsys_analyze_hint`) are pure functions and tested directly.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from serve.profile import (
    BINDING_RULES,
    classify_binding,
    nsys_analyze_hint,
    nsys_run,
)

# ---- classify_binding: regime classification table ----------------------


def test_classify_binding_draft_bound_regime() -> None:
    """draft_pct >= 0.60 → 'draft-bound'."""
    assert classify_binding(0.60) == "draft-bound"
    assert classify_binding(0.75) == "draft-bound"
    assert classify_binding(1.0) == "draft-bound"


def test_classify_binding_balanced_regime() -> None:
    """0.30 <= draft_pct < 0.60 → 'balanced'."""
    assert classify_binding(0.30) == "balanced"
    assert classify_binding(0.45) == "balanced"
    # upper bound is exclusive: 0.60 is draft-bound, not balanced
    assert classify_binding(0.59) == "balanced"


def test_classify_binding_verify_bound_regime() -> None:
    """draft_pct < 0.30 → 'verify-bound'."""
    assert classify_binding(0.0) == "verify-bound"
    assert classify_binding(0.15) == "verify-bound"
    # 0.30 is balanced, not verify-bound
    assert classify_binding(0.29) == "verify-bound"


def test_binding_rules_table_is_complete() -> None:
    """BINDING_RULES must cover all three regimes (no drift)."""
    assert set(BINDING_RULES.keys()) == {"draft-bound", "verify-bound", "balanced"}


# ---- nsys_analyze_hint: composition --------------------------------------


def test_nsys_analyze_hint_composes_stats_command(tmp_path: Path) -> None:
    """Hint includes the report path with the .nsys-rep suffix and the
    expected `--report` flags."""
    rep = tmp_path / "speculative"
    hint = nsys_analyze_hint(rep)
    assert hint.startswith("nsys stats ")
    assert "--report cuda_gpu_kern_sum,gputrace" in hint
    assert f"{rep}.nsys-rep" in hint


# ---- nsys_run: subprocess arg construction --------------------------------


def test_nsys_run_includes_trace_and_output(tmp_path: Path) -> None:
    """nsys_run composes: nsys profile --trace=X --output=Y -- target_cmd."""
    with mock.patch("serve.profile.subprocess.call", return_value=0) as m_call:
        out_path = tmp_path / "trace"
        rc = nsys_run(
            target_cmd=["python", "bench.py", "--workload", "chat"],
            output=out_path,
        )
    assert rc == 0
    args = m_call.call_args[0][0]
    # Front of the args list: nsys profile --trace=... --output=... --
    assert args[0] == "nsys"
    assert args[1] == "profile"
    assert any(a.startswith("--trace=") and "cuda" in a for a in args)
    assert any(a.startswith("--output=") and str(out_path) in a for a in args)
    assert "--" in args
    # Tail of the args list: the user's target_cmd verbatim
    assert args[-4:] == ["python", "bench.py", "--workload", "chat"]


def test_nsys_run_custom_trace_and_extra(tmp_path: Path) -> None:
    """Custom --trace value + extra nsys flags propagate into the args list."""
    with mock.patch("serve.profile.subprocess.call", return_value=0) as m_call:
        nsys_run(
            target_cmd=["echo", "hi"],
            output=tmp_path / "t",
            trace="cuda",
            extra=["--force-overwrite=true", "--capture-range=cudaProfilerApi"],
        )
    args = m_call.call_args[0][0]
    assert any(a == "--trace=cuda" for a in args)
    assert "--force-overwrite=true" in args
    assert "--capture-range=cudaProfilerApi" in args


def test_nsys_run_creates_parent_dir(tmp_path: Path) -> None:
    """nsys_run mkdir-p's the output's parent directory."""
    nested = tmp_path / "deep" / "nested" / "trace"
    assert not nested.parent.exists()
    with mock.patch("serve.profile.subprocess.call", return_value=0):
        nsys_run(target_cmd=["true"], output=nested)
    assert nested.parent.exists()


def test_nsys_run_propagates_subprocess_exit_code(tmp_path: Path) -> None:
    """Return value of nsys_run == return value of subprocess.call."""
    with mock.patch("serve.profile.subprocess.call", return_value=42):
        rc = nsys_run(target_cmd=["false"], output=tmp_path / "t")
    assert rc == 42
