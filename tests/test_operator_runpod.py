"""Tests for scripts/operator_runpod.py — pure-Python logic + argparse shape.

We mock urllib + subprocess (no real RunPod / SSH).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from scripts import operator_runpod as op


# ── _recommend_table: filter + sort ──────────────────────────────────────────


def _gpu(id_: str, name: str, mem_gb: int, community_price: float) -> dict:
    return {
        "id": id_,
        "displayName": name,
        "memoryInGb": mem_gb,
        "communityPrice": community_price,
        "securePrice": community_price + 0.5,
        "lowestPrice": {
            "minimumBidPrice": community_price,
            "uninterruptablePrice": community_price,
        },
    }


def test_recommend_table_filters_memory_and_price() -> None:
    """Memory < 80GB → excluded. Price > $3/hr → excluded."""
    fake_response = {
        "data": {
            "gpuTypes": [
                _gpu("NVIDIA A40", "A40", 48, 0.40),  # mem too low
                _gpu("NVIDIA A100 80GB", "A100 80GB", 80, 1.20),  # ok
                _gpu("NVIDIA H100 80GB", "H100 80GB", 80, 2.20),  # ok
                _gpu("NVIDIA H100 NVL", "H100 NVL", 94, 2.40),  # ok
                _gpu("NVIDIA B200", "B200 192GB", 192, 4.50),  # over budget
            ]
        }
    }
    with patch.object(op, "_runpod_gpu_types", return_value=fake_response["data"]["gpuTypes"]):
        rows = op._recommend_table(max_hr=3.0, min_mem_gb=80)
    ids = [r["id"] for r in rows]
    assert "NVIDIA A40" not in ids  # mem too low
    assert "NVIDIA B200" not in ids  # over budget
    assert "NVIDIA A100 80GB" in ids
    assert "NVIDIA H100 80GB" in ids
    assert "NVIDIA H100 NVL" in ids
    # Sorted by perf_per_dollar descending
    perf = [r["perf_per_dollar"] for r in rows]
    assert perf == sorted(perf, reverse=True)


def test_recommend_table_relaxes_thresholds() -> None:
    """Loosening min_mem_gb captures smaller cards; raising max_hr captures B200."""
    fake_response = {"data": {"gpuTypes": [
        _gpu("NVIDIA A40", "A40", 48, 0.40),
        _gpu("NVIDIA B200", "B200 192GB", 192, 4.50),
    ]}}
    with patch.object(op, "_runpod_gpu_types", return_value=fake_response["data"]["gpuTypes"]):
        relaxed = op._recommend_table(max_hr=10.0, min_mem_gb=24)
        strict = op._recommend_table(max_hr=3.0, min_mem_gb=80)
    assert {r["id"] for r in relaxed} == {"NVIDIA A40", "NVIDIA B200"}
    assert strict == []


def test_cmd_recommend_prints_table_on_success() -> None:
    """cmd_recommend prints header + rows on success."""
    with patch.object(op, "_recommend_table", return_value=[
        {"id": "NVIDIA H100 80GB", "displayName": "H100", "memoryInGb": 80,
         "communityPrice": 2.20, "perf_per_dollar": 36363.6},
    ]):
        rc = op.cmd_recommend(argparse.Namespace())
    assert rc == 0


def test_cmd_recommend_empty_when_no_match() -> None:
    """No GPU matches → rc=1 + helpful message."""
    with patch.object(op, "_recommend_table", return_value=[]):
        rc = op.cmd_recommend(argparse.Namespace())
    assert rc == 1


def test_cmd_recommend_handles_api_error(capsys) -> None:
    """RunPod API down → rc=2 + stderr message."""
    with patch.object(op, "_recommend_table",
                      side_effect=urllib.error.URLError("network down")):
        rc = op.cmd_recommend(argparse.Namespace())
    assert rc == 2
    captured = capsys.readouterr()
    assert "RunPod API unreachable" in captured.err


# ── spec: JSON shape ─────────────────────────────────────────────────────────


def test_cmd_spec_emits_required_fields() -> None:
    """spec JSON contains every key RunPod create-pod needs."""
    args = argparse.Namespace(
        gpu="NVIDIA H100 NVL",
        gpu_count=1,
        disk=400,
        image=op.DEFAULT_IMAGE,
        repo_url=op.DEFAULT_REPO_URL,
    )
    rc = op.cmd_spec(args)
    assert rc == 0
    # Capture stdout via redirect
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        op.cmd_spec(args)
    # Parse the first JSON block from output
    out = buf.getvalue()
    spec = json.loads(out.split("\n\n")[0])
    assert spec["gpuTypeId"] == "NVIDIA H100 NVL"
    assert spec["containerDiskInGb"] == 400  # ≥ MIN_GPU_DISK_GB
    assert spec["env"][0]["key"] == "DRAFTFORGE_REPO_URL"
    assert spec["ports"] == f"{op.SSH_PORT}/tcp"


def test_cmd_spec_clamps_disk_below_minimum() -> None:
    """disk < MIN_GPU_DISK_GB → bumped to min (RunPod would reject tiny disks)."""
    import io
    import contextlib
    args = argparse.Namespace(
        gpu="NVIDIA H100 80GB",
        gpu_count=1,
        disk=50,  # too small
        image=op.DEFAULT_IMAGE,
        repo_url=op.DEFAULT_REPO_URL,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        op.cmd_spec(args)
    spec = json.loads(buf.getvalue().split("\n\n")[0])
    assert spec["containerDiskInGb"] >= op.MIN_GPU_DISK_GB


# ── push / run / status / stop: subprocess plumbing ──────────────────────────


def test_cmd_push_runs_scp_then_ssh_onboard() -> None:
    """push = scp repo to pod, ssh onboard_pod.sh."""
    completed = MagicMock(returncode=0)
    with patch.object(op.subprocess, "run", return_value=completed) as run_mock:
        rc = op.cmd_push(argparse.Namespace(
            pod_id="abc123",
            ssh_host="1.2.3.4",
            ssh_port=22,
            ssh_key=None,
        ))
    assert rc == 0
    assert run_mock.call_count == 2  # scp + ssh
    # First call: scp
    scp_cmd = run_mock.call_args_list[0][0][0]
    assert scp_cmd[0] == "scp"
    assert str(op.REPO_ROOT) in scp_cmd
    # Second call: ssh
    ssh_cmd = run_mock.call_args_list[1][0][0]
    assert ssh_cmd[0] == "ssh"
    assert "onboard_pod.sh" in ssh_cmd[-1]


def test_cmd_push_returns_nonzero_on_ssh_failure() -> None:
    """ssh onboard fails (non-zero exit) → push returns same code."""
    ok = MagicMock(returncode=0)
    fail = subprocess.CalledProcessError(42, "ssh")
    with patch.object(op.subprocess, "run", side_effect=[ok, fail]):
        rc = op.cmd_push(argparse.Namespace(
            pod_id="abc123", ssh_host="1.2.3.4", ssh_port=22, ssh_key=None,
        ))
    assert rc == 42


def test_cmd_run_threads_env_overrides() -> None:
    """--skip-train / --skip-ablate / --n-seeds all flow into the SSH env block."""
    completed = MagicMock(returncode=0)
    with patch.object(op.subprocess, "run", return_value=completed) as run_mock:
        rc = op.cmd_run(argparse.Namespace(
            pod_id="abc123", ssh_host="1.2.3.4", ssh_port=22, ssh_key=None,
            skip_train=True, skip_ablate=False, skip_serve=True, n_seeds=2,
        ))
    assert rc == 0
    ssh_cmd = run_mock.call_args_list[0][0][0]
    ssh_payload = ssh_cmd[-1]
    assert "SKIP_TRAIN=1" in ssh_payload
    assert "SKIP_SERVE=1" in ssh_payload
    assert "SKIP_ABLATE=1" not in ssh_payload  # not requested
    assert "N_SEEDS=2" in ssh_payload


def test_cmd_status_calls_ssh_with_nvidia_smi() -> None:
    """status ssh payload includes nvidia-smi + log tail."""
    completed = MagicMock(returncode=0)
    with patch.object(op.subprocess, "run", return_value=completed) as run_mock:
        rc = op.cmd_status(argparse.Namespace(
            pod_id="abc123", ssh_host="1.2.3.4", ssh_port=22, ssh_key=None,
        ))
    assert rc == 0
    ssh_cmd = run_mock.call_args_list[0][0][0]
    payload = ssh_cmd[-1]
    assert "nvidia-smi" in payload
    assert "pipeline.log" in payload


def test_cmd_stop_sends_shutdown() -> None:
    """stop = ssh shutdown -h now."""
    completed = MagicMock(returncode=0)
    with patch.object(op.subprocess, "run", return_value=completed) as run_mock:
        rc = op.cmd_stop(argparse.Namespace(
            pod_id="abc123", ssh_host="1.2.3.4", ssh_port=22, ssh_key=None,
        ))
    assert rc == 0
    ssh_cmd = run_mock.call_args_list[0][0][0]
    assert ssh_cmd[-1] == "shutdown -h now"


def test_cmd_run_timeout_returns_1() -> None:
    """24h ceiling hit → rc=1 (operator does not silently kill)."""
    with patch.object(op.subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=86400)):
        rc = op.cmd_run(argparse.Namespace(
            pod_id="abc123", ssh_host="1.2.3.4", ssh_port=22, ssh_key=None,
            skip_train=False, skip_ablate=False, skip_serve=False, n_seeds=1,
        ))
    assert rc == 1


# ── one-liner: end-to-end sequence ───────────────────────────────────────────


def test_cmd_one_liner_prints_required_steps() -> None:
    """one-liner covers recommend → spec → push → run → status → stop."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = op.cmd_one_liner(argparse.Namespace())
    assert rc == 0
    out = buf.getvalue()
    for step in ("recommend", "spec", "push", "run", "status", "stop"):
        assert step in out, f"missing step {step!r}"


# ── argparse: every subcommand is reachable ──────────────────────────────────


@pytest.mark.parametrize("subcmd", ["recommend", "spec", "push", "run", "status", "stop", "one-liner"])
def test_argparse_subcommands_reachable(subcmd: str) -> None:
    parser = op.build_parser()
    if subcmd in {"recommend", "one-liner"}:
        args = parser.parse_args([subcmd])
    elif subcmd == "spec":
        args = parser.parse_args([subcmd, "--gpu", "NVIDIA H100 NVL"])
    else:
        args = parser.parse_args([subcmd, "abc123", "--ssh-host", "1.2.3.4"])
    assert args.cmd == subcmd


def test_argparse_rejects_unknown_subcommand() -> None:
    parser = op.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bogus"])


# ── main() dispatcher ───────────────────────────────────────────────────────


def test_main_dispatches_to_correct_handler(monkeypatch) -> None:
    """main() routes to cmd_recommend, cmd_spec, etc. via args.cmd."""
    called = {"recommend": 0, "spec": 0, "push": 0}
    monkeypatch.setattr(op, "cmd_recommend", lambda _a: called.__setitem__("recommend", 1) or 0)
    monkeypatch.setattr(op, "cmd_spec", lambda _a: called.__setitem__("spec", 1) or 0)
    monkeypatch.setattr(op, "cmd_push", lambda _a: called.__setitem__("push", 1) or 0)

    op.main(["recommend"])
    op.main(["spec", "--gpu", "X"])
    op.main(["push", "P", "--ssh-host", "1.1.1.1"])
    assert called == {"recommend": 1, "spec": 1, "push": 1}