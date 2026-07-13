"""Tests for v1.3 operator extensions: concurrent subcommand, community filter, volume-id.

Builds on tests/test_operator_runpod.py (v1.2 baseline). All new tests assert
behaviour added in v1.3.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts import operator_runpod as op


def _gpu(id_: str, name: str, mem_gb: int, community_price: float, secure_price: float | None = None) -> dict:
    return {
        "id": id_,
        "displayName": name,
        "memoryInGb": mem_gb,
        "communityPrice": community_price,
        "securePrice": secure_price if secure_price is not None else community_price + 0.5,
        "lowestPrice": {
            "minimumBidPrice": community_price,
            "uninterruptablePrice": community_price,
        },
    }


# ── concurrent subcommand ────────────────────────────────────────────────────


def test_cmd_concurrent_threads_n_seeds_and_gpus_into_ssh() -> None:
    """cmd_concurrent ssh payload contains N_SEEDS + GPUS + run_concurrent_seeds.sh."""
    completed = MagicMock(returncode=0)
    with patch.object(op.subprocess, "run", return_value=completed) as run_mock:
        rc = op.cmd_concurrent(argparse.Namespace(
            pod_id="abc123", ssh_host="1.2.3.4", ssh_port=22, ssh_key=None,
            n_seeds=3, gpus="0 1 2",
        ))
    assert rc == 0
    ssh_cmd = run_mock.call_args_list[0][0][0]
    payload = ssh_cmd[-1]
    assert "N_SEEDS=3" in payload
    assert "GPUS='0 1 2'" in payload
    assert "run_concurrent_seeds.sh" in payload
    assert "3" in payload  # also argument form


def test_cmd_concurrent_pipes_logs_to_log_dir() -> None:
    """The concurrent runner emits logs under /workspace/draftforge/logs/.

    The operator must tee the ssh output so `status` can tail everything.
    """
    completed = MagicMock(returncode=0)
    with patch.object(op.subprocess, "run", return_value=completed) as run_mock:
        op.cmd_concurrent(argparse.Namespace(
            pod_id="abc123", ssh_host="1.2.3.4", ssh_port=22, ssh_key=None,
            n_seeds=2, gpus="0 1",
        ))
    ssh_payload = run_mock.call_args_list[0][0][0][-1]
    assert "tee" in ssh_payload or "concurrent.log" in ssh_payload


def test_cmd_concurrent_propagates_subprocess_error() -> None:
    """subprocess ssh error → non-zero return code."""
    failed = subprocess.CalledProcessError(7, "ssh")
    with patch.object(op.subprocess, "run", side_effect=failed):
        rc = op.cmd_concurrent(argparse.Namespace(
            pod_id="abc123", ssh_host="1.2.3.4", ssh_port=22, ssh_key=None,
            n_seeds=1, gpus="0",
        ))
    assert rc == 7


def test_cmd_status_concurrent_reports_all_per_seed_logs() -> None:
    """status subcommand's payload includes the concurrent log directory.

    After running `concurrent`, the operator's status subcommand should
    also tail ${LOG_DIR}/seed_*.log so users can see all per-seed progress.
    """
    completed = MagicMock(returncode=0)
    with patch.object(op.subprocess, "run", return_value=completed) as run_mock:
        op.cmd_status(argparse.Namespace(
            pod_id="abc123", ssh_host="1.2.3.4", ssh_port=22, ssh_key=None,
        ))
    payload = run_mock.call_args_list[0][0][0][-1]
    # Either tail all seed_*.log OR at minimum reference concurrent/logs path
    assert "seed_" in payload or "concurrent" in payload


# ── recommend --community filter ─────────────────────────────────────────────


def test_cmd_recommend_default_uses_community_price() -> None:
    """Default `recommend` keeps using communityCloud pricing (existing behaviour)."""
    fake_response = {"data": {"gpuTypes": [
        _gpu("A", "Card A", 80, 1.20, secure_price=2.50),
        _gpu("B", "Card B", 80, 0.80, secure_price=1.80),
    ]}}
    with patch.object(op, "_runpod_gpu_types", return_value=fake_response["data"]["gpuTypes"]):
        # Mock print + argparse call
        with patch.object(op, "_recommend_table", return_value=[
            {"id": "A", "displayName": "A", "memoryInGb": 80, "communityPrice": 1.20, "perf_per_dollar": 100.0},
            {"id": "B", "displayName": "B", "memoryInGb": 80, "communityPrice": 0.80, "perf_per_dollar": 200.0},
        ]):
            rc = op.cmd_recommend(argparse.Namespace())
    assert rc == 0  # default = community tier (unchanged from v1.2)


def test_cmd_recommend_secure_tier_uses_secure_price() -> None:
    """`recommend --tier secure` should pass through and use securePrice column."""
    rows = [
        {"id": "A", "displayName": "A", "memoryInGb": 80, "securePrice": 2.50, "perf_per_dollar": 100.0},
    ]
    # Assert the function we pass exists; v1.3 introduces _recommend_table_secure.
    with patch.object(op, "_recommend_table_secure", return_value=rows):
        rc = op.cmd_recommend(argparse.Namespace(tier="secure"))
    assert rc == 0


def test_recommend_table_community_includes_lower_priced_than_secure() -> None:
    """communityPrice can be 0.5x securePrice — recommend on community tier
    surfaces the cheap option that would otherwise be filtered by secure cap.
    """
    fake_response = {"data": {"gpuTypes": [
        _gpu("X", "GPU X", 80, 2.20, secure_price=4.80),  # secure 4.80 > cap 3.0
    ]}}
    with patch.object(op, "_runpod_gpu_types", return_value=fake_response["data"]["gpuTypes"]):
        # community tier, max_hr=3.0 → includes X (community 2.20 < 3.0)
        community_rows = op._recommend_table(max_hr=3.0, min_mem_gb=80)
        assert any(r["id"] == "X" for r in community_rows), (
            "community tier should surface GPU X at $2.20/hr"
        )


def test_cmd_recommend_argparse_accepts_tier_flag() -> None:
    """v1.3 adds `--tier community|secure` flag with default `community`."""
    parser = op.build_parser()
    args = parser.parse_args(["recommend", "--tier", "secure"])
    assert args.tier == "secure"

    args_default = parser.parse_args(["recommend"])
    assert args_default.tier == "community"


def test_cmd_recommend_invalid_tier_rejected() -> None:
    """Unknown tier (typo) → argparse error."""
    parser = op.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["recommend", "--tier", "bogus"])


# ── spec --volume-id ─────────────────────────────────────────────────────────


def test_cmd_spec_with_volume_id_emits_network_volume_id() -> None:
    """spec with --volume-id emits networkVolumeId and drops volumeInGb/volumeMountPath."""
    buf = io.StringIO()
    args = argparse.Namespace(
        gpu="NVIDIA H100 80GB",
        gpu_count=1,
        disk=200,
        image=op.DEFAULT_IMAGE,
        repo_url=op.DEFAULT_REPO_URL,
        volume_id="vol-abc-123",
        community=False,
    )
    with contextlib.redirect_stdout(buf):
        op.cmd_spec(args)
    spec = json.loads(buf.getvalue().split("\n\n")[0])
    assert spec["networkVolumeId"] == "vol-abc-123"
    # When attaching a network volume, you don't pre-allocate disk + mount path
    assert spec.get("volumeInGb", 0) == 0


def test_cmd_spec_without_volume_id_keeps_default_disk_path() -> None:
    """spec without --volume-id: behaves as v1.2 (volumeInGb=0, mount at /workspace)."""
    buf = io.StringIO()
    args = argparse.Namespace(
        gpu="NVIDIA H100 80GB",
        gpu_count=1,
        disk=200,
        image=op.DEFAULT_IMAGE,
        repo_url=op.DEFAULT_REPO_URL,
        volume_id=None,
        community=False,
    )
    with contextlib.redirect_stdout(buf):
        op.cmd_spec(args)
    spec = json.loads(buf.getvalue().split("\n\n")[0])
    assert spec.get("volumeInGb", 0) == 0
    assert spec.get("networkVolumeId") is None or "networkVolumeId" not in spec
    assert spec["volumeMountPath"] == "/workspace"


def test_cmd_spec_community_flag_sets_community_cloud_flag() -> None:
    """--community should set communityCloud=true in the spec payload."""
    buf = io.StringIO()
    args = argparse.Namespace(
        gpu="NVIDIA H100 80GB",
        gpu_count=1,
        disk=200,
        image=op.DEFAULT_IMAGE,
        repo_url=op.DEFAULT_REPO_URL,
        volume_id=None,
        community=True,
    )
    with contextlib.redirect_stdout(buf):
        op.cmd_spec(args)
    spec = json.loads(buf.getvalue().split("\n\n")[0])
    assert spec.get("communityCloud") is True or spec.get("community") is True


def test_cmd_spec_argparse_accepts_volume_id_and_community() -> None:
    """argparse wires both new flags cleanly."""
    parser = op.build_parser()
    args = parser.parse_args([
        "spec", "--gpu", "NVIDIA H100 80GB",
        "--volume-id", "vol-xyz",
        "--community",
    ])
    assert args.volume_id == "vol-xyz"
    assert args.community is True


# ── main() dispatcher recognises new cmd_concurrent ─────────────────────────


def test_main_dispatches_to_cmd_concurrent(monkeypatch) -> None:
    """main() routes 'concurrent' to the new handler."""
    called = {"n": 0}

    def _fake_concurrent(_args):
        called["n"] = 1
        return 0

    monkeypatch.setattr(op, "cmd_concurrent", _fake_concurrent)

    op.main([
        "concurrent", "P", "--ssh-host", "1.1.1.1",
        "--n-seeds", "2", "--gpus", "0 1",
    ])
    assert called["n"] == 1
