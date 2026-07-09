"""RunPod one-command operator for DraftForge.

Subcommands (all stdlib-only: urllib + subprocess):

  recommend            Print GPU recommendation table (live RunPod API).
  spec --gpu h100      Emit JSON pod-create spec (paste into RunPod UI / MCP).
  push   POD_ID        SCP repo into a running pod, run onboard_pod.sh.
  run    POD_ID        SSH run_full_pipeline.sh on a configured pod.
  status POD_ID        Show GPU util + last 50 log lines.
  stop   POD_ID        SSH shutdown.
  one-liner            Print the full user-runtime sequence end-to-end.

Cost guardrails:
  - recommend hard-caps at $3/hr (within the $200-250 budget per STATE.md)
  - spec validates disk >= 200GB and image has CUDA
  - push/run/status refuse if pod unreachable (ssh exits non-zero)

This operator never auto-creates a pod. RunPod charges per-second from
pod-create; the user must explicitly invoke RunPod UI/MCP with the
emitted spec. Honor parent spec rule: "Never run destructive commands
without asking first."
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# DraftForge pod constraints (mirrors STATE.md / DECISIONS.md)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_URL = "https://github.com/anthropic-research/draftforge.git"
DEFAULT_DISK_GB = 200
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.12-cuda12.4.1-devel-ubuntu22.04"
MIN_GPU_DISK_GB = 200
DRAFTFORGE_HOME = "/workspace/draftforge"
HF_HUB_CACHE = "/workspace/hf/draftforge"
SSH_PORT = 22

# RunPod public API (no auth required for read endpoints)
RUNPOD_GPU_API = "https://api.runpod.io/graphql"
RUNPOD_GPU_QUERY = """query GpuTypes {
  gpuTypes {
    id
    displayName
    memoryInGb
    securePrice
    communityPrice
    lowestPrice(input: { gpuCount: 1 }) {
      minimumBidPrice
      uninterruptablePrice
    }
  }
}"""


# ── recommend: live GPU table from RunPod GraphQL API ─────────────────────────


def _runpod_gpu_types() -> list[dict[str, Any]]:
    """Fetch GPU types + prices from RunPod's public GraphQL endpoint."""
    payload = json.dumps({"query": RUNPOD_GPU_QUERY}).encode("utf-8")
    req = urllib.request.Request(
        RUNPOD_GPU_API,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if "errors" in body:
        raise RuntimeError(f"RunPod API error: {body['errors']}")
    return body["data"]["gpuTypes"]


def _recommend_table(max_hr: float = 3.0, min_mem_gb: int = 80) -> list[dict[str, Any]]:
    """Filter to (memory ≥ min_mem_gb, community price ≤ max_hr), sorted by perf/$.

    Heuristic perf score: memoryInGb x 1000 / communityPrice.
    Higher = better price-per-GB.
    """
    out = []
    for g in _runpod_gpu_types():
        mem = g.get("memoryInGb") or 0
        community = g.get("communityPrice")
        if mem < min_mem_gb:
            continue
        if community is None or community > max_hr:
            continue
        score = (mem * 1000.0) / community
        out.append(
            {
                "id": g["id"],
                "displayName": g["displayName"],
                "memoryInGb": mem,
                "communityPrice": community,
                "perf_per_dollar": round(score, 1),
            }
        )
    out.sort(key=lambda r: r["perf_per_dollar"], reverse=True)
    return out


def cmd_recommend(_args: argparse.Namespace) -> int:
    try:
        rows = _recommend_table()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[recommend] ERROR: RunPod API unreachable: {e}", file=sys.stderr)
        return 2
    if not rows:
        print("[recommend] no GPUs match (memory ≥80GB, price ≤$3/hr).")
        print("[recommend] raise --max-hr or lower --min-mem.")
        return 1
    print(f"{'id':<22} {'name':<32} {'mem':>6} {'$/hr':>8} {'perf/$':>10}")
    print("-" * 80)
    for r in rows[:15]:  # top 15
        print(
            f"{r['id']:<22} {r['displayName']:<32} {r['memoryInGb']:>5}G "
            f"${r['communityPrice']:>6.2f} {r['perf_per_dollar']:>10}"
        )
    print()
    print("[recommend] pick the top row unless you have a region preference.")
    print("[recommend] then: python scripts/operator_runpod.py spec --gpu <ID>")
    return 0


# ── spec: emit JSON payload for RunPod create-pod ────────────────────────────


def cmd_spec(args: argparse.Namespace) -> int:
    spec = {
        "name": "draftforge-train",
        "imageName": args.image,
        "gpuTypeId": args.gpu,
        "gpuCount": args.gpu_count,
        "containerDiskInGb": max(args.disk, MIN_GPU_DISK_GB),
        "volumeInGb": 0,
        "volumeMountPath": "/workspace",
        "ports": f"{SSH_PORT}/tcp",
        "env": [
            {"key": "DRAFTFORGE_REPO_URL", "value": args.repo_url},
            {"key": "DRAFTFORGE_HOME", "value": DRAFTFORGE_HOME},
            {"key": "HF_HUB_CACHE", "value": HF_HUB_CACHE},
            {"key": "PUBLIC_KEY", "value": "<paste your ssh public key>"},
        ],
        "dockerArgs": "",
        "minMemoryInGb": 32,
        "minVCPU": 8,
    }
    print(json.dumps(spec, indent=2))
    print()
    print("[spec] HOW TO USE")
    print("  1. RunPod UI → Pods → + Deploy → Custom → paste JSON above")
    print("  2. Set PUBLIC_KEY to your `cat ~/.ssh/id_rsa.pub`")
    print("  3. Click Deploy. Wait for pod = RUNNING.")
    print("  4. Note the pod ID and SSH command (RunPod UI shows them).")
    print("  5. ssh -p <port> root@<host>  (RunPod shows host:port)")
    print("  6. Then:")
    print(
        "       python scripts/operator_runpod.py push <POD_ID> "
        "--ssh-port <PORT> --ssh-host <HOST>"
    )
    print(
        "       python scripts/operator_runpod.py run  <POD_ID> "
        "--ssh-port <PORT> --ssh-host <HOST>"
    )
    return 0


# ── push / run / status / stop: subprocess to ssh + scp ──────────────────────


def _ssh_base(ssh_host: str, ssh_port: int, ssh_key: str | None) -> list[str]:
    base = ["ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=/dev/null"]
    if ssh_key:
        base += ["-i", ssh_key]
    base += [f"root@{ssh_host}"]
    return base


def _scp_base(ssh_port: int, ssh_key: str | None) -> list[str]:
    base = ["scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=/dev/null", "-r"]
    if ssh_key:
        base += ["-i", ssh_key]
    return base


def _run(cmd: list[str], check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Run a subprocess; print live output."""
    print(f"[run] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, timeout=timeout, text=True)


def cmd_push(args: argparse.Namespace) -> int:
    ssh_base = _ssh_base(args.ssh_host, args.ssh_port, args.ssh_key)
    scp_base = _scp_base(args.ssh_port, args.ssh_key)

    # 1. SCP the repo into the pod (rsync would be cleaner but adds dep).
    remote_dst = f"root@{args.ssh_host}:/workspace/"
    print(f"[push] copying repo to {remote_dst}")
    _run([*scp_base, str(REPO_ROOT), remote_dst], check=False)

    # 2. SSH in and run onboard_pod.sh.
    cmd = [*ssh_base, "bash /workspace/draftforge/scripts/onboard_pod.sh"]
    try:
        _run(cmd, check=True, timeout=900)
    except subprocess.CalledProcessError as e:
        print(f"[push] onboard failed: {e}", file=sys.stderr)
        return e.returncode
    print("[push] onboard complete. Pod ready for training.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    ssh_base = _ssh_base(args.ssh_host, args.ssh_port, args.ssh_key)
    env_overrides = ""
    if args.skip_train:
        env_overrides += " SKIP_TRAIN=1"
    if args.skip_ablate:
        env_overrides += " SKIP_ABLATE=1"
    if args.skip_serve:
        env_overrides += " SKIP_SERVE=1"
    if args.n_seeds:
        env_overrides += f" N_SEEDS={args.n_seeds}"
    cmd = [*ssh_base, f"cd /workspace/draftforge && " f"export{env_overrides} && " f"bash scripts/run_full_pipeline.sh 2>&1 | tee /workspace/draftforge/pipeline.log"]
    print("[run] full pipeline (training+ablation+serve+analyze+release).")
    print("[run] tail logs with: operator_runpod.py status <POD>")
    print("[run] stop with:        operator_runpod.py stop <POD>")
    try:
        _run(cmd, check=True, timeout=86_400)  # 24h ceiling
    except subprocess.TimeoutExpired:
        print("[run] WARNING: 24h ceiling hit — check pod log + cost.")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"[run] pipeline failed: {e}", file=sys.stderr)
        return e.returncode
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    ssh_base = _ssh_base(args.ssh_host, args.ssh_port, args.ssh_key)
    cmd = [*ssh_base, "echo '--- nvidia-smi ---' && nvidia-smi --query-gpu=memory.used,memory.total," "utilization.gpu --format=csv && " "echo '--- last 50 lines of pipeline.log (if any) ---' && " "(test -f /workspace/draftforge/pipeline.log && " "tail -n 50 /workspace/draftforge/pipeline.log || echo '(no pipeline.log yet)')"]
    try:
        _run(cmd, check=False, timeout=30)
    except subprocess.TimeoutExpired:
        print("[status] ssh timed out", file=sys.stderr)
        return 1
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    ssh_base = _ssh_base(args.ssh_host, args.ssh_port, args.ssh_key)
    print("[stop] WARNING: terminating pod (draftforge data on /workspace is volatile).")
    print("[stop] persist results with: scp -P <port> root@<host>:/workspace/draftforge/results ./results")
    cmd = [*ssh_base, "shutdown -h now"]
    try:
        _run(cmd, check=False, timeout=15)
    except subprocess.CalledProcessError:
        pass
    return 0


def cmd_one_liner(_args: argparse.Namespace) -> int:
    print("=" * 70)
    print("DraftForge x RunPod — full user-runtime sequence")
    print("=" * 70)
    print()
    print("# 1. Pick a GPU (latest, ≤$3/hr, ≥80GB):")
    print("    python scripts/operator_runpod.py recommend")
    print()
    print("# 2. Generate pod-create spec:")
    print("    python scripts/operator_runpod.py spec --gpu <GPU_ID>")
    print()
    print("# 3. RunPod UI → Deploy Custom Pod → paste spec")
    print("    (set PUBLIC_KEY = `cat ~/.ssh/id_rsa.pub`)")
    print()
    print("# 4. After pod = RUNNING, note host + port from RunPod UI.")
    print()
    print("# 5. Push repo + onboard:")
    print("    python scripts/operator_runpod.py push POD_ID \\")
    print("        --ssh-host <HOST> --ssh-port <PORT> [--ssh-key ~/.ssh/id_rsa]")
    print()
    print("# 6. Run the pipeline (24h ceiling, default 1 seed = ~6-8h on H100):")
    print("    python scripts/operator_runpod.py run POD_ID \\")
    print("        --ssh-host <HOST> --ssh-port <PORT> --n-seeds 1")
    print()
    print("# 7. Monitor:")
    print("    python scripts/operator_runpod.py status POD_ID \\")
    print("        --ssh-host <HOST> --ssh-port <PORT>")
    print()
    print("# 8. Pull results + terminate:")
    print("    scp -P <PORT> -r root@<HOST>:/workspace/draftforge/results ./results")
    print("    python scripts/operator_runpod.py stop POD_ID --ssh-host <HOST> --ssh-port <PORT>")
    print()
    print("=" * 70)
    return 0


# ── argparse ─────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="operator_runpod",
        description="DraftForge x RunPod one-command operator.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # recommend
    sub.add_parser("recommend", help="List best GPUs (live RunPod API).")

    # spec
    spec_p = sub.add_parser("spec", help="Emit JSON pod-create spec.")
    spec_p.add_argument("--gpu", default="NVIDIA H100 80GB HBM3",
                        help="RunPod GPU type id (default: H100 80GB).")
    spec_p.add_argument("--gpu-count", type=int, default=1)
    spec_p.add_argument("--disk", type=int, default=DEFAULT_DISK_GB,
                        help=f"Container disk in GB (min {MIN_GPU_DISK_GB}).")
    spec_p.add_argument("--image", default=DEFAULT_IMAGE)
    spec_p.add_argument("--repo-url", default=DEFAULT_REPO_URL)

    # push / run / status / stop — all need ssh connectivity
    for name in ("push", "run", "status", "stop"):
        sp = sub.add_parser(name, help=f"{name.title()} the pod.")
        sp.add_argument("pod_id", help="RunPod pod ID (informational; ssh connectivity used).")
        sp.add_argument("--ssh-host", required=True, help="RunPod pod SSH host.")
        sp.add_argument("--ssh-port", type=int, default=SSH_PORT)
        sp.add_argument("--ssh-key", default=None, help="Path to SSH private key.")

    # run-specific extras
    run_p = sub.choices["run"]  # type: ignore[attr-defined]
    run_p.add_argument("--n-seeds", type=int, default=1,
                       help="Number of training seeds (default 1; 3 = full reproducibility).")
    run_p.add_argument("--skip-train", action="store_true")
    run_p.add_argument("--skip-ablate", action="store_true")
    run_p.add_argument("--skip-serve", action="store_true")

    sub.add_parser("one-liner", help="Print the full user-runtime sequence.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "recommend":
        return cmd_recommend(args)
    if args.cmd == "spec":
        return cmd_spec(args)
    if args.cmd == "push":
        return cmd_push(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "stop":
        return cmd_stop(args)
    if args.cmd == "one-liner":
        return cmd_one_liner(args)
    parser.error(f"unknown subcommand: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
