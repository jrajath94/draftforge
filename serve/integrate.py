"""Construct CLI invocations that load the draft head in vLLM and SGLang.

The draft head path and target model name are injected into the verification
commands. Both commands are shell-invocation strings — the runner executes
them in subprocess and captures stdout/stderr for `make bench`.

References (verified 2026-07-08):
  vLLM:   docs.vllm.ai/en/latest/features/speculative_decoding/
  SGLang: docs.sglang.io/advanced_features/speculative_decoding.html

Pin versions at run time: `pip install "vllm>=0.10,<0.16"`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_vllm_invocation(
    *,
    target_model: str,
    draft_head_path: Path,
    num_speculative_tokens: int = 4,
    extra: str = "",
) -> str:
    """Compose vllm serve invocation with EAGLE-3 speculation."""
    config_json = json.dumps(
        {
            "method": "eagle3",
            "model": str(draft_head_path),
            "num_speculative_tokens": num_speculative_tokens,
        }
    )
    # Note: --speculative-config expects a JSON string (CLI form) or YAML file.
    return (
        f'vllm serve "{target_model}" '
        f'--host 127.0.0.1 --port 8000 '
        f"--speculative-config '{config_json}' "
        f"--trust-remote-code "
        f"{extra}"
    )


def build_sglang_invocation(
    *,
    target_model: str,
    draft_head_path: Path,
    num_speculative_tokens: int = 4,
    extra: str = "",
) -> str:
    """Compose sglang serve invocation with EAGLE-3 speculation."""
    return (
        f"python -m sglang.launch_server "
        f"--model-path {target_model} "
        f"--speculative-algorithm EAGLE3 "
        f"--speculative-draft-model-path {draft_head_path} "
        f"--speculative-num-steps {num_speculative_tokens} "
        f"--host 127.0.0.1 --port 30000 "
        f"{extra}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="HF model id or path")
    ap.add_argument("--draft", required=True, help="Path to trained draft head dir")
    ap.add_argument("--runtime", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--num-spec", type=int, default=4)
    ap.add_argument(
        "--out", type=Path, default=Path("results/serve/invocation.sh")
    )
    args = ap.parse_args()

    draft_p = Path(args.draft)
    if args.runtime == "vllm":
        cmd = build_vllm_invocation(
            target_model=args.target,
            draft_head_path=draft_p,
            num_speculative_tokens=args.num_spec,
        )
    else:
        cmd = build_sglang_invocation(
            target_model=args.target,
            draft_head_path=draft_p,
            num_speculative_tokens=args.num_spec,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{cmd}\n")
    args.out.chmod(0o755)
    print(cmd)


if __name__ == "__main__":
    main()
