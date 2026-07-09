"""Quickstart: render vLLM + SGLang invocations for a (hypothetical) head.

Run: .venv/bin/python examples/quickstart_serve.py

No GPU, no HF, no network. Demonstrates serve/integrate.py's invocation
builders — useful for sanity-checking the --speculative-config JSON shape
before going to a GPU pod.

The 'checkpoint_dir' below is a placeholder. Replace with a real path
once you have a trained head from train/run_all_seeds.sh.
"""

from __future__ import annotations

from pathlib import Path

from serve.integrate import build_sglang_invocation, build_vllm_invocation


def main() -> int:
    print("=" * 60)
    print("DraftForge quickstart: vLLM + SGLang invocation builders")
    print("=" * 60)

    target = "Qwen/Qwen3-14B"
    head_path = Path("results/train/tri_layer/42/best")
    num_spec = 4

    # 1. vLLM invocation
    print("\n[1] vLLM invocation")
    vllm_cmd = build_vllm_invocation(
        target_model=target,
        draft_head_path=head_path,
        num_speculative_tokens=num_spec,
    )
    print(f"    $ {vllm_cmd}")

    # 2. SGLang invocation
    print("\n[2] SGLang invocation")
    sglang_cmd = build_sglang_invocation(
        target_model=target,
        draft_head_path=head_path,
        num_speculative_tokens=num_spec,
    )
    print(f"    $ {sglang_cmd}")

    # 3. Write the invocations to a script (so you can `bash` it on a GPU pod)
    out_dir = Path("examples/_out")
    out_dir.mkdir(exist_ok=True)
    vllm_script = out_dir / "vllm_cmd.sh"
    vllm_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + vllm_cmd + "\n")
    vllm_script.chmod(0o755)
    sglang_script = out_dir / "sglang_cmd.sh"
    sglang_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + sglang_cmd + "\n")
    sglang_script.chmod(0o755)
    print("\n[3] Wrote executable scripts:")
    print(f"    {vllm_script}")
    print(f"    {sglang_script}")
    print("\n    Copy to a GPU pod and run with:")
    print("    $ scp examples/_out/vllm_cmd.sh user@pod:~/")
    print("    $ ssh user@pod 'bash vllm_cmd.sh'")

    print("\n" + "=" * 60)
    print("done. The --speculative-config JSON should mention eagle3,")
    print("num_speculative_tokens=4, and the draft model path.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
