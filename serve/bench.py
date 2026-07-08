"""vLLM/SGLang benchmark wrappers.

Generates load + latency commands using each runtime's `bench` tool:
  vLLM:   vllm bench latency / throughput
  SGLang: python -m sglang.bench_one_batch / sglang.bench_serving

Outputs to results/serve/<runtime>/baseline/ and .../speculative/ JSONL.
Comparison table comes from `eval/acceptance.py`.

Requires: GPU + the runtimes installed.
"""

from __future__ import annotations

from pathlib import Path


def vllm_bench_cmd(
    *,
    target_model: str,
    spec: bool = False,
    draft_head: Path | None = None,
    num_spec: int = 4,
    max_num_seqs: int = 16,
    output: Path,
) -> str:
    base = (
        f"vllm bench latency "
        f"--model {target_model} "
        f"--max-num-seqs {max_num_seqs} "
        f"--input-len 512 --output-len 256 "
        f"--save-json {output}"
    )
    if spec and draft_head is not None:
        import json

        cfg = json.dumps(
            {
                "method": "eagle3",
                "model": str(draft_head),
                "num_speculative_tokens": num_spec,
            }
        )
        return f"{base} --speculative-config '{cfg}' --trust-remote-code"
    return base


def sglang_bench_cmd(
    *,
    target_model: str,
    spec: bool = False,
    draft_head: Path | None = None,
    num_spec: int = 4,
    output: Path,
) -> str:
    base = (
        f"python -m sglang.bench_one_batch "
        f"--model-path {target_model} "
        f"--batch-size 16 "
        f"--input-len 512 --output-len 256 "
        f"--json {output}"
    )
    if spec and draft_head is not None:
        return (
            f"{base} "
            f"--speculative-algorithm EAGLE3 "
            f"--speculative-draft-model-path {draft_head} "
            f"--speculative-num-steps {num_spec}"
        )
    return base
