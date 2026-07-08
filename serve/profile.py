"""Nsight Systems wrapper for draft-verify loop profiling.

`nsys profile --trace=cuda,nvtx --output=... python ...` captures kernels
issued by the draft model + verify pass. Compare against the same
workload without speculation to see whether ITL bottleneck is
draft-bound or verify-bound.

Usage:
   nsys profile --trace=cuda,nvtx --output=results/serve/nsys/speculative \\
        -- python ...serve_speculative.py

This file wraps the nsys invocation so the bench script stays portable.

Addendum 5: kernel-attribution rules (BINDING_RULES below) map the
percentage of loop time spent in draft vs verify kernels to one of
three regimes. The exact thresholds depend on vLLM version and compute
capability — call `classify_binding()` and compare against measured
percentages, do not hard-code the regime assignment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Addendum 5 — kernel attribution table.
# Maps the % of loop time spent in the DRAFT kernel to a binding regime.
# Thresholds are empirical defaults; tune after first GPU run.
#
#   regime          | draft %   | verify % | remediation
#   ----------------|-----------|----------|----------------------------
#   draft-bound     | > 60%     | < 30%    | quantize/distill draft head
#   verify-bound    | < 30%     | > 60%    | improve acceptance rate
#   balanced        | else      | else     | no change; track over time
#
# Verify percentage = 100 - draft% - other_overhead. "other_overhead"
# covers KV-cache memory copies, tokenizer/detokenizer, and scheduler
# waits, so the three numbers need not sum to 100.
BINDING_RULES: dict[str, tuple[float, float]] = {
    "draft-bound": (0.60, 1.01),   # draft_pct >= 0.60
    "verify-bound": (-0.01, 0.30), # verify_pct > 0.60 → draft < 0.30
    "balanced": (0.30, 0.60),      # 0.30 <= draft < 0.60
}


def classify_binding(draft_pct: float) -> str:
    """Classify the loop-time regime from the draft kernel's share.

    Args:
        draft_pct: fraction of loop time in the draft forward pass (0.0-1.0).

    Returns:
        One of "draft-bound" | "verify-bound" | "balanced".
    """
    if draft_pct >= BINDING_RULES["draft-bound"][0]:
        return "draft-bound"
    if draft_pct < BINDING_RULES["balanced"][0]:
        return "verify-bound"
    return "balanced"


def nsys_run(
    target_cmd: list[str],
    output: Path,
    *,
    trace: str = "cuda,nvtx",
    extra: list[str] | None = None,
) -> int:
    """Invoke nsys profile. Returns subprocess exit code."""
    output.parent.mkdir(parents=True, exist_ok=True)
    nsys_args = [
        "nsys",
        "profile",
        f"--trace={trace}",
        f"--output={output}",
        *(extra or []),
        "--",
    ]
    return subprocess.call(nsys_args + target_cmd)


def nsys_analyze_hint(report_path: Path) -> str:
    """Suggest nsys stats command for the top kernels."""
    return (
        f"nsys stats --report cuda_gpu_kern_sum,gputrace {report_path}.nsys-rep"
    )
