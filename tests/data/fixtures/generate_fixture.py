"""Generate a synthetic tiny_traces.jsonl for offline tests.

50 examples: 40 general (mix of sharegpt/openhermes styles), 10 finance.
Includes 3 exact duplicates and 1 near-duplicate for dedup verification.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "tiny_traces.jsonl"


def _msgs(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def main() -> None:
    rows: list[dict] = []

    # 20 sharegpt-style general
    for i in range(20):
        rows.append(
            {
                "id": f"sharegpt-{i:03d}",
                "domain": "general",
                "source": "sharegpt",
                "messages": _msgs(
                    f"General question {i}: explain concept {i}?",
                    f"General answer {i}: explanation about concept {i}.",
                ),
                "meta": {},
            }
        )

    # 20 openhermes-style general
    for i in range(20):
        rows.append(
            {
                "id": f"openhermes-{i:03d}",
                "domain": "general",
                "source": "openhermes",
                "messages": _msgs(
                    f"OpenHermes q{i}: write snippet {i}.",
                    f"OpenHermes a{i}: snippet {i} code body.",
                ),
                "meta": {},
            }
        )

    # 10 finance
    for i in range(10):
        rows.append(
            {
                "id": f"finance-{i:03d}",
                "domain": "finance",
                "source": "finance",
                "messages": _msgs(
                    f"Finance question {i}: compare P/E ratios of X and Y.",
                    f"Finance answer {i}: P/E analysis with ratios.",
                ),
                "meta": {"topic": "equity"},
            }
        )

    # 3 exact duplicates (re-emit some general rows with different IDs)
    for src in ("sharegpt-000", "openhermes-000", "finance-000"):
        original = next(r for r in rows if r["id"] == src)
        rows.append(
            {
                **original,
                "id": f"{src}-dup",
            }
        )
    # 1 near-duplicate (rephrase openhermes-001)
    rows.append(
        {
            "id": "openhermes-001-near-dup",
            "domain": "general",
            "source": "openhermes",
            "messages": _msgs(
                "OpenHermes q1: write snippet 1.",
                "OpenHermes a1: snippet 1 code body.",
            ),
            "meta": {},
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
