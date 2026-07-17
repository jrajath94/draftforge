"""OpenHermes-2.5 loader.

Public dataset: teknium/OpenHermes-2.5. Subset for variety since
ShareGPT-derived is already the primary source.
"""

from __future__ import annotations

try:
    from datasets import load_dataset
except ModuleNotFoundError:  # pragma: no cover - depends on optional extra
    load_dataset = None  # type: ignore[assignment, unused-ignore]

from data.types import Example


def load_openhermes(
    hf_dataset_id: str = "teknium/OpenHermes-2.5",
    split: str = "train",
    max_examples: int = 50_000,
) -> list[Example]:
    """Load OpenHermes-2.5 instruction/response traces (subset)."""
    if load_dataset is None:
        raise ModuleNotFoundError(
            "datasets is required for OpenHermes loading; install the data "
            "extras or use a local-only config"
        )
    ds = load_dataset(hf_dataset_id, split=split, streaming=True)
    out: list[Example] = []
    for i, row in enumerate(ds):
        if i >= max_examples:
            break
        # OpenHermes uses "conversations" with "from"/"value"
        # OR a single "instruction"/"output" pair (newer versions).
        messages = _extract_messages(row)
        if not messages or len(messages) < 2:
            continue
        out.append(
            Example(
                id=f"openhermes-{i:06d}",
                domain="general",
                messages=messages,
                source="openhermes",
                meta={"hf_dataset_id": hf_dataset_id},
            )
        )
    return out


def _extract_messages(row: dict) -> list[dict[str, str]]:
    """Extract messages from either conversations or instruction/output format."""
    if "conversations" in row and isinstance(row["conversations"], list):
        msgs: list[dict[str, str]] = []
        for turn in row["conversations"]:
            frm = turn.get("from", "")
            val = turn.get("value", "")
            norm_role = "user" if frm in ("human", "user", "system") else "assistant"
            msgs.append({"role": norm_role, "content": str(val)})
        return msgs
    if "instruction" in row and "output" in row:
        return [
            {"role": "user", "content": str(row["instruction"])},
            {"role": "assistant", "content": str(row["output"])},
        ]
    return []
