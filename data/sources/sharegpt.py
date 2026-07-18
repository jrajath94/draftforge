"""ShareGPT-derived loader.

Default is Aeala/ShareGPT_Vicuna_unfiltered (~68K ShareGPT conversations,
the corpus family the EAGLE line of work trains on). Note the EAGLE-3
reference repo `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` is a MODEL repo —
`load_dataset` cannot read it (found by the rung-3 GPU smoke).
"""

from __future__ import annotations

try:
    from datasets import load_dataset
except ModuleNotFoundError:  # pragma: no cover - depends on optional extra
    load_dataset = None  # type: ignore[assignment, unused-ignore]

from data.types import Example


def load_sharegpt(
    hf_dataset_id: str = "Aeala/ShareGPT_Vicuna_unfiltered",
    split: str = "train",
    max_examples: int = 100_000,
) -> list[Example]:
    """Load ShareGPT-derived instruction/response traces."""
    if load_dataset is None:
        raise ModuleNotFoundError(
            "datasets is required for ShareGPT loading; install the data "
            "extras or use a local-only config"
        )
    ds = load_dataset(hf_dataset_id, split=split, streaming=True)
    out: list[Example] = []
    for i, row in enumerate(ds):
        if i >= max_examples:
            break
        # EAGLE3 dataset stores conversations under various keys depending
        # on version. Common keys: "conversations", "messages", "text".
        messages = _extract_messages(row)
        if not messages or len(messages) < 2:
            continue
        out.append(
            Example(
                id=f"sharegpt-{i:06d}",
                domain="general",
                messages=messages,
                source="sharegpt",
                meta={"hf_dataset_id": hf_dataset_id},
            )
        )
    return out


def _extract_messages(row: dict) -> list[dict[str, str]]:
    """Pull messages out of the various schema variants."""
    if "messages" in row and isinstance(row["messages"], list):
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in row["messages"]]
    if "conversations" in row and isinstance(row["conversations"], list):
        msgs: list[dict[str, str]] = []
        for turn in row["conversations"]:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            # ShareGPT uses "human"/"gpt"; normalize to OpenAI roles.
            norm_role = "user" if role in ("human", "user") else "assistant"
            msgs.append({"role": norm_role, "content": str(content)})
        return msgs
    if "text" in row:
        return [{"role": "user", "content": str(row["text"])}]
    return []
