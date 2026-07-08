"""Shared data types for DraftForge data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    """One instruction/response trace.

    `messages` is the canonical OpenAI-style chat log:
        [{"role": "user", "content": "..."},
         {"role": "assistant", "content": "..."}]
    """

    id: str
    domain: str  # "general" | "finance"
    messages: list[dict[str, str]]
    source: str
    meta: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Concatenate messages into a single string (for hashing/dedup).

        Whitespace-normalized so trivial formatting differences don't
        spawn duplicates.
        """
        parts: list[str] = []
        for m in self.messages:
            parts.append(f"{m['role']}: {m['content']}")
        return " ".join(" ".join(p.split()) for p in parts)
