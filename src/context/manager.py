"""Context window manager: token counting, auto-trim, priority eviction."""
from __future__ import annotations

import os
from typing import Any

import tiktoken


# Priority levels (higher = keep longer)
PRIORITY_SHORT_TERM_OLD = 4   # evict first
PRIORITY_SEMANTIC = 3
PRIORITY_EPISODIC = 2
PRIORITY_LONG_TERM = 1        # keep longest


class ContextManager:
    """Manages context assembly and token-budget enforcement.

    4-Level Priority Eviction Hierarchy:
        Level 4 (evict first) → Old short-term messages (> N turns ago)
        Level 3               → Semantic search results (re-fetchable)
        Level 2               → Episodic summaries
        Level 1 (keep last)   → Redis facts/preferences (user identity)
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        trim_threshold: float | None = None,
    ) -> None:
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "8192"))
        self.trim_threshold = trim_threshold or float(
            os.getenv("CONTEXT_TRIM_THRESHOLD", "0.8")
        )
        self._enc = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Return number of tokens in *text*."""
        return len(self._enc.encode(text))

    def count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens across a list of {role, content} dicts."""
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.get("content", ""))
            total += 4  # per-message overhead (role + framing)
        return total

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def build_context(
        self,
        short_term: str = "",
        long_term: str = "",
        episodic: str = "",
        semantic: str = "",
    ) -> str:
        """Assemble context string respecting token budget.

        Adds sections in priority order (lowest priority first so that
        high-priority sections always fit).
        """
        budget = int(self.max_tokens * self.trim_threshold)
        sections: list[tuple[int, str, str]] = []  # (priority, label, content)

        if short_term:
            sections.append((PRIORITY_SHORT_TERM_OLD, "Recent conversation", short_term))
        if semantic:
            sections.append((PRIORITY_SEMANTIC, "Related knowledge", semantic))
        if episodic:
            sections.append((PRIORITY_EPISODIC, "Past experiences", episodic))
        if long_term:
            sections.append((PRIORITY_LONG_TERM, "User facts", long_term))

        # Sort: keep highest-priority (low number) sections when budget is tight
        sections.sort(key=lambda x: x[0], reverse=True)  # high priority number = add first, trim first

        kept: list[str] = []
        used = 0
        # We want priority 1 (long-term) to survive — reverse sort keeps priority 4 first
        # so we evict them when over budget.
        # Rebuild: add in priority order high→low, trim from high end.

        # Strategy: add all, then evict from lowest priority (highest number) until fits
        all_parts: list[tuple[int, str]] = []
        for priority, label, content in sections:
            part = f"=== {label} ===\n{content}"
            all_parts.append((priority, part))

        all_parts.sort(key=lambda x: x[0])  # low number = high priority = keep
        result_parts: list[str] = [p for _, p in all_parts]

        # Trim from the end (lowest priority = highest number in sorted desc)
        all_parts_desc = sorted(all_parts, key=lambda x: x[0], reverse=True)  # evict first = high number

        while True:
            full_text = "\n\n".join(p for _, p in sorted(all_parts_desc, key=lambda x: x[0]))
            if self.count_tokens(full_text) <= budget or not all_parts_desc:
                return full_text
            all_parts_desc.pop(0)  # evict highest-number priority

        return ""

    # ------------------------------------------------------------------
    # Trim messages list
    # ------------------------------------------------------------------

    def trim_messages(
        self,
        messages: list[dict[str, str]],
        reserved_tokens: int = 1000,
    ) -> list[dict[str, str]]:
        """Trim oldest messages so total fits within budget.

        *reserved_tokens* is set aside for the system prompt and response.
        Always keeps the most recent message.
        """
        budget = self.max_tokens - reserved_tokens
        # Walk from newest to oldest
        kept: list[dict[str, str]] = []
        used = 0
        for msg in reversed(messages):
            t = self.count_tokens(msg.get("content", "")) + 4
            if used + t > budget and kept:
                break
            kept.append(msg)
            used += t
        return list(reversed(kept))
