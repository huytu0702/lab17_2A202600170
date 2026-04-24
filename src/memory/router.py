"""Memory router: classifies query intent and dispatches to the right backend."""
from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .short_term import ShortTermMemory
    from .long_term import LongTermMemory
    from .episodic import EpisodicMemory
    from .semantic import SemanticMemory


class MemoryIntent(str, Enum):
    USER_PREFERENCE = "user_preference"   # Redis
    FACTUAL_RECALL = "factual_recall"     # Chroma
    EXPERIENCE_RECALL = "experience_recall"  # Episodic JSON
    GENERAL = "general"                   # Short-term buffer


# ---------------------------------------------------------------------------
# Rule-based patterns
# ---------------------------------------------------------------------------

_PREFERENCE_PATTERNS = re.compile(
    r"\b(i like|i love|i prefer|i enjoy|i hate|i dislike|"
    r"my name is|call me|i am|remember (that )?i|"
    r"my (favorite|favourite)|i usually|i always|i never)\b",
    re.IGNORECASE,
)

_EXPERIENCE_PATTERNS = re.compile(
    r"\b(last time|previously|before|earlier|we (talked|discussed|spoke)|"
    r"you (said|told|mentioned)|what did (we|you)|in our (last|previous)|"
    r"do you remember when|the other day)\b",
    re.IGNORECASE,
)

_FACTUAL_PATTERNS = re.compile(
    r"\b(what is|what are|who is|who are|how does|how do|explain|"
    r"tell me about|describe|define|what does|why is|why are|"
    r"when (is|was|did)|where (is|are))\b",
    re.IGNORECASE,
)


class MemoryRouter:
    """Routes queries to the appropriate memory backend and retrieves context."""

    def __init__(
        self,
        short_term: "ShortTermMemory",
        long_term: "LongTermMemory",
        episodic: "EpisodicMemory",
        semantic: "SemanticMemory",
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term
        self.episodic = episodic
        self.semantic = semantic

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_intent(self, query: str) -> MemoryIntent:
        """Return the most appropriate MemoryIntent for *query*."""
        if _PREFERENCE_PATTERNS.search(query):
            return MemoryIntent.USER_PREFERENCE
        if _EXPERIENCE_PATTERNS.search(query):
            return MemoryIntent.EXPERIENCE_RECALL
        if _FACTUAL_PATTERNS.search(query):
            return MemoryIntent.FACTUAL_RECALL
        return MemoryIntent.GENERAL

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, user_id: str, thread_id: str) -> str:
        """Retrieve relevant context from the appropriate backend(s).

        Always includes short-term buffer. Adds specialised context based
        on intent.
        """
        intent = self.classify_intent(query)
        parts: list[str] = []

        # Short-term is always included
        short = self.short_term.get_formatted(thread_id)
        if short:
            parts.append(f"=== Recent conversation ===\n{short}")

        if intent == MemoryIntent.USER_PREFERENCE:
            long = self.long_term.get_formatted(user_id)
            if long:
                parts.append(f"=== User facts/preferences ===\n{long}")

        elif intent == MemoryIntent.EXPERIENCE_RECALL:
            episodes = self.episodic.get_formatted(query, user_id=user_id)
            if episodes:
                parts.append(f"=== Past experiences ===\n{episodes}")

        elif intent == MemoryIntent.FACTUAL_RECALL:
            semantic = self.semantic.get_formatted(query)
            if semantic:
                parts.append(f"=== Related knowledge ===\n{semantic}")

        else:  # GENERAL — check all briefly
            long = self.long_term.get_formatted(user_id)
            if long:
                parts.append(f"=== User facts/preferences ===\n{long}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        content: str,
        user_id: str,
        intent: MemoryIntent | None = None,
        key: str | None = None,
        turn: int = 0,
    ) -> None:
        """Persist *content* to the appropriate backend."""
        if intent is None:
            intent = self.classify_intent(content)

        if intent == MemoryIntent.USER_PREFERENCE and key:
            self.long_term.set_fact(user_id, key, content)
        elif intent == MemoryIntent.FACTUAL_RECALL:
            self.semantic.add_memory(content, metadata={"user_id": user_id})
        elif intent == MemoryIntent.EXPERIENCE_RECALL:
            self.episodic.log_episode(user_id=user_id, content=content, turn=turn)
        else:
            # General — log as episodic with low importance
            self.episodic.log_episode(
                user_id=user_id, content=content, turn=turn, importance_score=0.3
            )
