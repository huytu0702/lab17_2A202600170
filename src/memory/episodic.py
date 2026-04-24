"""Episodic memory backend — append-only JSON log."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


_DEFAULT_LOG_PATH = Path(__file__).parent.parent.parent / "data" / "episodic_log.json"


class EpisodicMemory:
    """Stores conversation episodes in a JSON file.

    Each episode record:
    {
        "id": str,
        "timestamp": ISO-8601 str,
        "user_id": str,
        "turn": int,
        "content": str,
        "tags": list[str],
        "importance_score": float  # 0.0 – 1.0
    }
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        self.log_path = Path(log_path or _DEFAULT_LOG_PATH)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text("[]", encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, episodes: list[dict[str, Any]]) -> None:
        self.log_path.write_text(
            json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_episode(
        self,
        user_id: str,
        content: str,
        turn: int = 0,
        tags: list[str] | None = None,
        importance_score: float = 0.5,
    ) -> str:
        """Append a new episode. Returns the episode id."""
        episode: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "turn": turn,
            "content": content,
            "tags": tags or [],
            "importance_score": max(0.0, min(1.0, importance_score)),
        }
        episodes = self._load()
        episodes.append(episode)
        self._save(episodes)
        return episode["id"]

    def search_episodes(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return up to *top_k* episodes matching *query* keywords.

        Scoring: +2 per keyword found in tags, +1 per keyword in content.
        """
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        episodes = self._load()
        if user_id:
            episodes = [e for e in episodes if e.get("user_id") == user_id]

        scored: list[tuple[float, dict[str, Any]]] = []
        for ep in episodes:
            score = ep.get("importance_score", 0.5)
            content_lower = ep.get("content", "").lower()
            tags_lower = [t.lower() for t in ep.get("tags", [])]
            for kw in keywords:
                if any(kw in tag for tag in tags_lower):
                    score += 2
                if kw in content_lower:
                    score += 1
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def get_all(self, user_id: str | None = None) -> list[dict[str, Any]]:
        episodes = self._load()
        if user_id:
            return [e for e in episodes if e.get("user_id") == user_id]
        return episodes

    def get_formatted(self, query: str, user_id: str, top_k: int = 3) -> str:
        episodes = self.search_episodes(query, user_id=user_id, top_k=top_k)
        if not episodes:
            return ""
        lines = []
        for ep in episodes:
            ts = ep.get("timestamp", "")[:19].replace("T", " ")
            lines.append(f"[{ts}] {ep['content']}")
        return "\n".join(lines)
