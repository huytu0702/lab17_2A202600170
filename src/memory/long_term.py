"""Long-term memory backend using Redis (cross-session user facts)."""
from __future__ import annotations

import json
import os
from typing import Any


class LongTermMemory:
    """Stores user facts/preferences in Redis as JSON hashes.

    Falls back to an in-process dict when Redis is unavailable so that
    development works without a running Redis instance.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis: Any = None
        self._mock: dict[str, dict[str, str]] = {}
        try:
            import redis  # type: ignore

            client = redis.from_url(url, decode_responses=True)
            client.ping()
            self._redis = client
        except Exception:
            # Redis not available — use in-memory fallback
            self._redis = None

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_fact(self, user_id: str, key: str, value: str) -> None:
        """Store a single fact for *user_id*."""
        if self._redis:
            self._redis.hset(f"user:{user_id}:facts", key, value)
        else:
            self._mock.setdefault(user_id, {})[key] = value

    def get_fact(self, user_id: str, key: str) -> str | None:
        """Retrieve a single fact by key."""
        if self._redis:
            return self._redis.hget(f"user:{user_id}:facts", key)
        return self._mock.get(user_id, {}).get(key)

    def get_facts(self, user_id: str) -> dict[str, str]:
        """Retrieve all facts for *user_id*."""
        if self._redis:
            return self._redis.hgetall(f"user:{user_id}:facts") or {}
        return dict(self._mock.get(user_id, {}))

    def search_facts(self, user_id: str, query: str) -> dict[str, str]:
        """Return facts whose key or value contains *query* (case-insensitive)."""
        q = query.lower()
        return {
            k: v
            for k, v in self.get_facts(user_id).items()
            if q in k.lower() or q in v.lower()
        }

    def delete_fact(self, user_id: str, key: str) -> None:
        if self._redis:
            self._redis.hdel(f"user:{user_id}:facts", key)
        else:
            self._mock.get(user_id, {}).pop(key, None)

    def clear_user(self, user_id: str) -> None:
        if self._redis:
            self._redis.delete(f"user:{user_id}:facts")
        else:
            self._mock.pop(user_id, None)

    def get_formatted(self, user_id: str) -> str:
        facts = self.get_facts(user_id)
        if not facts:
            return ""
        return "\n".join(f"- {k}: {v}" for k, v in facts.items())
