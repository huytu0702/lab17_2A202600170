"""Short-term memory backend using LangGraph MemorySaver."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver


class ShortTermMemory:
    """Wraps LangGraph MemorySaver for per-thread conversation buffer.

    Keeps last ``buffer_window`` human+AI turns in memory.
    Thread state is lost when the process restarts (MemorySaver is in-process).
    """

    def __init__(self, buffer_window: int = 10) -> None:
        self.checkpointer = MemorySaver()
        self.buffer_window = buffer_window
        # Local cache: thread_id -> list[BaseMessage]
        self._buffer: dict[str, list[BaseMessage]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(self, thread_id: str, message: BaseMessage) -> None:
        """Append a message and evict oldest if over window."""
        buf = self._buffer.setdefault(thread_id, [])
        buf.append(message)
        # Keep at most buffer_window * 2 messages (N human + N AI)
        max_msgs = self.buffer_window * 2
        if len(buf) > max_msgs:
            self._buffer[thread_id] = buf[-max_msgs:]

    def get_messages(self, thread_id: str) -> list[BaseMessage]:
        """Return stored messages for the given thread."""
        return list(self._buffer.get(thread_id, []))

    def get_formatted(self, thread_id: str) -> str:
        """Return a human-readable string of recent turns."""
        msgs = self.get_messages(thread_id)
        if not msgs:
            return ""
        lines: list[str] = []
        for msg in msgs:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def clear(self, thread_id: str) -> None:
        """Clear buffer for a specific thread."""
        self._buffer.pop(thread_id, None)

    def get_config(self, thread_id: str) -> dict[str, Any]:
        """Return LangGraph config dict for this thread."""
        return {"configurable": {"thread_id": thread_id}}
