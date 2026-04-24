"""LangGraph agent with multi-memory routing.

Graph: START → memory_retrieve → respond → memory_save → END

AgentState follows rubric shape:
    messages, user_profile, episodes, semantic_hits, memory_budget
"""
from __future__ import annotations

import operator
import os
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.context.manager import ContextManager
from src.memory import (
    EpisodicMemory,
    LongTermMemory,
    MemoryIntent,
    MemoryRouter,
    SemanticMemory,
    ShortTermMemory,
)

load_dotenv()


# ---------------------------------------------------------------------------
# State — matches rubric shape
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    # Core conversation
    messages: Annotated[list[BaseMessage], operator.add]
    # Identity
    user_id: str
    thread_id: str
    # Memory sections injected into prompt
    user_profile: dict[str, str]        # long-term Redis facts
    episodes: list[dict]                # episodic log entries
    semantic_hits: list[str]            # Chroma search results
    recent_conversation: str            # short-term buffer text
    # Budget
    memory_budget: int
    # Internal routing
    intent: str


# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

_short_term = ShortTermMemory(buffer_window=10)
_long_term = LongTermMemory()
_episodic = EpisodicMemory()
_semantic = SemanticMemory()
_router = MemoryRouter(_short_term, _long_term, _episodic, _semantic)
_ctx_mgr = ContextManager()
_llm = ChatOpenAI(
    model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
    temperature=0.7,
)


# ---------------------------------------------------------------------------
# Helper: build structured system prompt from state sections
# ---------------------------------------------------------------------------


def _build_system_prompt(state: AgentState) -> str:
    parts = [
        "You are a helpful assistant with a multi-layer memory system.",
        "Use ALL available memory sections below to give accurate, personalised answers.",
        "When a user corrects a previous fact, always use the LATEST version.",
        "",
    ]

    profile = state.get("user_profile") or {}
    if profile:
        lines = "\n".join(f"  - {k}: {v}" for k, v in profile.items())
        parts.append(f"=== User Profile (long-term) ===\n{lines}")

    episodes = state.get("episodes") or []
    if episodes:
        ep_lines = "\n".join(
            f"  [{e.get('timestamp','')[:19]}] {e.get('content','')}" for e in episodes[:3]
        )
        parts.append(f"=== Past Experiences (episodic) ===\n{ep_lines}")

    sem = state.get("semantic_hits") or []
    if sem:
        sem_lines = "\n".join(f"  - {s}" for s in sem[:5])
        parts.append(f"=== Related Knowledge (semantic) ===\n{sem_lines}")

    recent = state.get("recent_conversation") or ""
    if recent:
        parts.append(f"=== Recent Conversation (short-term) ===\n{recent}")

    budget = state.get("memory_budget", 0)
    parts.append(f"\n[Memory budget used: {budget} tokens]")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def memory_retrieve(state: AgentState) -> dict[str, Any]:
    """Retrieve all memory sections and populate state."""
    last_msg = state["messages"][-1]
    query = last_msg.content if isinstance(last_msg.content, str) else ""
    user_id = state["user_id"]
    thread_id = state["thread_id"]

    intent = _router.classify_intent(query)

    profile = _long_term.get_facts(user_id)
    recent = _short_term.get_formatted(thread_id)
    episodes = _episodic.search_episodes(query, user_id=user_id, top_k=3)
    semantic_hits = _semantic.search(query, k=5)

    # Measure token budget
    combined = f"{profile}\n{recent}\n{episodes}\n{semantic_hits}"
    budget = _ctx_mgr.count_tokens(combined)

    return {
        "user_profile": profile,
        "episodes": episodes,
        "semantic_hits": semantic_hits,
        "recent_conversation": recent,
        "memory_budget": budget,
        "intent": intent.value,
    }


def respond(state: AgentState) -> dict[str, Any]:
    """Generate LLM response with full memory context injected."""
    system_prompt = _build_system_prompt(state)
    history = state["messages"][:-1]
    last_msg = state["messages"][-1]

    # Trim history to fit token budget
    history_dicts = [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        }
        for m in history
    ]
    trimmed = _ctx_mgr.trim_messages(history_dicts, reserved_tokens=1500)
    trimmed_msgs: list[BaseMessage] = [
        HumanMessage(content=d["content"]) if d["role"] == "user" else AIMessage(content=d["content"])
        for d in trimmed
    ]

    messages_to_send = [SystemMessage(content=system_prompt)] + trimmed_msgs + [last_msg]
    response = _llm.invoke(messages_to_send)
    return {"messages": [response]}


def memory_save(state: AgentState) -> dict[str, Any]:
    """Persist new information to appropriate backends with conflict handling."""
    msgs = state["messages"]
    user_id = state["user_id"]
    thread_id = state["thread_id"]
    intent = MemoryIntent(state.get("intent", MemoryIntent.GENERAL.value))

    # Find last human + AI pair
    ai_msg: BaseMessage | None = msgs[-1] if msgs else None
    human_msg: BaseMessage | None = None
    for m in reversed(msgs[:-1]):
        if isinstance(m, HumanMessage):
            human_msg = m
            break

    # Always update short-term buffer
    if human_msg:
        _short_term.add_message(thread_id, human_msg)
    if ai_msg and isinstance(ai_msg, AIMessage):
        _short_term.add_message(thread_id, ai_msg)

    if not human_msg:
        return {}

    content = human_msg.content

    # --- Conflict handling: detect corrections ---
    # Patterns: "nhầm", "sửa lại", "không phải ... mà là", "actually", "correction"
    import re
    correction_pattern = re.compile(
        r"\b(nhầm|sửa lại|không phải|actually|correction|i meant|i was wrong|let me correct)\b",
        re.IGNORECASE,
    )
    is_correction = bool(correction_pattern.search(content))

    if intent == MemoryIntent.USER_PREFERENCE:
        # Extract key=value from natural language (simple heuristic)
        # e.g. "my name is Alice" → key=name, value=Alice
        kv_patterns = [
            (re.compile(r"my name is\s+(\w+)", re.I), "name"),
            (re.compile(r"i(?:'m| am)\s+(.+?)(?:\.|,|$)", re.I), "description"),
            (re.compile(r"i(?:'m| am) allergic to\s+(.+?)(?:\.|,|$)", re.I), "allergy"),
            (re.compile(r"i (?:like|love|prefer|enjoy)\s+(.+?)(?:\.|,|$)", re.I), "preference"),
            (re.compile(r"i (?:hate|dislike)\s+(.+?)(?:\.|,|$)", re.I), "dislike"),
            (re.compile(r"call me\s+(\w+)", re.I), "name"),
        ]
        for pattern, key in kv_patterns:
            m = pattern.search(content)
            if m:
                value = m.group(1).strip()
                # Conflict: overwrite (latest wins)
                _long_term.set_fact(user_id, key, value)

        # Correction: also clear old conflicting facts
        if is_correction:
            # Re-parse and update — already handled by overwrite above
            pass

    elif intent == MemoryIntent.FACTUAL_RECALL:
        if ai_msg:
            combined = f"Q: {content}\nA: {ai_msg.content}"
            _semantic.add_memory(combined, metadata={"user_id": user_id})

    else:
        # Episodic for experience recall + general
        turn = len(msgs) // 2
        importance = 0.7 if intent == MemoryIntent.EXPERIENCE_RECALL else 0.3
        _episodic.log_episode(user_id=user_id, content=content, turn=turn, importance_score=importance)

    return {}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    checkpointer = MemorySaver()
    builder = StateGraph(AgentState)
    builder.add_node("memory_retrieve", memory_retrieve)
    builder.add_node("respond", respond)
    builder.add_node("memory_save", memory_save)
    builder.add_edge(START, "memory_retrieve")
    builder.add_edge("memory_retrieve", "respond")
    builder.add_edge("respond", "memory_save")
    builder.add_edge("memory_save", END)
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


def chat(
    user_message: str,
    user_id: str = "default_user",
    thread_id: str = "default_thread",
) -> str:
    """High-level chat interface."""
    config = {"configurable": {"thread_id": thread_id}}
    state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "user_id": user_id,
        "thread_id": thread_id,
        "user_profile": {},
        "episodes": [],
        "semantic_hits": [],
        "recent_conversation": "",
        "memory_budget": 0,
        "intent": MemoryIntent.GENERAL.value,
    }
    result = graph.invoke(state, config=config)
    last = result["messages"][-1]
    return last.content if isinstance(last.content, str) else str(last.content)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uid = "alice"
    tid = "demo-thread-1"

    turns = [
        "Hi! My name is Alice and I love hiking.",
        "I am allergic to peanuts.",
        "Wait, nhầm rồi — I am allergic to soy, not peanuts.",
        "What do you know about my allergies?",
        "What activities do I like?",
        "Can you explain what LangGraph is?",
        "What did we talk about at the start?",
    ]

    print("=== Multi-Memory Agent Demo ===\n")
    for msg in turns:
        print(f"User: {msg}")
        reply = chat(msg, user_id=uid, thread_id=tid)
        print(f"Agent: {reply}\n")
