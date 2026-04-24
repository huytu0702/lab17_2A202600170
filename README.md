# Lab #17 — Multi-Memory Agent với LangGraph

**Sinh viên:** Nguyễn Huy Tú — MSV: 2A202600170

---

## Mô tả

Agent hội thoại đa tầng nhớ, xây dựng trên LangGraph. Agent tự động phân loại ý định của người dùng và lưu/truy xuất từ đúng backend tương ứng.

## Kiến trúc

```
User Query
    │
    ▼
MemoryRouter (intent classifier)
    ├── user_preference  → Redis (long-term profile)
    ├── factual_recall   → Chroma (semantic search)
    ├── experience_recall→ Episodic JSON log
    └── general          → MemorySaver (short-term buffer)
    │
    ▼
LangGraph StateGraph
  START → memory_retrieve → respond → memory_save → END
    │
    ▼
ContextManager (tiktoken, 4-level priority eviction)
```

## 4 Memory Backends

| Backend | Loại | Persistence |
|---------|------|-------------|
| `MemorySaver` (LangGraph) | Short-term buffer | In-process, per thread |
| Redis | Long-term profile | Cross-session, JSON hash |
| `episodic_log.json` | Episodic log | Append-only JSON file |
| Chroma + `text-embedding-3-small` | Semantic search | Persistent vector store |

## Conflict Handling

Khi user sửa thông tin cũ (ví dụ: dị ứng), profile tự động overwrite — "latest wins":

```
User: I am allergic to peanuts.   → Redis: allergy = peanuts
User: Nhầm — allergic to soy.     → Redis: allergy = soy   ✅
```

## Benchmark (thực tế)

10 multi-turn conversations × 5 turns × 2 agents (with/without memory):

| Metric | With Memory | No Memory |
|--------|-------------|-----------|
| Hit rate | **92%** | 90% |
| Total tokens | **11,802** | 17,142 |
| Token savings | **−31%** | — |

→ Chi tiết: [`BENCHMARK.md`](BENCHMARK.md) · [`benchmark_report.md`](benchmark_report.md)  
→ Privacy & limitations: [`REFLECTION.md`](REFLECTION.md)

## Cài đặt & Chạy

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
cp .env  # điền OPENAI_API_KEY

# Demo agent
python src/agent.py

# Chạy benchmark
python src/benchmark.py
```

> Redis optional — tự động fallback sang in-memory dict nếu không có Redis.

## Tài liệu tham khảo

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangChain Docs](https://python.langchain.com/docs/introduction/)
- [Chroma Docs](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Redis Docs](https://redis.io/docs/)

## Cấu trúc thư mục

```
├── src/
│   ├── memory/
│   │   ├── short_term.py   # MemorySaver wrapper
│   │   ├── long_term.py    # Redis backend
│   │   ├── episodic.py     # JSON episodic log
│   │   ├── semantic.py     # Chroma + embeddings
│   │   └── router.py       # Intent classifier
│   ├── context/
│   │   └── manager.py      # Token budget & trim
│   ├── agent.py            # LangGraph StateGraph
│   └── benchmark.py        # 10-scenario benchmark
├── BENCHMARK.md
├── REFLECTION.md
├── benchmark_report.md
└── requirements.txt
```
