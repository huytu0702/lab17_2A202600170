"""Semantic memory backend using Chroma + OpenAI embeddings."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_DEFAULT_CHROMA_DIR = str(Path(__file__).parent.parent.parent / "chroma_db")
_COLLECTION_NAME = "conversation_memories"


class SemanticMemory:
    """Stores and retrieves memories via vector similarity search.

    Uses ChromaDB (persistent) with OpenAI text-embedding-3-small.
    """

    def __init__(self, chroma_dir: str | None = None) -> None:
        self._dir = chroma_dir or _DEFAULT_CHROMA_DIR
        self._client: Any = None
        self._collection: Any = None
        self._vectorstore: Any = None
        self._ready = False
        try:
            self._init()
            self._ready = True
        except Exception as exc:
            print(f"[SemanticMemory] init failed ({exc}), semantic search disabled.")

    def _init(self) -> None:
        import chromadb  # type: ignore
        from langchain_chroma import Chroma  # type: ignore
        from langchain_openai import OpenAIEmbeddings  # type: ignore

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        Path(self._dir).mkdir(parents=True, exist_ok=True)
        self._vectorstore = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=self._dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Embed and store *text* with optional *metadata*."""
        if not self._ready:
            return
        from langchain_core.documents import Document  # type: ignore

        doc = Document(page_content=text, metadata=metadata or {})
        self._vectorstore.add_documents([doc])

    def search(self, query: str, k: int = 5) -> list[str]:
        """Return top-*k* relevant memory texts for *query*."""
        if not self._ready:
            return []
        try:
            docs = self._vectorstore.similarity_search(query, k=k)
            return [d.page_content for d in docs]
        except Exception:
            return []

    def get_formatted(self, query: str, k: int = 5) -> str:
        results = self.search(query, k=k)
        if not results:
            return ""
        return "\n".join(f"- {r}" for r in results)
