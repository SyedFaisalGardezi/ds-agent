"""Long-term semantic memory using ChromaDB + nomic-embed-text (via Ollama).

Stores tool outputs, synthesis replies, TASK_SPECs, and model results as
vector embeddings for cross-session semantic retrieval.

Graceful degradation: if Ollama embedding endpoint is unavailable (model not
pulled or Ollama down), all operations silently no-op so the rest of the agent
is unaffected. Pull the model once to enable LTM:
    ollama pull nomic-embed-text
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import httpx

CHROMA_PATH   = Path(os.environ.get("DSAGENT_MEMORY_DIR",
                                    str(Path.home() / ".ds-agent" / "memory"))) / "chroma"
OLLAMA_HOST   = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL   = os.environ.get("DSAGENT_EMBED_MODEL", "nomic-embed-text")
COLLECTION    = "agent_memory"
LTM_DISTANCE  = float(os.environ.get("DSAGENT_LTM_DISTANCE", "0.55"))
LTM_HITS      = int(os.environ.get("DSAGENT_LTM_HITS", "4"))


def _embed(text: str) -> list[float] | None:
    """Embed via Ollama nomic-embed-text. Returns None if unavailable."""
    try:
        r = httpx.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:4096]},
            timeout=30.0,
        )
        r.raise_for_status()
        emb = r.json().get("embedding")
        return emb if isinstance(emb, list) and emb else None
    except Exception:
        return None


class LongTermMemory:
    """Persistent vector store for cross-session semantic recall.

    Each document carries session_id metadata so isolation_mode can restrict
    retrieval to the current session only.
    """

    def __init__(self) -> None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self._col = self._client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        session_id: str,
        doc_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Embed and persist a memory chunk. Silently no-ops if LTM unavailable.

        doc_type: "tool_output" | "synthesis" | "task_spec" | "model_result" | "schema"
        """
        emb = _embed(content)
        if emb is None:
            return  # embedding unavailable — graceful skip

        doc_id = hashlib.sha256(
            f"{session_id}_{doc_type}_{content[:64]}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:24]

        meta: dict[str, Any] = {
            "session_id": session_id,
            "doc_type":   doc_type,
            "ts":         datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        try:
            self._col.add(
                ids=[doc_id],
                embeddings=[emb],
                documents=[content],
                metadatas=[meta],
            )
        except Exception:
            pass

    def store_tool_output(self, session_id: str, tool_name: str, output: str) -> None:
        self.store(
            content=f"[{tool_name}]\n{output[:2000]}",
            session_id=session_id,
            doc_type="tool_output",
            metadata={"tool": tool_name},
        )

    def store_model_result(
        self,
        session_id: str,
        model: str,
        metric: str,
        score: float,
        dataset: str = "",
    ) -> None:
        content = (
            f"Model: {model} | Metric: {metric} | Score: {score:.4f}"
            + (f" | Dataset: {dataset}" if dataset else "")
        )
        self.store(
            content=content,
            session_id=session_id,
            doc_type="model_result",
            metadata={"model": model, "metric": metric, "score": str(score)},
        )

    def store_task_spec(self, session_id: str, task_spec: dict[str, Any]) -> None:
        self.store(
            content=f"TASK_SPEC:\n{json.dumps(task_spec, indent=2)}",
            session_id=session_id,
            doc_type="task_spec",
        )

    def store_synthesis(self, session_id: str, text: str) -> None:
        self.store(content=text, session_id=session_id, doc_type="synthesis")

    # ── Read ──────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        n: int = LTM_HITS,
        session_id: str | None = None,
        isolation_mode: bool = False,
        doc_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top-n most relevant chunks. Returns [] if LTM unavailable."""
        emb = _embed(query)
        if emb is None:
            return []

        total = self._col.count()
        if total == 0:
            return []

        where: dict[str, Any] = {}
        if isolation_mode and session_id:
            where["session_id"] = {"$eq": session_id}
        if doc_types:
            if len(doc_types) == 1:
                where["doc_type"] = {"$eq": doc_types[0]}
            else:
                where["$or"] = [{"doc_type": {"$eq": t}} for t in doc_types]

        kwargs: dict[str, Any] = {
            "query_embeddings": [emb],
            "n_results": min(n, total),
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._col.query(**kwargs)
        except Exception:
            return []

        if not results.get("documents"):
            return []

        return [
            {"content": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def retrieve_as_context_string(
        self,
        query: str,
        n: int = LTM_HITS,
        session_id: str | None = None,
        isolation_mode: bool = False,
        distance_threshold: float = LTM_DISTANCE,
    ) -> str:
        """Formatted string of relevant memories, ready for LLM system prompt injection."""
        hits = self.retrieve(
            query=query,
            n=n,
            session_id=session_id,
            isolation_mode=isolation_mode,
        )
        hits = [h for h in hits if h["distance"] <= distance_threshold]
        if not hits:
            return ""

        parts = []
        for h in hits:
            meta = h["metadata"]
            label = (
                f"[{meta.get('doc_type', 'memory')} | "
                f"session {meta.get('session_id', '?')[:8]}]"
            )
            parts.append(f"{label}\n{h['content']}")

        return "--- Relevant past context ---\n" + "\n\n".join(parts) + "\n---"

    def count(self) -> int:
        return self._col.count()

    def delete_session_memories(self, session_id: str) -> None:
        try:
            self._col.delete(where={"session_id": {"$eq": session_id}})
        except Exception:
            pass

    def prune_old_memories(self, days: int = 90) -> None:
        """Remove memories older than `days` days."""
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        try:
            self._col.delete(where={"ts": {"$lt": cutoff}})
        except Exception:
            pass
