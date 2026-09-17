from __future__ import annotations

import os

import chromadb
from langchain.memory import ConversationBufferWindowMemory
from sentence_transformers import SentenceTransformer


class AgentMemory:
    def __init__(self, window_k: int = 10) -> None:
        self.short_term = ConversationBufferWindowMemory(k=window_k, return_messages=True)
        self._chroma = chromadb.PersistentClient(path=os.environ.get("CHROMA_PATH", ".chroma"))
        self._embed = SentenceTransformer("all-MiniLM-L6-v2")
        self._collection = self._chroma.get_or_create_collection("long_term")

    def save_analysis(self, run_id: str, summary: str) -> None:
        embedding = self._embed.encode(summary).tolist()
        self._collection.upsert(
            ids=[run_id],
            embeddings=[embedding],
            documents=[summary],
        )

    def retrieve_similar(self, query: str, n: int = 5) -> list[str]:
        embedding = self._embed.encode(query).tolist()
        results = self._collection.query(query_embeddings=[embedding], n_results=n)
        return results["documents"][0] if results["documents"] else []
