from __future__ import annotations

import os

import chromadb
from sentence_transformers import SentenceTransformer

from agent.connectors.base import DataConnector


class ConnectorRegistry:
    _instance: ConnectorRegistry | None = None

    def __init__(self) -> None:
        self._connectors: dict[str, DataConnector] = {}
        self._embed = SentenceTransformer("all-MiniLM-L6-v2")
        self._chroma = chromadb.PersistentClient(path=os.environ.get("CHROMA_PATH", ".chroma"))
        self._schema_col = self._chroma.get_or_create_collection("schema_cache")

    @classmethod
    def get(cls) -> ConnectorRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, connector: DataConnector) -> None:
        self._connectors[connector.name] = connector

    def index_all(self) -> None:
        for name, conn in self._connectors.items():
            schemas = conn.list_schemas()
            if schemas.status != "ok":
                continue
            for schema in schemas.data or []:
                desc = conn.describe_table(schema)
                if desc.status == "ok":
                    text = f"{name}.{schema}: {desc.data}"
                    embedding = self._embed.encode(text).tolist()
                    self._schema_col.upsert(
                        ids=[f"{name}.{schema}"],
                        embeddings=[embedding],
                        documents=[text],
                    )

    def search_schema(self, query: str, n: int = 5) -> list[str]:
        embedding = self._embed.encode(query).tolist()
        results = self._schema_col.query(query_embeddings=[embedding], n_results=n)
        return results["documents"][0] if results["documents"] else []

    def __getitem__(self, name: str) -> DataConnector:
        return self._connectors[name]
