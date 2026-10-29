from __future__ import annotations

from abc import ABC, abstractmethod

from agent.core.types import ToolResult


class DataConnector(ABC):
    name: str = ""

    @abstractmethod
    def connect(self) -> ToolResult: ...

    @abstractmethod
    def list_schemas(self) -> ToolResult: ...

    @abstractmethod
    def describe_table(self, table: str) -> ToolResult: ...

    @abstractmethod
    def query(self, query: str, params: dict | None = None) -> ToolResult: ...

    @abstractmethod
    def estimate_cost(self, query: str) -> ToolResult: ...

    @abstractmethod
    def detect_pii(self, table: str) -> ToolResult: ...
