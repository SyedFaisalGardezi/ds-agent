from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    task_type: str


class ToolRegistry:
    _instance: ToolRegistry | None = None

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    @classmethod
    def get(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, tool: Any) -> Any:
        self._tools[tool.name] = tool
        return tool

    def all(self) -> list[Any]:
        return list(self._tools.values())

    def __getitem__(self, name: str) -> Any:
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def register_tool(tool: Any) -> Any:
    return ToolRegistry.get().register(tool)


def tool(task_type: str = "default") -> Callable:
    def decorator(cls: type) -> type:
        cls.task_type = task_type  # type: ignore[attr-defined]
        instance = cls()
        register_tool(instance)
        return cls
    return decorator
