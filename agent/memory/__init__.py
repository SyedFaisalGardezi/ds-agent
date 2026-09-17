"""ds-agent memory subsystem.

Layers:
  1. PersistentSessionStore — SQLite session + message persistence (Layer 1)
  2. LongTermMemory         — ChromaDB semantic recall across sessions (Layer 2)
  3. SchemaCache            — SQLite dataset schema cache by file hash (Layer 3)
  4. build_context_messages — Smart context window with summary buffer (Layer 4)
  5. isolation              — Memory isolation mode detection (Layer 5)

All components degrade gracefully when their dependencies are unavailable.
"""
from agent.memory.isolation import apply_isolation_intent, detect_isolation_intent
from agent.memory.long_term import LongTermMemory
from agent.memory.schema_cache import SchemaCache
from agent.memory.session_memory import build_context_messages
from agent.memory.session_store import PersistentSessionStore

__all__ = [
    "LongTermMemory",
    "SchemaCache",
    "PersistentSessionStore",
    "build_context_messages",
    "detect_isolation_intent",
    "apply_isolation_intent",
]
