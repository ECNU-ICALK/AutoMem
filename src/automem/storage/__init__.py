"""
Memory Storage Backends.

Provides orthogonal, pluggable storage implementations for MemoryUnit persistence.
Decoupled from extraction, retrieval, and management layers.

Available backends:
  - JsonStorage:    Pure JSON file storage
  - VectorStorage:  FAISS-indexed vector storage with JSON metadata sidecar
  - HybridStorage:  JSON metadata + FAISS vector index combined
  - GraphStore:     Three-layer graph (Query/Content/Entity) with networkx

Usage:
  from automem.storage import JsonStorage, VectorStorage, HybridStorage, GraphStore

  store = GraphStore({"storage_dir": "./storage/my_graph"})
  store.initialize()
  store.upsert_memory_unit(unit, entities=[{"name": "web_search", "type": "tool"}])
"""

from .base_storage import BaseMemoryStorage
from .json_storage import JsonStorage
from .vector_storage import VectorStorage
from .hybrid_storage import HybridStorage
from .graph_storage import GraphStore
from .llm_graph_storage import LLMGraphStore

__all__ = [
    "BaseMemoryStorage",
    "JsonStorage",
    "VectorStorage",
    "HybridStorage",
    "GraphStore",
    "LLMGraphStore",
]
