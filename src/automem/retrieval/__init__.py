"""
Memory Retrieval Layer.

Provides 5 orthogonal retrieval strategies that work with any storage backend:

  - SemanticRetriever:     Cosine similarity on embeddings
  - KeywordRetriever:      TF-IDF text matching
  - HybridRetriever:       Weighted combination of sub-retrievers
  - GraphRetriever:        Vector seed + graph neighbor expansion
  - ContrastiveRetriever:  Success/failure differentiated search

All retrievers return a unified MemoryPack output that can be formatted
into a prompt string via `pack.to_prompt_string()`.

Usage:
  from automem.retrieval import SemanticRetriever, QueryContext

  retriever = SemanticRetriever(store, embedding_model)
  pack = retriever.retrieve(QueryContext(query="How to parse PDF?"))
  prompt = pack.to_prompt_string()
"""

from .base_retriever import (
    BaseRetriever,
    QueryContext,
    ScoredUnit,
    TraceEntry,
    EvidenceRef,
    MemoryPack,
)
from .semantic_retriever import SemanticRetriever
from .keyword_retriever import KeywordRetriever
from .hybrid_retriever import HybridRetriever
from .graph_retriever import GraphRetriever
from .contrastive_retriever import ContrastiveRetriever
from .hybrid_graph_retriever import HybridGraphRetriever
from .cbr_retriever import CBRRetriever
from .cbr_rerank_retriever import CBRRerankRetriever
from .multi_store_retriever import MultiStoreRetriever
from .tag_retriever import TagRetriever
from .tag_vocabulary import TagVocabulary
from .query_classifier import QueryClassifier
# Stage-1 (2026-05-17) adoptions
from .hyde_retriever import HydeRetriever
from .mmr_retriever import MmrRetriever, mmr_select

__all__ = [
    # Base types
    "BaseRetriever",
    "QueryContext",
    "ScoredUnit",
    "TraceEntry",
    "EvidenceRef",
    "MemoryPack",
    # Retrievers
    "SemanticRetriever",
    "KeywordRetriever",
    "HybridRetriever",
    "GraphRetriever",
    "ContrastiveRetriever",
    "HybridGraphRetriever",
    "CBRRetriever",
    "CBRRerankRetriever",
    "MultiStoreRetriever",
    "TagRetriever",
    "TagVocabulary",
    "QueryClassifier",
    # Stage-1 (2026-05-17)
    "HydeRetriever",
    "MmrRetriever",
    "mmr_select",
]
