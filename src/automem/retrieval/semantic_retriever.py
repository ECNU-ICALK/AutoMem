"""
Semantic Retriever — Cosine similarity on embeddings.

The simplest and most fundamental retrieval strategy. Encodes the query
into the same embedding space as stored MemoryUnits, then ranks by
cosine similarity.

Requires:
  - An embedding model (sentence-transformers) for query encoding
  - MemoryUnits with pre-computed embeddings in the storage backend
"""

from typing import Any, Dict, Optional

from sklearn.metrics.pairwise import cosine_similarity

from automem.retrieval.base_retriever import (
    BaseRetriever,
    MemoryPack,
    QueryContext,
    ScoredUnit,
    TraceEntry,
)


class SemanticRetriever(BaseRetriever):
    """
    Retrieves memories by cosine similarity between query embedding
    and stored unit embeddings.

    Config options:
        min_score (float): Minimum similarity threshold. Default 0.0.
        active_only (bool): Only consider active units. Default True.
    """

    def __init__(self, store, embedding_model=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(store, config)
        self.embedding_model = embedding_model
        self.min_score = self.config.get("min_score", 0.0)
        self.active_only = self.config.get("active_only", True)

    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        # Get query embedding
        query_emb = ctx.embedding
        if query_emb is None and self.embedding_model is not None:
            query_emb = self.embedding_model.encode(
                ctx.query, convert_to_numpy=True
            )
        if query_emb is None:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="semantic", candidates=0, selected=0,
                params={"error": "no_embedding"},
            )])

        query_emb = query_emb.reshape(1, -1)

        # Fast path: use the store's native FAISS search when available
        # (VectorStorage and HybridStorage implement .search() backed by FAISS,
        # which is O(log N) vs the O(N) numpy path below).
        if hasattr(self.store, 'search'):
            try:
                results = self.store.search(
                    query_emb.squeeze(0),
                    top_k=top_k,
                    active_only=self.active_only,
                )
                scored = [
                    ScoredUnit(unit=u, score=s, method="semantic")
                    for u, s in results
                    if s >= self.min_score
                ]
                trace = [TraceEntry(
                    step=1,
                    method="semantic_faiss",
                    candidates=self.store.count(),
                    selected=len(scored),
                    params={"min_score": self.min_score, "top_k": top_k,
                            "backend": "faiss"},
                )]
                return self._make_pack(ctx, scored, trace)
            except Exception:
                # FAISS path failed — fall through to numpy path
                pass

        # Fallback: full embedding matrix + numpy cosine similarity
        # Used for JsonStorage and GraphStore which don't implement .search().
        emb_matrix, units = self.store.get_embedding_index(
            active_only=self.active_only
        )
        if emb_matrix is None or len(units) == 0:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="semantic", candidates=0, selected=0,
            )])

        # Compute cosine similarity
        sims = cosine_similarity(query_emb, emb_matrix)[0]

        # Rank and select top-k above threshold
        top_k_actual = min(top_k, len(units))
        if top_k_actual <= 0:
            # Guard: [-0:] returns the WHOLE pool (numpy slice semantics),
            # silently inverting top_k=0 into inject-everything.
            top_indices = []
        else:
            top_indices = sims.argsort()[-top_k_actual:][::-1]

        scored = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < self.min_score:
                continue
            scored.append(ScoredUnit(
                unit=units[idx],
                score=score,
                method="semantic",
            ))

        trace = [TraceEntry(
            step=1,
            method="semantic",
            candidates=len(units),
            selected=len(scored),
            params={"min_score": self.min_score, "top_k": top_k},
        )]


        return self._make_pack(ctx, scored, trace)
