"""HyDE Retriever — Hypothetical Document Embeddings (Gao et al. 2022).

Stage 1 adoption (2026-05-17). Adapted from github.com/texttron/hyde
(MIT). We borrow the prompt-then-average pattern but rewrite the integration
to live inside our BaseRetriever interface and reuse the project's existing
embedding model (via automem.search.attribution._get_embedder) and storage
backend.

Core algorithm (verbatim from the paper):
  1. Build a task-specific "hypothesis" prompt: "Write a passage to answer
     the question: {query}".
  2. Call the LLM N times to generate hypothesis passages.
  3. Encode each hypothesis + the original query into the same embedding space.
  4. Average all (N+1) embeddings into a single ``hyde_vector``.
  5. Run dense retrieval against the pool using ``hyde_vector`` as the query.

This module ONLY implements step 1-4 and delegates step 5 to the store's
native search (FAISS for VectorStorage / HybridStorage). For stores without a
.search() method (JsonStorage, GraphStore), falls back to a numpy cosine path
identical to SemanticRetriever's fallback.

Cost: +1 LLM call per query (or +N for multi-sample, default N=1). Caller
is responsible for the LLM model (passed as ``hypothesis_model``).
"""

from typing import Any, Dict, List, Optional

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from automem.retrieval.base_retriever import (
    BaseRetriever,
    MemoryPack,
    QueryContext,
    ScoredUnit,
    TraceEntry,
)

logger = logging.getLogger(__name__)


# ── Prompt (adapted from hyde/src/hyde/promptor.py:WEB_SEARCH) ──────────
#
# The original repo has 8 task-specific prompts (SciFact, ArguAna, FIQA, ...).
# For GAIA-style fact-lookup / multi-hop reasoning, the WEB_SEARCH prompt
# generalizes well. We rewrite it slightly to encourage richer factual content
# and prevent overly-short hypotheses.

_HYDE_PROMPT = """Please write a passage of 2-4 sentences that would directly answer the question below.
Be as specific and factual as you can. The passage should look like an excerpt from a Wikipedia article, a research paper, or a Wikipedia talk page — something an information retrieval system would surface as evidence.

Question: {query}

Passage:"""


class HydeRetriever(BaseRetriever):
    """Hypothetical Document Embeddings retriever.

    Config options:
        hypothesis_model (Any):   LLM used to generate the hypothesis passage.
                                  If None, falls back to no-HyDE (semantic only).
        n_hypotheses (int):       How many hypothesis passages to sample (default 1).
                                  Higher N averages over more samples but costs more.
        min_score (float):        Minimum cosine similarity threshold (default 0.0).
        active_only (bool):       Only consider active units (default True).
        embedding_model:          Encoder for query + hypotheses (sentence-transformers).
                                  When None, falls back to the project-wide embedder.

    The retrieval API matches the rest of automem.retrieval: ``retrieve(ctx, top_k)``
    returns a ``MemoryPack`` with `scored_units`, `trace`, etc.
    """

    def __init__(
        self,
        store,
        embedding_model=None,
        hypothesis_model=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(store, config)
        self.embedding_model = embedding_model
        self.hypothesis_model = hypothesis_model
        self.n_hypotheses = int(self.config.get("n_hypotheses", 1))
        self.min_score = self.config.get("min_score", 0.0)
        self.active_only = self.config.get("active_only", True)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    def _encode(self, text: str) -> Optional[np.ndarray]:
        """Encode ``text`` with the configured embedding_model, or fall back
        to the project-wide embedder."""
        if self.embedding_model is not None:
            try:
                return self.embedding_model.encode(text, convert_to_numpy=True)
            except Exception as e:
                logger.warning("[HyDE] embedding_model.encode failed: %s; falling back", e)

        # Fall back to attribution.py's lazy-loaded embedder
        try:
            from automem.search.attribution import _get_embedder
            emb_model = _get_embedder()
            if emb_model is None:
                return None
            return emb_model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        except Exception as e:
            logger.warning("[HyDE] fallback embedder failed: %s", e)
            return None

    def _generate_hypotheses(self, query: str) -> List[str]:
        """Generate N hypothesis passages from the query via LLM."""
        if self.hypothesis_model is None:
            return []

        from automem.llm_utils import call_llm_text

        hypotheses: List[str] = []
        prompt_template = _HYDE_PROMPT
        for _ in range(self.n_hypotheses):
            try:
                text = call_llm_text(
                    self.hypothesis_model,
                    prompt_template,
                    {"query": query},
                    max_retries=2,
                )
                text = (text or "").strip()
                if text:
                    hypotheses.append(text)
            except Exception as e:
                logger.warning("[HyDE] hypothesis generation failed: %s", e)
                # Try once more without retry; if still failing, give up on this sample
                continue
        return hypotheses

    def _build_hyde_vector(
        self, query: str, hypotheses: List[str],
    ) -> Optional[np.ndarray]:
        """Encode query + all hypotheses, then average."""
        all_texts = [query] + hypotheses
        all_embs: List[np.ndarray] = []
        for t in all_texts:
            emb = self._encode(t)
            if emb is not None:
                all_embs.append(emb)
        if not all_embs:
            return None
        return np.mean(np.stack(all_embs), axis=0)

    # ──────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────
    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        # Step 1-2: generate hypothesis passages
        hypotheses = self._generate_hypotheses(ctx.query)
        n_hyp = len(hypotheses)

        # Step 3-4: build averaged hyde vector
        hyde_vec = self._build_hyde_vector(ctx.query, hypotheses)
        if hyde_vec is None:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="hyde", candidates=0, selected=0,
                params={"error": "no_embedding_or_no_hypothesis",
                        "n_hypotheses": n_hyp},
            )])

        # Step 5: dense retrieval against the pool, using hyde_vec as query
        # Fast path: store has a native FAISS .search()
        if hasattr(self.store, "search"):
            try:
                results = self.store.search(
                    hyde_vec, top_k=top_k, active_only=self.active_only,
                )
                scored = [
                    ScoredUnit(unit=u, score=s, method="hyde")
                    for u, s in results
                    if s >= self.min_score
                ]
                trace = [TraceEntry(
                    step=1, method="hyde_faiss",
                    candidates=self.store.count() if hasattr(self.store, "count") else len(results),
                    selected=len(scored),
                    params={"n_hypotheses": n_hyp, "top_k": top_k},
                )]
                return self._make_pack(ctx, scored, trace)
            except Exception:
                pass  # fall through to numpy path

        # Fallback: numpy cosine
        emb_matrix, units = self.store.get_embedding_index(
            active_only=self.active_only,
        )
        if emb_matrix is None or len(units) == 0:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="hyde", candidates=0, selected=0,
                params={"n_hypotheses": n_hyp},
            )])
        sims = cosine_similarity(hyde_vec.reshape(1, -1), emb_matrix)[0]
        top_k_actual = min(top_k, len(units))
        # Guard: [-0:] returns the WHOLE pool (numpy slice semantics).
        top_idx = sims.argsort()[-top_k_actual:][::-1] if top_k_actual > 0 else []
        scored = []
        for idx in top_idx:
            score = float(sims[idx])
            if score < self.min_score:
                continue
            scored.append(ScoredUnit(
                unit=units[idx], score=score, method="hyde",
            ))
        trace = [TraceEntry(
            step=1, method="hyde_numpy",
            candidates=len(units), selected=len(scored),
            params={"n_hypotheses": n_hyp, "top_k": top_k},
        )]
        return self._make_pack(ctx, scored, trace)


__all__ = ["HydeRetriever"]
