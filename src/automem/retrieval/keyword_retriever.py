"""
Keyword Retriever — TF-IDF text matching.

Retrieves memories by computing TF-IDF vectors from unit content text
and ranking by cosine similarity against the query. No embedding model
required — purely lexical matching.

Useful as a complement to semantic search: catches exact term matches
that embedding models may miss (e.g. tool names, error codes).
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from automem.retrieval.base_retriever import (
    BaseRetriever,
    MemoryPack,
    QueryContext,
    ScoredUnit,
    TraceEntry,
)


_LATIN_TOKEN_RE = re.compile(r"[a-z0-9_]+")
# CJK unified ideographs (+ extension A). Chinese has no whitespace word
# boundaries, so index each han character plus adjacent bigrams. Under the
# old latin-only regex, 27/100 official xBench queries tokenized to ZERO
# tokens and the whole lexical arm (keyword sub-retriever of hybrid, and the
# failure_bank fallback) returned nothing on Chinese benchmarks.
_CJK_RUN_RE = re.compile(r"[㐀-䶿一-鿿]+")


def _tokenize(text: str) -> List[str]:
    """Lowercased latin/numeric tokens plus CJK unigrams and bigrams."""
    text = text.lower()
    tokens = _LATIN_TOKEN_RE.findall(text)
    for run in _CJK_RUN_RE.findall(text):
        tokens.extend(run)
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class KeywordRetriever(BaseRetriever):
    """
    Retrieves memories by TF-IDF cosine similarity on content text.

    Config options:
        min_score (float): Minimum TF-IDF similarity. Default 0.01.
        active_only (bool): Only consider active units. Default True.
    """

    def __init__(self, store, config: Optional[Dict[str, Any]] = None):
        super().__init__(store, config)
        self.min_score = self.config.get("min_score", 0.01)
        self.active_only = self.config.get("active_only", True)

    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        units = self.store.get_all(active_only=self.active_only)
        if not units:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="keyword", candidates=0, selected=0,
            )])

        # Build corpus
        docs = []
        for u in units:
            docs.append(_tokenize(u.content_text()))

        query_tokens = _tokenize(ctx.query)
        if not query_tokens:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="keyword", candidates=len(units), selected=0,
                params={"error": "empty_query_tokens"},
            )])

        # Compute IDF
        n_docs = len(docs)
        all_tokens = set()
        for d in docs:
            all_tokens.update(d)
        all_tokens.update(query_tokens)

        doc_freq: Dict[str, int] = Counter()
        for d in docs:
            for token in set(d):
                doc_freq[token] += 1

        idf = {}
        for token in all_tokens:
            df = doc_freq.get(token, 0)
            idf[token] = math.log((n_docs + 1) / (df + 1)) + 1.0

        # TF-IDF vectors and cosine similarity
        def _tfidf_vec(tokens: List[str]) -> Dict[str, float]:
            tf = Counter(tokens)
            vec = {}
            for t, count in tf.items():
                if t in idf:
                    vec[t] = (1 + math.log(count)) * idf[t]
            return vec

        def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
            common = set(v1.keys()) & set(v2.keys())
            if not common:
                return 0.0
            dot = sum(v1[k] * v2[k] for k in common)
            norm1 = math.sqrt(sum(x * x for x in v1.values()))
            norm2 = math.sqrt(sum(x * x for x in v2.values()))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)

        query_vec = _tfidf_vec(query_tokens)

        scored_pairs = []
        for i, doc_tokens in enumerate(docs):
            doc_vec = _tfidf_vec(doc_tokens)
            sim = _cosine(query_vec, doc_vec)
            if sim >= self.min_score:
                scored_pairs.append((i, sim))

        # Sort by similarity descending
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        scored_pairs = scored_pairs[:top_k]

        scored = [
            ScoredUnit(unit=units[i], score=sim, method="keyword")
            for i, sim in scored_pairs
        ]

        trace = [TraceEntry(
            step=1,
            method="keyword",
            candidates=len(units),
            selected=len(scored),
            params={"min_score": self.min_score, "top_k": top_k,
                    "query_tokens": len(query_tokens)},
        )]


        return self._make_pack(ctx, scored, trace)
