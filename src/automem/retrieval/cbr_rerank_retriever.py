"""CBRRerankRetriever — CBR + LLM rerank (Memento-inspired).

Approximates Memento's neural case-selection policy without RL training.

Pipeline:
  1. Run CBRRetriever to get top-K candidates (K=top_k * 3, e.g. 15).
  2. For each candidate, ask a small LLM (gate client) to score
     whether it applies to the current query (1-10 integer).
  3. Combine: final_score = 0.5 * cbr_score + 0.5 * llm_score/10
  4. Return top top_k.

The rerank model is an explicit constructor dependency. Missing configuration is
an initialization error instead of silently changing the architecture to CBR.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Any, Callable, Dict, Optional


from automem.retrieval.base_retriever import (
    BaseRetriever,
    MemoryPack,
    QueryContext,
    ScoredUnit,
)
from automem.retrieval.cbr_retriever import CBRRetriever
from automem.model_io import invoke_text_model

logger = logging.getLogger(__name__)

_RERANK_SYSTEM = (
    "You are a CASE RELEVANCE SCORER. Given a user question and a candidate "
    "memory from a past task, rate how LIKELY the memory is to HELP answer "
    "the question on a 1-10 integer scale.\n\n"
    "10 = same task class, memory directly applies, no contradiction\n"
    "7  = similar task class, some aspects relevant\n"
    "4  = loosely related, generic advice\n"
    "1  = unrelated or contradicts the question\n\n"
    "Output EXACTLY one integer 1-10. No explanation."
)


class CBRRerankRetriever(BaseRetriever):
    def __init__(
        self,
        store,
        embedding_model=None,
        config: Optional[Dict[str, Any]] = None,
        model: Any = None,
        model_resolver: Optional[Callable[[], Any]] = None,
        usage_in_task_metrics: bool = False,
        usage_in_task_metrics_resolver: Optional[Callable[[Any], bool]] = None,
    ):
        super().__init__(store, config)
        self.embedding_model = embedding_model
        self._cbr = CBRRetriever(store, embedding_model, config)
        self.rerank_pool_size = int(self.config.get("rerank_pool_size", 15))
        self.cbr_weight = float(self.config.get("cbr_weight", 0.5))
        self.llm_weight = float(self.config.get("llm_weight", 0.5))
        self._model = model
        self._model_resolver = model_resolver
        self._usage_in_task_metrics = bool(usage_in_task_metrics)
        self._usage_in_task_metrics_resolver = usage_in_task_metrics_resolver
        if self._model is None and self._model_resolver is None:
            raise ValueError(
                "cbr_rerank requires an injected callable or OpenAI-compatible model"
            )
        if self._model is None and self._model_resolver() is None:
            raise ValueError("cbr_rerank model resolver returned no model at initialization")
        self.rerank_model_id = str(
            self.config.get("rerank_model_id")
            or getattr(model, "model_id", "")
            or ""
        )
        # Codex Q7-A4 fix (2026-04-28): track rerank LLM token usage so
        # it can be billed into token_eff. With rerank_pool_size=15 we
        # spend ~600 input × 15 = 9k tokens per query — without metering
        # the architect treats cbr_rerank as nearly-free vs raw CBR.
        #
        # Codex Q11-A1 fix (2026-04-28): under shared_memory_provider
        # + concurrency > 1, the rerank counters must be PER-THREAD;
        # otherwise Task A's snapshot includes Task B's rerank
        # increments accumulated between A.reset and A.snapshot.
        # We keep a global lifetime counter (`rerank_stats`) for
        # debugging plus a thread-local per-task counter that the
        # provider snapshot reads exclusively.
        self.rerank_stats = {
            "rerank_calls": 0,
            "rerank_input_tokens": 0,
            "rerank_output_tokens": 0,
        }
        self._rerank_thread_local = threading.local()

    def _resolve_model(self) -> Any:
        model = self._model_resolver() if self._model_resolver is not None else self._model
        if model is None:
            raise RuntimeError("cbr_rerank model resolver returned no model")
        return model

    def _usage_is_in_task_metrics(self, model: Any) -> bool:
        if self._usage_in_task_metrics_resolver is not None:
            return bool(self._usage_in_task_metrics_resolver(model))
        return self._usage_in_task_metrics

    def reset_usage_metrics(self) -> None:
        tl = self._rerank_thread_local
        tl.rerank_calls = 0
        tl.rerank_input_tokens = 0
        tl.rerank_output_tokens = 0
        tl.rerank_usage_in_task_metrics = False

    def get_usage_metrics(self) -> Dict[str, Any]:
        tl = self._rerank_thread_local
        return {
            "rerank_calls": int(getattr(tl, "rerank_calls", 0) or 0),
            "rerank_input_tokens": int(
                getattr(tl, "rerank_input_tokens", 0) or 0
            ),
            "rerank_output_tokens": int(
                getattr(tl, "rerank_output_tokens", 0) or 0
            ),
            "rerank_usage_in_task_metrics": bool(
                getattr(tl, "rerank_usage_in_task_metrics", False)
            ),
        }

    def _score_candidate(self, query: str, unit) -> float:
        """Return LLM score in [0,1]."""
        mem_text = unit.content_text()[:600]
        user_prompt = f"Question:\n{query[:600]}\n\nCandidate memory:\n{mem_text}"
        client = self._resolve_model()
        usage_in_task_metrics = self._usage_is_in_task_metrics(client)
        self.rerank_stats["rerank_calls"] += 1
        tl = self._rerank_thread_local
        prior_calls = int(getattr(tl, "rerank_calls", 0) or 0)
        tl.rerank_calls = prior_calls + 1
        tl.rerank_usage_in_task_metrics = (
            usage_in_task_metrics
            if prior_calls == 0
            else bool(getattr(tl, "rerank_usage_in_task_metrics", False))
            and usage_in_task_metrics
        )
        try:
            result = invoke_text_model(
                client,
                model=self.rerank_model_id or getattr(client, "model_id", ""),
                system=_RERANK_SYSTEM,
                user=user_prompt,
                max_tokens=16,
                temperature=0.0,
            )
        except Exception as exc:
            raise RuntimeError("cbr_rerank model call failed") from exc

        self.rerank_stats["rerank_input_tokens"] += result.input_tokens
        self.rerank_stats["rerank_output_tokens"] += result.output_tokens
        tl.rerank_input_tokens = (
            getattr(tl, "rerank_input_tokens", 0) + result.input_tokens
        )
        tl.rerank_output_tokens = (
            getattr(tl, "rerank_output_tokens", 0) + result.output_tokens
        )
        match = re.fullmatch(r"\s*(10|[1-9])\s*", result.text)
        if match is None:
            raise RuntimeError("cbr_rerank model must return exactly one integer from 1 to 10")
        return int(match.group(1)) / 10.0

    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        # Stage 1: get a larger candidate pool from CBR
        pool_size = max(self.rerank_pool_size, top_k)
        pack = self._cbr.retrieve(ctx, top_k=pool_size)
        if not pack.scored_units:
            return pack

        # Stage 2: LLM rerank. Construction fails when no model is configured.
        # FIX 2026-06-29: the LLM blend (cbr_weight*cosine + llm_weight*llm/10) is a
        # RERANK key only — it must not become the unit's final score, which is the
        # absolute CBR cosine the Tier-1 gate compares against the 0.40 floor. The
        # blend mixes a 0..1 cosine with a 1..10 LLM judgment (0.5 on parse-fail),
        # distorting the cosine scale so a relevant unit the LLM under-scored got
        # dropped by the gate. Blend ranks; raw CBR cosine stays the gate score.
        rescored = []
        for su in pack.scored_units:
            llm_score = self._score_candidate(ctx.query, su.unit)
            rank = self.cbr_weight * su.score + self.llm_weight * llm_score
            rescored.append((rank, ScoredUnit(unit=su.unit, score=su.score)))
        rescored.sort(key=lambda t: t[0], reverse=True)
        pack.scored_units = [su for _, su in rescored[:top_k]]
        pack.retriever_name = "cbr_rerank"
        logger.info(f"[cbr_rerank] reranked {pool_size} → {len(pack.scored_units)} units")
        return pack
