"""Observation Graph — a lightweight structural experience map.

The observation graph is a rule-based artifact (no LLM in its update path)
that captures, across evolution rounds:
  - which task patterns (by GAIA level) have been seen,
  - how each extraction-type / retriever combination performed on them,
  - which memory units carried the most attribution.

It is consumed ONLY as extra context for the architecture Proposer (when a
candidate is proposed in "observation-aware" mode). It does NOT change any
runtime behaviour of extraction / retrieval / storage / management.

See docs (design discussion) for the full rationale.
"""

from .graph import ObservationGraph, ObsNode, ObsEdge

__all__ = ["ObservationGraph", "ObsNode", "ObsEdge"]
