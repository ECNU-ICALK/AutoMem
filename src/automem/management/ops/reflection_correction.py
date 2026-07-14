"""
ReflectionCorrectionOp — Use LLM reflection to analyze and correct memories
that contributed to a failed task.

Part of the 'failure_adjustment' operation group.
"""

import json
import re
import time
import uuid
import logging
from typing import Any, Dict, List, Optional


from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit, MemoryRelation, RelationType

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> Optional[Any]:
    """Parse JSON from raw LLM output, handling code-block wrappers."""
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try extracting from code block
    match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return None


class ReflectionCorrectionOp(BaseManageOp):
    """
    After a failed task, use an LLM to reflect on the memories that were
    used and determine whether each should be updated, replaced, or kept.

    For UPDATE actions the existing unit's content is modified in-place.
    For REPLACE actions a new unit is created with a SUPERSEDES relation
    to the old one, and the old unit is deactivated.
    """

    op_name = "reflection_correction"
    op_group = "failure_adjustment"
    trigger_type = TriggerType.POST_TASK
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = True
    requires_embedding = True
    rl_action_id = 10

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name)

        try:
            # Only trigger on task failure
            if context.get("task_succeeded", True):
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            result.triggered = True

            used_unit_ids: List[str] = context.get("used_unit_ids", [])
            task_query = context.get("task_query", "")
            failure_context = context.get("failure_context", "")

            # Fetch the used units
            used_units: List[MemoryUnit] = []
            for uid in used_unit_ids:
                unit = self.store.get(uid)
                if unit is not None:
                    used_units.append(unit)

            if not used_units:
                logger.info("reflection_correction: no used units found, skipping")
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            # Build LLM prompt
            memory_descriptions = []
            for i, u in enumerate(used_units):
                memory_descriptions.append(
                    f"Memory {i+1} (id={u.id}, type={u.type.value}, "
                    f"keys={list(u.content.keys())}):\n"
                    f"  Content: {u.content_text()[:500]}"
                )

            prompt = (
                "You are a memory management assistant. A task has FAILED and the "
                "following memories were used during execution. Analyze whether each "
                "memory contributed to the failure and suggest corrections.\n\n"
                f"Task query: {task_query}\n"
                f"Failure context: {failure_context}\n\n"
                "Memories used:\n"
                + "\n".join(memory_descriptions)
                + "\n\n"
                "For each memory, respond with one of:\n"
                '- "UPDATE": the memory has useful info but needs correction. '
                "Provide corrected values using ONLY the existing content keys shown above.\n"
                '- "REPLACE": the memory is fundamentally wrong and should be '
                "replaced. Use the SAME keys as the original content.\n"
                '- "KEEP": the memory is fine and did not cause the failure.\n\n'
                "Respond with a JSON array where each element has:\n"
                '  {"id": "<memory_id>", "action": "UPDATE|REPLACE|KEEP", '
                '"new_content": {...} (only for UPDATE/REPLACE, use existing keys only)}\n'
                "Output ONLY the JSON array."
            )

            # LLM call
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            response = self.llm_client(messages)
            response_text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

            # Parse LLM response
            suggestions = _parse_json(response_text)
            if suggestions is None:
                logger.warning(
                    "reflection_correction: failed to parse LLM response"
                )
                result.details["error"] = "Failed to parse LLM JSON response"
                result.duration_ms = (time.time() - t0) * 1000
                return result

            units_modified = 0
            units_created = 0

            # Build a lookup for used units
            unit_map = {u.id: u for u in used_units}

            for suggestion in suggestions:
                unit_id = suggestion.get("id", "")
                action = suggestion.get("action", "KEEP").upper()
                new_content = suggestion.get("new_content", {})

                unit = unit_map.get(unit_id)
                if unit is None:
                    logger.warning(
                        "reflection_correction: unit %s not in used set, skipping",
                        unit_id,
                    )
                    continue

                if action == "KEEP":
                    continue

                elif action == "UPDATE":
                    # Only update keys that already exist in the unit's content
                    # to prevent schema corruption from LLM hallucinating wrong keys
                    safe_content = {k: v for k, v in new_content.items() if k in unit.content}
                    if not safe_content:
                        logger.warning(
                            "reflection_correction: UPDATE for unit %s produced no "
                            "valid keys (got: %s, unit keys: %s), skipping",
                            unit_id[:8], list(new_content.keys()), list(unit.content.keys()),
                        )
                        continue
                    unit.content.update(safe_content)
                    unit.compute_signature()

                    # Recompute embedding
                    if self.embedding_model is not None:
                        try:
                            unit.embedding = self.embedding_model.encode(
                                unit.content_text()
                            )
                        except Exception as e:
                            logger.warning(
                                "reflection_correction: embedding failed: %s", e
                            )

                    self.store.update(unit)
                    units_modified += 1

                    # Graph enhanced: update node attributes
                    if self._is_graph_store():
                        try:
                            nid = self.store._content_nid(unit.id)
                            graph = self.store._graph
                            if graph.has_node(nid):
                                graph.nodes[nid]["content"] = unit.content_text()[:200]
                                graph.nodes[nid]["signature"] = unit.signature
                        except Exception as e:
                            logger.warning(
                                "reflection_correction: graph node update failed: %s", e
                            )

                    logger.info(
                        "reflection_correction: updated unit %s", unit_id[:8]
                    )

                elif action == "REPLACE":
                    # Filter replacement content to known keys to avoid schema corruption
                    safe_content = {k: v for k, v in new_content.items() if k in unit.content}
                    if not safe_content:
                        logger.warning(
                            "reflection_correction: REPLACE for unit %s produced no "
                            "valid keys (got: %s, unit keys: %s), skipping",
                            unit_id[:8], list(new_content.keys()), list(unit.content.keys()),
                        )
                        continue
                    # Create a new replacement unit
                    new_unit = MemoryUnit(
                        id=str(uuid.uuid4()),
                        type=unit.type,
                        content=safe_content,
                        source_task_id=unit.source_task_id,
                        source_task_query=unit.source_task_query,
                        task_outcome=unit.task_outcome,
                        confidence=unit.confidence,
                        usage_count=0,
                        success_count=0,
                        decay_weight=1.0,
                        is_active=True,
                    )
                    new_unit.compute_signature()
                    new_unit.token_estimate()

                    # Compute embedding for new unit
                    if self.embedding_model is not None:
                        try:
                            new_unit.embedding = self.embedding_model.encode(
                                new_unit.content_text()
                            )
                        except Exception as e:
                            logger.warning(
                                "reflection_correction: embedding failed: %s", e
                            )

                    # Add SUPERSEDES relation from new to old
                    new_unit.relations.append(
                        MemoryRelation(
                            target_id=unit.id,
                            relation_type=RelationType.SUPERSEDES,
                            weight=1.0,
                        )
                    )

                    # Add the replacement FIRST; deactivate the old unit only
                    # once the new one is verifiably stored. The old order
                    # (deactivate -> add) lost the memory entirely when add()
                    # skipped the unit (embedding failed on a vector store,
                    # or the new signature collided with an existing one)
                    # while still reporting created=1.
                    added = self.store.add([new_unit])
                    if not added:
                        logger.warning(
                            "reflection_correction: replacement for unit %s was "
                            "not stored (add returned 0 — missing embedding or "
                            "duplicate signature); keeping the old unit active",
                            unit_id[:8],
                        )
                        continue
                    unit.is_active = False
                    self.store.update(unit)
                    units_created += 1
                    units_modified += 1

                    # Graph enhanced: add SUPERSEDES edge
                    if self._is_graph_store():
                        try:
                            graph = self.store._graph
                            old_nid = self.store._content_nid(unit.id)
                            new_nid = self.store._content_nid(new_unit.id)

                            # Mark old node as inactive
                            if graph.has_node(old_nid):
                                graph.nodes[old_nid]["is_active"] = False

                            # Add SUPERSEDES edge from new to old
                            graph.add_edge(
                                new_nid,
                                old_nid,
                                key="SUPERSEDES",
                                edge_type="SUPERSEDES",
                                weight=1.0,
                            )
                        except Exception as e:
                            logger.warning(
                                "reflection_correction: graph edge update failed: %s",
                                e,
                            )

                    logger.info(
                        "reflection_correction: replaced unit %s with %s",
                        unit_id[:8], new_unit.id[:8],
                    )

            # Rebuild FAISS index if units were modified (Vector/Hybrid stores)
            if units_modified > 0 or units_created > 0:
                self._rebuild_faiss_if_needed()

            result.units_modified = units_modified
            result.units_created = units_created
            result.units_affected = units_modified + units_created
            result.details = {
                "used_unit_count": len(used_units),
                "suggestions_parsed": len(suggestions) if suggestions else 0,
            }

        except Exception as e:
            logger.error(
                "reflection_correction: execution failed: %s", e, exc_info=True
            )
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        logger.info(
            "reflection_correction: completed in %.1fms, modified=%d created=%d",
            result.duration_ms, result.units_modified, result.units_created,
        )
        return result
