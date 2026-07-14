"""
ExperienceLedger — structured, evidence-graded experience accumulation
across rounds within a single AutoMem evolution run.

Replaces the freeform `cumulative_principles` string. The ledger stores:
  - principles[]: high-level claims with evidence trails and confidence
  - component_performance[]: granular per-(component,value) statistics
  - dead_ends[]: combos empirically shown not to work
  - open_questions[]: hypotheses raised but not yet tested

Each round, a *strong* diagnosis LLM (gpt-5.5 or similar) is given:
  - current ledger
  - this round's round_done.json
  - per-candidate attribution.json files

and outputs a structured delta JSON that mutates the ledger. Constraints:

  - At most LEDGER_MAX_PRINCIPLES principles total (lowest conf × recency dropped).
  - At most LEDGER_MAX_NEW_PER_ROUND new principles introduced per round.
  - At most LEDGER_MAX_DEAD_ENDS_PER_ROUND new dead ends per round.
  - Every principle MUST cite ≥ 1 evidence (round, candidate, metric, delta).
  - confidence > 0.5 requires ≥ LEDGER_MIN_EVIDENCE_FOR_HIGH_CONF supporting evidence.

The proposer reads the ledger via `render_for_prompt()` which formats a
Markdown view sorted by confidence × recency.

The whole subsystem can be disabled via `no_ledger=True` — `update_with_round`
becomes a no-op and `render_for_prompt` returns an empty placeholder.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

LEDGER_MAX_PRINCIPLES = 20
LEDGER_MAX_NEW_PER_ROUND = 3
LEDGER_MAX_DEAD_ENDS_PER_ROUND = 5
LEDGER_MAX_OPEN_QUESTIONS = 10
LEDGER_MIN_EVIDENCE_FOR_HIGH_CONF = 2
LEDGER_HIGH_CONF_THRESHOLD = 0.5
# Smart-6 fix (2026-05-16): automatic decay parameters chosen via AskUserQuestion.
LEDGER_VERIFIED_BUMP = 0.10        # confidence delta when LLM marks verified
LEDGER_REFUTED_PENALTY = 0.30      # confidence delta when LLM marks refuted
LEDGER_TIME_DECAY_PER_ROUND = 0.05 # confidence delta when principle silent N rounds
LEDGER_DORMANT_THRESHOLD = 0.30    # confidence below this → hidden from prompt

LEDGER_DIR_NAME = "ledger"
LEDGER_FILE_NAME = "ledger.json"
LEDGER_ARCHIVE_NAME = "ledger_archive.json"


# ── Public surface ────────────────────────────────────────────────────────

class ExperienceLedger:
    """Per-run, persistent, structured experience accumulator.

    Lifecycle:
      ledger = ExperienceLedger(run_dir, no_ledger=False)
      # ... per round:
      prompt_section = ledger.render_for_prompt()   # → §2F text
      # ... after round evaluation:
      ledger.update_with_round(round_id, round_summary, attributions,
                               diagnosis_model, prompt_template_path)
    """

    def __init__(self, run_dir: Path, no_ledger: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.no_ledger = no_ledger
        self.ledger_dir = self.run_dir / LEDGER_DIR_NAME
        self.ledger_path = self.ledger_dir / LEDGER_FILE_NAME
        self.archive_path = self.ledger_dir / LEDGER_ARCHIVE_NAME

        if no_ledger:
            self._data: Dict[str, Any] = _empty_ledger()
            return

        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        if self.ledger_path.exists():
            try:
                self._data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(
                    "Failed to load existing ledger at %s (%s); starting fresh.",
                    self.ledger_path, e,
                )
                self._data = _empty_ledger()
        else:
            self._data = _empty_ledger()

    # ── Read API ─────────────────────────────────────────────────────────

    @property
    def is_disabled(self) -> bool:
        return self.no_ledger

    @property
    def data(self) -> Dict[str, Any]:
        """Return the raw ledger data (read-only access for logging/inspection)."""
        return dict(self._data)

    def render_for_prompt(self) -> str:
        """Format the ledger as Markdown for proposer §2F. Returns empty
        placeholder when disabled or fully empty."""
        if self.no_ledger:
            return "(ledger disabled via --no_ledger; running without cross-round memory)"

        principles = self._data.get("principles", [])
        questions = self._data.get("open_questions", [])
        dead_ends = self._data.get("dead_ends", [])
        comp_perf = self._data.get("component_performance", [])

        if not (principles or questions or dead_ends or comp_perf):
            return ("(experience ledger is empty — round 1 has not yet completed, "
                    "or this is run round 1)")

        # Smart-6 fix (2026-05-16): filter dormant (confidence < DORMANT_THRESHOLD).
        # Dormant principles still live in ledger.json (for audit / future
        # revival) but don't pollute the proposer prompt.
        non_dormant = [p for p in principles
                       if float(p.get("confidence", 0.0)) >= LEDGER_DORMANT_THRESHOLD]
        dormant_count = len(principles) - len(non_dormant)

        active_principles = [p for p in non_dormant if p.get("status") == "active"]
        # Sort by confidence × recency (last_validated_round)
        last_round = self._data.get("last_updated_round", 0) or 1
        active_principles.sort(
            key=lambda p: (
                p.get("confidence", 0)
                * (1.0 - 0.2 * max(0, last_round - p.get("last_validated_round", 0)))
            ),
            reverse=True,
        )

        lines: List[str] = [
            f"_Ledger built across {self._data.get('last_updated_round', 0)} "
            f"completed round(s). "
            f"{dormant_count} principle(s) dormant (conf < {LEDGER_DORMANT_THRESHOLD:.2f}) — "
            f"not shown but preserved in ledger.json. "
            f"Treat low-confidence entries as tentative._",
            "",
        ]

        if active_principles:
            high_conf = [p for p in active_principles
                         if p.get("confidence", 0) >= LEDGER_HIGH_CONF_THRESHOLD]
            low_conf = [p for p in active_principles
                        if p.get("confidence", 0) < LEDGER_HIGH_CONF_THRESHOLD]
            if high_conf:
                lines.append("**Active high-confidence principles** "
                             f"(conf ≥ {LEDGER_HIGH_CONF_THRESHOLD}):")
                for p in high_conf:
                    lines.append(_format_principle(p))
                lines.append("")
            if low_conf:
                lines.append("**Tentative principles** (need more evidence):")
                for p in low_conf[:LEDGER_MAX_PRINCIPLES // 2]:
                    lines.append(_format_principle(p))
                lines.append("")

        if questions:
            untested = [q for q in questions if q.get("status") != "answered"][
                :LEDGER_MAX_OPEN_QUESTIONS
            ]
            if untested:
                lines.append("**Untested hypotheses** (consider testing):")
                for q in untested:
                    rec = _jinja_safe(q.get("recommended_test", ""))
                    qtxt = _jinja_safe(q.get("question", ""))
                    lines.append(f"  - [{q.get('id', '?')}] {qtxt}"
                                 f"{(' — ' + rec) if rec else ''}")
                lines.append("")

        if dead_ends:
            active_de = [d for d in dead_ends if d.get("active", True)]
            if active_de:
                lines.append("**Dead ends** (do NOT propose these combinations):")
                for d in active_de:
                    lines.append(
                        f"  - {_jinja_safe(d.get('combo', '?'))}"
                        f" — {_jinja_safe(d.get('outcome', ''))}"
                        f" (R{d.get('evidence_round', '?')})"
                    )
                lines.append("")

        if comp_perf:
            lines.append("**Component performance traces (raw stats)**:")
            for cp in comp_perf:
                lines.append(
                    f"  - {_jinja_safe(cp.get('component', '?'))} = "
                    f"`{_jinja_safe(cp.get('value', '?'))}`: "
                    f"n={cp.get('n_evaluations', 0)} "
                    f"mean_acc={cp.get('mean_acc', 0):.3f} "
                    f"mean_lift={cp.get('mean_lift', 0):+.3f}"
                )
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    # ── Write API ────────────────────────────────────────────────────────

    def update_with_round(
        self,
        round_id: int,
        round_summary: Dict[str, Any],
        attributions: List[Dict[str, Any]],
        diagnosis_model,
        prompt_template_path: str,
    ) -> Dict[str, Any]:
        """Invoke the diagnosis LLM to produce a structured ledger delta and
        apply it. Returns the delta dict (also persisted to round_N_delta.json).

        On any failure (LLM error, malformed JSON, schema violation), logs and
        returns an empty delta rather than crashing — search continues with
        the unchanged ledger.
        """
        if self.no_ledger:
            logger.info("[ledger] disabled (no_ledger=True); skipping update.")
            return {}

        # Idempotency guard (Codex C1 fix): if last_updated_round already
        # ≥ round_id, this round was previously applied. Resuming after a
        # crash that occurred AFTER ledger.update_with_round but BEFORE
        # state.write would otherwise re-enter this round (the for loop
        # restarts at the same `next_round`) and double-apply the delta —
        # duplicating principles, dead_ends and version-bumping twice.
        if int(self._data.get("last_updated_round", 0)) >= int(round_id):
            logger.info(
                "[ledger] R%d already applied (last_updated_round=%d); "
                "skipping update_with_round to preserve idempotency.",
                round_id, self._data.get("last_updated_round", 0),
            )
            return {}

        try:
            delta = self._call_diagnosis_for_delta(
                round_id, round_summary, attributions,
                diagnosis_model, prompt_template_path,
            )
        except Exception as e:
            logger.warning("[ledger] update_with_round LLM call failed: %s", e)
            return {}

        if not delta or not isinstance(delta, dict):
            logger.warning("[ledger] LLM returned empty/non-dict delta; keeping ledger unchanged.")
            return {}

        # Persist raw delta for audit trail before mutation
        delta_path = self.ledger_dir / f"round_{round_id}_delta.json"
        try:
            delta_path.write_text(json.dumps(delta, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        except Exception as e:
            logger.warning("[ledger] failed to persist round delta: %s", e)

        # Validate + apply
        try:
            applied = self._apply_delta(delta, round_id)
        except Exception as e:
            logger.warning("[ledger] _apply_delta failed: %s — ledger unchanged.", e)
            return {}

        # Smart-6 fix (2026-05-16): time-decay confidence for principles the
        # LLM did NOT mention this round. Without decay the ledger keeps stale
        # high-confidence claims that may no longer hold after the pool / agent
        # has evolved. Decay rate is small (-0.05/round) so a principle silent
        # for 4 rounds still hovers ≥ 0.30 — the dormancy threshold.
        decay_stats = self._decay_unmentioned_principles(round_id)

        self._data["last_updated_round"] = round_id
        self._data["version"] = int(self._data.get("version", 0)) + 1
        self._save()

        logger.info(
            "[ledger] R%d update: principles +%d/-%d, dead_ends +%d, "
            "open_questions +%d, decayed=%d, dormant_now=%d "
            "(total: %d principles, %d dead_ends, %d questions)",
            round_id,
            applied.get("principles_added", 0),
            applied.get("principles_archived", 0),
            applied.get("dead_ends_added", 0),
            applied.get("open_questions_added", 0),
            decay_stats.get("decayed", 0),
            decay_stats.get("dormant_after", 0),
            len(self._data.get("principles", [])),
            len(self._data.get("dead_ends", [])),
            len(self._data.get("open_questions", [])),
        )
        return delta

    def _decay_unmentioned_principles(self, round_id: int) -> Dict[str, int]:
        """Smart-6 fix (2026-05-16). Decay confidence for principles whose
        last_validated_round is older than this round (LLM did not surface
        them in evidence_updates).

        Returns {"decayed": N_principles_decayed, "dormant_after": N_below_threshold}.
        """
        principles = self._data.get("principles", []) or []
        n_decayed = 0
        for p in principles:
            # G2 fix (codex review, 2026-05-16): be defensive about field types
            # — JSON round-trip can leave values as strings like "5.0" or "5",
            # and int("5.0") raises ValueError. Use float-then-int so the decay
            # never crashes the whole round update on one malformed entry.
            try:
                last_val = int(float(p.get("last_validated_round", 0) or 0))
            except (ValueError, TypeError):
                last_val = 0
            if last_val >= round_id:
                continue  # mentioned this round — keep
            try:
                current = float(p.get("confidence", 0.0) or 0.0)
            except (ValueError, TypeError):
                current = 0.0
            new_conf = max(0.0, current - LEDGER_TIME_DECAY_PER_ROUND)
            if new_conf != current:
                p["confidence"] = new_conf
                n_decayed += 1
        n_dormant = sum(
            1 for p in principles
            if _safe_float(p.get("confidence", 0.0)) < LEDGER_DORMANT_THRESHOLD
        )
        return {"decayed": n_decayed, "dormant_after": n_dormant}

    # ── Internals ────────────────────────────────────────────────────────

    def _call_diagnosis_for_delta(
        self,
        round_id: int,
        round_summary: Dict[str, Any],
        attributions: List[Dict[str, Any]],
        diagnosis_model,
        prompt_template_path: str,
    ) -> Dict[str, Any]:
        """Render the ledger_update prompt and call the diagnosis LLM."""
        from automem.llm_utils import load_prompt, render_prompt, parse_json_response

        template = load_prompt(prompt_template_path)
        # Compact the inputs to keep prompt size sane
        round_summary_compact = _compact_round_summary(round_summary)
        attributions_compact = _compact_attributions(attributions)

        ctx = {
            "round_id": round_id,
            "current_ledger_json": json.dumps(self._data, indent=2, ensure_ascii=False),
            "round_summary_json": json.dumps(round_summary_compact, indent=2, ensure_ascii=False),
            "attributions_json": json.dumps(attributions_compact, indent=2, ensure_ascii=False),
            "max_new_principles": LEDGER_MAX_NEW_PER_ROUND,
            "max_new_dead_ends": LEDGER_MAX_DEAD_ENDS_PER_ROUND,
            "max_total_principles": LEDGER_MAX_PRINCIPLES,
            "min_evidence_for_high_conf": LEDGER_MIN_EVIDENCE_FOR_HIGH_CONF,
            "high_conf_threshold": LEDGER_HIGH_CONF_THRESHOLD,
        }
        filled = render_prompt(template, ctx)

        messages = [{"role": "user", "content": [{"type": "text", "text": filled}]}]
        response = diagnosis_model(messages)
        raw = response.content if hasattr(response, "content") else str(response)

        parsed = parse_json_response(raw)
        if isinstance(parsed, dict):
            return parsed
        # Some models wrap in a list — accept the first dict
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        logger.warning("[ledger] could not parse LLM output as dict: %r",
                       (raw or "")[:200])
        return {}

    def _apply_delta(self, delta: Dict[str, Any], round_id: int) -> Dict[str, int]:
        """Mutate self._data based on delta. Enforces all constraints.

        Delta schema (all fields optional):
          {
            "new_principles":   [{"claim": "...", "domain": "retrieval|storage|...",
                                  "evidence": [{"round": N, "candidate": "rX_cY",
                                                "metric": "acc|lift|fit",
                                                "delta": +0.04, "supports": true,
                                                "context": "..."}],
                                  "confidence": 0.0-1.0}],
            "evidence_updates": [{"id": "P001", "new_evidence": [{...}],
                                  "new_confidence": 0.65, "new_status":
                                  "active|refuted|superseded"}],
            "new_dead_ends":    [{"combo": "...", "outcome": "...",
                                  "evidence_round": N}],
            "answered_questions":[{"id": "Q003", "answer": "...",
                                   "supporting_round": N}],
            "new_questions":    [{"question": "...", "recommended_test": "..."}],
            "component_performance_updates": [{"component": "retrieval",
                                               "value": "graph",
                                               "n_evaluations": 3,
                                               "mean_acc": 0.66,
                                               "mean_lift": 0.03,
                                               "best_context": "..."}]
          }
        """
        stats = {
            "principles_added": 0, "principles_archived": 0,
            "dead_ends_added": 0, "open_questions_added": 0,
        }

        # 1. Apply evidence_updates (mutate existing principles in place)
        principles = self._data.setdefault("principles", [])
        by_id = {p["id"]: p for p in principles if "id" in p}
        for upd in delta.get("evidence_updates", []) or []:
            pid = upd.get("id")
            if not pid or pid not in by_id:
                continue
            p = by_id[pid]
            new_ev = [e for e in (upd.get("new_evidence") or [])
                      if _is_valid_evidence(e)]
            if new_ev:
                p.setdefault("evidence", []).extend(new_ev)
            if "new_confidence" in upd:
                p["confidence"] = _clip01(upd["new_confidence"])
            if "new_status" in upd and upd["new_status"] in {
                "active", "refuted", "superseded"
            }:
                p["status"] = upd["new_status"]
            # Codex I4 fix: only refresh recency when actual NEW evidence was
            # added; status/confidence-only touches must not artificially
            # boost the recency multiplier in the proposer's view.
            if new_ev:
                p["last_validated_round"] = round_id
            # Constraint: confidence > threshold requires ≥ N evidence
            if (p.get("confidence", 0) >= LEDGER_HIGH_CONF_THRESHOLD
                    and len(p.get("evidence", []))
                    < LEDGER_MIN_EVIDENCE_FOR_HIGH_CONF):
                p["confidence"] = LEDGER_HIGH_CONF_THRESHOLD - 0.01

        # 2. Add new_principles. Codex I5 fix: validate FIRST, then cap the
        # validated set — taking [:MAX] before validation discards good
        # principles when the LLM happens to emit junk earlier in the list.
        new_p = delta.get("new_principles", []) or []
        validated: List[Dict[str, Any]] = []
        for raw in new_p:
            np = _validate_new_principle(
                raw, round_id, len(principles) + len(validated)
            )
            if np is not None:
                validated.append(np)
            if len(validated) >= LEDGER_MAX_NEW_PER_ROUND:
                break
        for np in validated:
            principles.append(np)
            stats["principles_added"] += 1

        # 3. Cap principles total — drop lowest (confidence × recency)
        if len(principles) > LEDGER_MAX_PRINCIPLES:
            principles.sort(
                key=lambda p: (
                    p.get("confidence", 0)
                    * (1.0 - 0.2 * max(
                        0, round_id - p.get("last_validated_round", 0)
                    ))
                ),
                reverse=True,
            )
            archived = principles[LEDGER_MAX_PRINCIPLES:]
            self._archive_principles(archived, round_id)
            stats["principles_archived"] = len(archived)
            del principles[LEDGER_MAX_PRINCIPLES:]

        # 4. Apply dead_ends (≤ MAX_DEAD_ENDS_PER_ROUND new)
        dead_ends = self._data.setdefault("dead_ends", [])
        new_de = delta.get("new_dead_ends", []) or []
        for raw in new_de[:LEDGER_MAX_DEAD_ENDS_PER_ROUND]:
            if not isinstance(raw, dict) or not raw.get("combo"):
                continue
            # Dedup by combo
            if any(d.get("combo") == raw["combo"] for d in dead_ends):
                continue
            dead_ends.append({
                "combo": str(raw["combo"])[:200],
                "outcome": str(raw.get("outcome", ""))[:200],
                "evidence_round": int(raw.get("evidence_round", round_id)),
                "active": True,
            })
            stats["dead_ends_added"] += 1

        # 5. Answered questions
        questions = self._data.setdefault("open_questions", [])
        for upd in delta.get("answered_questions", []) or []:
            qid = upd.get("id")
            if not qid:
                continue
            for q in questions:
                if q.get("id") == qid:
                    q["status"] = "answered"
                    q["answer"] = str(upd.get("answer", ""))[:300]
                    q["answered_round"] = round_id
                    break

        # 6. New questions
        new_q = delta.get("new_questions", []) or []
        for raw in new_q[:LEDGER_MAX_OPEN_QUESTIONS]:
            if not isinstance(raw, dict) or not raw.get("question"):
                continue
            qid = f"Q{len(questions) + 1:03d}"
            questions.append({
                "id": qid,
                "question": str(raw["question"])[:300],
                "recommended_test": str(raw.get("recommended_test", ""))[:300],
                "raised_round": round_id,
                "status": "untested",
            })
            stats["open_questions_added"] += 1
        # Cap questions
        if len(questions) > LEDGER_MAX_OPEN_QUESTIONS:
            # Drop oldest answered first, then oldest untested
            questions.sort(key=lambda q: (
                q.get("status") == "answered",
                -(q.get("raised_round", 0)),
            ))
            del questions[LEDGER_MAX_OPEN_QUESTIONS:]

        # 7. Component performance — replace by (component, value) key
        cp_updates = delta.get("component_performance_updates", []) or []
        cp_list = self._data.setdefault("component_performance", [])
        cp_index = {(c.get("component"), c.get("value")): i
                    for i, c in enumerate(cp_list)}
        for upd in cp_updates:
            if not isinstance(upd, dict):
                continue
            key = (upd.get("component"), upd.get("value"))
            if not all(key):
                continue
            row = {
                "component": str(upd["component"])[:50],
                "value": str(upd["value"])[:50],
                "n_evaluations": int(upd.get("n_evaluations", 0)),
                "mean_acc": _clip01(upd.get("mean_acc", 0)),
                "mean_lift": float(upd.get("mean_lift", 0)),
                "best_context": str(upd.get("best_context", ""))[:200],
            }
            if key in cp_index:
                cp_list[cp_index[key]] = row
            else:
                cp_list.append(row)

        return stats

    def _archive_principles(self, archived: List[Dict[str, Any]],
                            round_id: int) -> None:
        """Persist dropped principles to ledger_archive.json (atomic write
        — Codex I2 fix: prior version did naive read-modify-write which
        could truncate on a kill mid-write)."""
        try:
            existing: List[Dict[str, Any]] = []
            if self.archive_path.exists():
                existing = json.loads(self.archive_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            for p in archived:
                p = dict(p)
                p["archived_at_round"] = round_id
                existing.append(p)
            tmp = self.archive_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, self.archive_path)
        except Exception as e:
            logger.warning("[ledger] failed to archive principles: %s", e)

    def _save(self) -> None:
        """Atomic write of ledger.json."""
        tmp = self.ledger_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, self.ledger_path)


# ── Helpers ───────────────────────────────────────────────────────────────

def _empty_ledger() -> Dict[str, Any]:
    return {
        "version": 0,
        "last_updated_round": 0,
        "principles": [],
        "component_performance": [],
        "dead_ends": [],
        "open_questions": [],
    }


def _clip01(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


def _safe_float(v: Any, default: float = 0.0) -> float:
    """G2 fix (codex review, 2026-05-16): coerce confidence-like values to
    float without crashing on strings like '5.0' or odd JSON round-trips."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _is_valid_evidence(e: Any) -> bool:
    """Validate an evidence dict: must have all four required keys, AND
    `delta` must be numeric (Codex M4 fix: previously only checked key
    presence, so an LLM emitting `"delta": "0.04"` (string) would pass and
    later break when sorting/aggregating)."""
    if not isinstance(e, dict):
        return False
    if not all(k in e for k in ("round", "candidate", "metric", "delta")):
        return False
    return isinstance(e["delta"], (int, float)) and not isinstance(e["delta"], bool)


def _validate_new_principle(raw: Any, round_id: int, n_existing: int
                            ) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    claim = (raw.get("claim") or "").strip()
    if not claim:
        return None
    evidence = [e for e in (raw.get("evidence") or []) if _is_valid_evidence(e)]
    if not evidence:
        return None  # ZERO-evidence principles are forbidden
    confidence = _clip01(raw.get("confidence", 0.3))
    if (confidence >= LEDGER_HIGH_CONF_THRESHOLD
            and len(evidence) < LEDGER_MIN_EVIDENCE_FOR_HIGH_CONF):
        # Force confidence below threshold if evidence is thin
        confidence = LEDGER_HIGH_CONF_THRESHOLD - 0.01
    domain = str(raw.get("domain", ""))[:30] or "general"
    pid = f"P{n_existing + 1:03d}"
    return {
        "id": pid,
        "claim": claim[:300],
        "domain": domain,
        "evidence": evidence,
        "confidence": confidence,
        "status": "active",
        "first_introduced_round": round_id,
        "last_validated_round": round_id,
    }


def _jinja_safe(s: Any) -> str:
    """Escape Jinja2 control sequences in user-content fields so the ledger
    render is safe to inline into another Jinja-rendered prompt
    (architecture_search.txt). Without this, a principle whose claim happens
    to contain `{{` or `{%` would either crash the proposer prompt render or
    silently get interpreted as a Jinja directive."""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    return (s.replace("{{", "{ {").replace("}}", "} }")
             .replace("{%", "{ %").replace("%}", "% }"))


def _format_principle(p: Dict[str, Any]) -> str:
    """Compact one-line render of a principle for the proposer prompt."""
    pid = p.get("id", "?")
    conf = p.get("confidence", 0)
    claim = _jinja_safe(p.get("claim", ""))
    n_ev = len(p.get("evidence") or [])
    return f"  - [{pid} conf={conf:.2f} n_ev={n_ev}] {claim}"


def _compact_round_summary(rs: Dict[str, Any]) -> Dict[str, Any]:
    """Trim round_summary so the ledger prompt stays small."""
    cands = []
    for c in (rs.get("candidate_results") or []):
        m = c.get("metrics") or {}
        cands.append({
            "config_id": c.get("config_id"),
            "diversity_role": c.get("diversity_role"),
            "added_to_front": c.get("added_to_front"),
            "failed": c.get("failed", False),
            "architecture": c.get("architecture"),
            "metrics": {
                "accuracy": m.get("accuracy"),
                "memory_lift": m.get("memory_lift"),
                "hit_rate": m.get("hit_rate"),
                "fitness": m.get("fitness"),
                "empty_retrieval_rate": m.get("empty_retrieval_rate"),
            },
        })
    return {
        "round_id": rs.get("round_id"),
        "batch_size": rs.get("batch_size"),
        "best_fitness": rs.get("best_fitness"),
        "pareto_front_size": rs.get("pareto_front_size"),
        "pool_size": rs.get("pool_size"),
        "candidates": cands,
        # G10 fix (codex review, 2026-05-16): pass through new Smart-N fields so
        # the ledger LLM can fold them into principles. Without this they're
        # invisible to ledger-update reasoning.
        "differential_diagnosis": rs.get("differential_diagnosis", {}),
    }


def _compact_attributions(atts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trim per-candidate attribution.json to the diagnosis-relevant subset.

    Codex I3 fix: `(s.get("diagnosis", "") or "")[:300]` — naive
    `s.get(k, default)` only returns `default` when the key is absent. If
    the saved attribution has `"diagnosis": null`, we got `None[:300]` and
    crashed — caught only by the outer `try/except`, silently disabling
    the ledger update for that round. Coalesce with `or ""` BEFORE slicing.
    """
    out = []
    for a in atts:
        s = (a or {}).get("summary") or {}
        # E11 fix (2026-05-16): prefer rule_diagnosis (new field) but fall back
        # to legacy diagnosis for old attribution.json files. After B2 rename
        # the writer emits both as alias; readers should use new key going
        # forward.
        rule_diag = (s.get("rule_diagnosis") or s.get("diagnosis") or "")[:300]
        # G7/G10 fix (codex review, 2026-05-16): also surface memory_compliance
        # — without this Smart-8's per-task LLM compliance check is invisible
        # to the ledger LLM and the principles can't reference instruction-
        # bypass patterns.
        mc = s.get("memory_compliance") or {}
        compact_mc: Dict[str, Any] = {}
        if mc and isinstance(mc, dict):
            compact_mc = {
                "avg_followed_score": mc.get("avg_followed_score"),
                "n_tasks_with_compliance_data": mc.get("n_tasks_with_compliance_data"),
                "violation_distribution": mc.get("violation_distribution") or {},
                "interpretation": (mc.get("interpretation") or "")[:240],
            }

        # H4 (2026-05-17): surface concrete per-task subclass examples so the
        # ledger LLM can learn "what task patterns trigger which subclass"
        # (e.g. "scope_misinterpretation often hits questions with 'between
        # Y1 and Y2'"). Without this the ledger only sees subclass counts and
        # cannot capture pattern-level principles.
        per_task = (a or {}).get("per_task") or []
        subclass_samples: Dict[str, List[Dict[str, Any]]] = {}
        for t in per_task:
            sub = t.get("reasoning_subclass")
            if not sub:
                continue
            if len(subclass_samples.get(sub, [])) >= 2:
                continue  # cap 2 per subclass to keep ledger prompt bounded
            subclass_samples.setdefault(sub, []).append({
                "task_id": (t.get("task_id") or "")[:24],
                "question": (t.get("question") or "")[:150],
                "evidence": (t.get("reasoning_subclass_evidence") or "")[:120],
            })

        out.append({
            "config_id": (a or {}).get("config_id"),
            "summary": {
                "rule_diagnosis": rule_diag,
                # Legacy alias for any downstream consumer reading "diagnosis".
                "diagnosis": rule_diag,
                "breakdown": s.get("breakdown") or {},
                "rates": s.get("rates") or {},
                "reasoning_error_subclasses":
                    s.get("reasoning_error_subclasses") or {},
                "memory_compliance": compact_mc,
                # H4 fix (2026-05-17): per-subclass task examples.
                "reasoning_subclass_examples": subclass_samples,
                # Smart-9a (2026-05-17): the LLM-synthesized verdict is the
                # single richest summary of this candidate — let the ledger
                # LLM learn from it too.
                "synthesized_verdict": s.get("synthesized_verdict") or {},
            },
            "layer_diagnosis": (s.get("layer_diagnosis")
                                or (a or {}).get("layer_diagnosis")
                                or {}),
        })
    return out
