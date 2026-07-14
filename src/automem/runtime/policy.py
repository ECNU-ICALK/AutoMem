"""The single, code-defined execution policy for AutoMem.

This policy is deliberately not part of the architecture search space. Every
candidate is evaluated with the same query planning, context composition, and
phase refresh behavior.
"""

from dataclasses import asdict, dataclass
import hashlib
import json


RUNTIME_POLICY_ID = "automem-runtime-v1"


@dataclass(frozen=True)
class RuntimePolicy:
    policy_id: str = RUNTIME_POLICY_ID
    max_injected_units: int = 3
    max_refreshes_per_task: int = 1
    planner_max_query_chars: int = 2000
    composer_max_candidate_chars: int = 1200
    composer_max_output_tokens: int = 500

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


DEFAULT_RUNTIME_POLICY = RuntimePolicy()
