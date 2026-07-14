"""Thread-safe phase and duplicate control for memory injection."""

from dataclasses import dataclass, field
import hashlib
import threading
from typing import Dict, Iterable, Set

from automem.memory_types import MemoryStatus

from .policy import DEFAULT_RUNTIME_POLICY, RuntimePolicy


@dataclass
class _Session:
    injected_unit_ids: Set[str] = field(default_factory=set)
    guidance_fingerprints: Set[str] = field(default_factory=set)
    refreshes: int = 0


class InjectionSessionRegistry:
    def __init__(self, policy: RuntimePolicy = DEFAULT_RUNTIME_POLICY):
        self.policy = policy
        self._sessions: Dict[str, _Session] = {}
        self._lock = threading.RLock()

    @staticmethod
    def key(query: str, task_id: str = "") -> str:
        return str(task_id or query or "").strip()

    def begin(self, key: str) -> None:
        with self._lock:
            self._sessions[key] = _Session()

    def phase_allowed(
        self,
        key: str,
        status: MemoryStatus,
        *,
        refresh_boundary: bool = False,
    ) -> bool:
        with self._lock:
            if status == MemoryStatus.BEGIN:
                self._sessions[key] = _Session()
                return True
            if status != MemoryStatus.IN or not refresh_boundary:
                return False
            session = self._sessions.setdefault(key, _Session())
            if session.refreshes >= self.policy.max_refreshes_per_task:
                return False
            # A refresh budget limits attempts, not only successful injections.
            # Otherwise an empty retrieval/composer result could allow repeated
            # IN calls at every later step in the same task.
            session.refreshes += 1
            return True

    def unseen_indices(self, key: str, unit_ids: Iterable[str]) -> list[int]:
        with self._lock:
            seen = self._sessions.setdefault(key, _Session()).injected_unit_ids
            return [index for index, unit_id in enumerate(unit_ids) if unit_id not in seen]

    def commit(
        self,
        key: str,
        status: MemoryStatus,
        unit_ids: Iterable[str],
        guidance: str,
    ) -> bool:
        fingerprint = hashlib.sha256(guidance.strip().encode("utf-8")).hexdigest()
        with self._lock:
            session = self._sessions.setdefault(key, _Session())
            if fingerprint in session.guidance_fingerprints:
                return False
            session.guidance_fingerprints.add(fingerprint)
            session.injected_unit_ids.update(str(value) for value in unit_ids)
            return True
