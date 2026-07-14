"""Command-line interface for inspecting and smoke-testing AutoMem."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from automem.architecture.models import (
    SCHEMA_VERSION,
    ArchitectureSpec,
    architecture_space_manifest,
)
from automem.memory_schema import MemoryUnit, MemoryUnitType
from automem.memory_types import MemoryStatus
from automem.retrieval.base_retriever import QueryContext
from automem.retrieval.keyword_retriever import KeywordRetriever
from automem.runtime.context_composer import MemoryContextComposer
from automem.runtime.session import InjectionSessionRegistry
from automem.storage.json_storage import JsonStorage


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _space_command(_: argparse.Namespace) -> int:
    print(json.dumps(architecture_space_manifest(), indent=2, sort_keys=True))
    return 0


def run_offline_smoke() -> dict[str, Any]:
    """Exercise the public spec and a persistent retrieval/injection lifecycle."""
    spec = ArchitectureSpec(
        schema_version=SCHEMA_VERSION,
        encode="tip",
        store="json",
        retrieve="hybrid",
        manage="lightweight",
    )
    _require(
        ArchitectureSpec.from_dict(spec.to_dict()).fingerprint == spec.fingerprint,
        "architecture serialization changed its fingerprint",
    )
    multi_encode = ArchitectureSpec(
        schema_version=SCHEMA_VERSION,
        encode=("workflow", "tip", "trajectory"),
        store="json",
        retrieve="hybrid",
        manage="lightweight",
    )
    _require(
        multi_encode.encode == ("tip", "trajectory", "workflow"),
        "multi-encode subset was not canonicalized to menu order",
    )
    _require(
        ArchitectureSpec.from_search_dict(multi_encode.to_search_dict())
        == multi_encode,
        "multi-encode search round-trip changed the architecture",
    )
    _require(
        ArchitectureSpec.from_dict(multi_encode.to_dict()).fingerprint
        == multi_encode.fingerprint,
        "multi-encode serialization changed its fingerprint",
    )

    with TemporaryDirectory(prefix="automem-smoke-") as temporary_dir:
        db_path = Path(temporary_dir) / "memory.json"
        store = JsonStorage({"db_path": str(db_path)})
        _require(store.initialize(), "JsonStorage failed to initialize")

        first = MemoryUnit(
            id="smoke-tip-1",
            type=MemoryUnitType.TIP,
            content={
                "topic": "atomic persistence",
                "principle": "Write memory updates atomically before reuse.",
                "micro_example": "Replace a completed temporary file.",
            },
            source_task_id="offline-smoke",
            task_outcome="success",
        )
        second = MemoryUnit(
            id="smoke-tip-2",
            type=MemoryUnitType.TIP,
            content={
                "topic": "refresh boundaries",
                "principle": "Refresh memory only at a planning boundary.",
                "micro_example": "Allow one refresh after a replan.",
            },
            source_task_id="offline-smoke",
            task_outcome="success",
        )
        first.compute_signature()
        second.compute_signature()
        _require(store.add([first, second]) == 2, "JsonStorage did not persist two units")

        reopened = JsonStorage({"db_path": str(db_path)})
        _require(reopened.initialize(), "JsonStorage failed to reopen")
        _require(reopened.count() == 2, "JsonStorage reload count mismatch")

        pack = KeywordRetriever(reopened).retrieve(
            QueryContext(query="atomic memory persistence"),
            top_k=2,
        )
        _require(pack.scored_units, "offline keyword retrieval returned no memories")
        _require(
            pack.scored_units[0].unit.id == first.id,
            "offline keyword retrieval ranked the unrelated memory first",
        )

        registry = InjectionSessionRegistry()
        composer = MemoryContextComposer()
        session_key = registry.key("atomic memory persistence", "offline-smoke")
        _require(
            registry.phase_allowed(session_key, MemoryStatus.BEGIN),
            "BEGIN injection was rejected",
        )
        begin_candidates = [
            {
                "id": pack.scored_units[0].unit.id,
                "score": pack.scored_units[0].score,
                "text": pack.scored_units[0].unit.content_text(),
            }
        ]
        begin_result = composer.compose(
            "atomic memory persistence",
            begin_candidates,
            client=None,
            model="",
        )
        _require(begin_result.used_fallback, "offline composer did not use its fallback")
        _require(begin_result.kept_indices == [0], "offline composer fallback kept wrong item")
        _require(
            registry.commit(
                session_key,
                MemoryStatus.BEGIN,
                [first.id],
                begin_result.guidance,
            ),
            "BEGIN guidance was not committed",
        )
        _require(
            not registry.commit(
                session_key,
                MemoryStatus.BEGIN,
                [first.id],
                begin_result.guidance,
            ),
            "duplicate guidance was accepted",
        )
        _require(
            not registry.phase_allowed(session_key, MemoryStatus.IN),
            "IN injection was allowed outside a refresh boundary",
        )
        _require(
            registry.phase_allowed(
                session_key,
                MemoryStatus.IN,
                refresh_boundary=True,
            ),
            "first boundary refresh was rejected",
        )
        _require(
            registry.unseen_indices(session_key, [first.id, second.id]) == [1],
            "refresh did not exclude an already injected memory",
        )
        refresh_result = composer.compose(
            "refresh memory after replanning",
            [{"id": second.id, "score": 1.0, "text": second.content_text()}],
            client=None,
            model="",
        )
        _require(
            registry.commit(
                session_key,
                MemoryStatus.IN,
                [second.id],
                refresh_result.guidance,
            ),
            "refresh guidance was not committed",
        )
        _require(
            not registry.phase_allowed(
                session_key,
                MemoryStatus.IN,
                refresh_boundary=True,
            ),
            "a second refresh exceeded the fixed runtime budget",
        )

    return {
        "architecture": spec.to_dict(),
        "architecture_fingerprint": spec.fingerprint,
        "checks": {
            "architecture_validation": "ok",
            "json_persistence": "ok",
            "offline_keyword_retrieval": "ok",
            "runtime_begin_refresh_dedup": "ok",
            "composer_fallback": "ok",
        },
        "status": "ok",
    }


def _smoke_command(_: argparse.Namespace) -> int:
    print(json.dumps(run_offline_smoke(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automem",
        description="Inspect and validate the AutoMem memory architecture.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    space_parser = subparsers.add_parser(
        "space",
        help="Print the sole public architecture space and its counts.",
    )
    space_parser.set_defaults(handler=_space_command)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run a fully offline persistence, retrieval, and runtime smoke test.",
    )
    smoke_parser.set_defaults(handler=_smoke_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"automem {args.command} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
