"""
Configuration for automem memory system
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from .memory_types import MemoryType
from .resources import prompt_path

# Default configuration for different memory providers
STORAGE_BASE_DIR = "./storage"

PROMPT_DIR = str(prompt_path())

DEFAULT_CONFIG = {
    "automem": {
        "default_top_k": 3,
        "active_provider": "modular",
        "storage_base_dir": STORAGE_BASE_DIR,
    },

    "providers": {
        MemoryType.MODULAR: {
            "storage_dir": "./storage/modular",
            "storage_type": "json",
            "retriever_type": "hybrid",
            "retriever_config": {},
            "enabled_prompts": ["tip"],
            "prompt_dir": PROMPT_DIR,
            "top_k": 5,
            "min_relevance": 0.0,  # Global injection threshold: skip memories below this (0=disabled, use retriever min_score instead)
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_cache_dir": "./storage/models",
        },
    },
}


def load_runtime_config(runtime_config_path: str | Path) -> Dict[str, Any]:
    """Load and validate the structured config emitted by ArchitectureCompiler."""

    from .architecture.compiler import RuntimeConfig

    path = Path(runtime_config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Runtime config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Runtime config JSON must contain an object")

    allowed = set(RuntimeConfig().to_dict())
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown runtime config fields: {sorted(unknown)}")
    return RuntimeConfig.from_dict(payload).to_dict()


def get_memory_config(
    provider_type: MemoryType,
    runtime_config_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Build provider config, optionally merging a compiled runtime config."""

    cfg = deepcopy(DEFAULT_CONFIG["providers"].get(provider_type, {}))
    if runtime_config_path is None:
        return cfg
    if provider_type is not MemoryType.MODULAR:
        raise ValueError("AutoMem ships only the modular provider")

    runtime = load_runtime_config(runtime_config_path)
    retrieval_config = dict(runtime["retrieval_config"])
    cfg.update(
        {
            "storage_dir": runtime["storage_dir"],
            "storage_type": runtime["primary_storage_type"],
            "additional_stores": dict(runtime["additional_stores"]),
            "retriever_type": runtime["retrieval_type"],
            "retriever_config": retrieval_config,
            "enabled_prompts": list(runtime["extract_plan"].get("extract_types", [])),
            "management_enabled": True,
            "management_preset": runtime["management_preset"],
            "management_config": dict(runtime["management_config"]),
            "runtime_schema_version": runtime["runtime_schema_version"],
        }
    )
    if "top_k" in retrieval_config:
        cfg["top_k"] = int(retrieval_config["top_k"])
    if "gate_threshold" in retrieval_config:
        cfg["gate_threshold"] = float(retrieval_config["gate_threshold"])
    return cfg


def get_automem_config() -> Dict[str, Any]:
    """Get configuration for the automem memory system"""
    return DEFAULT_CONFIG["automem"].copy()
