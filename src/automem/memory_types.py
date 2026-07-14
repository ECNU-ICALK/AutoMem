"""
Core types for the unified memory system
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryStatus(Enum):
    """Status of memory operation"""
    # only these two status
    BEGIN = "begin"
    IN = "in"


class MemoryType(Enum):
    """Provider implementations shipped by AutoMem."""

    MODULAR = "modular"

# Provider mapping for dynamic loading
# Format: MemoryType -> (ClassName, ModuleName)
PROVIDER_MAPPING = {
    MemoryType.MODULAR: ("ModularMemoryProvider", "modular_memory_provider"),
}


def get_provider_class(memory_type: MemoryType):
    """Resolve a shipped provider class from its installed package module."""

    from importlib import import_module

    class_name, module_name = PROVIDER_MAPPING[memory_type]
    module = import_module(f"automem.providers.{module_name}")
    return getattr(module, class_name)


class MemoryItemType(Enum):
    """Type of memory item content"""
    TEXT = "text"
    API = "api"


@dataclass
class MemoryRequest:
    """Request for memory retrieval"""
    query: str
    context: str
    status: MemoryStatus
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class MemoryItem:
    """Base memory item structure"""
    id: str
    content: Any
    metadata: Dict[str, Any]
    score: Optional[float] = None
    type: MemoryItemType = MemoryItemType.TEXT


@dataclass
class MemoryResponse:
    """Response containing retrieved memories"""
    memories: List[MemoryItem]
    memory_type: MemoryType
    total_count: int
    request_id: Optional[str] = None


@dataclass
class TrajectoryData:
    """Data structure for memory ingestion"""
    query: str
    trajectory: List[Dict[str, Any]]
    result: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
