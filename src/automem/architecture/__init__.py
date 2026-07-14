from .models import (
    ARCHITECTURE_CHOICES,
    SCHEMA_VERSION,
    SPACE_ID,
    ArchitectureSpec,
    architecture_space_manifest,
)


def __getattr__(name: str):
    """Load compiler types lazily to keep the space/model imports acyclic."""
    if name in {"ArchitectureCompiler", "CompilationError", "RuntimeConfig"}:
        from .compiler import ArchitectureCompiler, CompilationError, RuntimeConfig

        return {
            "ArchitectureCompiler": ArchitectureCompiler,
            "CompilationError": CompilationError,
            "RuntimeConfig": RuntimeConfig,
        }[name]
    raise AttributeError(name)

__all__ = [
    "ARCHITECTURE_CHOICES",
    "ArchitectureCompiler",
    "ArchitectureSpec",
    "CompilationError",
    "RuntimeConfig",
    "SCHEMA_VERSION",
    "SPACE_ID",
    "architecture_space_manifest",
]
