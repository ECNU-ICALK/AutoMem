"""Access package resources without depending on the process working directory."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Union


def prompt_resource(*parts: str):
    """Return a traversable prompt resource inside the installed package."""

    resource = files("automem").joinpath("prompts")
    for part in parts:
        resource = resource.joinpath(part)
    return resource


def prompt_path(*parts: str) -> Path:
    """Return a filesystem path for an installed, unpacked AutoMem resource.

    Python wheels are installed as unpacked files by standard installers.  A
    clear error is preferable to silently falling back to the current working
    directory when a non-filesystem importer is used.
    """

    resource = prompt_resource(*parts)
    path = Path(str(resource))
    if not path.exists():
        joined = "/".join(parts) or "."
        raise FileNotFoundError(f"Packaged AutoMem prompt resource is missing: {joined}")
    return path


def read_prompt_text(*parts: str, encoding: str = "utf-8") -> str:
    """Read a prompt directly through :mod:`importlib.resources`."""

    return prompt_resource(*parts).read_text(encoding=encoding)


def read_prompt_bytes(*parts: str) -> bytes:
    """Read prompt bytes directly through :mod:`importlib.resources`."""

    return prompt_resource(*parts).read_bytes()


PathLike = Union[str, Path]

