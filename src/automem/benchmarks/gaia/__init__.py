"""GAIA benchmark helpers."""

from pathlib import Path


def resolve_attachment_path(file_name, infile) -> str:
    """Resolve one attachment confined to the metadata file's directory."""

    input_path = Path(infile).expanduser().resolve(strict=True)
    input_root = input_path.parent
    attachment = Path(str(file_name)).expanduser()
    if attachment.is_absolute():
        raise ValueError("GAIA attachment paths must be relative to the input file")
    resolved = (input_root / attachment).resolve()
    try:
        resolved.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(
            f"GAIA attachment escapes the input directory: {file_name!r}"
        ) from exc
    resolved = resolved.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"GAIA attachment is not a regular file: {file_name!r}")
    return str(resolved)


__all__ = ["resolve_attachment_path"]
