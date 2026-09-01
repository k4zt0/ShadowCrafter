"""Create a deterministic size and SHA-256 inventory for a downloaded artifact tree."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import typer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(
    root: Path,
    output: Path,
    artifact_id: str,
    revision: str,
) -> None:
    """Hash regular files while excluding local Hub cache metadata."""
    resolved_root = root.resolve(strict=True)
    resolved_output = output.resolve()
    if not resolved_root.is_dir():
        raise typer.BadParameter("root must be a directory")
    if resolved_output.is_relative_to(resolved_root):
        raise typer.BadParameter("output must be outside the inventoried tree")

    files: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(resolved_root).parts:
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": path.relative_to(resolved_root).as_posix(),
                "size": size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": "1.0",
        "artifact_id": artifact_id,
        "revision": revision,
        "created_at": datetime.now(UTC).isoformat(),
        "root": str(resolved_root),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("artifact_id", "file_count", "total_bytes")}))


if __name__ == "__main__":
    typer.run(main)
