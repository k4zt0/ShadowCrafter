"""Safe workstation import of accountless Colab candidate/checkpoint exports."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from shadowcrafter.data.manifest import sha256_file, write_json_exclusive

_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
_MAX_EXPANDED_BYTES = 16 * 1024 * 1024 * 1024
_MAX_FILE_BYTES = 8 * 1024 * 1024 * 1024
_MAX_MEMBERS = 20_000
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ColabExportError(RuntimeError):
    """A downloaded Colab archive failed its local trust-boundary checks."""


@dataclass(frozen=True, slots=True)
class ImportedColabExport:
    kind: Literal["candidate", "checkpoints"]
    root: Path
    receipt: Path
    archive_sha256: str
    source_revision: str | None
    adapter_sha256: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "root": str(self.root),
            "receipt": str(self.receipt),
            "archive_sha256": self.archive_sha256,
            "source_revision": self.source_revision,
            "adapter_sha256": self.adapter_sha256,
        }


def _regular_archive(path: Path) -> Path:
    if path.is_symlink():
        raise ColabExportError("Colab export must not be a symbolic link")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size == 0
        or metadata.st_size > _MAX_ARCHIVE_BYTES
    ):
        raise ColabExportError("Colab export must be one bounded regular file")
    return resolved


def _safe_member_path(name: str) -> tuple[str, ...]:
    pure = PurePosixPath(name)
    parts = pure.parts
    if (
        pure.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} or _SAFE_COMPONENT.fullmatch(part) is None for part in parts)
    ):
        raise ColabExportError(f"unsafe Colab archive path: {name!r}")
    return parts


def _validate_members(
    members: list[tarfile.TarInfo],
) -> tuple[str, Literal["candidate", "checkpoints"]]:
    if not members or len(members) > _MAX_MEMBERS:
        raise ColabExportError("Colab archive member count is outside its bound")
    top_levels: set[str] = set()
    names: set[str] = set()
    total = 0
    has_manifest = False
    for member in members:
        parts = _safe_member_path(member.name)
        normalized = "/".join(parts)
        if normalized in names:
            raise ColabExportError(f"duplicate Colab archive path: {normalized!r}")
        names.add(normalized)
        top_levels.add(parts[0])
        if not (member.isdir() or member.isfile()):
            raise ColabExportError("Colab archive contains a link or special file")
        if member.size < 0 or member.size > _MAX_FILE_BYTES:
            raise ColabExportError("Colab archive member exceeds its size bound")
        total += member.size
        if total > _MAX_EXPANDED_BYTES:
            raise ColabExportError("Colab archive expands beyond its total bound")
        if len(parts) == 2 and parts[1] == "run-manifest.json":
            has_manifest = True
    if len(top_levels) != 1:
        raise ColabExportError("Colab archive must contain exactly one top-level directory")
    top_level = next(iter(top_levels))
    kind: Literal["candidate", "checkpoints"] = "candidate" if has_manifest else "checkpoints"
    return top_level, kind


def _extract(archive: tarfile.TarFile, members: list[tarfile.TarInfo], staging: Path) -> None:
    for member in members:
        parts = _safe_member_path(member.name)
        target = staging.joinpath(*parts)
        if member.isdir():
            target.mkdir(mode=0o700, parents=True, exist_ok=True)
            continue
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ColabExportError(f"unable to read Colab archive member: {member.name}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        finally:
            os.close(descriptor)
            source.close()
        if target.stat().st_size != member.size:
            raise ColabExportError(f"Colab archive member changed size: {member.name}")


def _json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ColabExportError(f"invalid Colab JSON control file: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ColabExportError(f"invalid Colab JSON: {path.name}") from error
    if not isinstance(payload, dict):
        raise ColabExportError(f"Colab JSON must be an object: {path.name}")
    return payload


def _verify_candidate(root: Path) -> tuple[str, str]:
    manifest = _json_object(root / "run-manifest.json")
    adapter = manifest.get("adapter")
    base = manifest.get("base_model")
    environment = manifest.get("environment")
    invariants = manifest.get("effective_training_invariants")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("version") != "v2.0-colab-candidate"
        or not isinstance(adapter, dict)
        or adapter.get("lora_only") is not True
        or adapter.get("finite") is not True
        or adapter.get("safe_serialization") is not True
        or not isinstance(base, dict)
        or base.get("id") != "ornith-ai/Ornith-1.5-9B"
        or base.get("revision") != "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
        or not isinstance(environment, dict)
        or not isinstance(invariants, dict)
        or invariants.get("checkpoint_storage") != "ephemeral"
    ):
        raise ColabExportError("candidate run-manifest violates the V2 accountless contract")
    source_revision = environment.get("git_revision")
    weights_sha256 = adapter.get("adapter_weights_sha256")
    config_sha256 = adapter.get("adapter_config_sha256")
    if (
        not isinstance(source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
        or not isinstance(weights_sha256, str)
        or _SHA256.fullmatch(weights_sha256) is None
        or not isinstance(config_sha256, str)
        or _SHA256.fullmatch(config_sha256) is None
    ):
        raise ColabExportError("candidate run-manifest contains invalid identity hashes")
    weights = root / "adapter/adapter_model.safetensors"
    config = root / "adapter/adapter_config.json"
    if (
        weights.is_symlink()
        or config.is_symlink()
        or not weights.is_file()
        or not config.is_file()
        or sha256_file(weights) != weights_sha256
        or sha256_file(config) != config_sha256
    ):
        raise ColabExportError("candidate adapter files differ from the run-manifest")
    forbidden = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".bin", ".pt", ".pth", ".onnx", ".gguf"}
    ]
    if forbidden:
        raise ColabExportError("candidate contains optimizer or unapproved weight formats")
    return source_revision, weights_sha256


def _inventory(root: Path) -> list[dict[str, str | int]]:
    files: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ColabExportError("imported Colab export contains a symbolic link")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return files


def import_colab_export(archive_path: Path, destination_root: Path) -> ImportedColabExport:
    """Import one downloaded candidate/checkpoint archive without overwriting local data."""

    source = _regular_archive(archive_path)
    archive_sha256 = sha256_file(source)
    destination_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination_root.is_symlink():
        raise ColabExportError("Colab import destination must not be a symbolic link")
    destination_root = destination_root.resolve(strict=True)
    with tarfile.open(source, mode="r:gz") as archive:
        members = archive.getmembers()
        top_level, kind = _validate_members(members)
        kind_root = destination_root / kind
        kind_root.mkdir(mode=0o700, exist_ok=True)
        if kind_root.is_symlink() or kind_root.resolve(strict=True).parent != destination_root:
            raise ColabExportError("Colab import kind directory is outside its root")
        destination = kind_root / f"{top_level}-{archive_sha256[:12]}"
        imported = destination / top_level
        receipt_path = imported / "import-receipt.json"
        if destination.exists():
            if destination.is_symlink() or imported.is_symlink():
                raise ColabExportError("existing Colab import must not contain directory links")
            receipt = _json_object(receipt_path)
            if receipt.get("archive_sha256") != archive_sha256 or receipt.get("kind") != kind:
                raise ColabExportError("existing Colab import has another identity")
            existing_source = receipt.get("source_revision")
            existing_adapter = receipt.get("adapter_sha256")
            if existing_source is not None and not isinstance(existing_source, str):
                raise ColabExportError("existing Colab import source revision is invalid")
            if existing_adapter is not None and not isinstance(existing_adapter, str):
                raise ColabExportError("existing Colab import adapter hash is invalid")
            return ImportedColabExport(
                kind=kind,
                root=imported,
                receipt=receipt_path,
                archive_sha256=archive_sha256,
                source_revision=existing_source,
                adapter_sha256=existing_adapter,
            )
        staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
        staging.mkdir(mode=0o700, parents=True, exist_ok=False)
        try:
            _extract(archive, members, staging)
            imported_root = staging / top_level
            source_revision: str | None = None
            adapter_sha256: str | None = None
            if kind == "candidate":
                source_revision, adapter_sha256 = _verify_candidate(imported_root)
            receipt = {
                "schema_version": 1,
                "kind": kind,
                "archive_filename": source.name,
                "archive_sha256": archive_sha256,
                "source_revision": source_revision,
                "adapter_sha256": adapter_sha256,
                "files": _inventory(imported_root),
            }
            write_json_exclusive(imported_root / "import-receipt.json", receipt)
            staging.rename(destination)
        except Exception:
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging, ignore_errors=True)
            raise
    return ImportedColabExport(
        kind=kind,
        root=imported,
        receipt=imported / "import-receipt.json",
        archive_sha256=archive_sha256,
        source_revision=source_revision,
        adapter_sha256=adapter_sha256,
    )
