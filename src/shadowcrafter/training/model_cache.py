"""Fail-closed provenance verification for pinned Hugging Face model caches.

The Hub commit SHA identifies metadata, not necessarily the bytes in a mutable
local cache.  This module binds every file in a complete snapshot to metadata
returned for that exact commit and emits a deterministic manifest that can be
pinned by the training operator.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from shadowcrafter.data.manifest import canonical_json_sha256, write_json_exclusive

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")
_CHUNK_SIZE = 8 * 1024 * 1024


class ModelCacheError(RuntimeError):
    """Raised when local base-model bytes cannot be proven from Hub metadata."""


@dataclass(frozen=True)
class PinnedCacheManifest:
    """An exact manifest byte stream supplied as training authority."""

    path: Path
    sha256: str
    payload: Mapping[str, Any]


ModelInfoLoader = Callable[[str, str], Any]


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _safe_repository_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ModelCacheError(f"unsafe repository filename in Hub metadata: {value!r}")
    if unicodedata.normalize("NFC", value) != value:
        raise ModelCacheError(f"non-canonical repository filename in Hub metadata: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ModelCacheError(f"unsafe repository filename in Hub metadata: {value!r}")
    if path.as_posix() != value:
        raise ModelCacheError(f"non-canonical repository filename in Hub metadata: {value!r}")
    return value


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ModelCacheError(f"{label} must be a non-negative integer")
    return value


def _load_model_info(
    model_id: str,
    revision: str,
    loader: ModelInfoLoader | None,
) -> Any:
    if loader is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:  # pragma: no cover - isolated runtime prerequisite
            raise ModelCacheError("huggingface-hub is required for cache verification") from error

        def loader(identifier: str, pinned_revision: str) -> Any:
            return HfApi().model_info(
                identifier,
                revision=pinned_revision,
                files_metadata=True,
            )

    try:
        info = loader(model_id, revision)
    except Exception as error:
        raise ModelCacheError(
            "could not obtain files_metadata for the pinned Hub revision; offline or partial "
            "metadata never authorizes training"
        ) from error
    if str(_field(info, "sha") or "") != revision:
        raise ModelCacheError(
            f"Hub resolved an unexpected revision: {_field(info, 'sha')!r}; expected {revision}"
        )
    return info


def _metadata_inventory(info: Any) -> dict[str, dict[str, Any]]:
    siblings = _field(info, "siblings")
    if not isinstance(siblings, (list, tuple)) or not siblings:
        raise ModelCacheError("Hub response contains no complete sibling files_metadata")
    inventory: dict[str, dict[str, Any]] = {}
    for sibling in siblings:
        path = _safe_repository_path(_field(sibling, "rfilename"))
        if path in inventory:
            raise ModelCacheError(f"duplicate repository filename in Hub metadata: {path}")
        size = _exact_nonnegative_int(_field(sibling, "size"), f"metadata size for {path}")
        blob_id = _field(sibling, "blob_id")
        if not isinstance(blob_id, str) or _GIT_SHA1.fullmatch(blob_id) is None:
            raise ModelCacheError(f"missing exact Git blob SHA-1 metadata for {path}")
        lfs = _field(sibling, "lfs")
        lfs_sha256: str | None = None
        if lfs is not None:
            lfs_sha = _field(lfs, "sha256")
            lfs_size = _exact_nonnegative_int(_field(lfs, "size"), f"LFS size for {path}")
            if not isinstance(lfs_sha, str) or _SHA256.fullmatch(lfs_sha) is None:
                raise ModelCacheError(f"missing exact LFS SHA-256 metadata for {path}")
            if lfs_size != size:
                raise ModelCacheError(f"Hub size and LFS size disagree for {path}")
            lfs_sha256 = lfs_sha
        if path.endswith((".safetensors", ".bin", ".pt", ".pth")) and lfs_sha256 is None:
            raise ModelCacheError(
                f"large model artifact {path} has no authoritative LFS SHA-256 metadata"
            )
        inventory[path] = {
            "size": size,
            "blob_id": blob_id,
            "lfs_sha256": lfs_sha256,
        }
    return inventory


def _snapshot_files(snapshot_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    try:
        root_mode = snapshot_root.lstat().st_mode
    except OSError as error:
        raise ModelCacheError(f"model snapshot is unavailable: {snapshot_root}") from error
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ModelCacheError("model snapshot root must be a real directory, not a symlink")
    for current, directory_names, file_names in os.walk(snapshot_root, followlinks=False):
        current_path = Path(current)
        for directory_name in directory_names:
            directory = current_path / directory_name
            if directory.is_symlink():
                raise ModelCacheError(f"snapshot contains a symbolic-link directory: {directory}")
        for file_name in file_names:
            path = current_path / file_name
            relative = path.relative_to(snapshot_root).as_posix()
            _safe_repository_path(relative)
            try:
                mode = path.lstat().st_mode
            except OSError as error:
                raise ModelCacheError(f"could not inspect cached file {relative}") from error
            if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                raise ModelCacheError(f"snapshot contains a special file: {relative}")
            files[relative] = path
    return files


def _hash_stable_file(path: Path, *, include_git_blob: bool) -> tuple[int, str, str | None]:
    """Hash one resolved regular file and reject replacement during the read."""
    try:
        resolved = path.resolve(strict=True)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise ModelCacheError(f"could not open cached model file safely: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ModelCacheError(f"cached model target is not a regular file: {path}")
        sha256 = hashlib.sha256()
        git_sha1 = hashlib.sha1(usedforsecurity=False) if include_git_blob else None
        if git_sha1 is not None:
            git_sha1.update(f"blob {before.st_size}\0".encode())
        while chunk := os.read(descriptor, _CHUNK_SIZE):
            sha256.update(chunk)
            if git_sha1 is not None:
                git_sha1.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ModelCacheError(f"cached model file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    if path.resolve(strict=True) != resolved:
        raise ModelCacheError(f"snapshot link changed while hashing: {path}")
    return before.st_size, sha256.hexdigest(), git_sha1.hexdigest() if git_sha1 else None


def verify_hf_cache_snapshot(
    snapshot_root: Path,
    *,
    model_id: str,
    revision: str,
    model_info: Any | None = None,
    model_info_loader: ModelInfoLoader | None = None,
) -> dict[str, Any]:
    """Stream-verify a complete pinned snapshot and return its deterministic manifest."""
    if _GIT_SHA1.fullmatch(revision) is None:
        raise ModelCacheError("model revision must be an exact lowercase Git SHA-1")
    info = model_info or _load_model_info(model_id, revision, model_info_loader)
    if str(_field(info, "sha") or "") != revision:
        raise ModelCacheError("provided Hub metadata does not match the pinned revision")
    metadata = _metadata_inventory(info)
    snapshot_root = snapshot_root.resolve(strict=True)
    expected_cache_name = f"models--{model_id.replace('/', '--')}"
    if (
        snapshot_root.name != revision
        or snapshot_root.parent.name != "snapshots"
        or snapshot_root.parent.parent.name != expected_cache_name
    ):
        raise ModelCacheError(
            "snapshot path is not the exact Hugging Face cache directory for the pinned model"
        )
    cache_root = snapshot_root.parent.parent.resolve(strict=True)
    files = _snapshot_files(snapshot_root)
    missing = sorted(set(metadata) - set(files))
    unexpected = sorted(set(files) - set(metadata))
    if missing or unexpected:
        raise ModelCacheError(
            f"cached snapshot is incomplete or has extra files; missing={missing[:12]}, "
            f"unexpected={unexpected[:12]}"
        )

    verified_files: list[dict[str, Any]] = []
    for relative in sorted(metadata):
        path = files[relative]
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ModelCacheError(f"cached snapshot entry is dangling: {relative}") from error
        if not resolved.is_relative_to(cache_root):
            raise ModelCacheError(f"cached snapshot entry escapes its model cache: {relative}")
        expected = metadata[relative]
        expected_lfs = expected["lfs_sha256"]
        size, sha256, git_sha1 = _hash_stable_file(
            path,
            include_git_blob=expected_lfs is None,
        )
        if size != expected["size"]:
            raise ModelCacheError(
                f"cached size mismatch for {relative}: expected {expected['size']}, observed {size}"
            )
        if expected_lfs is not None:
            if sha256 != expected_lfs:
                raise ModelCacheError(f"cached LFS SHA-256 mismatch for {relative}")
            algorithm = "sha256"
            digest = sha256
        else:
            if git_sha1 != expected["blob_id"]:
                raise ModelCacheError(f"cached Git blob SHA-1 mismatch for {relative}")
            algorithm = "git-blob-sha1"
            digest = str(git_sha1)
        verified_files.append(
            {
                "path": relative,
                "size": size,
                "identity": {"algorithm": algorithm, "digest": digest},
                "hub_blob_id": expected["blob_id"],
                "cache_target": resolved.relative_to(cache_root).as_posix(),
                "storage": "symlink" if path.is_symlink() else "regular",
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "snapshot_root": str(snapshot_root),
        "complete_repository_snapshot": True,
        "file_count": len(verified_files),
        "files_sha256": canonical_json_sha256(verified_files),
        "files": verified_files,
    }
    return manifest


def write_cache_manifest(path: Path, payload: Mapping[str, Any]) -> str:
    """Write verified cache evidence without ever replacing operator evidence."""
    if path.is_symlink() or path.exists():
        raise ModelCacheError(f"refusing to overwrite cache manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_json_exclusive(path, dict(payload))
    except (FileExistsError, OSError) as error:
        raise ModelCacheError(f"could not create cache manifest exclusively: {path}") from error
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pinned_cache_manifest(
    path: Path,
    expected_sha256: str,
    *,
    model_id: str,
    revision: str,
) -> PinnedCacheManifest:
    """Load exactly pinned manifest bytes and validate their self-consistency."""
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ModelCacheError("base cache manifest pin must be an exact lowercase SHA-256")
    if path.is_symlink() or not path.is_file():
        raise ModelCacheError("base cache manifest must be a regular non-symlink file")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ModelCacheError("could not read the pinned base cache manifest") from error
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ModelCacheError(
            f"base cache manifest SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    try:
        value: object = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCacheError("base cache manifest is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ModelCacheError("base cache manifest must contain a JSON object")
    if value.get("schema_version") != 1:
        raise ModelCacheError("unsupported base cache manifest schema")
    if value.get("model_id") != model_id or value.get("revision") != revision:
        raise ModelCacheError("base cache manifest does not describe the pinned model revision")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ModelCacheError("base cache manifest contains no verified files")
    if value.get("complete_repository_snapshot") is not True:
        raise ModelCacheError("base cache manifest does not prove a complete repository snapshot")
    if value.get("file_count") != len(files):
        raise ModelCacheError("base cache manifest file count is inconsistent")
    if value.get("files_sha256") != canonical_json_sha256(files):
        raise ModelCacheError("base cache manifest inventory hash is inconsistent")
    if hashlib.sha256(path.read_bytes()).hexdigest() != observed_sha256:
        raise ModelCacheError("base cache manifest changed while it was being validated")
    return PinnedCacheManifest(path.resolve(), observed_sha256, value)


__all__ = [
    "ModelCacheError",
    "ModelInfoLoader",
    "PinnedCacheManifest",
    "load_pinned_cache_manifest",
    "verify_hf_cache_snapshot",
    "write_cache_manifest",
]
