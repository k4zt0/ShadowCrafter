"""Atomic remote promotion of audited adapter artifacts into an immutable release tree."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shadowcrafter.automation.io import atomic_write_bytes

_SHA256 = r"^[0-9a-f]{64}$"
_RELEASE_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_ALLOWED_SUFFIXES = frozenset({".json", ".jinja", ".md", ".model", ".safetensors", ".txt"})
_DENIED_NAMES = re.compile(
    r"(?:optimizer|scheduler|trainer[_-]?state|rng[_-]?state|training_args|\.pickle$|\.pkl$)",
    re.I,
)
_MAX_FILE_BYTES = 2 * 1024**3
_MAX_TOTAL_BYTES = 4 * 1024**3
_MAX_SPEC_BYTES = 4 * 1024 * 1024


class PromotionError(RuntimeError):
    """Candidate artifacts cannot be promoted without weakening custody controls."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def _relative(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise ValueError("artifact path must be canonical POSIX syntax")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact path must be relative and traversal-free")
    return path


class PromotionFile(_StrictModel):
    source_path: str
    destination_path: str
    size: int = Field(ge=1, le=_MAX_FILE_BYTES)
    sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def validate_file(self) -> PromotionFile:
        source = _relative(self.source_path)
        destination = _relative(self.destination_path)
        if _DENIED_NAMES.search(self.source_path) or _DENIED_NAMES.search(self.destination_path):
            raise ValueError("training state and pickle artifacts are prohibited from releases")
        if source.suffix.lower() not in _ALLOWED_SUFFIXES:
            raise ValueError("source artifact suffix is not release-allowlisted")
        if destination.suffix.lower() not in _ALLOWED_SUFFIXES:
            raise ValueError("destination artifact suffix is not release-allowlisted")
        return self


class PromotionRequest(_StrictModel):
    schema_version: Literal[1]
    model: Literal["shadowcrafter-9b"]
    repo_id: Literal["KaztoRay/ShadowCrafter-9B"]
    release_id: str = Field(pattern=_RELEASE_ID)
    checkpoint_root: str
    remote_release_root: str
    candidate_checkpoint_sha256: str = Field(pattern=_SHA256)
    files: tuple[PromotionFile, ...] = Field(min_length=3, max_length=256)

    @model_validator(mode="after")
    def validate_request(self) -> PromotionRequest:
        checkpoint_prefix = f"/root/ShadowCrafter/artifacts/checkpoints/{self.model}/"
        release_prefix = f"/root/ShadowCrafter/artifacts/releases/{self.model}/"
        if not self.checkpoint_root.startswith(checkpoint_prefix):
            raise ValueError("checkpoint root is outside the model-specific namespace")
        expected_root = f"{release_prefix}{self.release_id}"
        if self.remote_release_root != expected_root:
            raise ValueError("release root is not the exact model/release-id destination")
        destinations = [entry.destination_path for entry in self.files]
        if destinations != sorted(destinations) or len(destinations) != len(set(destinations)):
            raise ValueError("promotion destinations must be sorted and unique")
        expected_prefix = f"releases/{self.release_id}/"
        for destination in destinations:
            if destination != "README.md" and not destination.startswith(expected_prefix):
                raise ValueError("release artifact is outside its namespaced Hub path")
        if destinations.count("README.md") != 1:
            raise ValueError("promotion requires exactly one root model card")
        scoped = [name for name in destinations if name.startswith(expected_prefix)]
        if not any(name.endswith(".safetensors") for name in scoped):
            raise ValueError("promotion contains no safetensors adapter")
        if not any(name.endswith(("adapter_config.json", "config.json")) for name in scoped):
            raise ValueError("promotion contains no adapter/model configuration")
        if sum(entry.size for entry in self.files) > _MAX_TOTAL_BYTES:
            raise ValueError("promotion exceeds the private publisher total-byte bound")
        return self


class PromotedFile(_StrictModel):
    path: str
    size: int
    sha256: str


class PromotionManifest(_StrictModel):
    schema_version: Literal[1]
    release_id: str
    repo_id: str
    model: str
    remote_root: str
    candidate_checkpoint_sha256: str
    total_bytes: int
    files: tuple[PromotedFile, ...]


def _open_source(root_fd: int, relative: PurePosixPath) -> int:
    current = os.dup(root_fd)
    try:
        for index, part in enumerate(relative.parts):
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if index < len(relative.parts) - 1:
                flags |= getattr(os, "O_DIRECTORY", 0)
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        return current
    except Exception:
        os.close(current)
        raise


def _copy_verified(source_fd: int, destination: Path, expected: PromotionFile) -> None:
    before = os.fstat(source_fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected.size:
        raise PromotionError(f"source size or file type changed: {expected.source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    digest = hashlib.sha256()
    size = 0
    try:
        while block := os.read(source_fd, 8 * 1024 * 1024):
            size += len(block)
            if size > expected.size:
                raise PromotionError(f"source grew while copying: {expected.source_path}")
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    after = os.fstat(source_fd)
    if (
        size != expected.size
        or digest.hexdigest() != expected.sha256
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise PromotionError(f"source failed stable checksum verification: {expected.source_path}")


def promote_release(request: PromotionRequest, output_manifest: Path) -> PromotionManifest:
    """Copy only pre-hashed approved files, fsync, and atomically expose a fresh release."""

    source_root = Path(request.checkpoint_root)
    source_metadata = source_root.lstat()
    if not stat.S_ISDIR(source_metadata.st_mode) or stat.S_ISLNK(source_metadata.st_mode):
        raise PromotionError("checkpoint root must be a real directory")
    target = Path(request.remote_release_root)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists() or target.is_symlink():
        raise PromotionError("release target already exists; releases are immutable")
    staging = parent / f".{target.name}.staging-{os.getpid()}"
    staging.mkdir(mode=0o700)
    source_fd = os.open(source_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        for entry in request.files:
            descriptor = _open_source(source_fd, _relative(entry.source_path))
            try:
                destination = staging.joinpath(*_relative(entry.destination_path).parts)
                _copy_verified(descriptor, destination, entry)
            finally:
                os.close(descriptor)
        for directory, _, _ in os.walk(staging, topdown=False):
            descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.rename(staging, target)
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        os.close(source_fd)
    manifest = PromotionManifest(
        schema_version=1,
        release_id=request.release_id,
        repo_id=request.repo_id,
        model=request.model,
        remote_root=request.remote_release_root,
        candidate_checkpoint_sha256=request.candidate_checkpoint_sha256,
        total_bytes=sum(entry.size for entry in request.files),
        files=tuple(
            PromotedFile(path=entry.destination_path, size=entry.size, sha256=entry.sha256)
            for entry in request.files
        ),
    )
    atomic_write_bytes(output_manifest, manifest.model_dump_json(indent=2).encode() + b"\n")
    return manifest


def load_promotion_request(path: Path, expected_sha256: str) -> PromotionRequest:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_size > _MAX_SPEC_BYTES
    ):
        raise PromotionError("promotion request must be a bounded regular file")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise PromotionError("promotion request SHA-256 mismatch")
    return PromotionRequest.model_validate_json(content)
