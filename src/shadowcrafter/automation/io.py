"""Bounded configuration/state I/O and a portable exclusive controller lock."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from shadowcrafter.automation.models import AutomationConfig, WorkflowState

_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_STATE_BYTES = 4 * 1024 * 1024
ModelT = TypeVar("ModelT", bound=BaseModel)


class AutomationIOError(RuntimeError):
    """Raised when durable automation metadata is unsafe or malformed."""


def _regular_file_bytes(path: Path, maximum: int, *, missing_ok: bool = False) -> bytes | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise AutomationIOError(f"required file does not exist: {path}") from None
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise AutomationIOError(f"expected a regular non-symlink file: {path}")
    if metadata.st_size > maximum:
        raise AutomationIOError(f"file exceeds its safety bound: {path}")
    before = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
    content = path.read_bytes()
    after_metadata = path.lstat()
    after = (
        after_metadata.st_dev,
        after_metadata.st_ino,
        after_metadata.st_size,
        after_metadata.st_mtime_ns,
    )
    if before != after:
        raise AutomationIOError(f"file changed while it was read: {path}")
    return content


def load_config(path: Path) -> tuple[AutomationConfig, str]:
    content = _regular_file_bytes(path, _MAX_CONFIG_BYTES)
    assert content is not None
    try:
        payload = yaml.safe_load(content)
        config = AutomationConfig.model_validate(payload)
    except Exception as error:
        raise AutomationIOError(f"invalid automation config: {error}") from error
    return config, hashlib.sha256(content).hexdigest()


def load_state(path: Path) -> WorkflowState | None:
    content = _regular_file_bytes(path, _MAX_STATE_BYTES, missing_ok=True)
    if content is None:
        return None
    try:
        return WorkflowState.model_validate_json(content)
    except Exception as error:
        raise AutomationIOError(f"invalid workflow state: {error}") from error


def atomic_write_bytes(path: Path, content: bytes, *, mode: int = 0o600) -> None:
    """Write and fsync a fresh file, then atomically replace the destination."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() and path.is_symlink():
        raise AutomationIOError(f"refusing to replace a symbolic link: {path}")
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, mode)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AutomationIOError(f"short write while creating {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_state(path: Path, state: WorkflowState) -> None:
    content = state.model_dump_json(indent=2).encode() + b"\n"
    if len(content) > _MAX_STATE_BYTES:
        raise AutomationIOError("workflow state exceeds its safety bound")
    atomic_write_bytes(path, content)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    atomic_write_bytes(path, content)


class ExclusiveLock(AbstractContextManager["ExclusiveLock"]):
    """Use advisory flock; fall back to atomic mkdir on platforms without it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._descriptor: int | None = None
        self._mkdir_path = path.with_suffix(path.suffix + ".d")

    def __enter__(self) -> ExclusiveLock:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(descriptor)
                raise AutomationIOError("another automation controller owns the lock") from None
            self._descriptor = descriptor
            return self
        except (AttributeError, ImportError):  # pragma: no cover - non-POSIX fallback
            try:
                self._mkdir_path.mkdir(mode=0o700)
            except FileExistsError:
                raise AutomationIOError("another automation controller owns the lock") from None
            return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None
        else:  # pragma: no cover - non-POSIX fallback
            self._mkdir_path.rmdir()


def sha256_file(path: Path, *, maximum: int = 256 * 1024 * 1024) -> tuple[int, str]:
    content = _regular_file_bytes(path, maximum)
    assert content is not None
    return len(content), hashlib.sha256(content).hexdigest()
