"""Exact-argv subprocess boundary used by all local orchestration."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_MAX_CAPTURE_BYTES = 4 * 1024 * 1024
_DENIED_ENV_PARTS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "PRIVATE_KEY", "CREDENTIAL")


class ProcessError(RuntimeError):
    """An exact command failed or violated its execution contract."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes


class ProcessRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: int,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult: ...


class SubprocessRunner:
    """Run without a shell and keep captured output bounded."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: int,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        exact = tuple(argv)
        if not exact or any(not isinstance(item, str) or "\x00" in item for item in exact):
            raise ProcessError("argv must contain non-empty NUL-free strings")
        environment = os.environ.copy()
        if env is not None:
            for name, value in env.items():
                if any(part in name.upper() for part in _DENIED_ENV_PARTS):
                    raise ProcessError(f"secret-like environment override is prohibited: {name}")
                environment[name] = value
        try:
            completed = subprocess.run(  # noqa: S603 - exact argv, never a shell.
                exact,
                cwd=cwd,
                input=stdin,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                shell=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ProcessError(f"command could not complete: {exact[0]}") from error
        if len(completed.stdout) > _MAX_CAPTURE_BYTES or len(completed.stderr) > _MAX_CAPTURE_BYTES:
            raise ProcessError("command output exceeded its capture bound")
        result = CommandResult(exact, completed.returncode, completed.stdout, completed.stderr)
        if result.returncode != 0:
            raise ProcessError(f"command exited {result.returncode}: {exact[0]}")
        return result
