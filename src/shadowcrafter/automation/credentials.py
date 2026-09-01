"""Resolve credential locations without persisting credential values."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from shadowcrafter.automation.process import ProcessRunner


class CredentialError(RuntimeError):
    """A required local credential cannot be safely resolved."""


def resolve_ssh_identity(root: Path, alias: str, runner: ProcessRunner) -> Path:
    """Use a runtime override or OpenSSH config; return only a validated key path."""

    configured = os.environ.get("SHADOWCRAFTER_SSH_KEY")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    else:
        result = runner.run(
            ("ssh", "-G", alias),
            cwd=root,
            timeout_seconds=30,
        )
        for line in result.stdout.decode().splitlines():
            name, separator, value = line.partition(" ")
            if separator and name.lower() == "identityfile" and value:
                candidates.append(Path(value.replace("%d", str(Path.home()))).expanduser())
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            metadata = resolved.lstat()
        except OSError:
            continue
        if (
            stat.S_ISREG(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_mode & 0o077 == 0
            and not resolved.is_relative_to(root)
        ):
            return resolved
    raise CredentialError(
        "no regular mode-0600 SSH identity outside the repository was found at runtime"
    )
