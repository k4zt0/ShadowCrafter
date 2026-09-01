"""Local SSH client for the immutable detached-worker protocol."""

from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass
from pathlib import Path

from shadowcrafter.automation.io import atomic_write_bytes
from shadowcrafter.automation.models import (
    RemoteHandle,
    RemoteJobDocument,
    RemoteJobStatus,
    SSHSpec,
)
from shadowcrafter.automation.process import ProcessRunner


class RemoteProtocolError(RuntimeError):
    """The remote worker returned invalid, inconsistent, or unsafe state."""


@dataclass(frozen=True, slots=True)
class RemoteInvocation:
    remote_argv: tuple[str, ...]
    ssh_argv: tuple[str, ...]


def _remote_invocation(
    spec: SSHSpec,
    revision: str,
    arguments: tuple[str, ...],
) -> RemoteInvocation:
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RemoteProtocolError("remote helper revision must be an exact Git SHA-1")
    helper = f"{spec.helper_root}/{revision}/scripts/automation/remote-worker.py"
    remote_argv = ("/root/ShadowCrafter/.venv/bin/python", helper, *arguments)
    # OpenSSH exposes one remote command string. Quoting every already-validated argv
    # element preserves exact argument boundaries without interpolating shell syntax.
    command = shlex.join(remote_argv)
    ssh_argv = (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ForwardAgent=no",
        "-o",
        "StrictHostKeyChecking=yes",
        spec.alias,
        command,
    )
    return RemoteInvocation(remote_argv, ssh_argv)


class RemoteWorkerClient:
    def __init__(self, root: Path, spec: SSHSpec, runner: ProcessRunner) -> None:
        self.root = root
        self.spec = spec
        self.runner = runner

    def _call(
        self,
        revision: str,
        arguments: tuple[str, ...],
        *,
        stdin: bytes | None = None,
        timeout: int = 300,
    ) -> bytes:
        invocation = _remote_invocation(self.spec, revision, arguments)
        return self.runner.run(
            invocation.ssh_argv,
            cwd=self.root,
            timeout_seconds=timeout,
            stdin=stdin,
        ).stdout

    def source_status(self, revision: str) -> bool:
        try:
            result = self._call(revision, ("source-status", "--revision", revision))
        except Exception:
            return False
        return result.decode().strip() == revision

    def install_and_launch(self, document: RemoteJobDocument) -> RemoteHandle:
        payload = document.model_dump_json().encode()
        self._call(
            document.git_revision,
            (
                "install",
                "--job-id",
                document.job_id,
                "--job-root",
                self.spec.job_root,
            ),
            stdin=payload,
        )
        raw = self._call(
            document.git_revision,
            (
                "launch",
                "--job-id",
                document.job_id,
                "--job-root",
                self.spec.job_root,
            ),
        )
        try:
            return RemoteHandle.model_validate_json(raw)
        except Exception as error:
            raise RemoteProtocolError("remote launch returned an invalid handle") from error

    def status(self, revision: str, job_id: str) -> RemoteJobStatus:
        raw = self._call(
            revision,
            ("status", "--job-id", job_id, "--job-root", self.spec.job_root),
        )
        try:
            return RemoteJobStatus.model_validate_json(raw)
        except Exception as error:
            raise RemoteProtocolError("remote worker returned invalid status") from error

    def fetch_evidence(
        self,
        revision: str,
        status: RemoteJobStatus,
        destination_by_index: dict[int, Path],
    ) -> None:
        if status.status != "succeeded":
            raise RemoteProtocolError("evidence is available only after remote success")
        expected = {entry.index: entry for entry in status.evidence}
        if set(expected) != set(destination_by_index):
            raise RemoteProtocolError("local evidence map does not match frozen remote inventory")
        for index in sorted(expected):
            entry = expected[index]
            content = self._call(
                revision,
                (
                    "read-evidence",
                    "--job-id",
                    status.job_id,
                    "--job-root",
                    self.spec.job_root,
                    "--index",
                    str(index),
                ),
                timeout=900,
            )
            if len(content) != entry.size or hashlib.sha256(content).hexdigest() != entry.sha256:
                raise RemoteProtocolError("remote evidence changed after its success inventory")
            atomic_write_bytes(destination_by_index[index], content)
