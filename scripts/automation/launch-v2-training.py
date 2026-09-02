#!/usr/bin/env python3
"""Install and launch the immutable ShadowCrafter-9B V2 training job."""

from __future__ import annotations

import argparse
import shlex
import stat
import subprocess
import sys
from pathlib import Path

from shadowcrafter.automation.models import (
    CommandSpec,
    EnvironmentVariable,
    RemoteEvidence,
    RemoteJobDocument,
)

_HOST = "capella.cloud.vessl.ai"
_PORT = 31044
_USER = "root"
_PROJECT = Path("/root/ShadowCrafter")
_SOURCE_ROOT = Path("/root/ShadowCrafter-source")
_JOB_ROOT = _PROJECT / "artifacts/automation"


class V2LaunchError(RuntimeError):
    """The V2 remote job could not be installed or launched."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    return parser.parse_args()


def _key(path: Path) -> Path:
    if path.is_symlink():
        raise V2LaunchError("SSH key must not be a symbolic link")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
        raise V2LaunchError("SSH key must be a private regular file with mode 0600 or narrower")
    return resolved


def _ssh_prefix(key: Path) -> tuple[str, ...]:
    return (
        "ssh",
        "-F",
        "/dev/null",
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
        "-i",
        str(key),
        "-p",
        str(_PORT),
        f"{_USER}@{_HOST}",
    )


def _remote_worker(
    key: Path,
    revision: str,
    arguments: tuple[str, ...],
    *,
    stdin: bytes | None = None,
) -> bytes:
    helper = _SOURCE_ROOT / revision / "scripts/automation/remote-worker.py"
    remote = (
        str(_PROJECT / ".venv/bin/python"),
        str(helper),
        *arguments,
    )
    completed = subprocess.run(  # noqa: S603 - fixed SSH executable and bounded remote argv.
        (*_ssh_prefix(key), shlex.join(remote)),
        input=stdin,
        capture_output=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip()
        raise V2LaunchError(message or "remote worker call failed")
    return completed.stdout


def _job(revision: str, attempt: int) -> RemoteJobDocument:
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise V2LaunchError("revision must be an exact lowercase 40-character Git SHA")
    if not 1 <= attempt <= 9:
        raise V2LaunchError("attempt must be between 1 and 9")
    short = revision[:7]
    job_id = f"shadowcrafter-9b-v2-train-{short}-r{attempt}"
    checkpoint = (
        _PROJECT
        / "artifacts/checkpoints/shadowcrafter-9b"
        / f"v2-expanded-juliet-20260902-{short}-r1"
    )
    report = (
        _PROJECT / "artifacts/iterations/shadowcrafter-9b/v2.0" / f"training-inputs-{short}.json"
    )
    environment = (
        EnvironmentVariable(name="PYTHONPATH", value=str(_SOURCE_ROOT / revision / "src")),
        EnvironmentVariable(name="PYTHONNOUSERSITE", value="1"),
    )
    return RemoteJobDocument(
        schema_version=1,
        job_id=job_id,
        git_revision=revision,
        model="9b",
        commands=(
            CommandSpec(
                id="prepare-and-train-v2",
                argv=(
                    str(_PROJECT / ".venv/bin/python"),
                    str(_SOURCE_ROOT / revision / "scripts/remote/run-v2-training.py"),
                    "--source-revision",
                    revision,
                ),
                timeout_seconds=604800,
                environment=environment,
            ),
        ),
        evidence=(
            RemoteEvidence(
                remote_path=str(checkpoint / "run-manifest.json"),
                local_path=(
                    "reports/private/version-loop/shadowcrafter-9b/v2.0/training-run-manifest.json"
                ),
            ),
            RemoteEvidence(
                remote_path=str(report),
                local_path=(
                    "reports/private/version-loop/shadowcrafter-9b/v2.0/training-inputs.json"
                ),
            ),
            RemoteEvidence(
                remote_path=str(_JOB_ROOT / job_id / "worker.log"),
                local_path=(
                    "reports/private/version-loop/shadowcrafter-9b/v2.0/training-worker.log"
                ),
                max_bytes=256 * 1024 * 1024,
            ),
        ),
        min_remote_free_bytes=250 * 1024**3,
        max_runtime_seconds=1209600,
    )


def main() -> int:
    args = _arguments()
    try:
        key = _key(args.ssh_key)
        document = _job(args.revision, args.attempt)
        common = ("--job-id", document.job_id, "--job-root", str(_JOB_ROOT))
        _remote_worker(
            key,
            args.revision,
            ("install", *common),
            stdin=document.model_dump_json().encode(),
        )
        launched = _remote_worker(key, args.revision, ("launch", *common))
        print(launched.decode().strip())
        return 0
    except (OSError, subprocess.SubprocessError, ValueError, V2LaunchError) as error:
        print(f"V2 launch refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
