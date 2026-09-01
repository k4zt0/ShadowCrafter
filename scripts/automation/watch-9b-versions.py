#!/usr/bin/env python3
"""Publish each completed private 9B version and enqueue the next one until 95%."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token

from shadowcrafter.automation.iterations import decide_quality, version_for
from shadowcrafter.automation.models import (
    CommandSpec,
    EnvironmentVariable,
    RemoteEvidence,
    RemoteJobDocument,
    RemoteJobStatus,
)
from shadowcrafter.release.remote_huggingface import publish_remote_experimental_release

_HOST = "capella.cloud.vessl.ai"
_PORT = 31044
_USER = "root"
_REMOTE_PROJECT = Path("/root/ShadowCrafter")
_SOURCE_ROOT = Path("/root/ShadowCrafter-source")
_JOB_ROOT = _REMOTE_PROJECT / "artifacts/automation"
_ITERATION_ROOT = _REMOTE_PROJECT / "artifacts/iterations/shadowcrafter-9b"
_REPO_ID = "KaztoRay/ShadowCrafter-9B"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SAFE_VERSION = re.compile(r"^v[1-9][0-9]*\.0$")
_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx"})


class VersionWatcherError(RuntimeError):
    """The local publication/continuation controller failed closed."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--first-source-revision", required=True)
    parser.add_argument("--first-dependency-job", required=True)
    parser.add_argument("--first-checkpoint", required=True)
    parser.add_argument("--start-version", type=int, default=1)
    parser.add_argument("--max-version", type=int, default=32)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("reports/private/version-loop/shadowcrafter-9b"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        content = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode()
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _key(path: Path) -> Path:
    if path.is_symlink():
        raise VersionWatcherError("SSH key must not be a symbolic link")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
        raise VersionWatcherError("SSH key must be a private regular file")
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


def _remote_call(
    key: Path,
    revision: str,
    arguments: tuple[str, ...],
    *,
    stdin: bytes | None = None,
    timeout: int = 300,
) -> bytes:
    helper = _SOURCE_ROOT / revision / "scripts/automation/remote-worker.py"
    remote = (str(_REMOTE_PROJECT / ".venv/bin/python"), str(helper), *arguments)
    command = shlex.join(remote)
    completed = subprocess.run(  # noqa: S603 - SSH argv and remote helper are fixed/pinned.
        (*_ssh_prefix(key), command),
        input=stdin,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise VersionWatcherError(
            f"remote worker call failed ({arguments[0]}, exit={completed.returncode})"
        )
    return completed.stdout


def _job(
    *,
    version: str,
    revision: str,
    source_revision: str,
    dependency: str | None,
    checkpoint: str | None,
) -> RemoteJobDocument:
    major = int(version.removeprefix("v").removesuffix(".0"))
    job_id = f"shadowcrafter-9b-v{major}-{revision[:7]}-r1"
    script = str(_SOURCE_ROOT / revision / "scripts/remote/run-9b-version.py")
    python = str(_REMOTE_PROJECT / ".venv/bin/python")
    environment = (
        EnvironmentVariable(name="PYTHONPATH", value=str(_SOURCE_ROOT / revision / "src")),
        EnvironmentVariable(name="PYTHONNOUSERSITE", value="1"),
    )
    commands: list[CommandSpec] = []
    if dependency is not None:
        commands.append(
            CommandSpec(
                id="wait-previous-training",
                argv=(
                    python,
                    str(_SOURCE_ROOT / revision / "scripts/remote/wait-for-job.py"),
                    "--job-id",
                    dependency,
                    "--job-root",
                    str(_JOB_ROOT),
                    "--timeout-seconds",
                    "345600",
                    "--poll-seconds",
                    "30",
                ),
                timeout_seconds=345900,
                environment=environment,
            )
        )
    argv = [
        python,
        script,
        "--version",
        version,
        "--source-revision",
        source_revision,
        "--protocol-revision",
        revision,
    ]
    if checkpoint is not None:
        argv.extend(("--checkpoint", checkpoint))
    commands.append(
        CommandSpec(
            id=f"run-version-{major}",
            argv=tuple(argv),
            timeout_seconds=604800,
            environment=environment,
        )
    )
    version_root = _ITERATION_ROOT / version
    return RemoteJobDocument(
        schema_version=1,
        job_id=job_id,
        git_revision=revision,
        model="9b",
        commands=tuple(commands),
        evidence=(
            RemoteEvidence(
                remote_path=str(version_root / "publication/ready.json"),
                local_path=f"reports/private/version-loop/shadowcrafter-9b/{version}/ready.json",
            ),
            RemoteEvidence(
                remote_path=str(version_root / "gate-report.json"),
                local_path=f"reports/private/version-loop/shadowcrafter-9b/{version}/gate-report.json",
            ),
            RemoteEvidence(
                remote_path=str(version_root / "evidence/release-evidence.json"),
                local_path=(
                    f"reports/private/version-loop/shadowcrafter-9b/{version}/"
                    "evidence/release-evidence.json"
                ),
            ),
            RemoteEvidence(
                remote_path=str(_JOB_ROOT / job_id / "worker.log"),
                local_path=(
                    f"reports/private/version-loop/shadowcrafter-9b/{version}/worker.log"
                ),
                max_bytes=64 * 1024 * 1024,
            ),
        ),
        min_remote_free_bytes=100 * 1024**3,
        max_runtime_seconds=1209600,
    )


def _launch(key: Path, document: RemoteJobDocument) -> None:
    payload = document.model_dump_json().encode()
    arguments = ("--job-id", document.job_id, "--job-root", str(_JOB_ROOT))
    _remote_call(key, document.git_revision, ("install", *arguments), stdin=payload)
    _remote_call(key, document.git_revision, ("launch", *arguments))


def _wait(key: Path, document: RemoteJobDocument, poll_seconds: int) -> RemoteJobStatus:
    if not 10 <= poll_seconds <= 600:
        raise VersionWatcherError("poll interval must be in [10, 600] seconds")
    while True:
        raw = _remote_call(
            key,
            document.git_revision,
            ("status", "--job-id", document.job_id, "--job-root", str(_JOB_ROOT)),
        )
        status = RemoteJobStatus.model_validate_json(raw)
        if status.status == "succeeded":
            return status
        if status.status == "failed":
            raise VersionWatcherError(status.failure or "remote version job failed")
        time.sleep(poll_seconds)


def _fetch_control(
    key: Path,
    document: RemoteJobDocument,
    status: RemoteJobStatus,
    local_version: Path,
) -> None:
    expected = {entry.index: entry for entry in status.evidence}
    if set(expected) != set(range(4)):
        raise VersionWatcherError("remote evidence inventory differs from the version contract")
    destinations = (
        local_version / "control/ready.json",
        local_version / "control/gate-report.json",
        local_version / "control/release-evidence.json",
        local_version / "control/worker.log",
    )
    for index, destination in enumerate(destinations):
        content = _remote_call(
            key,
            document.git_revision,
            (
                "read-evidence",
                "--job-id",
                document.job_id,
                "--job-root",
                str(_JOB_ROOT),
                "--index",
                str(index),
            ),
            timeout=900,
        )
        entry = expected[index]
        if len(content) != entry.size or hashlib.sha256(content).hexdigest() != entry.sha256:
            raise VersionWatcherError("remote control evidence failed checksum verification")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(destination, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())


def _scp_directory(key: Path, remote: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise VersionWatcherError(f"refusing to overwrite local evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    scp = shutil.which("scp")
    if scp is None:
        raise VersionWatcherError("scp is required for private evidence transfer")
    completed = subprocess.run(  # noqa: S603 - scp executable is resolved locally.
        (
            scp,
            "-F",
            "/dev/null",
            "-r",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-i",
            str(key),
            "-P",
            str(_PORT),
            f"{_USER}@{_HOST}:{remote}",
            str(destination),
        ),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=3600,
    )
    if completed.returncode != 0:
        raise VersionWatcherError("private evidence transfer failed")
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise VersionWatcherError("downloaded evidence contains a symbolic link")
        if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES:
            raise VersionWatcherError("model weights crossed the local custody boundary")


def _publish(
    *,
    key: Path,
    version: str,
    local_version: Path,
    gate_config: Path,
) -> dict[str, Any]:
    ready = json.loads((local_version / "control/ready.json").read_text(encoding="utf-8"))
    if ready.get("version") != version or not _SAFE_VERSION.fullmatch(version):
        raise VersionWatcherError("ready evidence belongs to another version")
    api = HfApi(token=get_token())
    before = api.model_info(_REPO_ID, files_metadata=False)
    if before.private is not True or not isinstance(before.sha, str):
        raise VersionWatcherError("Hugging Face repository is not private or has no parent SHA")
    publication = local_version / "publication"
    approvals = {
        name: {
            "path": f"publication/approvals/{name}.json",
            "sha256": ready["approvals"][name]["sha256"],
        }
        for name in ("artifact_integrity", "provenance", "license", "privacy", "safety")
    }
    manifest = {
        "schema_version": 1,
        "release_id": version,
        "repo_id": _REPO_ID,
        "release_tier": "Experimental Release",
        "visibility": "private",
        "commercial_release": False,
        "parent_commit": before.sha,
        "candidate_checkpoint_sha256": ready["candidate_checkpoint_sha256"],
        "remote_root": ready["remote_root"],
        "ssh": {"host": _HOST, "port": _PORT, "user": _USER},
        "files": ready["files"],
        "total_bytes": ready["total_bytes"],
        "evaluation": {
            "status": "measured",
            "evidence_manifest_sha256": ready["evaluation"]["evidence_manifest_sha256"],
            "reason": None,
        },
        "approvals": approvals,
    }
    manifest_path = local_version / "remote-release-manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    result = publish_remote_experimental_release(
        manifest_path,
        manifest_sha256=_sha256(manifest_path),
        ssh_key=key,
        evidence_path=local_version / "evidence/release-evidence.json",
        gate_config=gate_config,
    )
    api.create_tag(
        _REPO_ID,
        tag=version,
        revision=result.commit_sha,
        repo_type="model",
        exist_ok=False,
    )
    receipt = result.as_dict()
    receipt["tag"] = version
    _write_json_exclusive(local_version / "publication-receipt.json", receipt)
    if not publication.is_dir():
        raise VersionWatcherError("release approval directory was not transferred")
    return receipt


def main() -> int:
    args = _arguments()
    try:
        if _SHA40.fullmatch(args.revision) is None or _SHA40.fullmatch(
            args.first_source_revision
        ) is None:
            raise VersionWatcherError("source revisions must be exact Git SHA-1 values")
        if not 1 <= args.start_version <= args.max_version <= 128:
            raise VersionWatcherError("version range must be within [1, 128]")
        key = _key(args.ssh_key)
        root = args.local_root.resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        gate_config = Path("configs/eval/release-gates.yaml").resolve(strict=True)
        for index in range(args.start_version, args.max_version + 1):
            version = version_for(index)
            local_version = root / version
            if local_version.exists() or local_version.is_symlink():
                raise VersionWatcherError("local version state already exists; use --start-version")
            source_revision = args.first_source_revision if index == 1 else args.revision
            document = _job(
                version=version,
                revision=args.revision,
                source_revision=source_revision,
                dependency=args.first_dependency_job if index == 1 else None,
                checkpoint=args.first_checkpoint if index == 1 else None,
            )
            _launch(key, document)
            status = _wait(key, document, args.poll_seconds)
            local_version.mkdir(mode=0o700)
            _fetch_control(key, document, status, local_version)
            _scp_directory(key, _ITERATION_ROOT / version / "evidence", local_version / "evidence")
            # The evidence manifest fetched through the worker must equal the directory copy.
            if status.evidence[2].sha256 != _sha256(
                local_version / "evidence/release-evidence.json"
            ) or status.evidence[2].sha256 != _sha256(
                local_version / "control/release-evidence.json"
            ):
                raise VersionWatcherError("evidence directory copy changed after job completion")
            _scp_directory(
                key,
                _ITERATION_ROOT / version / "publication",
                local_version / "publication",
            )
            if status.evidence[1].sha256 != _sha256(
                local_version / "control/gate-report.json"
            ):
                raise VersionWatcherError("gate report changed after job completion")
            report = json.loads(
                (local_version / "control/gate-report.json").read_text(encoding="utf-8")
            )
            decision = decide_quality(report, version)
            receipt = _publish(
                key=key,
                version=version,
                local_version=local_version,
                gate_config=gate_config,
            )
            print(
                json.dumps(
                    {
                        "version": version,
                        "accuracy": decision.overall["accuracy"],
                        "target_met": decision.target_met,
                        "huggingface_commit": receipt["commit_sha"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if decision.target_met:
                _write_json_exclusive(
                    root / "completed.json",
                    {
                        "schema_version": 1,
                        "version": version,
                        "target": decision.target,
                        "overall": decision.overall,
                        "huggingface_commit": receipt["commit_sha"],
                    },
                )
                return 0
        raise VersionWatcherError("configured version ceiling reached before the 95% target")
    except Exception as error:
        print(f"version watcher stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
