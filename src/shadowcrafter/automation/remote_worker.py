"""Detached Linux worker with immutable job specs and verified evidence export."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from shadowcrafter.automation.io import ExclusiveLock, atomic_write_bytes, sha256_file
from shadowcrafter.automation.models import (
    EvidenceInventoryEntry,
    RemoteHandle,
    RemoteJobDocument,
    RemoteJobStatus,
)

_JOB_ROOT = Path("/root/ShadowCrafter/artifacts/automation")
_PROJECT_ROOT = Path("/root/ShadowCrafter")
_MAX_SPEC_BYTES = 1024 * 1024
_LOG_MODE = 0o600


class WorkerError(RuntimeError):
    """A detached job failed a runtime or integrity boundary."""


def _now() -> datetime:
    return datetime.now(UTC)


def _job_directory(root: Path, job_id: str) -> Path:
    if root != _JOB_ROOT:
        raise WorkerError("job root is outside the dedicated automation namespace")
    directory = root / job_id
    if directory.parent != root:
        raise WorkerError("job id escaped its root")
    return directory


def _read_spec(directory: Path) -> RemoteJobDocument:
    path = directory / "spec.json"
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise WorkerError("job spec is not an immutable regular file")
    if metadata.st_size > _MAX_SPEC_BYTES:
        raise WorkerError("job spec exceeds its size bound")
    return RemoteJobDocument.model_validate_json(path.read_bytes())


def _read_status(directory: Path) -> RemoteJobStatus:
    path = directory / "status.json"
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise WorkerError("job status is not a regular file")
    return RemoteJobStatus.model_validate_json(path.read_bytes())


def _write_status(directory: Path, status: RemoteJobStatus) -> None:
    atomic_write_bytes(directory / "status.json", status.model_dump_json(indent=2).encode() + b"\n")


def _process_start_ticks(pid: int) -> int:
    content = Path(f"/proc/{pid}/stat").read_text()
    # The comm field can contain spaces and parentheses; fields after the last ')' are stable.
    fields = content.rsplit(")", 1)[1].strip().split()
    if len(fields) < 20:
        raise WorkerError("remote /proc status is truncated")
    return int(fields[19])


def install(job_id: str, root: Path) -> None:
    payload = sys.stdin.buffer.read(_MAX_SPEC_BYTES + 1)
    if len(payload) > _MAX_SPEC_BYTES:
        raise WorkerError("job spec exceeds its input bound")
    spec = RemoteJobDocument.model_validate_json(payload)
    if spec.job_id != job_id:
        raise WorkerError("job id does not match its immutable spec")
    directory = _job_directory(root, job_id)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    canonical = spec.model_dump_json(indent=2).encode() + b"\n"
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        existing = (directory / "spec.json").read_bytes()
        if existing != canonical:
            raise WorkerError("existing job id is bound to a different spec") from None
        return
    atomic_write_bytes(directory / "spec.json", canonical)
    _write_status(
        directory,
        RemoteJobStatus(schema_version=1, job_id=job_id, status="pending"),
    )


def _systemd_handle(directory: Path, spec: RemoteJobDocument) -> RemoteHandle | None:
    systemd_run = shutil.which("systemd-run")
    systemctl = shutil.which("systemctl")
    if systemd_run is None or systemctl is None:
        return None
    probe = subprocess.run(  # noqa: S603 - fixed executable and argv.
        (systemctl, "is-system-running"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=10,
    )
    if probe.returncode not in {0, 1}:
        return None
    safe_job = spec.job_id.replace(".", "-")
    if len(safe_job) > 80:
        safe_job = safe_job[:63] + "-" + hashlib.sha256(safe_job.encode()).hexdigest()[:16]
    unit = f"shadowcrafter-{safe_job}.service"
    argv = (
        systemd_run,
        "--unit",
        unit,
        "--collect",
        "--no-block",
        f"--property=RuntimeMaxSec={spec.max_runtime_seconds}",
        "--property=Type=exec",
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--job-id",
        spec.job_id,
        "--job-root",
        str(_JOB_ROOT),
    )
    completed = subprocess.run(  # noqa: S603 - exact argv, no shell.
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        return None
    for _ in range(50):
        show = subprocess.run(  # noqa: S603 - fixed executable and unit.
            (systemctl, "show", unit, "--property=MainPID", "--value"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        text = show.stdout.decode(errors="replace").strip()
        if text.isdecimal() and int(text) > 0:
            return RemoteHandle(job_id=spec.job_id, backend="systemd", pid=int(text), unit=unit)
        time.sleep(0.1)
    raise WorkerError("systemd accepted the unit but exposed no live MainPID")


def _nohup_handle(directory: Path, spec: RemoteJobDocument) -> RemoteHandle:
    nohup = shutil.which("nohup")
    if nohup is None:
        raise WorkerError("neither systemd-run nor nohup is available")
    log_path = directory / "worker.log"
    descriptor = os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
        _LOG_MODE,
    )
    try:
        process = subprocess.Popen(  # noqa: S603 - exact argv, no shell.
            (
                nohup,
                sys.executable,
                str(Path(__file__).resolve()),
                "run",
                "--job-id",
                spec.job_id,
                "--job-root",
                str(_JOB_ROOT),
            ),
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=descriptor,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(descriptor)
    start_ticks = _process_start_ticks(process.pid)
    os.kill(process.pid, 0)
    return RemoteHandle(
        job_id=spec.job_id,
        backend="nohup",
        pid=process.pid,
        process_start_ticks=start_ticks,
    )


def launch(job_id: str, root: Path) -> RemoteHandle:
    directory = _job_directory(root, job_id)
    spec = _read_spec(directory)
    status = _read_status(directory)
    if status.status == "succeeded":
        raise WorkerError("completed jobs are immutable and cannot be relaunched")
    if status.status == "running" and status.pid is not None and status.backend is not None:
        return RemoteHandle(
            job_id=job_id,
            backend=status.backend,
            pid=status.pid,
            process_start_ticks=status.process_start_ticks,
            unit=status.unit,
        )
    if status.status == "failed":
        raise WorkerError("failed job ids are immutable; retry with a new attempt id")
    launched_at = _now()
    _write_status(
        directory,
        RemoteJobStatus(
            schema_version=1,
            job_id=job_id,
            status="running",
            started_at=launched_at,
        ),
    )
    handle = _systemd_handle(directory, spec) or _nohup_handle(directory, spec)
    current = _read_status(directory)
    if current.status in {"succeeded", "failed"}:
        return handle
    _write_status(
        directory,
        RemoteJobStatus(
            schema_version=1,
            job_id=job_id,
            status="running",
            backend=handle.backend,
            pid=handle.pid,
            process_start_ticks=handle.process_start_ticks,
            unit=handle.unit,
            started_at=launched_at,
        ),
    )
    return handle


def _stable_evidence(spec: RemoteJobDocument) -> tuple[EvidenceInventoryEntry, ...]:
    entries: list[EvidenceInventoryEntry] = []
    for index, item in enumerate(spec.evidence):
        path = Path(item.remote_path)
        size, digest = sha256_file(path, maximum=item.max_bytes)
        entries.append(
            EvidenceInventoryEntry(
                index=index,
                remote_path=item.remote_path,
                size=size,
                sha256=digest,
            )
        )
    return tuple(entries)


def run_job(job_id: str, root: Path) -> None:
    directory = _job_directory(root, job_id)
    spec = _read_spec(directory)
    with ExclusiveLock(directory / "run.lock"):
        status = _read_status(directory)
        if status.status == "succeeded":
            return
        started = status.started_at or _now()
        common = {
            "schema_version": 1,
            "job_id": job_id,
            "status": "running",
            "backend": status.backend,
            "pid": os.getpid(),
            "process_start_ticks": _process_start_ticks(os.getpid()),
            "unit": status.unit,
            "started_at": started,
        }
        _write_status(directory, RemoteJobStatus.model_validate(common))
        try:
            if shutil.disk_usage(_PROJECT_ROOT).free < spec.min_remote_free_bytes:
                raise WorkerError("remote disk free space is below the immutable job budget")
            deadline = time.monotonic() + spec.max_runtime_seconds
            log_descriptor = os.open(
                directory / "worker.log",
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
                _LOG_MODE,
            )
            try:
                for command in spec.commands:
                    remaining = int(deadline - time.monotonic())
                    if remaining <= 0:
                        raise WorkerError("remote job exhausted its total runtime budget")
                    current = RemoteJobStatus.model_validate({**common, "current_step": command.id})
                    _write_status(directory, current)
                    cwd = _PROJECT_ROOT if command.cwd == "." else _PROJECT_ROOT / command.cwd
                    if not cwd.resolve(strict=True).is_relative_to(_PROJECT_ROOT):
                        raise WorkerError("remote command cwd escaped the project workspace")
                    environment = {
                        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                        "HOME": "/root",
                        "PYTHONNOUSERSITE": "1",
                    }
                    environment.update(
                        {variable.name: variable.value for variable in command.environment}
                    )
                    result = subprocess.run(  # noqa: S603 - schema-validated exact argv.
                        command.argv,
                        cwd=cwd,
                        stdin=subprocess.DEVNULL,
                        stdout=log_descriptor,
                        stderr=log_descriptor,
                        shell=False,
                        check=False,
                        timeout=min(command.timeout_seconds, remaining),
                        env=environment,
                    )
                    if result.returncode != 0:
                        raise WorkerError(
                            f"remote step {command.id!r} exited with code {result.returncode}"
                        )
            finally:
                os.close(log_descriptor)
            evidence = _stable_evidence(spec)
            _write_status(
                directory,
                RemoteJobStatus(
                    schema_version=1,
                    job_id=job_id,
                    status="succeeded",
                    backend=status.backend,
                    pid=os.getpid(),
                    process_start_ticks=_process_start_ticks(os.getpid()),
                    unit=status.unit,
                    started_at=started,
                    finished_at=_now(),
                    evidence=evidence,
                ),
            )
        except Exception as error:
            _write_status(
                directory,
                RemoteJobStatus(
                    schema_version=1,
                    job_id=job_id,
                    status="failed",
                    backend=status.backend,
                    pid=os.getpid(),
                    process_start_ticks=_process_start_ticks(os.getpid()),
                    unit=status.unit,
                    started_at=started,
                    finished_at=_now(),
                    failure=str(error)[:1000],
                ),
            )
            raise


def status(job_id: str, root: Path) -> RemoteJobStatus:
    directory = _job_directory(root, job_id)
    observed = _read_status(directory)
    if observed.status != "running" or observed.pid is None:
        return observed
    live = False
    if observed.backend == "systemd" and observed.unit:
        systemctl = shutil.which("systemctl")
        if systemctl:
            result = subprocess.run(  # noqa: S603 - fixed argv.
                (systemctl, "show", observed.unit, "--property=ActiveState", "--value"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
            live = result.returncode == 0 and result.stdout.decode().strip() in {
                "active",
                "activating",
            }
    elif observed.backend == "nohup" and observed.process_start_ticks is not None:
        try:
            os.kill(observed.pid, 0)
            live = _process_start_ticks(observed.pid) == observed.process_start_ticks
        except (OSError, ValueError, WorkerError):
            live = False
    if live:
        return observed
    # Avoid a launch/status race before the child has replaced the launcher's state.
    if observed.started_at and (_now() - observed.started_at).total_seconds() < 15:
        return observed
    failed = observed.model_copy(
        update={
            "status": "failed",
            "finished_at": _now(),
            "failure": "detached worker handle disappeared before a terminal status was written",
        }
    )
    _write_status(directory, failed)
    return failed


def read_evidence(job_id: str, root: Path, index: int) -> None:
    directory = _job_directory(root, job_id)
    spec = _read_spec(directory)
    observed = _read_status(directory)
    if observed.status != "succeeded" or index < 0 or index >= len(observed.evidence):
        raise WorkerError("requested evidence is not in a completed frozen inventory")
    entry = observed.evidence[index]
    declared = spec.evidence[index]
    if entry.remote_path != declared.remote_path:
        raise WorkerError("evidence inventory no longer matches the immutable job spec")
    path = Path(entry.remote_path)
    size, digest = sha256_file(path, maximum=declared.max_bytes)
    if size != entry.size or digest != entry.sha256:
        raise WorkerError("remote evidence changed after job completion")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        remaining = size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise WorkerError("short evidence read")
            sys.stdout.buffer.write(block)
            remaining -= len(block)
        sys.stdout.buffer.flush()
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source-status")
    source.add_argument("--revision", required=True)
    for name in ("install", "launch", "run", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--job-id", required=True)
        command.add_argument("--job-root", type=Path, required=True)
    read = subparsers.add_parser("read-evidence")
    read.add_argument("--job-id", required=True)
    read.add_argument("--job-root", type=Path, required=True)
    read.add_argument("--index", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "source-status":
            source_root = Path(__file__).resolve().parents[3]
            git = shutil.which("git")
            if git is None:
                raise WorkerError("git is unavailable for source verification")
            result = subprocess.run(  # noqa: S603 - fixed git query against own snapshot.
                (git, "-C", str(source_root), "rev-parse", "HEAD"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=30,
            )
            revision = result.stdout.decode().strip()
            if result.returncode != 0 or revision != args.revision:
                raise WorkerError("remote source snapshot revision is not exact")
            print(revision)
        elif args.command == "install":
            install(args.job_id, args.job_root)
        elif args.command == "launch":
            print(launch(args.job_id, args.job_root).model_dump_json())
        elif args.command == "run":
            run_job(args.job_id, args.job_root)
        elif args.command == "status":
            print(status(args.job_id, args.job_root).model_dump_json())
        elif args.command == "read-evidence":
            read_evidence(args.job_id, args.job_root, args.index)
        return 0
    except Exception as error:
        print(f"remote worker refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
