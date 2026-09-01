"""Strict schemas shared by the local controller and detached remote worker."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_IDENTIFIER = r"^[a-z0-9][a-z0-9-]{0,62}$"
_SHA256 = r"^[0-9a-f]{64}$"
_GIT_SHA = r"^[0-9a-f]{40}$"
_PLACEHOLDER = re.compile(
    r"\$\{(?:GIT_REVISION|RUN_ID|PROJECT_ROOT|MANIFEST_PATH|MANIFEST_SHA256|"
    r"EVIDENCE_PATH|SSH_KEY)\}"
)
_SECRET_NAME = re.compile(r"(?:TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE|CREDENTIAL|API_KEY)", re.I)
_SAFE_ENVIRONMENT_NAMES = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "LD_LIBRARY_PATH",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_CACHE",
        "TRANSFORMERS_OFFLINE",
        "XDG_CACHE_HOME",
    }
)
_REMOTE_EVIDENCE_ROOTS = (
    "/root/ShadowCrafter/artifacts/manifests/",
    "/root/ShadowCrafter/artifacts/preflight/",
    "/root/ShadowCrafter/artifacts/environment/",
    "/root/ShadowCrafter/artifacts/iterations/",
    "/root/ShadowCrafter/artifacts/checkpoints/",
    "/root/ShadowCrafter/artifacts/automation/",
    "/root/ShadowCrafter/reports/",
)
_REMOTE_EVIDENCE_SUFFIXES = (".json", ".jsonl", ".log", ".txt")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def _safe_relative(value: str, *, label: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must not be absolute or traverse directories")
    if path.as_posix() != value:
        raise ValueError(f"{label} must use canonical POSIX syntax")
    return value


def _safe_argument(value: str) -> str:
    if not value or len(value) > 4096 or any(ord(character) < 32 for character in value):
        raise ValueError("command arguments must be non-empty, bounded, printable strings")
    remainder = _PLACEHOLDER.sub("", value)
    if "${" in remainder:
        raise ValueError("command contains an unsupported placeholder")
    return value


class EnvironmentVariable(StrictModel):
    name: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    value: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_variable(self) -> EnvironmentVariable:
        if self.name not in _SAFE_ENVIRONMENT_NAMES or _SECRET_NAME.search(self.name):
            raise ValueError(f"environment variable is not allowlisted: {self.name}")
        _safe_argument(self.value)
        return self


class CommandSpec(StrictModel):
    """One exact argv invocation. Shell syntax is never interpreted."""

    id: str = Field(pattern=_IDENTIFIER)
    argv: tuple[Annotated[str, Field(min_length=1, max_length=4096)], ...] = Field(
        min_length=1,
        max_length=128,
    )
    cwd: str = "."
    timeout_seconds: int = Field(default=3600, ge=1, le=604800)
    environment: tuple[EnvironmentVariable, ...] = ()

    @model_validator(mode="after")
    def validate_command(self) -> CommandSpec:
        for argument in self.argv:
            _safe_argument(argument)
        if self.cwd != ".":
            _safe_relative(self.cwd, label="command cwd")
        if _SECRET_NAME.search(self.argv[0]):
            raise ValueError("command executable has a secret-like name")
        names = [variable.name for variable in self.environment]
        if len(names) != len(set(names)):
            raise ValueError("command environment names must be unique")
        return self


class FileExpectation(StrictModel):
    path: str
    max_bytes: int = Field(default=16 * 1024 * 1024, ge=1, le=256 * 1024 * 1024)
    sha256: str | None = Field(default=None, pattern=_SHA256)

    @model_validator(mode="after")
    def validate_path(self) -> FileExpectation:
        _safe_relative(self.path, label="expected file")
        if self.path.lower().endswith(_WEIGHT_SUFFIXES):
            raise ValueError("local expected files may not be model weights")
        return self


class CommitPolicy(StrictModel):
    paths: tuple[str, ...] = Field(min_length=1, max_length=256)
    message: str = Field(min_length=3, max_length=120)

    @model_validator(mode="after")
    def validate_paths(self) -> CommitPolicy:
        for path in self.paths:
            _safe_relative(path, label="commit allowlist path")
            if path.lower().endswith(_WEIGHT_SUFFIXES):
                raise ValueError("model weights may not enter the source repository")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("commit allowlist paths must be unique")
        if any(ord(character) < 32 for character in self.message):
            raise ValueError("commit message contains control characters")
        return self


class ResourceBudget(StrictModel):
    min_local_free_bytes: int = Field(default=2 * 1024**3, ge=0)
    min_remote_free_bytes: int = Field(default=20 * 1024**3, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=20)
    max_runtime_seconds: int = Field(default=172800, ge=60, le=1209600)


class RemoteEvidence(StrictModel):
    remote_path: str
    local_path: str
    max_bytes: int = Field(default=16 * 1024 * 1024, ge=1, le=256 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_paths(self) -> RemoteEvidence:
        _safe_argument(self.remote_path)
        rendered = _PLACEHOLDER.sub("safe", self.remote_path)
        if not rendered.startswith(_REMOTE_EVIDENCE_ROOTS):
            raise ValueError("remote evidence path is outside the metadata/report allowlist")
        if not rendered.endswith(_REMOTE_EVIDENCE_SUFFIXES) or rendered.lower().endswith(
            _WEIGHT_SUFFIXES
        ):
            raise ValueError("remote evidence must be a bounded text or JSON artifact")
        _safe_relative(self.local_path, label="local evidence path")
        if self.local_path.lower().endswith(_WEIGHT_SUFFIXES):
            raise ValueError("remote weights may never be materialized locally")
        return self


class RemoteJobSpec(StrictModel):
    model: Literal["9b"]
    commands: tuple[CommandSpec, ...] = Field(min_length=1, max_length=32)
    evidence: tuple[RemoteEvidence, ...] = Field(min_length=1, max_length=64)
    min_remote_free_bytes: int = Field(default=20 * 1024**3, ge=0)
    max_runtime_seconds: int = Field(default=172800, ge=60, le=1209600)

    @model_validator(mode="after")
    def validate_remote_job(self) -> RemoteJobSpec:
        command_ids = [command.id for command in self.commands]
        if len(command_ids) != len(set(command_ids)):
            raise ValueError("remote command ids must be unique")
        local_targets = [entry.local_path for entry in self.evidence]
        if len(local_targets) != len(set(local_targets)):
            raise ValueError("remote evidence destinations must be unique")
        return self


class TaskSpec(StrictModel):
    id: str = Field(pattern=_IDENTIFIER)
    kind: Literal["local", "remote_model"]
    depends_on: tuple[str, ...] = ()
    commands: tuple[CommandSpec, ...] = ()
    expected_files: tuple[FileExpectation, ...] = ()
    commit: CommitPolicy
    budget: ResourceBudget = ResourceBudget()
    deploy_command: CommandSpec | None = None
    remote_job: RemoteJobSpec | None = None
    finalize_commands: tuple[CommandSpec, ...] = ()
    publish_command: CommandSpec | None = None
    publish_receipt: FileExpectation | None = None
    release_manifest_path: str | None = None
    evaluation_evidence_path: str | None = None

    @model_validator(mode="after")
    def validate_task(self) -> TaskSpec:
        if self.id in self.depends_on or len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("task dependencies must be unique and may not reference itself")
        if self.kind == "local":
            if not self.commands:
                raise ValueError("local task requires at least one command")
            if (
                any(
                    item is not None
                    for item in (self.deploy_command, self.remote_job, self.publish_command)
                )
                or self.finalize_commands
                or self.publish_receipt is not None
            ):
                raise ValueError("local task contains remote-model-only fields")
            if self.release_manifest_path is not None or self.evaluation_evidence_path is not None:
                raise ValueError("local task contains release-only paths")
        else:
            if self.commands:
                raise ValueError("remote model task uses remote_job commands, not local commands")
            if self.deploy_command is None or self.remote_job is None:
                raise ValueError("remote model task requires deploy_command and remote_job")
            if self.publish_command is None or self.publish_receipt is None:
                raise ValueError("remote model task requires a local publisher and receipt")
            if self.release_manifest_path is None:
                raise ValueError("remote model task requires a local release manifest path")
            _safe_relative(self.release_manifest_path, label="release manifest path")
            if self.evaluation_evidence_path is not None:
                _safe_relative(self.evaluation_evidence_path, label="evaluation evidence path")
            expected = {item.path for item in self.expected_files}
            if self.publish_receipt.path not in expected:
                raise ValueError("publication receipt must also be an expected file")
        return self


class GitHubSpec(StrictModel):
    repository: Literal["Odytssey/ShadowCrafter"]
    expected_url: Literal[
        "https://github.com/Odytssey/ShadowCrafter.git",
        "git@github.com:Odytssey/ShadowCrafter.git",
    ]
    remote: Literal["origin"] = "origin"
    branch: Literal["main"] = "main"


class SSHSpec(StrictModel):
    alias: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    helper_root: Literal["/root/ShadowCrafter-source"] = "/root/ShadowCrafter-source"
    job_root: Literal["/root/ShadowCrafter/artifacts/automation"] = (
        "/root/ShadowCrafter/artifacts/automation"
    )


class AutomationConfig(StrictModel):
    schema_version: Literal[1]
    github: GitHubSpec
    ssh: SSHSpec
    state_path: str = ".automation/state.json"
    lock_path: str = ".automation/controller.lock"
    stop_path: str = ".automation/STOP"
    tasks: tuple[TaskSpec, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_workflow(self) -> AutomationConfig:
        for path in (self.state_path, self.lock_path, self.stop_path):
            _safe_relative(path, label="automation runtime path")
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task ids must be unique")
        known: set[str] = set()
        model_9b: str | None = None
        for task in self.tasks:
            missing = set(task.depends_on) - known
            if missing:
                raise ValueError(
                    f"task {task.id!r} has forward, missing, or cyclic dependencies: "
                    f"{sorted(missing)}"
                )
            known.add(task.id)
            if task.remote_job and task.remote_job.model == "9b":
                if model_9b is not None:
                    raise ValueError("workflow may contain only one 9B model task")
                model_9b = task.id
        return self


class WorkflowStatus(StrEnum):
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"


class RemoteHandle(StrictModel):
    job_id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]{0,126}$")
    backend: Literal["systemd", "nohup"]
    pid: int = Field(ge=1)
    process_start_ticks: int | None = Field(default=None, ge=1)
    unit: str | None = Field(default=None, pattern=r"^shadowcrafter-[a-z0-9-]{1,96}\.service$")


class TaskState(StrictModel):
    id: str = Field(pattern=_IDENTIFIER)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    git_revision: str | None = Field(default=None, pattern=_GIT_SHA)
    source_revision: str | None = Field(default=None, pattern=_GIT_SHA)
    remote_handle: RemoteHandle | None = None
    phase: Literal["start", "poll", "finalize", "publish", "commit"] = "start"
    published_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40,64}$")
    last_error: str | None = Field(default=None, max_length=1000)


class WorkflowState(StrictModel):
    schema_version: Literal[1]
    config_sha256: str = Field(pattern=_SHA256)
    status: WorkflowStatus = WorkflowStatus.ACTIVE
    created_at: datetime
    updated_at: datetime
    tasks: tuple[TaskState, ...]


class RemoteJobDocument(StrictModel):
    schema_version: Literal[1]
    job_id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]{0,126}$")
    git_revision: str = Field(pattern=_GIT_SHA)
    model: Literal["9b"]
    commands: tuple[CommandSpec, ...] = Field(min_length=1, max_length=32)
    evidence: tuple[RemoteEvidence, ...] = Field(min_length=1, max_length=64)
    min_remote_free_bytes: int = Field(ge=0)
    max_runtime_seconds: int = Field(ge=60, le=1209600)


class EvidenceInventoryEntry(StrictModel):
    index: int = Field(ge=0)
    remote_path: str
    size: int = Field(ge=0, le=256 * 1024 * 1024)
    sha256: str = Field(pattern=_SHA256)


class RemoteJobStatus(StrictModel):
    schema_version: Literal[1]
    job_id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    backend: Literal["systemd", "nohup"] | None = None
    pid: int | None = Field(default=None, ge=1)
    process_start_ticks: int | None = Field(default=None, ge=1)
    unit: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    current_step: str | None = None
    failure: str | None = Field(default=None, max_length=1000)
    evidence: tuple[EvidenceInventoryEntry, ...] = ()
