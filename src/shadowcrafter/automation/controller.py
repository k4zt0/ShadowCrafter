"""One-tick durable workflow controller for launchd supervision."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shadowcrafter.automation.credentials import resolve_ssh_identity
from shadowcrafter.automation.gitops import GitPublisher
from shadowcrafter.automation.io import (
    ExclusiveLock,
    atomic_write_bytes,
    load_config,
    load_state,
    sha256_file,
    write_state,
)
from shadowcrafter.automation.models import (
    CommandSpec,
    FileExpectation,
    RemoteJobDocument,
    TaskSpec,
    TaskState,
    TaskStatus,
    WorkflowState,
    WorkflowStatus,
)
from shadowcrafter.automation.process import ProcessError, ProcessRunner, SubprocessRunner
from shadowcrafter.automation.remote import RemoteWorkerClient


class AutomationError(RuntimeError):
    """A workflow transition failed closed."""


def _now() -> datetime:
    return datetime.now(UTC)


def _replace(value: str, context: dict[str, str]) -> str:
    rendered = value
    for name, replacement in context.items():
        rendered = rendered.replace("${" + name + "}", replacement)
    if "${" in rendered:
        raise AutomationError(f"command has an unresolved placeholder: {rendered!r}")
    return rendered


def _render_command(command: CommandSpec, context: dict[str, str]) -> CommandSpec:
    return command.model_copy(
        update={
            "argv": tuple(_replace(argument, context) for argument in command.argv),
            "cwd": _replace(command.cwd, context),
            "environment": tuple(
                variable.model_copy(update={"value": _replace(variable.value, context)})
                for variable in command.environment
            ),
        }
    )


class AutomationController:
    """Advance at most one bounded task phase and persist every transition."""

    def __init__(
        self,
        root: Path,
        config_path: Path,
        *,
        runner: ProcessRunner | None = None,
        git: GitPublisher | None = None,
        remote: RemoteWorkerClient | None = None,
    ) -> None:
        self.root = root.resolve(strict=True)
        self.config_path = config_path.resolve(strict=True)
        self.runner = runner or SubprocessRunner()
        self.config, self.config_sha256 = load_config(self.config_path)
        self.git = git or GitPublisher(self.root, self.config.github, self.runner)
        self.remote = remote or RemoteWorkerClient(self.root, self.config.ssh, self.runner)
        self.state_path = self._runtime_path(self.config.state_path)
        self.lock_path = self._runtime_path(self.config.lock_path)
        self.stop_path = self._runtime_path(self.config.stop_path)

    def _runtime_path(self, relative: str) -> Path:
        path = self.root.joinpath(*Path(relative).parts)
        if not path.resolve(strict=False).is_relative_to(self.root):
            raise AutomationError("automation runtime path escaped the repository")
        return path

    def _initial_state(self) -> WorkflowState:
        timestamp = _now()
        return WorkflowState(
            schema_version=1,
            config_sha256=self.config_sha256,
            status=WorkflowStatus.ACTIVE,
            created_at=timestamp,
            updated_at=timestamp,
            tasks=tuple(TaskState(id=task.id) for task in self.config.tasks),
        )

    def _load_or_initialize(self) -> WorkflowState:
        observed = load_state(self.state_path)
        if observed is None:
            observed = self._initial_state()
            write_state(self.state_path, observed)
        if observed.config_sha256 != self.config_sha256:
            raise AutomationError(
                "automation config changed after state initialization; "
                "create a reviewed new workflow"
            )
        observed_ids = tuple(task.id for task in observed.tasks)
        configured_ids = tuple(task.id for task in self.config.tasks)
        if observed_ids != configured_ids:
            raise AutomationError("persistent state task order differs from the immutable config")
        return observed

    def _save_task(
        self,
        state: WorkflowState,
        updated_task: TaskState,
        *,
        workflow_status: WorkflowStatus | None = None,
    ) -> WorkflowState:
        tasks = tuple(updated_task if item.id == updated_task.id else item for item in state.tasks)
        next_state = state.model_copy(
            update={
                "tasks": tasks,
                "status": workflow_status or state.status,
                "updated_at": _now(),
            }
        )
        write_state(self.state_path, next_state)
        return next_state

    def _run_command(self, command: CommandSpec, context: dict[str, str]) -> bytes:
        rendered = _render_command(command, context)
        cwd = self.root if rendered.cwd == "." else self._runtime_path(rendered.cwd)
        environment = {item.name: item.value for item in rendered.environment}
        return self.runner.run(
            rendered.argv,
            cwd=cwd,
            timeout_seconds=rendered.timeout_seconds,
            env=environment,
        ).stdout

    def _verify_expected(self, expected: tuple[FileExpectation, ...]) -> None:
        for item in expected:
            path = self._runtime_path(item.path)
            size, digest = sha256_file(path, maximum=item.max_bytes)
            if size > item.max_bytes or (item.sha256 is not None and digest != item.sha256):
                raise AutomationError(f"expected artifact failed verification: {item.path}")

    def _context(
        self,
        task: TaskSpec,
        current: TaskState,
        *,
        run_id: str | None = None,
    ) -> dict[str, str]:
        revision = current.source_revision or self.git.head()
        context = {
            "GIT_REVISION": revision,
            "RUN_ID": run_id or f"{task.id}-a{max(current.attempts, 1)}-{revision[:12]}",
            "PROJECT_ROOT": str(self.root),
        }
        if task.release_manifest_path is not None:
            manifest = self._runtime_path(task.release_manifest_path)
            context["MANIFEST_PATH"] = str(manifest)
            if manifest.exists():
                _, digest = sha256_file(manifest, maximum=16 * 1024 * 1024)
                context["MANIFEST_SHA256"] = digest
        if task.evaluation_evidence_path is not None:
            context["EVIDENCE_PATH"] = str(self._runtime_path(task.evaluation_evidence_path))
        return context

    def _task_by_id(self, task_id: str) -> TaskState:
        state = load_state(self.state_path)
        if state is None:
            raise AutomationError("workflow state disappeared")
        return next(task for task in state.tasks if task.id == task_id)

    def _mark_failure(
        self,
        state: WorkflowState,
        task: TaskSpec,
        current: TaskState,
        error: Exception,
        *,
        retry_remote: bool,
    ) -> WorkflowState:
        attempts = current.attempts + (0 if current.phase == "start" else 1)
        terminal = attempts >= task.budget.max_attempts
        update: dict[str, Any] = {
            "attempts": attempts,
            "last_error": str(error)[:1000],
            "finished_at": _now() if terminal else None,
        }
        if terminal:
            update["status"] = TaskStatus.FAILED
        elif retry_remote:
            update.update(
                {
                    "status": TaskStatus.PENDING,
                    "phase": "start",
                    "remote_handle": None,
                    "source_revision": None,
                    "started_at": None,
                }
            )
        else:
            update["status"] = TaskStatus.RUNNING
        return self._save_task(
            state,
            current.model_copy(update=update),
            workflow_status=WorkflowStatus.FAILED if terminal else None,
        )

    def _advance_local(
        self,
        state: WorkflowState,
        task: TaskSpec,
        current: TaskState,
    ) -> WorkflowState:
        attempt = current.attempts + 1
        running = current.model_copy(
            update={"status": TaskStatus.RUNNING, "attempts": attempt, "started_at": _now()}
        )
        state = self._save_task(state, running)
        try:
            if shutil.disk_usage(self.root).free < task.budget.min_local_free_bytes:
                raise AutomationError("local free space is below the task budget")
            context = self._context(task, running)
            for command in task.commands:
                self._run_command(command, context)
            self._verify_expected(task.expected_files)
            revision = self.git.commit_and_push(task.commit)
            complete = running.model_copy(
                update={
                    "status": TaskStatus.SUCCEEDED,
                    "finished_at": _now(),
                    "git_revision": revision,
                    "last_error": None,
                }
            )
            return self._save_task(state, complete)
        except Exception as error:
            terminal = attempt >= task.budget.max_attempts
            failed = running.model_copy(
                update={
                    "status": TaskStatus.FAILED if terminal else TaskStatus.PENDING,
                    "finished_at": _now() if terminal else None,
                    "last_error": str(error)[:1000],
                }
            )
            return self._save_task(
                state,
                failed,
                workflow_status=WorkflowStatus.FAILED if terminal else None,
            )

    def _render_remote_document(
        self,
        task: TaskSpec,
        current: TaskState,
        context: dict[str, str],
    ) -> RemoteJobDocument:
        assert task.remote_job is not None
        commands = tuple(_render_command(command, context) for command in task.remote_job.commands)
        evidence = tuple(
            item.model_copy(update={"remote_path": _replace(item.remote_path, context)})
            for item in task.remote_job.evidence
        )
        return RemoteJobDocument(
            schema_version=1,
            job_id=context["RUN_ID"],
            git_revision=context["GIT_REVISION"],
            model=task.remote_job.model,
            commands=commands,
            evidence=evidence,
            min_remote_free_bytes=max(
                task.budget.min_remote_free_bytes,
                task.remote_job.min_remote_free_bytes,
            ),
            max_runtime_seconds=min(
                task.budget.max_runtime_seconds,
                task.remote_job.max_runtime_seconds,
            ),
        )

    def _start_remote(
        self,
        state: WorkflowState,
        task: TaskSpec,
        current: TaskState,
    ) -> WorkflowState:
        assert task.deploy_command is not None
        attempt = current.attempts + 1
        revision = self.git.head()
        run_id = f"{task.id}-a{attempt}-{revision[:12]}"
        starting = current.model_copy(
            update={
                "status": TaskStatus.RUNNING,
                "attempts": attempt,
                "started_at": _now(),
                "source_revision": revision,
                "phase": "start",
            }
        )
        state = self._save_task(state, starting)
        context = self._context(task, starting, run_id=run_id)
        try:
            if shutil.disk_usage(self.root).free < task.budget.min_local_free_bytes:
                raise AutomationError("local free space is below the model-task budget")
            try:
                self._run_command(task.deploy_command, context)
            except (ProcessError, AutomationError):
                if not self.remote.source_status(revision):
                    raise
            document = self._render_remote_document(task, starting, context)
            handle = self.remote.install_and_launch(document)
            polling = starting.model_copy(
                update={"phase": "poll", "remote_handle": handle, "last_error": None}
            )
            return self._save_task(state, polling)
        except Exception as error:
            terminal = attempt >= task.budget.max_attempts
            failed = starting.model_copy(
                update={
                    "status": TaskStatus.FAILED if terminal else TaskStatus.PENDING,
                    "phase": "start",
                    "source_revision": None,
                    "remote_handle": None,
                    "finished_at": _now() if terminal else None,
                    "last_error": str(error)[:1000],
                }
            )
            return self._save_task(
                state,
                failed,
                workflow_status=WorkflowStatus.FAILED if terminal else None,
            )

    def _poll_remote(
        self,
        state: WorkflowState,
        task: TaskSpec,
        current: TaskState,
    ) -> WorkflowState:
        if current.remote_handle is None or current.source_revision is None:
            raise AutomationError("running remote task has no immutable handle/source revision")
        observed = self.remote.status(current.source_revision, current.remote_handle.job_id)
        if observed.status in {"pending", "running"}:
            return state
        if observed.status == "failed":
            return self._mark_failure(
                state,
                task,
                current,
                AutomationError(observed.failure or "remote worker failed"),
                retry_remote=True,
            )
        assert task.remote_job is not None
        destinations = {
            index: self._runtime_path(entry.local_path)
            for index, entry in enumerate(task.remote_job.evidence)
        }
        self.remote.fetch_evidence(current.source_revision, observed, destinations)
        return self._save_task(state, current.model_copy(update={"phase": "finalize"}))

    def _finalize_remote(
        self,
        state: WorkflowState,
        task: TaskSpec,
        current: TaskState,
    ) -> WorkflowState:
        try:
            context = self._context(task, current)
            for command in task.finalize_commands:
                self._run_command(command, context)
            return self._save_task(state, current.model_copy(update={"phase": "publish"}))
        except Exception as error:
            return self._mark_failure(state, task, current, error, retry_remote=False)

    def _publish_remote(
        self,
        state: WorkflowState,
        task: TaskSpec,
        current: TaskState,
    ) -> WorkflowState:
        assert task.publish_command is not None and task.publish_receipt is not None
        try:
            context = self._context(task, current)
            context["SSH_KEY"] = str(
                resolve_ssh_identity(self.root, self.config.ssh.alias, self.runner)
            )
            raw = self._run_command(task.publish_command, context)
            payload = json.loads(raw)
            if (
                not isinstance(payload, dict)
                or payload.get("visibility") != "private"
                or payload.get("release_tier") != "Experimental Release"
                or not isinstance(payload.get("commit_sha"), str)
            ):
                raise AutomationError("publisher did not return a verified private release receipt")
            receipt_path = self._runtime_path(task.publish_receipt.path)
            atomic_write_bytes(
                receipt_path,
                json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n",
            )
            updated = current.model_copy(
                update={"phase": "commit", "published_commit": payload["commit_sha"]}
            )
            return self._save_task(state, updated)
        except Exception as error:
            return self._mark_failure(state, task, current, error, retry_remote=False)

    def _commit_remote(
        self,
        state: WorkflowState,
        task: TaskSpec,
        current: TaskState,
    ) -> WorkflowState:
        try:
            self._verify_expected(task.expected_files)
            revision = self.git.commit_and_push(task.commit)
            succeeded = current.model_copy(
                update={
                    "status": TaskStatus.SUCCEEDED,
                    "finished_at": _now(),
                    "git_revision": revision,
                    "last_error": None,
                }
            )
            return self._save_task(state, succeeded)
        except Exception as error:
            return self._mark_failure(state, task, current, error, retry_remote=False)

    def _advance_remote(
        self,
        state: WorkflowState,
        task: TaskSpec,
        current: TaskState,
    ) -> WorkflowState:
        if current.status == TaskStatus.PENDING or current.phase == "start":
            return self._start_remote(state, task, current)
        if (
            current.started_at is not None
            and (_now() - current.started_at).total_seconds() > task.budget.max_runtime_seconds
        ):
            return self._mark_failure(
                state,
                task,
                current,
                AutomationError("model task exhausted its total runtime budget"),
                retry_remote=False,
            )
        if current.phase == "poll":
            return self._poll_remote(state, task, current)
        if current.phase == "finalize":
            return self._finalize_remote(state, task, current)
        if current.phase == "publish":
            return self._publish_remote(state, task, current)
        if current.phase == "commit":
            return self._commit_remote(state, task, current)
        raise AutomationError(f"unknown remote task phase: {current.phase}")

    def run_once(self) -> WorkflowState:
        with ExclusiveLock(self.lock_path):
            state = self._load_or_initialize()
            if state.status in {WorkflowStatus.COMPLETE, WorkflowStatus.FAILED}:
                return state
            if self.stop_path.exists():
                stopped_tasks = tuple(
                    task.model_copy(update={"status": TaskStatus.STOPPED})
                    if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}
                    else task
                    for task in state.tasks
                )
                stopped = state.model_copy(
                    update={
                        "status": WorkflowStatus.STOPPED,
                        "tasks": stopped_tasks,
                        "updated_at": _now(),
                    }
                )
                write_state(self.state_path, stopped)
                return stopped
            if state.status == WorkflowStatus.STOPPED:
                return state

            succeeded = {task.id for task in state.tasks if task.status == TaskStatus.SUCCEEDED}
            for task_spec, task_state in zip(self.config.tasks, state.tasks, strict=True):
                if task_state.status == TaskStatus.SUCCEEDED:
                    continue
                if not set(task_spec.depends_on).issubset(succeeded):
                    continue
                if task_spec.kind == "local":
                    state = self._advance_local(state, task_spec, task_state)
                else:
                    state = self._advance_remote(state, task_spec, task_state)
                break

            if all(task.status == TaskStatus.SUCCEEDED for task in state.tasks):
                state = state.model_copy(
                    update={"status": WorkflowStatus.COMPLETE, "updated_at": _now()}
                )
                write_state(self.state_path, state)
            return state

    def dry_run(self) -> dict[str, Any]:
        """Validate and describe the next transition without writing state or invoking commands."""

        state = load_state(self.state_path) or self._initial_state()
        succeeded = {task.id for task in state.tasks if task.status == TaskStatus.SUCCEEDED}
        next_task = next(
            (
                spec.id
                for spec, observed in zip(self.config.tasks, state.tasks, strict=True)
                if observed.status != TaskStatus.SUCCEEDED
                and set(spec.depends_on).issubset(succeeded)
            ),
            None,
        )
        return {
            "config_sha256": self.config_sha256,
            "workflow_status": state.status,
            "next_task": next_task,
            "state_exists": self.state_path.exists(),
            "would_mutate": False,
            "tasks": [
                {"id": spec.id, "kind": spec.kind, "status": observed.status}
                for spec, observed in zip(self.config.tasks, state.tasks, strict=True)
            ],
        }
