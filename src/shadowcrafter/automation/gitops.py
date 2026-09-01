"""Exact-path source commits and private GitHub verification."""

from __future__ import annotations

import http.client
import json
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from shadowcrafter.automation.models import CommitPolicy, GitHubSpec
from shadowcrafter.automation.process import ProcessRunner


class GitSafetyError(RuntimeError):
    """Git state cannot be safely attributed to the current automation task."""


def _decode_paths(content: bytes) -> tuple[str, ...]:
    try:
        values = [value.decode("utf-8") for value in content.split(b"\0") if value]
    except UnicodeDecodeError as error:
        raise GitSafetyError("Git returned a non-UTF-8 path") from error
    return tuple(values)


def _status_paths(content: bytes) -> tuple[str, ...]:
    records = [record for record in content.split(b"\0") if record]
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) < 4 or record[2:3] != b" ":
            raise GitSafetyError("unexpected Git porcelain status record")
        status = record[:2]
        try:
            paths.append(record[3:].decode("utf-8"))
        except UnicodeDecodeError as error:
            raise GitSafetyError("Git returned a non-UTF-8 path") from error
        if b"R" in status or b"C" in status:
            index += 1
            if index >= len(records):
                raise GitSafetyError("truncated Git rename/copy status record")
            try:
                paths.append(records[index].decode("utf-8"))
            except UnicodeDecodeError as error:
                raise GitSafetyError("Git returned a non-UTF-8 path") from error
        index += 1
    return tuple(paths)


def _covered(path: str, allowlist: tuple[str, ...]) -> bool:
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        return False
    return any(
        path == allowed or path.startswith(allowed.rstrip("/") + "/") for allowed in allowlist
    )


class GitPublisher:
    def __init__(
        self,
        root: Path,
        spec: GitHubSpec,
        runner: ProcessRunner,
        private_checker: Callable[[], bool] | None = None,
    ) -> None:
        self.root = root
        self.spec = spec
        self.runner = runner
        self.private_checker = private_checker or self._github_private_via_credential_helper

    def _run(self, *argv: str, timeout: int = 300) -> bytes:
        return self.runner.run(argv, cwd=self.root, timeout_seconds=timeout).stdout

    def head(self) -> str:
        revision = self._run("git", "rev-parse", "HEAD").decode().strip()
        valid_hex = all(character in "0123456789abcdef" for character in revision)
        if len(revision) != 40 or not valid_hex:
            raise GitSafetyError("Git HEAD is not an exact SHA-1")
        return revision

    def verify_private_destination(self) -> None:
        configured = self._run("git", "remote", "get-url", self.spec.remote).decode().strip()
        if configured != self.spec.expected_url:
            raise GitSafetyError("Git origin does not match the pinned private repository")
        if not self.private_checker():
            raise GitSafetyError("GitHub destination is not private")

    def _github_private_via_credential_helper(self) -> bool:
        """Resolve credentials on stdin/stdout and use them only in one HTTPS request."""

        result = self.runner.run(
            ("git", "credential", "fill"),
            cwd=self.root,
            timeout_seconds=30,
            stdin=b"protocol=https\nhost=github.com\n\n",
        )
        fields: dict[str, str] = {}
        for line in result.stdout.decode().splitlines():
            name, separator, value = line.partition("=")
            if separator and name in {"protocol", "host", "username", "password"}:
                fields[name] = value
        if (
            fields.get("protocol") != "https"
            or fields.get("host") != "github.com"
            or not fields.get("password")
        ):
            raise GitSafetyError("Git credential helper returned no scoped GitHub credential")
        connection = http.client.HTTPSConnection("api.github.com", timeout=30)
        try:
            connection.request(
                "GET",
                f"/repos/{self.spec.repository}",
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {fields['password']}",
                    "User-Agent": "ShadowCrafter-background-controller/1",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            response = connection.getresponse()
            content = response.read(1024 * 1024 + 1)
        finally:
            connection.close()
        if response.status != 200 or len(content) > 1024 * 1024:
            raise GitSafetyError("GitHub privacy verification request failed closed")
        try:
            payload = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GitSafetyError("GitHub returned invalid repository metadata") from error
        return (
            isinstance(payload, dict)
            and payload.get("full_name") == self.spec.repository
            and payload.get("private") is True
            and payload.get("visibility") == "private"
        )

    def commit_and_push(self, policy: CommitPolicy) -> str:
        """Refuse unrelated dirt, stage exact paths, commit, push, and verify remote HEAD."""

        self.verify_private_destination()
        status = self._run(
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        dirty = _status_paths(status)
        outside = sorted(path for path in dirty if not _covered(path, policy.paths))
        if outside:
            raise GitSafetyError(f"unrelated dirty paths block automation: {outside[:20]}")

        if dirty:
            self._run("git", "add", "--", *policy.paths)
            staged = _decode_paths(
                self._run("git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACDMRTUXB")
            )
            outside_staged = sorted(path for path in staged if not _covered(path, policy.paths))
            if outside_staged:
                raise GitSafetyError(f"staged paths escape task allowlist: {outside_staged[:20]}")
            if staged:
                self._run("git", "commit", "-m", policy.message, "--", *policy.paths)

        revision = self.head()
        self._run(
            "git",
            "push",
            self.spec.remote,
            f"HEAD:refs/heads/{self.spec.branch}",
            timeout=900,
        )
        remote = self._run(
            "git",
            "ls-remote",
            "--exit-code",
            self.spec.remote,
            f"refs/heads/{self.spec.branch}",
        ).decode()
        expected_line = f"{revision}\trefs/heads/{self.spec.branch}"
        if remote.strip() != expected_line:
            raise GitSafetyError("private GitHub branch does not match the committed task revision")
        return revision
