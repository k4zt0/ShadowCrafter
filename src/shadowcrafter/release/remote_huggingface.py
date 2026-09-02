"""Memory-only remote artifact publication to the public ShadowCrafter model repository.

Model bytes are read from an immutable remote release directory through a fixed
SSH command, verified against a local allowlist manifest, and submitted in one
Hub commit. No model byte is written to the workstation filesystem and the Hub
token is never sent to the remote host.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from shadowcrafter.evaluation.gate import load_and_evaluate

_ALLOWED_REPOS = frozenset({"KaztoRay/ShadowCrafter-9B"})
_SHA256 = r"^[0-9a-f]{64}$"
_COMMIT = r"^[0-9a-f]{40,64}$"
_RELEASE_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_REMOTE_ROOT = re.compile(
    r"^/root/ShadowCrafter/artifacts/releases/"
    r"shadowcrafter-9b/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)
_SAFE_REMOTE_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_REVIEW_BYTES = 4 * 1024 * 1024
_MAX_CARD_BYTES = 2 * 1024 * 1024
_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
_SSH_STDERR_LIMIT = 64 * 1024
_SSH_TIMEOUT_SECONDS = 900
_CHUNK_BYTES = 1024 * 1024
_ALLOWED_SUFFIXES = frozenset({".json", ".jinja", ".md", ".model", ".safetensors", ".txt"})
_HF_ENDPOINT = "https://huggingface.co"
_REMOTE_FAMILY_BY_REPO = {
    "KaztoRay/ShadowCrafter-9B": "shadowcrafter-9b",
}

_REMOTE_READER_CODE = r"""
import json, os, stat, sys
request = json.loads(sys.stdin.buffer.read())
root = request["root"]
relative = request["path"]
expected = request["expected_size"]
maximum = request["maximum_size"]
if not isinstance(root, str) or not root.startswith("/root/ShadowCrafter/artifacts/releases/"):
    raise SystemExit(90)
if not isinstance(relative, str) or relative.startswith("/"):
    raise SystemExit(91)
parts = relative.split("/")
if not parts or any(not part or part in (".", "..") for part in parts):
    raise SystemExit(92)
if (
    not isinstance(expected, int)
    or not isinstance(maximum, int)
    or expected < 0
    or expected > maximum
):
    raise SystemExit(93)
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
root_parts = root.split("/")[1:]
if not root_parts or any(not part or part in (".", "..") for part in root_parts):
    raise SystemExit(97)
directory = os.open("/", flags | getattr(os, "O_DIRECTORY", 0))
current = None
try:
    for part in root_parts:
        child = os.open(
            part,
            flags | getattr(os, "O_DIRECTORY", 0),
            dir_fd=directory,
        )
        os.close(directory)
        directory = child
    current = directory
    for index, part in enumerate(parts):
        child_flags = flags
        if index < len(parts) - 1:
            child_flags |= getattr(os, "O_DIRECTORY", 0)
        child = os.open(part, child_flags, dir_fd=current)
        if current != directory:
            os.close(current)
        current = child
    metadata = os.fstat(current)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected:
        raise SystemExit(94)
    output = sys.stdout.buffer
    remaining = expected
    while remaining:
        block = os.read(current, min(1048576, remaining))
        if not block:
            raise SystemExit(95)
        output.write(block)
        remaining -= len(block)
    output.flush()
    final = os.fstat(current)
    if (final.st_size, final.st_mtime_ns) != (metadata.st_size, metadata.st_mtime_ns):
        raise SystemExit(96)
finally:
    try:
        if current is not None and current != directory:
            os.close(current)
    finally:
        os.close(directory)
""".strip()

# This string is constant. All remote root/path values travel as JSON on stdin,
# never as shell text or remote argv. Isolated stdlib Python avoids user/site hooks.
_REMOTE_COMMAND = "/usr/bin/python3 -I -S -c " + repr(_REMOTE_READER_CODE)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class LocalFileReference(_StrictModel):
    path: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=_SHA256)


class RemoteFileEntry(_StrictModel):
    path: str = Field(min_length=1, max_length=512)
    size: int = Field(ge=1, le=_MAX_FILE_BYTES)
    sha256: str = Field(pattern=_SHA256)


class SSHConnection(_StrictModel):
    host: Literal["capella.cloud.vessl.ai"]
    port: Literal[31044]
    user: Literal["root"]


class ApprovalReferences(_StrictModel):
    artifact_integrity: LocalFileReference
    provenance: LocalFileReference
    license: LocalFileReference
    privacy: LocalFileReference
    safety: LocalFileReference


class EvaluationPublication(_StrictModel):
    status: Literal["measured", "not-yet-evaluated"]
    evidence_manifest_sha256: str | None = Field(default=None, pattern=_SHA256)
    reason: str | None = Field(default=None, min_length=10, max_length=1000)

    @model_validator(mode="after")
    def validate_status(self) -> EvaluationPublication:
        if self.status == "measured":
            if self.evidence_manifest_sha256 is None or self.reason is not None:
                raise ValueError("measured evaluation requires only evidence_manifest_sha256")
        elif self.evidence_manifest_sha256 is not None or self.reason is None:
            raise ValueError("not-yet-evaluated requires a reason and no evidence hash")
        return self


class RemoteReleaseManifest(_StrictModel):
    schema_version: Literal[1]
    release_id: str = Field(pattern=_RELEASE_ID)
    repo_id: Literal["KaztoRay/ShadowCrafter-9B"]
    release_tier: Literal["Official Release"]
    visibility: Literal["public"]
    commercial_release: Literal[False]
    parent_commit: str = Field(pattern=_COMMIT)
    candidate_checkpoint_sha256: str = Field(pattern=_SHA256)
    remote_root: str = Field(min_length=1, max_length=512)
    ssh: SSHConnection
    files: list[RemoteFileEntry] = Field(min_length=3, max_length=256)
    total_bytes: int = Field(ge=1, le=_MAX_TOTAL_BYTES)
    evaluation: EvaluationPublication
    approvals: ApprovalReferences

    @model_validator(mode="after")
    def validate_inventory(self) -> RemoteReleaseManifest:
        family = _REMOTE_FAMILY_BY_REPO[self.repo_id]
        expected_root = f"/root/ShadowCrafter/artifacts/releases/{family}/{self.release_id}"
        if not _REMOTE_ROOT.fullmatch(self.remote_root) or self.remote_root != expected_root:
            raise ValueError("remote_root is outside the dedicated immutable release namespace")
        paths = [entry.path for entry in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("release file paths must be unique and sorted")
        if sum(entry.size for entry in self.files) != self.total_bytes:
            raise ValueError("release total_bytes does not equal its file inventory")
        prefix = f"releases/{self.release_id}/"
        for entry in self.files:
            pure = PurePosixPath(entry.path)
            components = entry.path.split("/")
            if (
                not _SAFE_REMOTE_PATH.fullmatch(entry.path)
                or pure.is_absolute()
                or any(
                    part in {"", ".", ".."}
                    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", part) is None
                    for part in components
                )
                or (entry.path != "README.md" and not entry.path.startswith(prefix))
                or Path(entry.path).suffix.lower() not in _ALLOWED_SUFFIXES
            ):
                raise ValueError(f"unsafe or unscoped release path: {entry.path!r}")
        if paths.count("README.md") != 1:
            raise ValueError("release inventory must contain exactly one root README.md")
        card_entry = next(entry for entry in self.files if entry.path == "README.md")
        if card_entry.size > _MAX_CARD_BYTES:
            raise ValueError("release model card exceeds its safety bound")
        scoped = [path.removeprefix(prefix) for path in paths if path.startswith(prefix)]
        if not any(path.endswith(".safetensors") for path in scoped):
            raise ValueError("release inventory contains no safetensors adapter/weight file")
        if not any(path.endswith(("adapter_config.json", "config.json")) for path in scoped):
            raise ValueError("release inventory contains no model or adapter configuration")
        return self


@dataclass(frozen=True, slots=True)
class PublishResult:
    repo_id: str
    commit_sha: str
    release_id: str
    manifest_sha256: str
    evaluation_status: str
    quality_target_met: bool | None
    total_bytes: int
    file_count: int

    def as_dict(self) -> dict[str, str | bool | int | None]:
        return {
            "repo_id": self.repo_id,
            "commit_sha": self.commit_sha,
            "release_id": self.release_id,
            "manifest_sha256": self.manifest_sha256,
            "evaluation_status": self.evaluation_status,
            "quality_target_met": self.quality_target_met,
            "total_bytes": self.total_bytes,
            "file_count": self.file_count,
            "visibility": "public",
            "release_tier": "Official Release",
        }


RemoteReader = Callable[[SSHConnection, str, RemoteFileEntry, Path], bytes]
HubStreamer = Callable[[str, str, str, str], Iterable[bytes]]
OperationFactory = Callable[[str, bytes], Any]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_json_object(content: bytes, description: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {description}: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(content, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _read_bounded_regular_file(path: Path, maximum: int, description: str) -> bytes:
    """Read a small local control file without following its final symlink."""

    if path.is_symlink():
        raise ValueError(f"{description} must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot safely open {description}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError(f"{description} must be a bounded regular file")
        content = bytearray()
        while len(content) < before.st_size:
            block = os.read(descriptor, min(_CHUNK_BYTES, before.st_size - len(content)))
            if not block:
                raise ValueError(f"{description} changed while it was read")
            content.extend(block)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ValueError(f"{description} changed while it was read")
        return bytes(content)
    finally:
        os.close(descriptor)


def _safe_local_reference(root: Path, reference: LocalFileReference) -> bytes:
    pure = PurePosixPath(reference.path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe local evidence path: {reference.path!r}")
    candidate = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"local evidence path contains a symlink: {reference.path!r}")
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if resolved_root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"local evidence path escapes its bundle: {reference.path!r}")
    content = _read_bounded_regular_file(
        resolved,
        _MAX_REVIEW_BYTES,
        f"local evidence file {reference.path!r}",
    )
    if _sha256_bytes(content) != reference.sha256:
        raise ValueError(f"local evidence checksum mismatch: {reference.path!r}")
    return content


def load_remote_release_manifest(
    path: Path,
    expected_sha256: str,
) -> tuple[RemoteReleaseManifest, str]:
    """Load an exact locally pinned remote release allowlist."""

    if re.fullmatch(_SHA256, expected_sha256) is None:
        raise ValueError("remote release manifest SHA-256 must be 64 lowercase hex characters")
    content = _read_bounded_regular_file(
        path,
        _MAX_MANIFEST_BYTES,
        "remote release manifest",
    )
    observed_sha256 = _sha256_bytes(content)
    if observed_sha256 != expected_sha256:
        raise ValueError("remote release manifest SHA-256 does not match its immutable pin")
    payload = _load_json_object(content, "remote release manifest")
    try:
        manifest = RemoteReleaseManifest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid remote release manifest: {exc}") from exc
    return manifest, observed_sha256


def _remote_inventory_sha256(manifest: RemoteReleaseManifest) -> str:
    payload = {
        "remote_root": manifest.remote_root,
        "files": [entry.model_dump(mode="json") for entry in manifest.files],
        "total_bytes": manifest.total_bytes,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(canonical)


def _verify_approvals(manifest: RemoteReleaseManifest, root: Path) -> dict[str, str]:
    results: dict[str, str] = {}
    inventory_sha256 = _remote_inventory_sha256(manifest)
    for name in ("artifact_integrity", "provenance", "license", "privacy", "safety"):
        reference = getattr(manifest.approvals, name)
        payload = _load_json_object(_safe_local_reference(root, reference), f"{name} approval")
        if (
            payload.get("schema_version") != 1
            or payload.get("review") != name
            or payload.get("passed") is not True
            or payload.get("repo_id") != manifest.repo_id
            or payload.get("release_id") != manifest.release_id
            or payload.get("candidate_checkpoint_sha256") != manifest.candidate_checkpoint_sha256
            or payload.get("remote_inventory_sha256") != inventory_sha256
            or payload.get("private_official_release_authorized") is not False
            or payload.get("public_release_authorized") is not True
        ):
            raise ValueError(f"{name} approval is missing, failed, or bound to another release")
        if name == "license" and (
            payload.get("commercial_release_authorized") is not False
            or payload.get("benchmark_material_sharing_authorized") is not False
        ):
            raise ValueError("license approval violates noncommercial benchmark constraints")
        results[name] = reference.sha256
    return results


def build_ssh_reader_argv(connection: SSHConnection, key_path: Path) -> tuple[str, ...]:
    if key_path.is_symlink():
        raise ValueError("SSH key must be a regular non-symlink file")
    resolved = key_path.resolve(strict=True)
    metadata = resolved.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("SSH key must be a regular non-symlink file")
    if metadata.st_mode & 0o077:
        raise ValueError("SSH key permissions must not grant group or other access")
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
        "IdentityAgent=none",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "ProxyCommand=none",
        "-o",
        "ProxyJump=none",
        "-o",
        "ControlMaster=no",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "RequestTTY=no",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "StrictHostKeyChecking=yes",
        "-i",
        str(resolved),
        "-p",
        str(connection.port),
        f"{connection.user}@{connection.host}",
        _REMOTE_COMMAND,
    )


def read_remote_file_via_ssh(
    connection: SSHConnection,
    remote_root: str,
    entry: RemoteFileEntry,
    key_path: Path,
) -> bytes:
    """Read one bounded remote file without putting its path in remote shell text."""

    argv = build_ssh_reader_argv(connection, key_path)
    request = json.dumps(
        {
            "root": remote_root,
            "path": entry.path,
            "expected_size": entry.size,
            "maximum_size": _MAX_FILE_BYTES,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    process = subprocess.Popen(  # noqa: S603 - fixed ssh executable and constant remote command.
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    try:
        process.stdin.write(request)
        process.stdin.close()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            output = bytearray()
            errors = bytearray()
            deadline = time.monotonic() + _SSH_TIMEOUT_SECONDS
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("remote artifact read timed out")
                events = selector.select(timeout=min(remaining, 1.0))
                if not events and process.poll() is not None:
                    break
                for key, _ in events:
                    block = os.read(key.fd, _CHUNK_BYTES)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stdout":
                        output.extend(block)
                        if len(output) > entry.size:
                            raise ValueError(
                                "remote host returned more bytes than its frozen manifest"
                            )
                    else:
                        errors.extend(block)
                        if len(errors) > _SSH_STDERR_LIMIT:
                            raise ValueError("remote SSH stderr exceeded its safety bound")
        return_code = process.wait(timeout=max(1.0, deadline - time.monotonic()))
        if return_code != 0:
            raise RuntimeError(f"remote artifact reader failed with exit code {return_code}")
        content = bytes(output)
        if len(content) != entry.size or _sha256_bytes(content) != entry.sha256:
            raise ValueError(f"remote artifact size or checksum mismatch: {entry.path}")
        return content
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if not stream.closed:
                stream.close()


def _evaluation_report(
    manifest: RemoteReleaseManifest,
    evidence_path: Path | None,
    gate_config: Path,
) -> Mapping[str, Any] | None:
    if manifest.evaluation.status == "not-yet-evaluated":
        if evidence_path is not None:
            raise ValueError("not-yet-evaluated release must not supply a metrics evidence file")
        return None
    if evidence_path is None:
        raise ValueError("measured release requires frozen evaluation evidence")
    evidence_content = _read_bounded_regular_file(
        evidence_path,
        _MAX_MANIFEST_BYTES,
        "evaluation evidence manifest",
    )
    if _sha256_bytes(evidence_content) != manifest.evaluation.evidence_manifest_sha256:
        raise ValueError("evaluation evidence hash does not match release manifest")
    result = load_and_evaluate(evidence_path, gate_config)
    if not result.passed or result.report is None:
        raise ValueError("evaluation integrity gate failed: " + "; ".join(result.failures))
    if (
        result.report.get("evidence_manifest_sha256")
        != manifest.evaluation.evidence_manifest_sha256
    ):
        raise ValueError("evaluation gate output is not bound to the frozen evidence manifest")
    candidate = result.report.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("model_id") != manifest.repo_id
        or candidate.get("checkpoint_sha256") != manifest.candidate_checkpoint_sha256
    ):
        raise ValueError("evaluation evidence belongs to another model or checkpoint")
    authorization = result.report.get("authorization")
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("model_publication_authorized") is not True
        or authorization.get("required_visibility") != "public"
        or authorization.get("public_publication_authorized") is not True
        or authorization.get("release_tier") != "Official Release"
        or authorization.get("commercial_use_permitted") is not False
    ):
        raise ValueError("evaluation evidence does not authorize public Official Release")
    if not isinstance(result.report.get("quality_target_met"), bool):
        raise ValueError("evaluation report must explicitly report quality_target_met")
    return result.report


def _front_matter(card: bytes) -> tuple[dict[str, Any], str]:
    try:
        text = card.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("model card must be UTF-8") from exc
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError("model card requires YAML front matter")
    header, body = text[4:].split("\n---\n", 1)
    metadata = yaml.safe_load(header)
    if not isinstance(metadata, dict):
        raise ValueError("model card front matter must be an object")
    return metadata, body


def _validate_model_card(
    card: bytes,
    manifest: RemoteReleaseManifest,
    report: Mapping[str, Any] | None,
) -> None:
    metadata, body = _front_matter(card)
    release = metadata.get("shadowcrafter_release")
    evaluation = metadata.get("shadowcrafter_evaluation")
    if not isinstance(release, dict) or not isinstance(evaluation, dict):
        raise ValueError("model card lacks machine-readable release/evaluation metadata")
    if (
        release.get("status") != "Official Release"
        or release.get("visibility") != "public"
        or release.get("commercial_use") is not False
        or release.get("release_id") != manifest.release_id
        or release.get("repository") != manifest.repo_id
        or release.get("candidate_checkpoint_sha256") != manifest.candidate_checkpoint_sha256
        or "Official Release" not in body
    ):
        raise ValueError("model card must prominently identify a public Official Release")
    if report is None:
        if (
            evaluation.get("status") != "not-yet-evaluated"
            or evaluation.get("accuracy") is not None
            or evaluation.get("balanced_accuracy") is not None
            or evaluation.get("macro_f1") is not None
            or evaluation.get("quality_target_met") is not None
            or not isinstance(manifest.evaluation.reason, str)
            or manifest.evaluation.reason not in body
        ):
            raise ValueError(
                "unevaluated model card must explicitly report null metrics and reason"
            )
        return
    benchmark = report.get("benchmark")
    overall = report.get("overall")
    metrics = overall.get("metrics") if isinstance(overall, Mapping) else None
    if not isinstance(benchmark, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError("evaluation report lacks benchmark or measured metrics")
    expected = {
        "status": "measured",
        "benchmark": benchmark.get("repository_id"),
        "revision": benchmark.get("upstream_revision"),
        "dataset_sha256": benchmark.get("dataset_sha256"),
        "sample_count": benchmark.get("sample_count"),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "quality_target_met": report.get("quality_target_met"),
    }
    if any(evaluation.get(key) != value for key, value in expected.items()):
        raise ValueError("model card evaluation values do not match frozen gate output")


def _default_operation(path: str, content: bytes) -> Any:
    from huggingface_hub import CommitOperationAdd

    return CommitOperationAdd(path_in_repo=path, path_or_fileobj=content)


def _local_hub_token() -> str:
    """Resolve Hugging Face credentials from its local standard auth sources."""

    from huggingface_hub import get_token

    token = get_token()
    if not isinstance(token, str) or not token:
        raise ValueError("local Hugging Face authentication is required for publication")
    return token


def _default_hub_stream(repo_id: str, revision: str, path: str, token: str) -> Iterator[bytes]:
    import httpx
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(
        repo_id,
        path,
        repo_type="model",
        revision=revision,
        endpoint=_HF_ENDPOINT,
    )
    headers = {"Accept-Encoding": "identity", "Authorization": f"Bearer {token}"}
    with (
        httpx.Client(follow_redirects=True, timeout=120.0) as client,
        client.stream("GET", url, headers=headers) as response,
    ):
        response.raise_for_status()
        content_encoding = response.headers.get("content-encoding")
        if content_encoding not in (None, "identity"):
            raise RuntimeError(
                "Hub returned content encoding that prevents byte-exact verification"
            )
        yield from response.iter_raw(_CHUNK_BYTES)


def _commit_sha(value: Any) -> str:
    for name in ("oid", "commit_id", "sha"):
        observed = getattr(value, name, None)
        if isinstance(observed, str) and re.fullmatch(_COMMIT, observed):
            return observed
    raise ValueError("Hugging Face commit response lacks an immutable commit SHA")


def publish_remote_official_release(
    manifest_path: Path,
    *,
    manifest_sha256: str,
    ssh_key: Path,
    gate_config: Path = Path("configs/eval/release-gates.yaml"),
    evidence_path: Path | None = None,
    api: Any | None = None,
    remote_reader: RemoteReader = read_remote_file_via_ssh,
    hub_streamer: HubStreamer = _default_hub_stream,
    operation_factory: OperationFactory = _default_operation,
) -> PublishResult:
    """Publish one manifest-verified public Official Release without local weight files."""

    manifest, observed_manifest_sha256 = load_remote_release_manifest(
        manifest_path,
        manifest_sha256,
    )
    if manifest.repo_id not in _ALLOWED_REPOS:
        raise ValueError("Hugging Face destination is not an approved ShadowCrafter repository")
    root = manifest_path.parent.resolve(strict=True)
    _verify_approvals(manifest, root)
    report = _evaluation_report(manifest, evidence_path, gate_config)

    hub_token = _local_hub_token()
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(endpoint=_HF_ENDPOINT, token=hub_token)

    before = api.model_info(manifest.repo_id, token=hub_token, files_metadata=False)
    if getattr(before, "private", None) is not False:
        raise ValueError("destination repository is not public before publication")
    if getattr(before, "sha", None) != manifest.parent_commit:
        raise ValueError("destination parent commit changed; refusing a raced publication")

    contents: dict[str, bytes] = {}
    observed_total = 0
    for entry in manifest.files:
        content = remote_reader(manifest.ssh, manifest.remote_root, entry, ssh_key)
        if len(content) != entry.size or _sha256_bytes(content) != entry.sha256:
            raise ValueError(f"remote reader returned unverified content: {entry.path}")
        observed_total += len(content)
        if observed_total > _MAX_TOTAL_BYTES:
            raise ValueError("release content exceeds the local memory-only total bound")
        contents[entry.path] = content
    if observed_total != manifest.total_bytes:
        raise ValueError("remote release bytes do not match manifest total")
    _validate_model_card(contents["README.md"], manifest, report)

    operations = [operation_factory(path, contents[path]) for path in sorted(contents)]
    commit = api.create_commit(
        manifest.repo_id,
        operations,
        repo_type="model",
        revision="main",
        parent_commit=manifest.parent_commit,
        commit_message=f"{manifest.release_tier}: {manifest.release_id}",
        commit_description=(
            f"remote_release_manifest_sha256: {observed_manifest_sha256}\n"
            f"candidate_checkpoint_sha256: {manifest.candidate_checkpoint_sha256}\n"
            f"evaluation_status: {manifest.evaluation.status}"
        ),
        token=hub_token,
        num_threads=1,
    )
    commit_sha = _commit_sha(commit)
    after = api.model_info(
        manifest.repo_id,
        revision=commit_sha,
        token=hub_token,
        files_metadata=False,
    )
    if getattr(after, "private", None) is not False or getattr(after, "sha", None) != commit_sha:
        raise RuntimeError("destination repository lost public visibility or commit identity")

    for entry in manifest.files:
        digest = hashlib.sha256()
        size = 0
        for block in hub_streamer(manifest.repo_id, commit_sha, entry.path, hub_token):
            if not isinstance(block, bytes):
                raise TypeError("Hub streamer must yield bytes")
            size += len(block)
            if size > entry.size:
                raise RuntimeError("Hub verification stream exceeded frozen file size")
            digest.update(block)
        if size != entry.size or digest.hexdigest() != entry.sha256:
            raise RuntimeError(f"post-upload Hub checksum mismatch: {entry.path}")

    quality_target_met = report.get("quality_target_met") if report is not None else None
    if quality_target_met is not None and not isinstance(quality_target_met, bool):
        raise TypeError("quality_target_met must be a boolean")
    return PublishResult(
        repo_id=manifest.repo_id,
        commit_sha=commit_sha,
        release_id=manifest.release_id,
        manifest_sha256=observed_manifest_sha256,
        evaluation_status=manifest.evaluation.status,
        quality_target_met=quality_target_met,
        total_bytes=manifest.total_bytes,
        file_count=len(manifest.files),
    )
