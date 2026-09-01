"""Bounded Python AST assessment that never imports or executes target code."""

from __future__ import annotations

import ast
import hashlib
import hmac
import io
import json
import os
import stat
import tokenize
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from shadowcrafter.integrations.contracts import (
    Confidence,
    EvidenceReference,
    Severity,
    SourceLocation,
    WhiteBoxFinding,
    WhiteBoxScope,
)
from shadowcrafter.whitebox.models import (
    StaticEvidenceRecord,
    WhiteBoxAssessmentResult,
    WhiteBoxAuthorizationArtifact,
    validate_immutable_revision,
    validate_repository_uri,
    validate_scope_path,
)

MAX_AUTHORIZATION_BYTES = 65_536
MAX_FILES = 2_000
MAX_FILE_BYTES = 1_048_576
MAX_TOTAL_BYTES = 64 * 1_048_576
MAX_WALK_ENTRIES = 100_000
_IGNORED_METADATA_DIRECTORIES = frozenset({".git", ".hg", ".svn", "__pycache__"})


class WhiteBoxAuthorizationError(ValueError):
    """Authorization proof, source identity, or requested source scope is invalid."""


class WhiteBoxLimitError(WhiteBoxAuthorizationError):
    """The approved source snapshot exceeds a deterministic static-analysis limit."""


class _StrictJSONError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Rule:
    rule_id: str
    title: str
    summary: str
    severity: Severity
    confidence: Confidence
    cwe: str
    remediation: str


@dataclass(frozen=True, slots=True)
class _SourceFile:
    relative: PurePosixPath
    content: bytes


_RULES = {
    "SC-PY-001": _Rule(
        "SC-PY-001",
        "Shell-enabled process candidate",
        "Static analysis found a process call that enables or inherently uses a command shell. "
        "Untrusted input reaching this call could create command-injection risk.",
        Severity.HIGH,
        Confidence.HIGH,
        "CWE-78",
        "Avoid a command shell, pass a fixed argument vector, and validate any variable input.",
    ),
    "SC-PY-002": _Rule(
        "SC-PY-002",
        "Dynamic code evaluation candidate",
        "Static analysis found direct use of a dynamic code-evaluation primitive. "
        "Untrusted input reaching it could execute unintended code.",
        Severity.HIGH,
        Confidence.MEDIUM,
        "CWE-95",
        "Replace dynamic evaluation with a constrained parser and explicit data validation.",
    ),
    "SC-PY-003": _Rule(
        "SC-PY-003",
        "Unsafe YAML loader candidate",
        "Static analysis found a YAML loader that is not explicitly constrained to a safe loader.",
        Severity.MEDIUM,
        Confidence.MEDIUM,
        "CWE-502",
        "Use a safe YAML loader and treat parsed values as untrusted data.",
    ),
    "SC-PY-004": _Rule(
        "SC-PY-004",
        "Unsafe deserialization candidate",
        "Static analysis found an executable-object deserialization call. Untrusted serialized "
        "data could cause unintended code execution.",
        Severity.HIGH,
        Confidence.MEDIUM,
        "CWE-502",
        "Use a non-executable serialization format with a strict schema and integrity checks.",
    ),
    "SC-PY-005": _Rule(
        "SC-PY-005",
        "Insecure temporary-file candidate",
        "Static analysis found a temporary-name API with a race-prone creation pattern.",
        Severity.LOW,
        Confidence.HIGH,
        "CWE-377",
        "Create the temporary file atomically with restrictive permissions.",
    ),
    "SC-PY-006": _Rule(
        "SC-PY-006",
        "Hard-coded secret candidate",
        "Static analysis found a credential-like variable assigned a literal value. "
        "The value is not retained in evidence.",
        Severity.HIGH,
        Confidence.MEDIUM,
        "CWE-798",
        "Load secrets from an approved secret manager and rotate any exposed credential.",
    ),
}


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _shell_enabled(keyword: ast.keyword) -> bool:
    if keyword.arg != "shell":
        return False
    if isinstance(keyword.value, ast.Constant):
        return bool(keyword.value.value)
    return True


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.matches: list[tuple[str, int]] = []
        self.aliases: dict[str, str] = {}

    def collect_import_aliases(self, tree: ast.AST) -> None:
        """Resolve module aliases before rule traversal, independent of source order."""

        imports = sorted(
            (node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))),
            key=lambda node: getattr(node, "lineno", 0),
        )
        for node in imports:
            if isinstance(node, ast.Import):
                self.visit_Import(node)
            else:
                self.visit_ImportFrom(node)

    def _resolve_name(self, node: ast.AST) -> str:
        name = _qualified_name(node)
        head, separator, tail = name.partition(".")
        resolved = self.aliases.get(head, head)
        return f"{resolved}.{tail}" if separator else resolved

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast visitor API.
        for item in node.names:
            local = item.asname or item.name.partition(".")[0]
            self.aliases[local] = item.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - AST API.
        if node.level != 0 or node.module is None:
            return
        for item in node.names:
            if item.name == "*":
                continue
            self.aliases[item.asname or item.name] = f"{node.module}.{item.name}"

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast visitor API.
        name = self._resolve_name(node.func)
        subprocess_calls = {
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.Popen",
            "subprocess.run",
        }
        inherent_shell_calls = {
            "os.popen",
            "os.system",
            "subprocess.getoutput",
            "subprocess.getstatusoutput",
        }
        if name in inherent_shell_calls or (
            name in subprocess_calls and any(_shell_enabled(item) for item in node.keywords)
        ):
            self.matches.append(("SC-PY-001", node.lineno))
        if name in {"eval", "exec", "builtins.eval", "builtins.exec"}:
            self.matches.append(("SC-PY-002", node.lineno))
        if name in {"yaml.unsafe_load", "yaml.full_load"}:
            self.matches.append(("SC-PY-003", node.lineno))
        elif name == "yaml.load":
            loader_nodes = [item.value for item in node.keywords if item.arg == "Loader"]
            if len(node.args) >= 2:
                loader_nodes.append(node.args[1])
            loaders = {self._resolve_name(item) for item in loader_nodes}
            if not loaders.intersection(
                {"yaml.SafeLoader", "yaml.CSafeLoader", "SafeLoader", "CSafeLoader"}
            ):
                self.matches.append(("SC-PY-003", node.lineno))
        unsafe_deserializers = {
            "cPickle.load",
            "cPickle.loads",
            "cloudpickle.load",
            "cloudpickle.loads",
            "dill.load",
            "dill.loads",
            "joblib.load",
            "pickle.load",
            "pickle.loads",
        }
        if name in unsafe_deserializers:
            self.matches.append(("SC-PY-004", node.lineno))
        if name == "tempfile.mktemp":
            self.matches.append(("SC-PY-005", node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 - ast visitor API.
        if _is_credential_literal(node.value):
            for target in node.targets:
                if _credential_name(target):
                    self.matches.append(("SC-PY-006", node.lineno))
                    break
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802 - ast visitor API.
        if (
            node.value is not None
            and _is_credential_literal(node.value)
            and _credential_name(node.target)
        ):
            self.matches.append(("SC-PY-006", node.lineno))
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:  # noqa: N802 - ast visitor API.
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and _credential_text(key.value)
                and _is_credential_literal(value)
            ):
                self.matches.append(("SC-PY-006", node.lineno))
                break
        self.generic_visit(node)


def _credential_name(node: ast.AST) -> bool:
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return _credential_text(node.slice.value)
    name = _qualified_name(node).lower()
    return _credential_text(name)


def _credential_text(value: str) -> bool:
    lowered = value.lower()
    return any(term in lowered for term in ("password", "passwd", "secret", "api_key", "token"))


def _is_credential_literal(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (str, bytes))
        and len(node.value) >= 8
    )


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError(f"duplicate authorization key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise _StrictJSONError(f"non-standard authorization constant: {value}")


def _safe_relative_path(path: str) -> PurePosixPath:
    try:
        return PurePosixPath(validate_scope_path(path))
    except ValueError as exc:
        raise WhiteBoxAuthorizationError(str(exc)) from exc


def _beneath(path: PurePosixPath, prefix: PurePosixPath) -> bool:
    return prefix == PurePosixPath(".") or path == prefix or prefix in path.parents


def _path_selected(
    relative: PurePosixPath,
    included: tuple[PurePosixPath, ...],
    excluded: tuple[PurePosixPath, ...],
) -> bool:
    return any(_beneath(relative, prefix) for prefix in included) and not any(
        _beneath(relative, prefix) for prefix in excluded
    )


def _directory_relevant(
    relative: PurePosixPath,
    included: tuple[PurePosixPath, ...],
    excluded: tuple[PurePosixPath, ...],
) -> bool:
    if any(_beneath(relative, prefix) for prefix in excluded):
        return False
    return any(_beneath(relative, prefix) or _beneath(prefix, relative) for prefix in included)


def _bounded_read_regular_file(path: Path) -> bytes:
    if not getattr(os, "O_NOFOLLOW", 0):
        raise WhiteBoxAuthorizationError("platform lacks no-follow source-file support")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise WhiteBoxAuthorizationError("approved source file could not be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise WhiteBoxAuthorizationError("approved source entry is not a regular file")
        if before.st_size > MAX_FILE_BYTES:
            raise WhiteBoxLimitError("an approved Python source file exceeds the 1 MiB limit")
        chunks: list[bytes] = []
        remaining = MAX_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(content) > MAX_FILE_BYTES:
            raise WhiteBoxLimitError("an approved Python source file exceeds the 1 MiB limit")
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise WhiteBoxAuthorizationError("approved source changed while it was being read")
        return content
    finally:
        os.close(descriptor)


def _collect_python_sources(
    repository_root: Path,
    included_paths: tuple[str, ...],
    excluded_paths: tuple[str, ...],
) -> tuple[_SourceFile, ...]:
    root = repository_root.resolve(strict=True)
    if not root.is_dir():
        raise WhiteBoxAuthorizationError("repository root must be an existing directory")
    included = tuple(_safe_relative_path(value) for value in included_paths)
    excluded = tuple(_safe_relative_path(value) for value in excluded_paths)
    if not included:
        raise WhiteBoxAuthorizationError("at least one included source path is required")

    for prefix in included:
        candidate = root if prefix == PurePosixPath(".") else root.joinpath(*prefix.parts)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise WhiteBoxAuthorizationError(
                "included source path escapes repository root"
            ) from exc
        if not candidate.exists():
            raise WhiteBoxAuthorizationError("an included source path does not exist")
        current = root
        for part in () if prefix == PurePosixPath(".") else prefix.parts:
            current /= part
            if current.is_symlink():
                raise WhiteBoxAuthorizationError("included source path contains a symlink")

    def walk_error(error: OSError) -> None:
        raise WhiteBoxAuthorizationError("approved source tree could not be traversed") from error

    sources: list[_SourceFile] = []
    total_bytes = 0
    walked_entries = 0
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=walk_error,
        followlinks=False,
    ):
        directory_path = Path(directory)
        relative_directory = PurePosixPath(directory_path.relative_to(root).as_posix())
        walked_entries += len(directory_names) + len(file_names)
        if walked_entries > MAX_WALK_ENTRIES:
            raise WhiteBoxLimitError("approved source traversal exceeds the entry limit")
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            if name in _IGNORED_METADATA_DIRECTORIES:
                continue
            relative = relative_directory / name
            if not _directory_relevant(relative, included, excluded):
                continue
            candidate = directory_path / name
            if candidate.is_symlink():
                raise WhiteBoxAuthorizationError(
                    "included source tree contains a symlink directory"
                )
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            relative = relative_directory / name
            if relative.suffix != ".py" or not _path_selected(relative, included, excluded):
                continue
            if len(sources) >= MAX_FILES:
                raise WhiteBoxLimitError("approved Python source count exceeds 2000 files")
            path = directory_path / name
            if path.is_symlink():
                raise WhiteBoxAuthorizationError("included Python source is a symlink")
            content = _bounded_read_regular_file(path)
            total_bytes += len(content)
            if total_bytes > MAX_TOTAL_BYTES:
                raise WhiteBoxLimitError("approved Python source exceeds the 64 MiB total limit")
            sources.append(_SourceFile(relative=relative, content=content))

    sources.sort(key=lambda item: str(item.relative))
    if not sources:
        raise WhiteBoxAuthorizationError("approved scope contains no Python source files")
    return tuple(sources)


def _snapshot_sha256(sources: tuple[_SourceFile, ...]) -> str:
    digest = hashlib.sha256(b"ShadowCrafter-Python-Source-Snapshot-v1\x00")
    for source in sources:
        path = str(source.relative).encode("utf-8")
        content_digest = hashlib.sha256(source.content).digest()
        digest.update(len(path).to_bytes(4, "big"))
        digest.update(path)
        digest.update(len(source.content).to_bytes(8, "big"))
        digest.update(content_digest)
    return digest.hexdigest()


def compute_python_source_snapshot_sha256(
    repository_root: Path,
    *,
    included_paths: tuple[str, ...] = (".",),
    excluded_paths: tuple[str, ...] = (),
) -> str:
    """Compute the deterministic digest an owner records in an approval artifact."""

    return _snapshot_sha256(
        _collect_python_sources(repository_root, included_paths, excluded_paths)
    )


def _decode_python_source(raw: bytes) -> str:
    encoding, _consumed = tokenize.detect_encoding(io.BytesIO(raw).readline)
    return raw.decode(encoding)


class AuthorizedWhiteBoxAssessor:
    """Review one immutable, authorized Python source snapshot without execution."""

    def __init__(
        self,
        *,
        scope: WhiteBoxScope,
        authorization_artifact: bytes,
        repository_root: Path,
        repository_uri: str,
        revision: str,
    ) -> None:
        if not authorization_artifact or len(authorization_artifact) > MAX_AUTHORIZATION_BYTES:
            raise WhiteBoxAuthorizationError(
                "a bounded explicit authorization artifact is required"
            )
        for value in (*scope.included_paths, *scope.excluded_paths):
            _safe_relative_path(value)
        if not scope.included_paths:
            raise WhiteBoxAuthorizationError("at least one included source path is required")
        if len(set(scope.repository_uris)) != len(scope.repository_uris):
            raise WhiteBoxAuthorizationError("approved repository URIs must be unique")
        if len(set(scope.revisions)) != len(scope.revisions):
            raise WhiteBoxAuthorizationError("approved revisions must be unique")
        try:
            validate_repository_uri(repository_uri)
            validate_immutable_revision(revision)
        except ValueError as exc:
            raise WhiteBoxAuthorizationError(str(exc)) from exc

        digest = hashlib.sha256(authorization_artifact).hexdigest()
        if not hmac.compare_digest(digest, scope.authorization.evidence_sha256):
            raise WhiteBoxAuthorizationError("authorization artifact digest does not match scope")
        try:
            raw_artifact = json.loads(
                authorization_artifact.decode("utf-8"),
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
            artifact = WhiteBoxAuthorizationArtifact.model_validate(raw_artifact)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValidationError,
            _StrictJSONError,
        ) as exc:
            raise WhiteBoxAuthorizationError(
                "authorization artifact is not valid strict JSON"
            ) from exc

        proof = scope.authorization
        if (
            artifact.authorization_id != proof.authorization_id
            or artifact.scope_id != scope.scope_id
            or artifact.approved_by != proof.approved_by
        ):
            raise WhiteBoxAuthorizationError("authorization identity does not match scope")
        if artifact.valid_from != proof.valid_from or artifact.valid_until != proof.valid_until:
            raise WhiteBoxAuthorizationError("authorization validity does not match scope")
        current = datetime.now(UTC)
        if (
            proof.valid_from.tzinfo is None
            or proof.valid_until.tzinfo is None
            or not proof.valid_from <= current < proof.valid_until
        ):
            raise WhiteBoxAuthorizationError("authorization is not currently valid")
        if repository_uri not in scope.repository_uris or revision not in scope.revisions:
            raise WhiteBoxAuthorizationError("repository or revision is outside the approved scope")
        if artifact.repository_uri != repository_uri or artifact.revision != revision:
            raise WhiteBoxAuthorizationError("authorization is bound to another source identity")
        if artifact.included_paths != scope.included_paths:
            raise WhiteBoxAuthorizationError("authorized included paths do not match scope")
        if artifact.excluded_paths != scope.excluded_paths:
            raise WhiteBoxAuthorizationError("authorized excluded paths do not match scope")

        root = repository_root.resolve(strict=True)
        sources = _collect_python_sources(root, scope.included_paths, scope.excluded_paths)
        snapshot_digest = _snapshot_sha256(sources)
        if not hmac.compare_digest(snapshot_digest, artifact.python_source_snapshot_sha256):
            raise WhiteBoxAuthorizationError("local Python source snapshot is not authorized")

        self.scope = scope
        self.root = root
        self.repository_uri = repository_uri
        self.revision = revision
        self._artifact = artifact
        self._sources = sources
        self._evidence_key = hashlib.sha256(
            b"ShadowCrafter-WhiteBox-Evidence-v1\x00" + authorization_artifact
        ).digest()

    def _require_current_authorization(self) -> None:
        if not self._artifact.valid_from <= datetime.now(UTC) < self._artifact.valid_until:
            raise WhiteBoxAuthorizationError("authorization expired during static assessment")

    def assess(self) -> WhiteBoxAssessmentResult:
        self._require_current_authorization()
        started = datetime.now(UTC)
        findings: list[WhiteBoxFinding] = []
        evidence: list[StaticEvidenceRecord] = []
        files_reviewed = 0
        files_skipped = 0

        for source_file in self._sources:
            self._require_current_authorization()
            try:
                source = _decode_python_source(source_file.content)
                tree = ast.parse(source, filename=str(source_file.relative))
            except (LookupError, UnicodeDecodeError, SyntaxError):
                files_skipped += 1
                continue
            files_reviewed += 1
            lines = source.splitlines(keepends=True)
            visitor = _Visitor()
            visitor.collect_import_aliases(tree)
            visitor.visit(tree)
            for rule_id, line_number in sorted(set(visitor.matches)):
                rule = _RULES[rule_id]
                line = lines[line_number - 1].encode() if line_number <= len(lines) else b""
                evidence_id = hashlib.sha256(
                    (
                        f"{self.scope.scope_id}:{self.repository_uri}:{self.revision}:"
                        f"{source_file.relative}:{line_number}:{rule_id}"
                    ).encode()
                ).hexdigest()[:24]
                evidence_payload = b"\x00".join(
                    (
                        str(source_file.relative).encode(),
                        str(line_number).encode(),
                        rule_id.encode(),
                        line,
                    )
                )
                evidence_digest = hmac.new(
                    self._evidence_key,
                    evidence_payload,
                    digestmod=hashlib.sha256,
                ).hexdigest()
                evidence.append(
                    StaticEvidenceRecord(
                        evidence_id=evidence_id,
                        rule_id=rule_id,
                        path=str(source_file.relative),
                        start_line=line_number,
                        evidence_digest=evidence_digest,
                    )
                )
                reference = EvidenceReference(
                    evidence_id=evidence_id,
                    source_uri=f"evidence://white-box/{evidence_id}",
                    description=(
                        f"Static rule {rule_id} matched at the recorded source location; "
                        "source content is omitted and the digest is authorization-keyed."
                    ),
                    sha256=evidence_digest,
                    observed_at=started,
                )
                findings.append(
                    WhiteBoxFinding(
                        finding_id=f"wb-{evidence_id}",
                        scope_id=self.scope.scope_id,
                        title=rule.title,
                        summary=rule.summary,
                        location=SourceLocation(
                            repository_uri=self.repository_uri,
                            revision=self.revision,
                            path=str(source_file.relative),
                            start_line=line_number,
                            end_line=line_number,
                        ),
                        severity=rule.severity,
                        confidence=rule.confidence,
                        cwe_candidates=(rule.cwe,),
                        evidence=(reference,),
                        remediation=(rule.remediation,),
                    )
                )

        limitations = (
            "Version 1 applies bounded Python AST rules and does not execute target code.",
            "The approved Python source snapshot was hash-verified before static parsing.",
            "Candidate findings require human data-flow validation and may contain false "
            "positives.",
            "Generated code, dependencies, runtime configuration, and unsupported languages "
            "require separate review.",
        )
        return WhiteBoxAssessmentResult(
            scope_id=self.scope.scope_id,
            repository_uri=self.repository_uri,
            revision=self.revision,
            started_at=started,
            completed_at=datetime.now(UTC),
            findings=tuple(findings),
            evidence=tuple(evidence),
            files_reviewed=files_reviewed,
            files_skipped=files_skipped,
            files_authorized=len(self._sources),
            limitations=limitations,
        )
