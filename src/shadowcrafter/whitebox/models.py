"""Result models for source-only white-box assessment."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from shadowcrafter.integrations.contracts import ContractModel, WhiteBoxFinding
from shadowcrafter.integrations.validators import reject_executable_content, validate_sha256

_IMMUTABLE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def validate_scope_path(value: str) -> str:
    """Require one canonical POSIX-relative path without traversal or controls."""

    if not value or "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError("scope paths must be non-empty canonical POSIX-relative paths")
    normalized = PurePosixPath(value)
    canonical = str(normalized)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("scope paths must be relative and cannot traverse parents")
    if canonical != value and not (value == "." and canonical == "."):
        raise ValueError("scope paths must use their canonical POSIX representation")
    return canonical


def validate_immutable_revision(value: str) -> str:
    """Accept canonical SHA-1 or SHA-256 object identifiers, never moving refs."""

    if not _IMMUTABLE_REVISION_PATTERN.fullmatch(value):
        raise ValueError("revision must be a lowercase 40- or 64-character object identifier")
    return value


def validate_repository_uri(value: str) -> str:
    """Allow source repository identifiers while excluding embedded secrets."""

    if any(ord(character) < 32 for character in value):
        raise ValueError("repository URI contains control characters")
    parsed = urlsplit(value)
    if parsed.scheme not in {"file", "git", "https", "ssh"}:
        raise ValueError("repository URI must use file, git, https, or ssh")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("repository URI must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("repository URI must not contain query or fragment data")
    if parsed.scheme != "file" and not parsed.hostname:
        raise ValueError("repository URI must contain an exact hostname")
    return value


class WhiteBoxAuthorizationArtifact(ContractModel):
    """Immutable approval bound to one source snapshot and assessment invocation."""

    schema_version: Literal["1.0"] = "1.0"
    authorization_id: str = Field(min_length=3, max_length=160)
    scope_id: str = Field(min_length=1, max_length=160)
    approved_by: str = Field(min_length=2, max_length=200)
    purpose: str = Field(min_length=8, max_length=2000)
    repository_uri: str = Field(min_length=1, max_length=2048)
    revision: str = Field(min_length=1, max_length=200)
    included_paths: tuple[str, ...] = Field(min_length=1)
    excluded_paths: tuple[str, ...] = ()
    python_source_snapshot_sha256: str
    valid_from: datetime
    valid_until: datetime
    static_analysis_only: Literal[True]
    target_code_execution_allowed: Literal[False]
    exploit_execution_allowed: Literal[False]

    _valid_snapshot = field_validator("python_source_snapshot_sha256")(validate_sha256)
    _valid_revision = field_validator("revision")(validate_immutable_revision)
    _valid_repository_uri = field_validator("repository_uri")(validate_repository_uri)
    _safe_purpose = field_validator("purpose")(reject_executable_content)
    _valid_included = field_validator("included_paths")(
        lambda values: tuple(validate_scope_path(value) for value in values)
    )
    _valid_excluded = field_validator("excluded_paths")(
        lambda values: tuple(validate_scope_path(value) for value in values)
    )

    @model_validator(mode="after")
    def validate_authorization(self) -> WhiteBoxAuthorizationArtifact:
        if self.valid_from.tzinfo is None or self.valid_until.tzinfo is None:
            raise ValueError("authorization timestamps must be timezone-aware")
        if self.valid_until <= self.valid_from:
            raise ValueError("authorization validity window must be positive")
        if len(set(self.included_paths)) != len(self.included_paths):
            raise ValueError("included paths must be unique")
        if len(set(self.excluded_paths)) != len(self.excluded_paths):
            raise ValueError("excluded paths must be unique")
        return self


class StaticEvidenceRecord(ContractModel):
    """Hashed source evidence without embedding source or secret values."""

    evidence_id: str = Field(min_length=1, max_length=160)
    rule_id: str = Field(pattern=r"^SC-PY-[0-9]{3}$")
    path: str = Field(min_length=1, max_length=2048)
    start_line: int = Field(ge=1)
    evidence_digest: str
    digest_kind: Literal["authorization-keyed-hmac-sha256"] = "authorization-keyed-hmac-sha256"
    source_content_included: Literal[False] = False
    target_code_executed: Literal[False] = False

    _valid_evidence_digest = field_validator("evidence_digest")(validate_sha256)
    _valid_path = field_validator("path")(validate_scope_path)


class WhiteBoxAssessmentResult(ContractModel):
    """Static candidate findings requiring human verification."""

    schema_version: Literal["1.0"] = "1.0"
    scope_id: str = Field(min_length=1, max_length=160)
    repository_uri: str = Field(min_length=1, max_length=2048)
    revision: str = Field(min_length=1, max_length=200)
    started_at: datetime
    completed_at: datetime
    findings: tuple[WhiteBoxFinding, ...]
    evidence: tuple[StaticEvidenceRecord, ...]
    files_reviewed: int = Field(ge=0)
    files_skipped: int = Field(ge=0)
    files_authorized: int = Field(ge=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    static_analysis_only: Literal[True] = True
    target_code_executed: Literal[False] = False
    exploit_attempted: Literal[False] = False
    human_review_required: Literal[True] = True
    source_snapshot_verified: Literal[True] = True
    analysis_complete: Literal[True] = True
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    _valid_repository_uri = field_validator("repository_uri")(validate_repository_uri)
    _valid_revision = field_validator("revision")(validate_immutable_revision)

    @model_validator(mode="after")
    def validate_evidence_links(self) -> WhiteBoxAssessmentResult:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("assessment timestamps must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.files_reviewed + self.files_skipped != self.files_authorized:
            raise ValueError("reviewed and skipped counts must cover the authorized snapshot")
        evidence_ids = {item.evidence_id for item in self.evidence}
        finding_ids = {item.finding_id for item in self.findings}
        if len(evidence_ids) != len(self.evidence):
            raise ValueError("evidence identifiers must be unique")
        if len(finding_ids) != len(self.findings):
            raise ValueError("finding identifiers must be unique")
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        for finding in self.findings:
            if finding.scope_id != self.scope_id:
                raise ValueError("finding scope does not match result scope")
            if (
                finding.location.repository_uri != self.repository_uri
                or finding.location.revision != self.revision
            ):
                raise ValueError("finding source identity does not match result source")
            if not {item.evidence_id for item in finding.evidence}.issubset(evidence_ids):
                raise ValueError("finding references evidence absent from the result")
            for reference in finding.evidence:
                record = evidence_by_id[reference.evidence_id]
                if (
                    record.path != finding.location.path
                    or record.start_line != finding.location.start_line
                    or record.evidence_digest != reference.sha256
                ):
                    raise ValueError("finding location or digest does not match evidence")
        return self
