"""Strict models for passive black-box assessment runtime state."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from shadowcrafter.integrations.contracts import (
    BlackBoxFinding,
    ContractModel,
)
from shadowcrafter.integrations.validators import (
    reject_executable_content,
    validate_allowlist_entry,
    validate_safe_http_method,
    validate_sha256,
)

_PASSIVE_PATH_PATTERN = re.compile(r"^/[A-Za-z0-9._~/-]*$")


def validate_passive_path(value: str, *, max_length: int = 4096) -> str:
    """Accept only unambiguous exact paths, never encoded probes or parameters."""

    segments = value.split("/")
    invalid = (
        len(value) > max_length
        or not _PASSIVE_PATH_PATTERN.fullmatch(value)
        or "//" in value
        or any(segment in {".", ".."} for segment in segments)
    )
    if invalid:
        raise ValueError(
            "passive paths allow only unambiguous ASCII letters, digits, dot, dash, underscore, "
            "tilde, and slash"
        )
    return value


class AuthorizationArtifact(ContractModel):
    """Contents of the immutable authorization file supplied to the runner."""

    schema_version: Literal["1.0"] = "1.0"
    authorization_id: str = Field(min_length=3, max_length=160)
    scope_id: str = Field(min_length=1, max_length=160)
    approved_by: str = Field(min_length=2, max_length=200)
    purpose: str = Field(min_length=8, max_length=2000)
    allowed_targets: tuple[str, ...] = Field(min_length=1)
    allowed_paths: tuple[str, ...] = Field(min_length=1)
    safe_methods: tuple[str, ...] = Field(min_length=1)
    valid_from: datetime
    valid_until: datetime
    passive_read_only_only: Literal[True]
    payloads_allowed: Literal[False]
    redirects_allowed: Literal[False]
    brute_force_allowed: Literal[False]
    credential_testing_allowed: Literal[False]
    exploit_execution_allowed: Literal[False]
    denial_of_service_allowed: Literal[False]
    state_changing_requests_allowed: Literal[False]

    _safe_purpose = field_validator("purpose")(reject_executable_content)
    _valid_targets = field_validator("allowed_targets")(
        lambda values: tuple(validate_allowlist_entry(value) for value in values)
    )
    _safe_methods = field_validator("safe_methods")(
        lambda values: tuple(validate_safe_http_method(value) for value in values)
    )

    @field_validator("allowed_paths")
    @classmethod
    def validate_allowed_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("at least one exact passive path must be authorized")
        return tuple(validate_passive_path(value) for value in values)

    @model_validator(mode="after")
    def validate_artifact(self) -> AuthorizationArtifact:
        if self.valid_from.tzinfo is None or self.valid_until.tzinfo is None:
            raise ValueError("authorization artifact timestamps must be timezone-aware")
        if self.valid_until <= self.valid_from:
            raise ValueError("authorization artifact has a non-positive validity window")
        if len(set(self.allowed_targets)) != len(self.allowed_targets):
            raise ValueError("authorization artifact targets must be unique")
        if len(set(self.safe_methods)) != len(self.safe_methods):
            raise ValueError("authorization artifact methods must be unique")
        if len(set(self.allowed_paths)) != len(self.allowed_paths):
            raise ValueError("authorization artifact paths must be unique")
        return self


class SafetyLimits(ContractModel):
    """Hard upper bounds independent of model or caller output."""

    request_timeout_seconds: float = Field(default=5.0, ge=0.25, le=10.0)
    max_response_bytes: int = Field(default=65_536, ge=0, le=65_536)
    max_header_bytes: int = Field(default=16_384, ge=1024, le=16_384)
    max_targets: int = Field(default=8, ge=1, le=16)
    max_requests: int = Field(default=24, ge=1, le=48)
    max_dns_answers: int = Field(default=8, ge=1, le=16)
    max_path_length: int = Field(default=2048, ge=1, le=4096)


class TLSMetadata(ContractModel):
    """Non-secret TLS connection metadata; certificate contents are not retained."""

    protocol: str = Field(min_length=1, max_length=40)
    cipher: str | None = Field(default=None, max_length=160)
    certificate_sha256: str | None = None
    certificate_not_after: datetime | None = None

    _valid_certificate_sha256 = field_validator("certificate_sha256")(
        lambda value: None if value is None else validate_sha256(value)
    )


class ObservedHeader(ContractModel):
    """One allowlisted response header with sensitive values redacted."""

    name: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=2000)


class EvidenceRecord(ContractModel):
    """Bounded response evidence that deliberately omits response body content."""

    evidence_id: str = Field(min_length=1, max_length=160)
    target: str = Field(min_length=1, max_length=2048)
    method: Literal["GET", "HEAD", "OPTIONS"]
    status_code: int = Field(ge=100, le=599)
    peer_ip: str = Field(min_length=2, max_length=64)
    response_headers: tuple[ObservedHeader, ...]
    body_prefix_sha256: str
    body_bytes_captured: int = Field(ge=0, le=65_536)
    body_truncated: bool
    elapsed_ms: float = Field(ge=0, le=10_000)
    tls: TLSMetadata | None = None
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    response_body_included: Literal[False] = False
    credentials_included: Literal[False] = False

    _valid_body_sha256 = field_validator("body_prefix_sha256")(validate_sha256)


class BlackBoxAssessmentResult(ContractModel):
    """Evidence-grounded result; every issue remains a candidate for human review."""

    schema_version: Literal["1.0"] = "1.0"
    scope_id: str = Field(min_length=1, max_length=160)
    authorization_id: str = Field(min_length=3, max_length=160)
    started_at: datetime
    completed_at: datetime
    findings: tuple[BlackBoxFinding, ...]
    evidence: tuple[EvidenceRecord, ...]
    passive_observation_only: Literal[True] = True
    payloads_sent: Literal[False] = False
    credentials_sent: Literal[False] = False
    redirects_followed: Literal[False] = False
    human_review_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_result(self) -> BlackBoxAssessmentResult:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        evidence_ids = {record.evidence_id for record in self.evidence}
        for finding in self.findings:
            if finding.scope_id != self.scope_id:
                raise ValueError("finding scope does not match result scope")
            if not {item.evidence_id for item in finding.evidence}.issubset(evidence_ids):
                raise ValueError("finding references evidence absent from the result")
        return self
