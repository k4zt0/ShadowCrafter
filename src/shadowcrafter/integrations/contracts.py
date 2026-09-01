"""Typed, defensive-only contracts for security product integrations.

These models describe candidate findings and plans.  They intentionally expose
no command, exploit, raw-malware, or autonomous execution field.  SIEM queries
are read-only; SOAR plans remain unexecuted until an external human-controlled
system approves them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shadowcrafter.integrations.validators import (
    reject_executable_content,
    target_is_allowlisted,
    validate_allowlist_entry,
    validate_cve_id,
    validate_cwe_id,
    validate_detection_query,
    validate_safe_http_method,
    validate_sha256,
    validate_stix_id,
)


class ContractModel(BaseModel):
    """Strict and immutable base for model-generated integration artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Severity(StrEnum):
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvidenceReference(ContractModel):
    evidence_id: str = Field(min_length=1, max_length=160)
    source_uri: str = Field(min_length=1, max_length=2048)
    description: str = Field(min_length=1, max_length=2000)
    sha256: str | None = None
    observed_at: datetime | None = None

    _safe_description = field_validator("description")(reject_executable_content)
    _valid_sha256 = field_validator("sha256")(
        lambda value: None if value is None else validate_sha256(value)
    )


class AuthorizationEvidence(ContractModel):
    """Auditable proof that a named owner approved a bounded assessment."""

    authorization_id: str = Field(min_length=3, max_length=160)
    approved_by: str = Field(min_length=2, max_length=200)
    evidence_uri: str = Field(min_length=1, max_length=2048)
    evidence_sha256: str = Field(description="Digest of the immutable authorization record")
    valid_from: datetime
    valid_until: datetime

    _valid_sha256 = field_validator("evidence_sha256")(validate_sha256)

    @model_validator(mode="after")
    def require_positive_window(self) -> AuthorizationEvidence:
        if self.valid_until <= self.valid_from:
            raise ValueError("authorization valid_until must be after valid_from")
        return self


class STIXReference(ContractModel):
    kind: Literal["stix"] = "stix"
    stix_id: str
    object_type: str = Field(min_length=1, max_length=80)
    spec_version: Literal["2.0", "2.1"] = "2.1"
    source_uri: str | None = Field(default=None, max_length=2048)

    _valid_stix_id = field_validator("stix_id")(validate_stix_id)


class SARIFReference(ContractModel):
    kind: Literal["sarif"] = "sarif"
    artifact_uri: str = Field(min_length=1, max_length=2048)
    rule_id: str = Field(min_length=1, max_length=200)
    result_index: int = Field(ge=0)
    run_id: str | None = Field(default=None, max_length=200)
    region_start_line: int | None = Field(default=None, ge=1)


SecurityReference = Annotated[STIXReference | SARIFReference, Field(discriminator="kind")]


class VulnerabilityFinding(ContractModel):
    finding_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=6000)
    severity: Severity
    confidence: Confidence
    cve_candidates: tuple[str, ...] = ()
    cwe_candidates: tuple[str, ...] = ()
    affected_components: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    remediation: tuple[str, ...] = Field(min_length=1)
    references: tuple[SecurityReference, ...] = ()

    _safe_text = field_validator("title", "summary")(reject_executable_content)
    _valid_cves = field_validator("cve_candidates")(
        lambda values: tuple(validate_cve_id(value) for value in values)
    )
    _valid_cwes = field_validator("cwe_candidates")(
        lambda values: tuple(validate_cwe_id(value) for value in values)
    )
    _safe_remediation = field_validator("remediation")(
        lambda values: tuple(reject_executable_content(value) for value in values)
    )


class BlackBoxScope(ContractModel):
    """Read-only, allowlisted HTTP observation scope with explicit approval."""

    scope_id: str = Field(min_length=1, max_length=160)
    authorization: AuthorizationEvidence
    allowed_targets: tuple[str, ...] = Field(min_length=1)
    safe_methods: tuple[str, ...] = ("GET", "HEAD", "OPTIONS")
    requests_per_minute: int = Field(default=30, ge=1, le=60)
    max_concurrency: int = Field(default=2, ge=1, le=4)
    allowlist_only: Literal[True] = True
    follow_redirects: Literal[False] = False
    deny_off_scope_redirects: Literal[True] = True
    brute_force_allowed: Literal[False] = False
    denial_of_service_allowed: Literal[False] = False
    credential_testing_allowed: Literal[False] = False
    exploit_execution_allowed: Literal[False] = False
    state_changing_requests_allowed: Literal[False] = False

    _valid_targets = field_validator("allowed_targets")(
        lambda values: tuple(validate_allowlist_entry(value) for value in values)
    )
    _safe_methods = field_validator("safe_methods")(
        lambda values: tuple(validate_safe_http_method(value) for value in values)
    )

    @model_validator(mode="after")
    def require_unique_scope(self) -> BlackBoxScope:
        if len(set(self.allowed_targets)) != len(self.allowed_targets):
            raise ValueError("allowed_targets must not contain duplicates")
        if len(set(self.safe_methods)) != len(self.safe_methods):
            raise ValueError("safe_methods must not contain duplicates")
        return self


class BlackBoxFinding(ContractModel):
    assessment_type: Literal["black_box"] = "black_box"
    finding_id: str = Field(min_length=1, max_length=160)
    scope_id: str = Field(min_length=1, max_length=160)
    target: str = Field(min_length=1, max_length=2048)
    assessed_methods: tuple[str, ...] = Field(min_length=1)
    title: str = Field(min_length=1, max_length=300)
    observation: str = Field(min_length=1, max_length=6000)
    severity: Severity
    confidence: Confidence
    cve_candidates: tuple[str, ...] = ()
    cwe_candidates: tuple[str, ...] = ()
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    remediation: tuple[str, ...] = Field(min_length=1)
    exploit_attempted: Literal[False] = False
    payload_included: Literal[False] = False

    _safe_methods = field_validator("assessed_methods")(
        lambda values: tuple(validate_safe_http_method(value) for value in values)
    )
    _safe_text = field_validator("title", "observation")(reject_executable_content)
    _safe_remediation = field_validator("remediation")(
        lambda values: tuple(reject_executable_content(value) for value in values)
    )
    _valid_cves = field_validator("cve_candidates")(
        lambda values: tuple(validate_cve_id(value) for value in values)
    )
    _valid_cwes = field_validator("cwe_candidates")(
        lambda values: tuple(validate_cwe_id(value) for value in values)
    )

    def validate_against_scope(self, scope: BlackBoxScope) -> None:
        """Enforce target, method, and authorization linkage before acceptance."""

        if self.scope_id != scope.scope_id:
            raise ValueError("finding scope_id does not match the authorized scope")
        if not target_is_allowlisted(self.target, scope.allowed_targets):
            raise ValueError("finding target is outside the authorized allowlist")
        if not set(self.assessed_methods).issubset(scope.safe_methods):
            raise ValueError("finding includes a method not authorized by the scope")


class SourceLocation(ContractModel):
    repository_uri: str = Field(min_length=1, max_length=2048)
    revision: str = Field(min_length=1, max_length=200)
    path: str = Field(min_length=1, max_length=2048)
    start_line: int = Field(ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def line_order(self) -> SourceLocation:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must not precede start_line")
        return self


class WhiteBoxScope(ContractModel):
    """Authorized source/configuration review without executing target code."""

    scope_id: str = Field(min_length=1, max_length=160)
    authorization: AuthorizationEvidence
    repository_uris: tuple[str, ...] = Field(min_length=1)
    revisions: tuple[str, ...] = Field(min_length=1)
    included_paths: tuple[str, ...] = (".",)
    excluded_paths: tuple[str, ...] = ()
    static_analysis_only: Literal[True] = True
    target_code_execution_allowed: Literal[False] = False
    exploit_execution_allowed: Literal[False] = False


class WhiteBoxFinding(ContractModel):
    assessment_type: Literal["white_box"] = "white_box"
    finding_id: str = Field(min_length=1, max_length=160)
    scope_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=6000)
    location: SourceLocation
    severity: Severity
    confidence: Confidence
    cve_candidates: tuple[str, ...] = ()
    cwe_candidates: tuple[str, ...] = ()
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    remediation: tuple[str, ...] = Field(min_length=1)
    proof_by_static_reasoning_only: Literal[True] = True
    payload_included: Literal[False] = False

    _safe_text = field_validator("title", "summary")(reject_executable_content)
    _safe_remediation = field_validator("remediation")(
        lambda values: tuple(reject_executable_content(value) for value in values)
    )
    _valid_cves = field_validator("cve_candidates")(
        lambda values: tuple(validate_cve_id(value) for value in values)
    )
    _valid_cwes = field_validator("cwe_candidates")(
        lambda values: tuple(validate_cwe_id(value) for value in values)
    )


AssessmentFinding = Annotated[
    BlackBoxFinding | WhiteBoxFinding, Field(discriminator="assessment_type")
]


class VulnerabilityReport(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    report_id: str = Field(min_length=1, max_length=160)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    title: str = Field(min_length=1, max_length=300)
    executive_summary: str = Field(min_length=1, max_length=10000)
    scope_summary: str = Field(min_length=1, max_length=6000)
    findings: tuple[VulnerabilityFinding | AssessmentFinding, ...]
    limitations: tuple[str, ...] = Field(min_length=1)
    references: tuple[SecurityReference, ...] = ()
    generated_by: Literal["Odytssey/ShadowCrafter"] = "Odytssey/ShadowCrafter"
    human_review_required: Literal[True] = True

    _safe_text = field_validator("title", "executive_summary", "scope_summary")(
        reject_executable_content
    )
    _safe_limitations = field_validator("limitations")(
        lambda values: tuple(reject_executable_content(value) for value in values)
    )


class QueryLanguage(StrEnum):
    SPL = "spl"
    KQL = "kql"
    LUCENE = "lucene"
    EQL = "eql"
    SQL_READ_ONLY = "sql_read_only"


class SIEMQueryCandidate(ContractModel):
    candidate_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=300)
    language: QueryLanguage
    query: str = Field(min_length=1, max_length=12000)
    data_sources: tuple[str, ...] = Field(min_length=1)
    required_fields: tuple[str, ...] = Field(min_length=1)
    false_positive_notes: tuple[str, ...] = Field(min_length=1)
    validation_notes: tuple[str, ...] = Field(min_length=1)
    read_only: Literal[True] = True
    candidate_only: Literal[True] = True
    human_review_required: Literal[True] = True

    _safe_title = field_validator("title")(reject_executable_content)
    _read_only_query = field_validator("query")(validate_detection_query)


class SigmaOperator(StrEnum):
    EQUALS = "equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"


class SigmaCondition(ContractModel):
    field: str = Field(min_length=1, max_length=300)
    operator: SigmaOperator
    values: tuple[str, ...] = Field(min_length=1)


class SigmaLogSource(ContractModel):
    category: str | None = Field(default=None, max_length=160)
    product: str | None = Field(default=None, max_length=160)
    service: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def require_log_source(self) -> SigmaLogSource:
        if not any((self.category, self.product, self.service)):
            raise ValueError("at least one Sigma log-source selector is required")
        return self


class SigmaRuleCandidate(ContractModel):
    candidate_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=6000)
    status: Literal["experimental"] = "experimental"
    log_source: SigmaLogSource
    conditions: tuple[SigmaCondition, ...] = Field(min_length=1)
    condition_expression: str = Field(min_length=1, max_length=500)
    false_positive_notes: tuple[str, ...] = Field(min_length=1)
    level: Severity
    candidate_only: Literal[True] = True
    human_review_required: Literal[True] = True

    _safe_text = field_validator("title", "description", "condition_expression")(
        reject_executable_content
    )


class SIEMSigmaCandidate(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    siem_query: SIEMQueryCandidate | None = None
    sigma_rule: SigmaRuleCandidate | None = None
    stix_references: tuple[STIXReference, ...] = ()
    sarif_references: tuple[SARIFReference, ...] = ()
    candidate_only: Literal[True] = True
    human_review_required: Literal[True] = True

    @model_validator(mode="after")
    def require_candidate(self) -> SIEMSigmaCandidate:
        if self.siem_query is None and self.sigma_rule is None:
            raise ValueError("at least one SIEM query or Sigma rule candidate is required")
        return self


class SOARActionType(StrEnum):
    ENRICH = "enrich"
    COLLECT_EVIDENCE = "collect_evidence"
    NOTIFY = "notify"
    OPEN_CASE = "open_case"
    PROPOSE_BLOCK = "propose_block"
    PROPOSE_ISOLATION = "propose_isolation"
    PROPOSE_QUARANTINE = "propose_quarantine"


class RollbackPlan(ContractModel):
    description: str = Field(min_length=1, max_length=3000)
    verification: str = Field(min_length=1, max_length=3000)
    owner_role: str = Field(min_length=1, max_length=160)

    _safe_text = field_validator("description", "verification")(reject_executable_content)


class ApprovalGate(ContractModel):
    required: Literal[True] = True
    status: Literal["pending"] = "pending"
    approver_role: str = Field(min_length=1, max_length=160)
    separation_of_duties: Literal[True] = True


class SOARStep(ContractModel):
    step_id: str = Field(min_length=1, max_length=160)
    action_type: SOARActionType
    description: str = Field(min_length=1, max_length=3000)
    target_reference: str = Field(min_length=1, max_length=2048)
    depends_on: tuple[str, ...] = ()
    approval: ApprovalGate
    rollback: RollbackPlan
    execution_status: Literal["not_executed"] = "not_executed"
    executable_content_included: Literal[False] = False

    _safe_description = field_validator("description")(reject_executable_content)


class SOARPlan(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    plan_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=300)
    incident_reference: str = Field(min_length=1, max_length=2048)
    rationale: str = Field(min_length=1, max_length=6000)
    steps: tuple[SOARStep, ...] = Field(min_length=1)
    human_approval_required: Literal[True] = True
    dry_run_only: Literal[True] = True
    autonomous_execution_allowed: Literal[False] = False
    status: Literal["draft_pending_approval"] = "draft_pending_approval"

    _safe_text = field_validator("title", "rationale")(reject_executable_content)

    @model_validator(mode="after")
    def validate_step_graph(self) -> SOARPlan:
        identifiers = tuple(step.step_id for step in self.steps)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("SOAR step_id values must be unique")
        known: set[str] = set()
        for step in self.steps:
            if not set(step.depends_on).issubset(known):
                raise ValueError("SOAR dependencies must refer to earlier steps")
            known.add(step.step_id)
        return self


class IndicatorType(StrEnum):
    DOMAIN = "domain"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    URL = "url"
    FILE_HASH = "file_hash"
    EMAIL = "email"
    MUTEX = "mutex"
    REGISTRY_KEY = "registry_key"


class Indicator(ContractModel):
    indicator_type: IndicatorType
    value: str = Field(min_length=1, max_length=4096)
    confidence: Confidence
    source_evidence_id: str = Field(min_length=1, max_length=160)


class MalwareFileMetadata(ContractModel):
    sha256: str
    size_bytes: int = Field(ge=0)
    file_name: str | None = Field(default=None, max_length=512)
    mime_type: str | None = Field(default=None, max_length=300)
    file_type: str | None = Field(default=None, max_length=300)

    _valid_sha256 = field_validator("sha256")(validate_sha256)


class MalwareAnalysis(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    analysis_id: str = Field(min_length=1, max_length=160)
    sample: MalwareFileMetadata
    family_candidates: tuple[str, ...] = ()
    classification: str = Field(min_length=1, max_length=300)
    confidence: Confidence
    observed_behaviors: tuple[str, ...]
    indicators: tuple[Indicator, ...]
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    recommended_response: tuple[str, ...] = Field(min_length=1)
    stix_references: tuple[STIXReference, ...] = ()
    handling_mode: Literal["metadata_only"] = "metadata_only"
    raw_sample_included: Literal[False] = False
    executable_content_included: Literal[False] = False
    human_review_required: Literal[True] = True

    _safe_classification = field_validator("classification")(reject_executable_content)
    _safe_behaviors = field_validator("observed_behaviors")(
        lambda values: tuple(reject_executable_content(value) for value in values)
    )
    _safe_response = field_validator("recommended_response")(
        lambda values: tuple(reject_executable_content(value) for value in values)
    )


# Readable compatibility names for downstream integrations.
MalwareAnalysisReport = MalwareAnalysis
SOARPlanCandidate = SOARPlan
VulnerabilityReportContract = VulnerabilityReport
