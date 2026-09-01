import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from shadowcrafter.integrations.contracts import (
    ApprovalGate,
    AuthorizationEvidence,
    BlackBoxFinding,
    BlackBoxScope,
    Confidence,
    EvidenceReference,
    MalwareAnalysis,
    MalwareFileMetadata,
    QueryLanguage,
    RollbackPlan,
    Severity,
    SIEMQueryCandidate,
    SOARActionType,
    SOARPlan,
    SOARStep,
    WhiteBoxScope,
)


def authorization() -> AuthorizationEvidence:
    now = datetime.now(UTC)
    return AuthorizationEvidence(
        authorization_id="AUTH-2026-001",
        approved_by="asset-owner@example.test",
        evidence_uri="vault://authorizations/AUTH-2026-001",
        evidence_sha256="a" * 64,
        valid_from=now - timedelta(hours=1),
        valid_until=now + timedelta(days=1),
    )


def evidence() -> EvidenceReference:
    return EvidenceReference(
        evidence_id="ev-1",
        source_uri="evidence://http-observation/1",
        description="Response headers omit a recommended transport policy.",
        sha256="b" * 64,
    )


def test_black_box_scope_is_authorized_read_only_and_bounded() -> None:
    scope = BlackBoxScope(
        scope_id="scope-1",
        authorization=authorization(),
        allowed_targets=("api.example.test", "10.20.0.0/24"),
        safe_methods=("GET", "HEAD"),
        requests_per_minute=10,
    )

    assert scope.allowlist_only
    assert not scope.follow_redirects
    assert not scope.exploit_execution_allowed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("safe_methods", ("POST",)),
        ("allowed_targets", ("0.0.0.0/0",)),
        ("allowed_targets", ("*.example.test",)),
        ("follow_redirects", True),
        ("brute_force_allowed", True),
        ("denial_of_service_allowed", True),
        ("credential_testing_allowed", True),
        ("exploit_execution_allowed", True),
        ("state_changing_requests_allowed", True),
    ],
)
def test_black_box_scope_rejects_unsafe_capabilities(field: str, value: object) -> None:
    values: dict[str, object] = {
        "scope_id": "scope-1",
        "authorization": authorization(),
        "allowed_targets": ("api.example.test",),
    }
    values[field] = value
    with pytest.raises(ValidationError):
        BlackBoxScope.model_validate(values)


def test_black_box_finding_is_evidence_only_and_scope_checked() -> None:
    scope = BlackBoxScope(
        scope_id="scope-1",
        authorization=authorization(),
        allowed_targets=("api.example.test",),
        safe_methods=("GET",),
    )
    finding = BlackBoxFinding(
        finding_id="bb-1",
        scope_id="scope-1",
        target="https://api.example.test/status",
        assessed_methods=("GET",),
        title="Missing transport policy header",
        observation="The observed response did not include the expected policy header.",
        severity=Severity.LOW,
        confidence=Confidence.HIGH,
        cwe_candidates=("CWE-319",),
        evidence=(evidence(),),
        remediation=("Add and verify the policy header at the application boundary.",),
    )

    finding.validate_against_scope(scope)
    outside = finding.model_copy(update={"target": "https://outside.example.test/"})
    with pytest.raises(ValueError, match="outside"):
        outside.validate_against_scope(scope)


def test_contracts_reject_commands_and_payload_fields() -> None:
    with pytest.raises(ValidationError, match="commands|payloads"):
        EvidenceReference(
            evidence_id="bad",
            source_uri="evidence://bad",
            description="bash -c id",
        )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        MalwareAnalysis.model_validate(
            {
                "analysis_id": "mal-1",
                "sample": {"sha256": "c" * 64, "size_bytes": 100},
                "classification": "Suspicious downloader metadata",
                "confidence": "medium",
                "observed_behaviors": [],
                "indicators": [],
                "evidence": [evidence().model_dump()],
                "recommended_response": ["Keep the sample quarantined for expert review."],
                "raw_sample": "opaque-binary-data",
            }
        )


def test_siem_query_must_be_read_only() -> None:
    common = {
        "candidate_id": "siem-1",
        "title": "Repeated authentication failures",
        "language": QueryLanguage.SPL,
        "data_sources": ("identity",),
        "required_fields": ("user", "result"),
        "false_positive_notes": ("Approved resilience testing may resemble this pattern.",),
        "validation_notes": ("Validate against a frozen historical log sample.",),
    }
    candidate = SIEMQueryCandidate(
        query="index=identity result=failure | stats count by user", **common
    )
    assert candidate.read_only

    with pytest.raises(ValidationError, match="read-only"):
        SIEMQueryCandidate(query="index=identity | delete", **common)


def test_soar_plan_requires_pending_human_approval_and_rollback() -> None:
    step = SOARStep(
        step_id="step-1",
        action_type=SOARActionType.NOTIFY,
        description="Notify the incident response lead with the evidence reference.",
        target_reference="case://IR-2026-001",
        approval=ApprovalGate(approver_role="incident-commander"),
        rollback=RollbackPlan(
            description="Withdraw the draft notification if the case mapping is incorrect.",
            verification="Confirm that no active notification remains associated with the draft.",
            owner_role="incident-commander",
        ),
    )
    plan = SOARPlan(
        plan_id="plan-1",
        title="Evidence notification proposal",
        incident_reference="case://IR-2026-001",
        rationale="The assigned responder needs the validated evidence reference.",
        steps=(step,),
    )

    assert plan.human_approval_required
    assert plan.dry_run_only
    assert plan.status == "draft_pending_approval"

    with pytest.raises(ValidationError):
        ApprovalGate(approver_role="incident-commander", status="approved")


def test_malware_contract_contains_metadata_not_sample_bytes() -> None:
    report = MalwareAnalysis(
        analysis_id="mal-1",
        sample=MalwareFileMetadata(sha256="c" * 64, size_bytes=4096),
        classification="Suspicious executable metadata",
        confidence=Confidence.MEDIUM,
        observed_behaviors=("The sandbox report records an unusual child-process pattern.",),
        indicators=(),
        evidence=(evidence(),),
        recommended_response=("Retain quarantine and request analyst validation.",),
    )

    assert report.handling_mode == "metadata_only"
    assert not report.raw_sample_included


def test_white_box_scope_cannot_enable_target_execution() -> None:
    with pytest.raises(ValidationError):
        WhiteBoxScope(
            scope_id="wb-1",
            authorization=authorization(),
            repository_uris=("git://internal/app",),
            revisions=("abc123",),
            target_code_execution_allowed=True,
        )


def test_checked_in_json_schemas_are_valid_and_identified() -> None:
    schema_directory = Path(__file__).parents[1] / "schemas"
    schema_paths = sorted(schema_directory.glob("*.schema.json"))

    assert {path.name for path in schema_paths} >= {
        "black-box-assessment.schema.json",
        "malware-analysis.schema.json",
        "security-references.schema.json",
        "siem-sigma-candidate.schema.json",
        "soar-plan.schema.json",
        "vulnerability-report.schema.json",
        "white-box-assessment.schema.json",
    }
    for path in schema_paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"].endswith(path.name)
