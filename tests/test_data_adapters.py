import base64
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shadowcrafter.data.adapters import (
    AdapterKind,
    canonicalize_downloaded_source,
    canonicalize_nist_juliet,
    canonicalize_ocsf_schema,
    canonicalize_splunk_security_content,
)
from shadowcrafter.data.manifest import sha256_file
from shadowcrafter.data.prepare import TemporalSplit, prepare_jsonl
from shadowcrafter.schemas import RiskTier, SecurityRecord, TaskType

REGISTRY = Path("configs/data/sources.yaml")
RETRIEVED_AT = datetime(2026, 8, 1, tzinfo=UTC)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _read_records(path: Path) -> list[SecurityRecord]:
    return [
        SecurityRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _juliet_fixture(path: Path, *, unsafe_name: str | None = None) -> None:
    single = """/* TEMPLATE GENERATED TESTCASE FILE */
#include <string.h>
void CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01_bad(void)
{
    char data[10];
    memcpy(data, "AAAAAAAAAAAA", 12);
}
void CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01_good(void)
{
    char data[10];
    memcpy(data, "AAAA", 4);
}
"""
    first = """void CWE78_OS_Command_Injection__char_file_system_51_badSink(char * data)
{
    system(data);
}
"""
    second = """void CWE78_OS_Command_Injection__char_file_system_51_goodSink(char * data)
{
    printLine(data);
}
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "C/testcases/CWE121/s01/"
            "CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01.c",
            single,
        )
        archive.writestr(
            "C/testcases/CWE78/s01/CWE78_OS_Command_Injection__char_file_system_51a.c",
            first,
        )
        archive.writestr(
            "C/testcases/CWE78/s01/CWE78_OS_Command_Injection__char_file_system_51b.c",
            second,
        )
        archive.writestr("C/testcases/CWE78/s01/main.c", "int main(void) { return 0; }")
        archive.writestr("C/Makefile", "all:\n\t@true\n")
        if unsafe_name is not None:
            archive.writestr(unsafe_name, "unsafe")


def test_nist_juliet_adapter_groups_lineage_and_never_extracts_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "juliet.zip"
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    _juliet_fixture(archive_path)
    archive_sha256 = sha256_file(archive_path)

    manifest = canonicalize_nist_juliet(
        archive_path,
        first_output,
        upstream_revision="juliet-cpp-1.3-test",
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
        expected_archive_sha256=archive_sha256,
        expected_case_count=2,
    )
    canonicalize_nist_juliet(
        archive_path,
        second_output,
        upstream_revision="juliet-cpp-1.3-test",
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
        expected_archive_sha256=archive_sha256,
        expected_case_count=2,
    )

    assert first_output.read_bytes() == second_output.read_bytes()
    records = _read_records(first_output)
    assert len(records) == 2
    by_cwe = {record.labels["cwe_id"]: record for record in records}
    assert by_cwe["CWE-121"].task == TaskType.SECURE_CODE_REVIEW
    assert by_cwe["CWE-121"].labels["source_file_count"] == 1
    assert by_cwe["CWE-78"].labels["source_file_count"] == 2
    assert "51a.c" in by_cwe["CWE-78"].messages[0].content
    assert "51b.c" in by_cwe["CWE-78"].messages[0].content
    assert by_cwe["CWE-78"].provenance.content_sha256 == by_cwe["CWE-78"].canonical_hash()
    assert manifest["output"]["record_count"] == 2
    assert manifest["statistics"] == {
        "test_case_count": 2,
        "selected_source_file_count": 3,
        "ignored_non_case_source_count": 1,
    }
    assert manifest["controls"]["archive_never_extracted"] is True
    assert not (tmp_path / "C").exists()


def test_nist_juliet_adapter_rejects_archive_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "juliet-unsafe.zip"
    _juliet_fixture(archive_path, unsafe_name="../outside.c")

    with pytest.raises(ValueError, match="unsafe entry"):
        canonicalize_nist_juliet(
            archive_path,
            tmp_path / "unsafe.jsonl",
            upstream_revision="juliet-cpp-1.3-test",
            retrieved_at=RETRIEVED_AT,
            registry_path=REGISTRY,
            expected_archive_sha256=sha256_file(archive_path),
            expected_case_count=2,
        )


def _attack_bundle() -> dict[str, object]:
    technique = {
        "type": "attack-pattern",
        "id": "attack-pattern--technique-one",
        "name": "Credential Access Example",
        "created": "2024-01-01T00:00:00Z",
        "modified": "2025-01-01T00:00:00Z",
        "external_references": [{"source_name": "mitre-attack", "external_id": "T1000"}],
    }
    safe_mitigation = {
        "type": "course-of-action",
        "id": "course-of-action--safe",
        "name": "Credential Rotation",
        "description": "Rotate exposed credentials and invalidate active sessions.",
        "created": "2024-02-01T00:00:00Z",
        "modified": "2025-02-01T00:00:00Z",
        "external_references": [{"source_name": "mitre-attack", "external_id": "M1000"}],
    }
    unsafe_mitigation = {
        "type": "course-of-action",
        "id": "course-of-action--unsafe",
        "name": "Unsafe Example",
        "description": "```powershell -enc AAAA``` exploit payload",
        "created": "2024-02-01T00:00:00Z",
    }
    relationships = [
        {
            "type": "relationship",
            "id": "relationship--safe",
            "relationship_type": "mitigates",
            "source_ref": "course-of-action--safe",
            "target_ref": "attack-pattern--technique-one",
            "created": "2024-03-01T00:00:00Z",
            "modified": "2025-03-01T00:00:00Z",
        },
        {
            "type": "relationship",
            "id": "relationship--unsafe",
            "relationship_type": "mitigates",
            "source_ref": "course-of-action--unsafe",
            "target_ref": "attack-pattern--technique-one",
            "created": "2024-03-01T00:00:00Z",
        },
    ]
    return {
        "type": "bundle",
        "id": "bundle--one",
        "objects": [technique, safe_mitigation, unsafe_mitigation, *relationships],
    }


def test_attack_adapter_is_deterministic_source_grounded_and_prepare_compatible(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "attack.json"
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    _write_json(source_path, _attack_bundle())

    first_manifest = canonicalize_downloaded_source(
        source_path,
        first_output,
        source_id="mitre-attack-enterprise",
        upstream_revision="attack-v19.1",
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
    )
    canonicalize_downloaded_source(
        source_path,
        second_output,
        source_id="mitre-attack-enterprise",
        upstream_revision="attack-v19.1",
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
    )

    assert first_output.read_bytes() == second_output.read_bytes()
    records = _read_records(first_output)
    assert len(records) == 1
    record = records[0]
    assert record.task == TaskType.DETECTION_ENGINEERING
    assert record.risk_tier == RiskTier.DEFENSIVE
    assert record.messages[-1].content == (
        "Credential Rotation (M1000): Rotate exposed credentials and invalidate active sessions."
    )
    assert record.labels["model_generated"] is False
    assert record.labels["requires_human_review"] is True
    assert record.labels["published_at"] == "2024-03-01T00:00:00+00:00"
    assert record.split_group == "mitre-attack:mitigation:M1000"
    assert record.provenance.license == "MITRE-ATTACK-Terms-of-Use"
    assert record.provenance.upstream_revision == "attack-v19.1"
    assert record.provenance.content_sha256 == record.canonical_hash()
    assert first_manifest["statistics"] == {
        "candidate_count": 2,
        "unsafe_skipped": 1,
        "invalid_skipped": 0,
    }

    prepared_dir = tmp_path / "prepared"
    prepared = prepare_jsonl(
        first_output,
        prepared_dir,
        registry_path=REGISTRY,
        temporal_split=TemporalSplit(
            validation_after=datetime(2025, 6, 1, tzinfo=UTC),
            test_after=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )
    assert prepared["split_counts"]["train"] == 1


def test_attack_adapter_emits_detection_analytics_and_safe_procedure_mappings(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "attack-expanded.json"
    output_path = tmp_path / "attack-expanded.jsonl"
    technique = {
        "type": "attack-pattern",
        "id": "attack-pattern--expanded-technique",
        "name": "Suspicious DNS Activity",
        "created": "2025-01-01T00:00:00Z",
        "modified": "2025-02-01T00:00:00Z",
        "external_references": [{"external_id": "T1999"}],
    }
    analytic = {
        "type": "x-mitre-analytic",
        "id": "x-mitre-analytic--expanded",
        "name": "Analytic 1999",
        "description": "Correlate repeated failed DNS lookups with unusual process lineage.",
        "created": "2025-01-02T00:00:00Z",
        "external_references": [{"external_id": "AN1999"}],
        "x_mitre_log_source_references": [{"name": "linux:syslog", "channel": "DNS query"}],
    }
    strategy = {
        "type": "x-mitre-detection-strategy",
        "id": "x-mitre-detection-strategy--expanded",
        "name": "Detect suspicious resolver behavior",
        "created": "2025-01-02T00:00:00Z",
        "external_references": [{"external_id": "DET1999"}],
        "x_mitre_analytic_refs": [analytic["id"]],
    }
    group = {
        "type": "intrusion-set",
        "id": "intrusion-set--expanded",
        "name": "Example Group",
        "created": "2025-01-03T00:00:00Z",
    }
    relationships = [
        {
            "type": "relationship",
            "id": "relationship--detects-expanded",
            "relationship_type": "detects",
            "source_ref": strategy["id"],
            "target_ref": technique["id"],
            "created": "2025-01-04T00:00:00Z",
        },
        {
            "type": "relationship",
            "id": "relationship--uses-expanded",
            "relationship_type": "uses",
            "source_ref": group["id"],
            "target_ref": technique["id"],
            "description": "Example Group generated periodic high-entropy DNS queries.",
            "created": "2025-01-05T00:00:00Z",
        },
    ]
    _write_json(
        source_path,
        {
            "type": "bundle",
            "objects": [technique, analytic, strategy, group, *relationships],
        },
    )

    manifest = canonicalize_downloaded_source(
        source_path,
        output_path,
        source_id="mitre-attack-enterprise",
        upstream_revision="attack-expanded",
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
    )

    records = _read_records(output_path)
    assert len(records) == 2
    by_kind = {record.labels["record_kind"]: record for record in records}
    detection = by_kind["attack_detection_analytic"]
    assert detection.task == TaskType.DETECTION_ENGINEERING
    assert "Analytic 1999 (AN1999)" in detection.messages[-1].content
    assert "linux:syslog: DNS query" in detection.messages[-1].content
    procedure = by_kind["attack_procedure_mapping"]
    assert procedure.task == TaskType.THREAT_INTELLIGENCE
    assert procedure.messages[-1].content == "Suspicious DNS Activity (T1999)"
    assert procedure.split_group == detection.split_group
    assert manifest["statistics"] == {
        "candidate_count": 2,
        "unsafe_skipped": 0,
        "invalid_skipped": 0,
    }


def test_attack_adapter_filters_actionable_procedure_payloads(tmp_path: Path) -> None:
    source_path = tmp_path / "attack-unsafe-procedure.json"
    output_path = tmp_path / "attack-unsafe-procedure.jsonl"
    _write_json(
        source_path,
        {
            "type": "bundle",
            "objects": [
                {
                    "type": "attack-pattern",
                    "id": "attack-pattern--unsafe",
                    "name": "Unsafe Example",
                    "created": "2025-01-01T00:00:00Z",
                    "external_references": [{"external_id": "T1998"}],
                },
                {
                    "type": "malware",
                    "id": "malware--unsafe",
                    "name": "Unsafe Malware",
                    "created": "2025-01-01T00:00:00Z",
                },
                {
                    "type": "relationship",
                    "id": "relationship--unsafe-procedure",
                    "relationship_type": "uses",
                    "source_ref": "malware--unsafe",
                    "target_ref": "attack-pattern--unsafe",
                    "description": "powershell -enc AAAA",
                    "created": "2025-01-01T00:00:00Z",
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="adapter produced no safe source-grounded records"):
        canonicalize_downloaded_source(
            source_path,
            output_path,
            source_id="mitre-attack-enterprise",
            upstream_revision="attack-expanded",
            retrieved_at=RETRIEVED_AT,
            registry_path=REGISTRY,
        )


def test_cwe_xml_adapter_emits_only_official_mitigation_text(tmp_path: Path) -> None:
    source_path = tmp_path / "cwe.xml"
    output_path = tmp_path / "cwe.jsonl"
    source_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<Weakness_Catalog xmlns="http://cwe.mitre.org/cwe-7" Date="2024-11-19">
  <Weaknesses>
    <Weakness ID="79" Name="Improper Neutralization" Status="Stable">
      <Potential_Mitigations>
        <Mitigation Mitigation_ID="MIT-79-1">
          <Phase>Implementation</Phase>
          <Description>Use context-aware output encoding.</Description>
        </Mitigation>
        <Mitigation Mitigation_ID="MIT-79-2">
          <Phase>Testing</Phase>
          <Description>```curl -x exploit payload```</Description>
        </Mitigation>
      </Potential_Mitigations>
      <Content_History>
        <Modification><Modification_Date>2025-01-10</Modification_Date></Modification>
      </Content_History>
    </Weakness>
  </Weaknesses>
</Weakness_Catalog>
"""
    )

    manifest = canonicalize_downloaded_source(
        source_path,
        output_path,
        source_id="mitre-cwe",
        upstream_revision="cwe-4.20",
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
    )

    records = _read_records(output_path)
    assert len(records) == 1
    record = records[0]
    assert record.task == TaskType.SECURE_CODE_REVIEW
    assert record.messages[-1].content == "Implementation: Use context-aware output encoding."
    assert record.labels["cwe_id"] == "CWE-79"
    assert record.labels["modified_at"] == "2025-01-10T00:00:00+00:00"
    assert record.split_group == "mitre-cwe:CWE-79"
    assert manifest["statistics"]["unsafe_skipped"] == 1


def test_cwe_xml_adapter_safely_reads_single_catalog_zip(tmp_path: Path) -> None:
    source_path = tmp_path / "cwe.zip"
    output_path = tmp_path / "cwe.jsonl"
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<Weakness_Catalog xmlns="http://cwe.mitre.org/cwe-7" Date="2025-01-01">
  <Weaknesses>
    <Weakness ID="22" Name="Path Traversal" Status="Stable">
      <Potential_Mitigations>
        <Mitigation><Phase>Implementation</Phase>
          <Description>Resolve paths beneath a trusted base directory.</Description>
        </Mitigation>
      </Potential_Mitigations>
    </Weakness>
  </Weaknesses>
</Weakness_Catalog>"""
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cwec_v-test.xml", xml)
    source_path.write_bytes(archive_bytes.getvalue())

    canonicalize_downloaded_source(
        source_path,
        output_path,
        source_id="mitre-cwe",
        upstream_revision="cwe-zip-test",
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
    )

    record = _read_records(output_path)[0]
    assert record.labels["cwe_id"] == "CWE-22"
    assert record.messages[-1].content == (
        "Implementation: Resolve paths beneath a trusted base directory."
    )


def test_capec_xml_adapter_emits_mitigations_and_cwe_mappings(tmp_path: Path) -> None:
    source_path = tmp_path / "capec.xml"
    output_path = tmp_path / "capec.jsonl"
    source_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<Attack_Pattern_Catalog xmlns="http://capec.mitre.org/capec-3"
                        xmlns:xhtml="http://www.w3.org/1999/xhtml"
                        Date="2025-01-01">
  <Attack_Patterns>
    <Attack_Pattern ID="100" Name="Example Pattern" Status="Stable">
      <Description>An application accepts data without sufficient validation.</Description>
      <Mitigations>
        <Mitigation><xhtml:p>Validate input using a strict allowlist.</xhtml:p></Mitigation>
      </Mitigations>
      <Related_Weaknesses>
        <Related_Weakness CWE_ID="20"/>
        <Related_Weakness CWE_ID="79"/>
      </Related_Weaknesses>
      <Content_History>
        <Submission><Submission_Date>2024-01-01</Submission_Date></Submission>
        <Modification><Modification_Date>2025-02-01</Modification_Date></Modification>
      </Content_History>
    </Attack_Pattern>
  </Attack_Patterns>
</Attack_Pattern_Catalog>
"""
    )

    manifest = canonicalize_downloaded_source(
        source_path,
        output_path,
        source_id="mitre-capec",
        upstream_revision="capec-3.9",
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
    )

    records = _read_records(output_path)
    assert len(records) == 2
    by_kind = {record.labels["record_kind"]: record for record in records}
    mitigation = by_kind["capec_mitigation"]
    assert mitigation.task == TaskType.SECURE_CODE_REVIEW
    assert mitigation.messages[-1].content == "Validate input using a strict allowlist."
    mapping = by_kind["capec_cwe_mapping"]
    assert mapping.task == TaskType.CWE_MAPPING
    assert mapping.messages[-1].content == "CWE-20, CWE-79"
    assert mapping.labels["modified_at"] == "2025-02-01T00:00:00+00:00"
    assert mapping.split_group == mitigation.split_group == "mitre-capec:CAPEC-100"
    assert manifest["statistics"] == {
        "candidate_count": 2,
        "unsafe_skipped": 0,
        "invalid_skipped": 0,
    }


def test_cwe_json_adapter_preserves_phase_and_lineage(tmp_path: Path) -> None:
    source_path = tmp_path / "cwe.json"
    output_path = tmp_path / "cwe.jsonl"
    _write_json(
        source_path,
        {
            "date": "2024-11-19",
            "weaknesses": [
                {
                    "id": "89",
                    "name": "SQL command neutralization",
                    "status": "Stable",
                    "modified": "2025-02-01T00:00:00Z",
                    "potential_mitigations": [
                        {
                            "phase": ["Architecture", "Implementation"],
                            "description": "Use parameterized database queries.",
                        },
                        {
                            "phase": ["Architecture", "Implementation"],
                            "description": "Use parameterized database queries.",
                        },
                    ],
                }
            ],
        },
    )
    canonicalize_downloaded_source(
        source_path,
        output_path,
        source_id="mitre-cwe",
        upstream_revision="cwe-json-4.20",
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
    )
    records = _read_records(output_path)
    assert len(records) == 1
    record = records[0]
    assert record.messages[-1].content == (
        "Architecture Implementation: Use parameterized database queries."
    )
    assert record.split_group == "mitre-cwe:CWE-89"
    assert record.labels["modified_at"] == "2025-02-01T00:00:00+00:00"


def test_cwe_xml_adapter_rejects_dtd_or_entities(tmp_path: Path) -> None:
    source_path = tmp_path / "cwe.xml"
    source_path.write_text(
        '<!DOCTYPE x [<!ENTITY secret SYSTEM "file:///etc/passwd">]><Weakness_Catalog/>'
    )
    with pytest.raises(ValueError, match="DTD and entity declarations are prohibited"):
        canonicalize_downloaded_source(
            source_path,
            tmp_path / "cwe.jsonl",
            source_id="mitre-cwe",
            upstream_revision="cwe-4.20",
            retrieved_at=RETRIEVED_AT,
            registry_path=REGISTRY,
        )


def test_defensive_catalog_adapter_supports_reviewed_owasp_style_shape(tmp_path: Path) -> None:
    source_path = tmp_path / "requirements.json"
    output_path = tmp_path / "requirements.jsonl"
    _write_json(
        source_path,
        {
            "requirements": [
                {
                    "id": "REQ-1",
                    "title": "Session invalidation",
                    "requirement": "Invalidate server-side sessions after credential rotation.",
                    "published_at": "2025-01-01T00:00:00Z",
                    "model_generated": False,
                    "human_reviewed": True,
                },
                {
                    "id": "REQ-GENERATED",
                    "title": "Unverified generated item",
                    "requirement": "This item must not enter the canonical corpus.",
                    "published_at": "2025-01-01T00:00:00Z",
                    "model_generated": True,
                    "human_reviewed": False,
                },
                {
                    "id": "REQ-BLACKBOX",
                    "title": "Passive evidence",
                    "requirement": "Retain redacted response metadata and require human review.",
                    "task": "black_box_assessment",
                    "published_at": "2025-01-01T00:00:00Z",
                    "model_generated": False,
                    "human_reviewed": True,
                },
                {
                    "id": "REQ-UNKNOWN-TASK",
                    "title": "Unknown task",
                    "requirement": "This item must not enter the canonical corpus.",
                    "task": "unsupported_task",
                    "published_at": "2025-01-01T00:00:00Z",
                    "model_generated": False,
                    "human_reviewed": True,
                },
            ]
        },
    )
    manifest = canonicalize_downloaded_source(
        source_path,
        output_path,
        source_id="odytssey-curated-security-instructions",
        upstream_revision="reviewed-set-1",
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
        adapter=AdapterKind.DEFENSIVE_CATALOG_JSON,
    )
    records = _read_records(output_path)
    record = next(item for item in records if item.record_id.endswith("REQ-1"))
    assert record.messages[-1].content == (
        "Session invalidation (REQ-1): Invalidate server-side sessions after credential rotation."
    )
    assert record.labels["generation_method"] == "deterministic_source_template"
    blackbox = next(item for item in records if item.record_id.endswith("REQ-BLACKBOX"))
    assert blackbox.task == TaskType.BLACK_BOX_ASSESSMENT
    assert blackbox.labels["record_kind"] == "blackbox_defensive_requirement"
    assert manifest["statistics"]["invalid_skipped"] == 2


def test_splunk_adapter_emits_only_production_detections(tmp_path: Path) -> None:
    repository = tmp_path / "splunk-security-content"
    detections = repository / "detections" / "endpoint"
    detections.mkdir(parents=True)
    (detections / "production.yml").write_text(
        """name: Suspicious Authentication Pattern
id: 11111111-1111-1111-1111-111111111111
version: 1
creation_date: '2025-01-01'
modification_date: '2025-02-01'
author: Security Researcher
status: production
type: Anomaly
description: Detect repeated authentication failures followed by a success.
data_source:
  - Authentication Logs
search: index=auth action=failure | stats count by user
how_to_implement: Ingest normalized authentication events and tune the index macro.
known_false_positives: Password resets and approved authentication testing.
mitre_attack_id:
  - T1110
security_domain: identity
tests:
  - name: Ignored test metadata
"""
    )
    (detections / "experimental.yml").write_text(
        """name: Experimental Rule
id: 22222222-2222-2222-2222-222222222222
status: experimental
"""
    )
    output_path = tmp_path / "splunk.jsonl"

    manifest = canonicalize_splunk_security_content(
        repository,
        output_path,
        upstream_revision="a" * 40,
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
    )

    records = _read_records(output_path)
    assert len(records) == 1
    record = records[0]
    assert record.task == TaskType.SIEM_QUERY
    assert record.labels["record_kind"] == "splunk_production_detection"
    assert record.labels["mitre_attack_ids"] == ["T1110"]
    assert "index=auth action=failure" in record.messages[-1].content
    assert "Ignored test metadata" not in output_path.read_text()
    assert manifest["statistics"] == {
        "candidate_count": 2,
        "unsafe_skipped": 0,
        "invalid_skipped": 1,
    }
    assert manifest["controls"]["attack_simulation_excluded"] is True


def test_splunk_adapter_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    repository = tmp_path / "splunk-security-content"
    detections = repository / "detections"
    detections.mkdir(parents=True)
    (detections / "ambiguous.yml").write_text(
        """name: First
name: Second
id: 11111111-1111-1111-1111-111111111111
status: production
"""
    )

    with pytest.raises(ValueError, match="duplicate YAML mapping key"):
        canonicalize_splunk_security_content(
            repository,
            tmp_path / "splunk.jsonl",
            upstream_revision="a" * 40,
            retrieved_at=RETRIEVED_AT,
            registry_path=REGISTRY,
        )


def test_ocsf_adapter_emits_dictionary_category_and_event_definitions(tmp_path: Path) -> None:
    repository = tmp_path / "ocsf-schema"
    events = repository / "events" / "identity"
    events.mkdir(parents=True)
    _write_json(repository / "version.json", {"version": "1.2.3"})
    _write_json(
        repository / "dictionary.json",
        {
            "attributes": {
                "user": {
                    "caption": "User",
                    "description": "The user associated with the security activity.",
                    "type": "user",
                }
            }
        },
    )
    _write_json(
        repository / "categories.json",
        {
            "attributes": {
                "iam": {
                    "uid": 3,
                    "caption": "Identity and Access Management",
                    "description": "Events related to authentication and access control.",
                }
            }
        },
    )
    _write_json(
        events / "authentication.json",
        {
            "name": "authentication",
            "caption": "Authentication",
            "description": "An authentication attempt and its outcome.",
            "attributes": {
                "user": {"requirement": "required"},
                "src_endpoint": {"requirement": "recommended"},
            },
        },
    )
    output_path = tmp_path / "ocsf.jsonl"

    manifest = canonicalize_ocsf_schema(
        repository,
        output_path,
        upstream_revision="b" * 40,
        upstream_committed_at=datetime(2026, 8, 31, tzinfo=UTC),
        retrieved_at=RETRIEVED_AT,
        registry_path=REGISTRY,
    )

    records = _read_records(output_path)
    assert len(records) == 3
    by_kind = {record.labels["schema_kind"]: record for record in records}
    assert by_kind["attribute"].messages[-1].content.endswith("Type: user; cardinality: scalar.")
    assert by_kind["category"].messages[-1].content.endswith("Category UID: 3.")
    event = by_kind["event"]
    assert event.task == TaskType.SIEM_QUERY
    assert "Required attributes: user." in event.messages[-1].content
    assert "Recommended attributes: src_endpoint." in event.messages[-1].content
    assert manifest["statistics"] == {
        "candidate_count": 3,
        "unsafe_skipped": 0,
        "invalid_skipped": 0,
    }
    assert manifest["controls"]["examples_and_logs_excluded"] is True


@pytest.mark.parametrize(
    "source_id",
    ["owasp-asvs", "cisa-kev", "ctibench", "primevul", "raw-malware-binaries"],
)
def test_adapter_refuses_every_non_train_policy_class(tmp_path: Path, source_id: str) -> None:
    source_path = tmp_path / f"{source_id}.json"
    _write_json(source_path, [])
    with pytest.raises(ValueError, match="does not permit train"):
        canonicalize_downloaded_source(
            source_path,
            tmp_path / f"{source_id}.jsonl",
            source_id=source_id,
            upstream_revision="revision",
            retrieved_at=RETRIEVED_AT,
            registry_path=REGISTRY,
            adapter=AdapterKind.DEFENSIVE_CATALOG_JSON,
        )


def test_adapter_rejects_binary_hidden_in_unused_source_field(tmp_path: Path) -> None:
    source_path = tmp_path / "attack.json"
    payload = _attack_bundle()
    payload["unused_payload"] = base64.b64encode(b"MZ" + (b"\x00" * 512)).decode()
    _write_json(source_path, payload)
    with pytest.raises(ValueError, match="raw executable or archive payload"):
        canonicalize_downloaded_source(
            source_path,
            tmp_path / "output.jsonl",
            source_id="mitre-attack-enterprise",
            upstream_revision="attack-v19.1",
            retrieved_at=RETRIEVED_AT,
            registry_path=REGISTRY,
        )


def test_adapter_enforces_input_size_limit(tmp_path: Path) -> None:
    source_path = tmp_path / "requirements.json"
    _write_json(source_path, {"requirements": []})
    with pytest.raises(ValueError, match="exceeds size limit"):
        canonicalize_downloaded_source(
            source_path,
            tmp_path / "output.jsonl",
            source_id="odytssey-curated-security-instructions",
            upstream_revision="revision",
            retrieved_at=RETRIEVED_AT,
            registry_path=REGISTRY,
            max_input_bytes=10,
        )
