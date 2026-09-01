import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shadowcrafter.data.adapters import AdapterKind, canonicalize_downloaded_source
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
                        }
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
    record = _read_records(output_path)[0]
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
    record = _read_records(output_path)[0]
    assert record.messages[-1].content == (
        "Session invalidation (REQ-1): Invalidate server-side sessions after credential rotation."
    )
    assert record.labels["generation_method"] == "deterministic_source_template"
    assert manifest["statistics"]["invalid_skipped"] == 1


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
