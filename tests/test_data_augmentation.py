from __future__ import annotations

import json
from pathlib import Path

import pytest

from shadowcrafter.data.augmentation import (
    derive_attack_technique_id_jsonl,
    derive_juliet_cwe_mapping_jsonl,
)
from shadowcrafter.data.prepare import SplitMode, prepare_jsonl_many
from shadowcrafter.schemas import Message, Provenance, RiskTier, SecurityRecord, TaskType


def _parent() -> SecurityRecord:
    record = SecurityRecord(
        record_id="nist-juliet-sard:CWE121_demo_01",
        task=TaskType.SECURE_CODE_REVIEW,
        risk_tier=RiskTier.DEFENSIVE,
        language="en",
        messages=[
            Message(
                role="user",
                content=(
                    "Perform a defensive static review. Do not compile or run it.\n\n"
                    "FILE CWE121_demo.c\n"
                    "void CWE121_demo_bad(void) { char x[4]; x[8] = 1; }"
                ),
            ),
            Message(role="assistant", content="Classification: CWE-121 (stack overflow)."),
        ],
        labels={
            "record_kind": "nist_juliet_cpp_test_case",
            "cwe_id": "CWE-121",
            "weakness_name": "stack based buffer overflow",
            "scenario": "demo",
            "flow_variant": "01",
            "suite_version": "1.3",
            "source_files": ["C/testcases/CWE121_demo_01.c"],
            "source_file_count": 1,
            "source_excerpt_truncated": False,
            "source_excerpt_sha256": "1" * 64,
            "source_grounded": True,
            "generation_method": "deterministic_source_template",
            "model_generated": False,
            "requires_human_review": True,
        },
        provenance=Provenance(
            source_id="nist-juliet-sard",
            source_url="https://samate.nist.gov/SARD/test-suites/112",
            license="CC0-1.0",
            retrieved_at="2026-09-02T00:00:00+00:00",
            upstream_revision="nist-sard-suite-112-juliet-cpp-1.3",
            record_key="CWE121_demo_01",
            content_sha256="0" * 64,
        ),
        split_group="nist-juliet-sard:CWE121_demo_01",
        benchmark_holdout=False,
        quality_score=0.9,
    )
    record.provenance.content_sha256 = record.canonical_hash()
    return record


def test_juliet_cwe_mapping_view_preserves_lineage_and_exact_answer(tmp_path: Path) -> None:
    parent = _parent()
    input_path = tmp_path / "juliet.jsonl"
    input_path.write_text(parent.model_dump_json() + "\n")
    output_path = tmp_path / "juliet-cwe.jsonl"

    manifest = derive_juliet_cwe_mapping_jsonl(
        input_path, output_path, expected_record_count=1
    )

    derived = SecurityRecord.model_validate_json(output_path.read_text())
    assert derived.task == TaskType.CWE_MAPPING
    assert derived.messages[-1].content == "CWE-121"
    assert "CWE121" not in derived.messages[0].content
    assert "CWE-121" not in derived.messages[0].content
    assert "CWE_REDACTED" in derived.messages[0].content
    assert derived.labels["target_id_redacted"] is True
    assert derived.labels["parent_record_id"] == parent.record_id
    assert derived.split_group == parent.split_group
    assert derived.provenance.content_sha256 == derived.canonical_hash()
    assert manifest["output"]["record_count"] == 1
    assert json.loads((tmp_path / "juliet-cwe.jsonl.manifest.json").read_text()) == manifest
    prepared = prepare_jsonl_many(
        [input_path, output_path],
        tmp_path / "processed",
        registry_path=Path("configs/data/sources.yaml"),
        split_mode=SplitMode.TRAIN_ONLY,
    )
    assert prepared["record_count"] == 2
    assert prepared["split_counts"]["train"] == 2
    assert prepared["exact_duplicate_count"] == 0
    assert prepared["normalized_duplicate_count"] == 0


def test_juliet_cwe_mapping_view_rejects_noncanonical_parent(tmp_path: Path) -> None:
    parent = _parent()
    parent.task = TaskType.CWE_MAPPING
    parent.provenance.content_sha256 = parent.canonical_hash()
    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text(parent.model_dump_json() + "\n")

    with pytest.raises(ValueError, match="canonical trainable review"):
        derive_juliet_cwe_mapping_jsonl(
            input_path, tmp_path / "derived.jsonl", expected_record_count=1
        )


def test_attack_technique_view_filters_prompt_target_leakage(tmp_path: Path) -> None:
    clean = SecurityRecord(
        record_id="mitre-attack-enterprise:relationship-clean",
        task=TaskType.THREAT_INTELLIGENCE,
        messages=[
            Message(role="user", content="A tool collected the host MAC address."),
            Message(role="assistant", content="System Network Configuration Discovery (T1016)"),
        ],
        labels={
            "record_kind": "attack_procedure_mapping",
            "technique_id": "T1016",
            "relationship_id": "relationship-clean",
            "source_grounded": True,
        },
        provenance=Provenance(
            source_id="mitre-attack-enterprise",
            source_url="https://example.test/attack.json",
            license="MITRE-ATTACK-Terms-of-Use",
            retrieved_at="2026-09-02T00:00:00+00:00",
            upstream_revision="fixture",
            record_key="relationship-clean",
            content_sha256="0" * 64,
        ),
        split_group="mitre-attack:technique:T1016",
    )
    clean.provenance.content_sha256 = clean.canonical_hash()
    leaking = clean.model_copy(deep=True)
    leaking.record_id = "mitre-attack-enterprise:relationship-leaking"
    leaking.messages[0].content = "This prompt already says T1016."
    leaking.provenance.record_key = "relationship-leaking"
    leaking.provenance.content_sha256 = leaking.canonical_hash()
    input_path = tmp_path / "v1.jsonl"
    input_path.write_text(clean.model_dump_json() + "\n" + leaking.model_dump_json() + "\n")
    output_path = tmp_path / "attack-id.jsonl"

    manifest = derive_attack_technique_id_jsonl(
        input_path,
        output_path,
        expected_input_count=2,
        expected_output_count=1,
    )

    derived = SecurityRecord.model_validate_json(output_path.read_text())
    assert derived.messages[-1].content == "T1016"
    assert derived.labels["parent_record_id"] == clean.record_id
    assert derived.split_group == clean.split_group
    assert manifest["filters"]["target_id_already_in_prompt"] == 1
