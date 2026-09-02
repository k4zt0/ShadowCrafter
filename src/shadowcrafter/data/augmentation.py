"""Deterministic, lineage-preserving multitask views of approved training records."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shadowcrafter.data.hygiene import record_contains_raw_binary
from shadowcrafter.data.manifest import sha256_file, write_json_exclusive
from shadowcrafter.schemas import Message, Provenance, SecurityRecord, TaskType

_JULIET_SOURCE_ID = "nist-juliet-sard"
_JULIET_INPUT_KIND = "nist_juliet_cpp_test_case"
_JULIET_OUTPUT_KIND = "nist_juliet_cpp_cwe_mapping"
_MAX_RECORDS = 100_000
_ATTACK_PROCEDURE_KIND = "attack_procedure_mapping"
_ATTACK_SOURCES = frozenset(
    {"mitre-attack-enterprise", "mitre-attack-ics", "mitre-attack-mobile"}
)
_ATTACK_ID = re.compile(r"^T[0-9]{4}(?:\.[0-9]{3})?$")


def _redact_target_cwe(text: str, cwe_id: str) -> str:
    number = str(int(cwe_id.removeprefix("CWE-")))
    pattern = re.compile(
        rf"(?<![A-Z0-9])CWE[-_ ]?0*{re.escape(number)}(?![0-9])",
        flags=re.IGNORECASE,
    )
    redacted, count = pattern.subn("CWE_REDACTED", text)
    if count == 0:
        raise ValueError("canonical Juliet source does not contain its expected CWE marker")
    if pattern.search(redacted):
        raise ValueError("Juliet target CWE marker remained after redaction")
    return redacted


def _juliet_cwe_mapping_view(record: SecurityRecord) -> SecurityRecord:
    if (
        record.provenance.source_id != _JULIET_SOURCE_ID
        or record.task != TaskType.SECURE_CODE_REVIEW
        or record.labels.get("record_kind") != _JULIET_INPUT_KIND
        or record.benchmark_holdout
        or len(record.messages) != 2
        or record.messages[0].role != "user"
        or record.messages[1].role != "assistant"
    ):
        raise ValueError("Juliet augmentation accepts only canonical trainable review records")
    cwe_id = record.labels.get("cwe_id")
    if not isinstance(cwe_id, str) or re.fullmatch(r"CWE-[0-9]+", cwe_id) is None:
        raise ValueError("canonical Juliet record lacks a valid CWE label")
    source_question = record.messages[0].content
    separator = "\n\n"
    if separator not in source_question:
        raise ValueError("canonical Juliet prompt lacks its bounded source excerpt")
    source_excerpt = _redact_target_cwe(
        source_question.split(separator, maxsplit=1)[1],
        cwe_id,
    )
    question = (
        "Map this non-executable NIST Juliet C/C++ source family to its primary CWE. "
        "Return exactly one identifier in CWE-### form and no prose. Do not compile or run it."
        f"\n\n{source_excerpt}"
    )
    labels = {
        key: value
        for key, value in record.labels.items()
        if key
        in {
            "cwe_id",
            "flow_variant",
            "scenario",
            "source_excerpt_sha256",
            "source_excerpt_truncated",
            "source_file_count",
            "source_files",
            "suite_version",
            "weakness_name",
        }
    }
    labels.update(
        {
            "record_kind": _JULIET_OUTPUT_KIND,
            "parent_record_id": record.record_id,
            "source_grounded": True,
            "target_id_redacted": True,
            "generation_method": "deterministic_multitask_view",
            "model_generated": False,
            "requires_human_review": True,
        }
    )
    derived = SecurityRecord(
        record_id=f"{record.record_id}:cwe-mapping",
        task=TaskType.CWE_MAPPING,
        risk_tier=record.risk_tier,
        language=record.language,
        messages=[
            Message(role="user", content=question),
            Message(role="assistant", content=cwe_id),
        ],
        labels=labels,
        provenance=Provenance(
            source_id=record.provenance.source_id,
            source_url=record.provenance.source_url,
            license=record.provenance.license,
            retrieved_at=record.provenance.retrieved_at.astimezone(UTC),
            upstream_revision=record.provenance.upstream_revision,
            record_key=f"{record.provenance.record_key}:cwe-mapping",
            content_sha256="0" * 64,
        ),
        split_group=record.split_group,
        benchmark_holdout=False,
        quality_score=record.quality_score,
    )
    derived.provenance.content_sha256 = derived.canonical_hash()
    if record_contains_raw_binary(derived):
        raise ValueError("derived Juliet mapping record contains a prohibited binary payload")
    return derived


def derive_juliet_cwe_mapping_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    expected_record_count: int = 64_099,
) -> dict[str, Any]:
    """Create one concise CWE classification view per canonical Juliet family."""

    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if input_path.is_symlink() or not input_path.is_file():
        raise ValueError("canonical Juliet input must be one regular file")
    if not 1 <= expected_record_count <= _MAX_RECORDS:
        raise ValueError("expected Juliet augmentation count is outside its bound")
    existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite Juliet augmentation artifacts: {existing}")
    input_sha256 = sha256_file(input_path)
    records: list[SecurityRecord] = []
    parent_ids: set[str] = set()
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(records) >= _MAX_RECORDS:
                raise ValueError("canonical Juliet input exceeds its record bound")
            try:
                parent = SecurityRecord.model_validate_json(line)
            except Exception as error:
                raise ValueError(f"invalid Juliet record at line {line_number}") from error
            if parent.record_id in parent_ids:
                raise ValueError("canonical Juliet input contains duplicate record IDs")
            parent_ids.add(parent.record_id)
            records.append(_juliet_cwe_mapping_view(parent))
    if len(records) != expected_record_count:
        raise ValueError(
            "Juliet augmentation count differs from its immutable input contract: "
            f"expected {expected_record_count}, observed {len(records)}"
        )
    if sha256_file(input_path) != input_sha256:
        raise ValueError("canonical Juliet input changed while augmentation was generated")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as writer:
        for record in records:
            writer.write(record.model_dump_json() + "\n")
    manifest = {
        "schema_version": 1,
        "adapter": "nist_juliet_cpp_cwe_mapping_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_id": _JULIET_SOURCE_ID,
        "license_id": records[0].provenance.license,
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "record_count": expected_record_count,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "record_count": len(records),
        },
        "controls": {
            "deterministic": True,
            "one_view_per_parent": True,
            "parent_split_group_preserved": True,
            "benchmark_content_used": False,
            "source_never_executed": True,
            "target_id_redacted": True,
        },
    }
    write_json_exclusive(manifest_path, manifest)
    return manifest


def _attack_technique_id_view(record: SecurityRecord) -> SecurityRecord | None:
    if record.labels.get("record_kind") != _ATTACK_PROCEDURE_KIND:
        return None
    technique_id = record.labels.get("technique_id")
    if (
        record.provenance.source_id not in _ATTACK_SOURCES
        or record.task != TaskType.THREAT_INTELLIGENCE
        or record.benchmark_holdout
        or len(record.messages) != 2
        or record.messages[0].role != "user"
        or record.messages[1].role != "assistant"
        or not isinstance(technique_id, str)
        or _ATTACK_ID.fullmatch(technique_id) is None
    ):
        raise ValueError("ATT&CK augmentation found a malformed canonical procedure record")
    source_prompt = record.messages[0].content
    suffix = r"(?![A-Z0-9])" if "." in technique_id else r"(?![A-Z0-9]|\.[0-9]{3})"
    if re.search(
        rf"(?<![A-Z0-9]){re.escape(technique_id)}{suffix}",
        source_prompt,
        flags=re.IGNORECASE,
    ):
        return None
    question = (
        "Map this source-grounded defensive procedure to its MITRE ATT&CK technique. "
        "Return only one technique or sub-technique identifier in T#### or T####.### form "
        "and no prose.\n\n"
        f"{source_prompt}"
    )
    labels = {
        "record_kind": "attack_procedure_id_mapping",
        "technique_id": technique_id,
        "parent_record_id": record.record_id,
        "source_grounded": True,
        "generation_method": "deterministic_multitask_view",
        "model_generated": False,
        "requires_human_review": True,
    }
    for key in ("relationship_id", "source_excerpt_sha256", "source_object_type"):
        if key in record.labels:
            labels[key] = record.labels[key]
    derived = SecurityRecord(
        record_id=f"{record.record_id}:technique-id",
        task=TaskType.THREAT_INTELLIGENCE,
        risk_tier=record.risk_tier,
        language=record.language,
        messages=[
            Message(role="user", content=question),
            Message(role="assistant", content=technique_id),
        ],
        labels=labels,
        provenance=Provenance(
            source_id=record.provenance.source_id,
            source_url=record.provenance.source_url,
            license=record.provenance.license,
            retrieved_at=record.provenance.retrieved_at.astimezone(UTC),
            upstream_revision=record.provenance.upstream_revision,
            record_key=f"{record.provenance.record_key}:technique-id",
            content_sha256="0" * 64,
        ),
        split_group=record.split_group,
        benchmark_holdout=False,
        quality_score=record.quality_score,
    )
    derived.provenance.content_sha256 = derived.canonical_hash()
    if record_contains_raw_binary(derived):
        raise ValueError("derived ATT&CK mapping record contains a prohibited binary payload")
    return derived


def derive_attack_technique_id_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    expected_input_count: int = 28_140,
    expected_output_count: int = 17_639,
) -> dict[str, Any]:
    """Create concise ID-only views from non-leaking canonical ATT&CK procedures."""

    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if input_path.is_symlink() or not input_path.is_file():
        raise ValueError("canonical V1 input must be one regular file")
    if not (
        1 <= expected_input_count <= _MAX_RECORDS
        and 1 <= expected_output_count <= _MAX_RECORDS
    ):
        raise ValueError("expected ATT&CK augmentation counts are outside their bounds")
    existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite ATT&CK augmentation artifacts: {existing}")
    input_sha256 = sha256_file(input_path)
    input_count = 0
    procedure_count = 0
    target_leak_count = 0
    records: list[SecurityRecord] = []
    source_counts: dict[str, int] = {}
    license_ids: set[str] = set()
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            input_count += 1
            if input_count > _MAX_RECORDS:
                raise ValueError("canonical V1 input exceeds its record bound")
            try:
                parent = SecurityRecord.model_validate_json(line)
            except Exception as error:
                raise ValueError(f"invalid canonical V1 record at line {line_number}") from error
            if parent.labels.get("record_kind") == _ATTACK_PROCEDURE_KIND:
                procedure_count += 1
            derived = _attack_technique_id_view(parent)
            if derived is None:
                if parent.labels.get("record_kind") == _ATTACK_PROCEDURE_KIND:
                    target_leak_count += 1
                continue
            records.append(derived)
            source_id = derived.provenance.source_id
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
            license_ids.add(derived.provenance.license)
    if input_count != expected_input_count or len(records) != expected_output_count:
        raise ValueError(
            "ATT&CK augmentation counts differ from the immutable V1 contract: "
            f"input={input_count}, output={len(records)}"
        )
    if procedure_count != len(records) + target_leak_count:
        raise ValueError("ATT&CK augmentation accounting is inconsistent")
    if sha256_file(input_path) != input_sha256:
        raise ValueError("canonical V1 input changed while augmentation was generated")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as writer:
        for record in records:
            writer.write(record.model_dump_json() + "\n")
    manifest = {
        "schema_version": 1,
        "adapter": "mitre_attack_procedure_id_mapping_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_ids": sorted(source_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "license_ids": sorted(license_ids),
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "record_count": input_count,
            "procedure_record_count": procedure_count,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "record_count": len(records),
        },
        "filters": {"target_id_already_in_prompt": target_leak_count},
        "controls": {
            "deterministic": True,
            "parent_split_group_preserved": True,
            "benchmark_content_used": False,
            "target_leakage_filtered": True,
        },
    }
    write_json_exclusive(manifest_path, manifest)
    return manifest


__all__ = ["derive_attack_technique_id_jsonl", "derive_juliet_cwe_mapping_jsonl"]
