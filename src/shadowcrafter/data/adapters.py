"""Bounded, deterministic adapters from approved defensive sources to SecurityRecord."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from shadowcrafter.data.hygiene import (
    contains_executable_binary_payload,
    record_contains_raw_binary,
)
from shadowcrafter.data.manifest import sha256_file, write_json_exclusive
from shadowcrafter.data.registry import (
    ContentKind,
    DataSource,
    Purpose,
    SourceRegistry,
    load_registry,
)
from shadowcrafter.schemas import (
    Message,
    Provenance,
    RiskTier,
    SecurityRecord,
    TaskType,
)

_DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024
_MAX_OBJECTS = 1_000_000
_MAX_OUTPUT_RECORDS = 250_000
_MAX_TEXT_CHARS = 20_000
_BLOCKED_MARKUP = re.compile(r"(?is)<(?:[A-Za-z0-9_-]+:)?(?:script|iframe|object|embed|code)\b|```")
_ACTIONABLE_PAYLOAD = (
    re.compile(r"(?im)^\s*(?:curl|wget|powershell|pwsh|cmd\.exe|bash|sh)\s+[-/]"),
    re.compile(r"(?i)(?:^|[.;:]\s+)(?:curl|wget|powershell|pwsh|cmd\.exe|bash|sh)\s+[-/]"),
    re.compile(r"(?i)\b(?:msfvenom|meterpreter|reverse\s+shell|bind\s+shell)\b"),
    re.compile(r"(?i)\b(?:shellcode|weaponized\s+payload|exploit\s+payload)\b"),
    re.compile(r"(?i)https?://\S+\.(?:exe|dll|ps1|bat|cmd|sh|bin)(?:\W|$)"),
)


class AdapterKind(StrEnum):
    ATTACK_STIX = "attack_stix"
    CWE_XML = "cwe_xml"
    CWE_JSON = "cwe_json"
    DEFENSIVE_CATALOG_JSON = "defensive_catalog_json"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _read_bounded(path: Path, max_bytes: int) -> bytes:
    if max_bytes <= 0 or max_bytes > _DEFAULT_MAX_INPUT_BYTES:
        raise ValueError(f"max_input_bytes must be between 1 and {_DEFAULT_MAX_INPUT_BYTES}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"canonicalization input exceeds size limit: {size} > {max_bytes}")
    with path.open("rb") as handle:
        content = handle.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"canonicalization input exceeds size limit: > {max_bytes}")
    return content


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json(content: bytes) -> Any:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("canonicalization input must be UTF-8") from exc
    if contains_executable_binary_payload(text):
        raise ValueError("raw executable or archive payload detected in source input")
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise ValueError("canonicalization input is not valid JSON") from exc


def _flatten_text(value: Any, *, depth: int = 0) -> str:
    if depth > 12:
        raise ValueError("source text nesting exceeds limit")
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, Mapping):
        preferred = ("#text", "text", "value", "description")
        for key in preferred:
            if key in value:
                return _flatten_text(value[key], depth=depth + 1)
        return " ".join(_flatten_text(item, depth=depth + 1) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_flatten_text(item, depth=depth + 1) for item in value)
    return ""


def _source_text(value: Any) -> str | None:
    raw = html.unescape(_flatten_text(value)).strip()
    if not raw or len(raw) > _MAX_TEXT_CHARS:
        return None
    if contains_executable_binary_payload(raw) or _BLOCKED_MARKUP.search(raw):
        return None
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        parser.close()
    except ValueError:
        return None
    plain = " ".join(" ".join(parser.parts).split())
    if not plain or len(plain) > _MAX_TEXT_CHARS:
        return None
    if any(pattern.search(plain) for pattern in _ACTIONABLE_PAYLOAD):
        return None
    return plain


def _parse_event_time(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _first_event_time(*values: Any) -> str | None:
    for value in values:
        parsed = _parse_event_time(value)
        if parsed is not None:
            return parsed
    return None


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return None


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.casefold())


def _pick(record: Mapping[str, Any], *names: str) -> Any | None:
    normalized = {_normalized_key(str(key)): value for key, value in record.items()}
    for name in names:
        key = _normalized_key(name)
        if key in normalized:
            return normalized[key]
    return None


def _external_id(stix_object: Mapping[str, Any], prefix: str) -> str | None:
    references = _sequence(stix_object.get("external_references")) or []
    for reference_value in references:
        reference = _mapping(reference_value)
        if reference is None:
            continue
        external_id = reference.get("external_id")
        if isinstance(external_id, str) and external_id.startswith(prefix):
            return external_id
    return None


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip("-")
    if cleaned:
        if len(cleaned) <= 180:
            return cleaned
        return f"{cleaned[:155]}-{hashlib.sha256(value.encode()).hexdigest()[:16]}"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _build_record(
    *,
    source: DataSource,
    upstream_revision: str,
    retrieved_at: datetime,
    record_key: str,
    record_id: str,
    split_group: str,
    task: TaskType,
    question: str,
    answer: str,
    labels: dict[str, Any],
) -> SecurityRecord:
    if not upstream_revision.strip():
        raise ValueError("upstream_revision must not be blank")
    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware")
    safe_question = _source_text(question)
    safe_answer = _source_text(answer)
    if safe_question is None or safe_answer is None:
        raise ValueError("adapter attempted to emit unsafe or unbounded text")
    if source.url is None:
        raise ValueError(f"source {source.id} does not have a canonical URL")

    record = SecurityRecord(
        record_id=record_id,
        task=task,
        risk_tier=RiskTier.DEFENSIVE,
        language="en",
        messages=[
            Message(role="user", content=safe_question),
            Message(role="assistant", content=safe_answer),
        ],
        labels={
            **labels,
            "source_grounded": True,
            "generation_method": "deterministic_source_template",
            "model_generated": False,
            "requires_human_review": True,
        },
        provenance=Provenance(
            source_id=source.id,
            source_url=str(source.url),
            license=source.license.id,
            retrieved_at=retrieved_at.astimezone(UTC),
            upstream_revision=upstream_revision,
            record_key=record_key,
            content_sha256="0" * 64,
        ),
        split_group=split_group,
        benchmark_holdout=False,
        quality_score=0.8,
    )
    record.provenance.content_sha256 = record.canonical_hash()
    if record_contains_raw_binary(record):
        raise ValueError("adapter produced a prohibited binary-bearing record")
    return record


def _adapt_attack_stix(
    payload: Any,
    source: DataSource,
    upstream_revision: str,
    retrieved_at: datetime,
) -> tuple[list[SecurityRecord], dict[str, int]]:
    bundle = _mapping(payload)
    if bundle is None or bundle.get("type") != "bundle":
        raise ValueError("ATT&CK adapter requires a STIX bundle object")
    object_values = _sequence(bundle.get("objects"))
    if object_values is None:
        raise ValueError("ATT&CK STIX bundle must contain an objects array")
    if len(object_values) > _MAX_OBJECTS:
        raise ValueError("ATT&CK STIX object count exceeds limit")

    objects: dict[str, Mapping[str, Any]] = {}
    relationships: list[Mapping[str, Any]] = []
    for value in object_values:
        item = _mapping(value)
        if item is None:
            continue
        object_id = item.get("id")
        if isinstance(object_id, str):
            if object_id in objects:
                raise ValueError(f"duplicate STIX object id: {object_id}")
            objects[object_id] = item
        if item.get("type") == "relationship" and item.get("relationship_type") == "mitigates":
            relationships.append(item)

    records: list[SecurityRecord] = []
    stats = {"candidate_count": len(relationships), "unsafe_skipped": 0, "invalid_skipped": 0}
    for relationship in sorted(relationships, key=lambda item: str(item.get("id", ""))):
        source_ref = relationship.get("source_ref")
        target_ref = relationship.get("target_ref")
        relationship_id = relationship.get("id")
        if not all(isinstance(value, str) for value in (source_ref, target_ref, relationship_id)):
            stats["invalid_skipped"] += 1
            continue
        mitigation = objects.get(str(source_ref))
        technique = objects.get(str(target_ref))
        if mitigation is None or technique is None:
            stats["invalid_skipped"] += 1
            continue
        if (
            mitigation.get("type") != "course-of-action"
            or technique.get("type") != "attack-pattern"
        ):
            stats["invalid_skipped"] += 1
            continue
        if any(
            item.get("revoked") is True or item.get("x_mitre_deprecated") is True
            for item in (relationship, mitigation, technique)
        ):
            stats["invalid_skipped"] += 1
            continue

        mitigation_name = _source_text(mitigation.get("name"))
        technique_name = _source_text(technique.get("name"))
        description = _source_text(mitigation.get("description"))
        published_at = _first_event_time(
            relationship.get("created"), mitigation.get("created"), technique.get("created")
        )
        modified_at = _first_event_time(
            relationship.get("modified"), mitigation.get("modified"), technique.get("modified")
        )
        if not mitigation_name or not technique_name or not description or not published_at:
            stats["unsafe_skipped"] += 1
            continue

        technique_id = _external_id(technique, "T") or str(target_ref)
        mitigation_id = _external_id(mitigation, "M") or str(source_ref)
        question = (
            f"What defensive mitigation does MITRE ATT&CK specify for "
            f"{technique_name} ({technique_id})?"
        )
        answer = f"{mitigation_name} ({mitigation_id}): {description}"
        labels = {
            "record_kind": "attack_mitigation",
            "technique_id": technique_id,
            "mitigation_id": mitigation_id,
            "relationship_id": relationship_id,
            "published_at": published_at,
            "modified_at": modified_at or published_at,
            "source_excerpt_sha256": hashlib.sha256(description.encode()).hexdigest(),
        }
        records.append(
            _build_record(
                source=source,
                upstream_revision=upstream_revision,
                retrieved_at=retrieved_at,
                record_key=str(relationship_id),
                record_id=f"{source.id}:{_safe_identifier(str(relationship_id))}",
                # A mitigation can apply to many techniques and can appear in more
                # than one ATT&CK domain. Keep the entire mitigation lineage in one
                # split so its identical defensive answer cannot leak across gates.
                split_group=f"mitre-attack:mitigation:{_safe_identifier(mitigation_id)}",
                task=TaskType.DETECTION_ENGINEERING,
                question=question,
                answer=answer,
                labels=labels,
            )
        )
        if len(records) > _MAX_OUTPUT_RECORDS:
            raise ValueError("canonical output record count exceeds limit")
    return records, stats


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(part.strip() for part in element.itertext() if part.strip())


def _descendants(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element.iter() if _local_name(child.tag) == name]


def _adapt_cwe_entries(
    entries: Sequence[Mapping[str, Any]],
    source: DataSource,
    upstream_revision: str,
    retrieved_at: datetime,
    catalog_date: Any,
) -> tuple[list[SecurityRecord], dict[str, int]]:
    if len(entries) > _MAX_OBJECTS:
        raise ValueError("CWE entry count exceeds limit")
    records: list[SecurityRecord] = []
    stats = {"candidate_count": 0, "unsafe_skipped": 0, "invalid_skipped": 0}
    for entry in sorted(entries, key=lambda item: str(_pick(item, "id", "cwe_id") or "")):
        weakness_id = _source_text(_pick(entry, "id", "cwe_id"))
        weakness_name = _source_text(_pick(entry, "name", "title"))
        status = _source_text(_pick(entry, "status")) or ""
        if not weakness_id or not weakness_name or status.casefold() == "deprecated":
            stats["invalid_skipped"] += 1
            continue
        cwe_id = weakness_id if weakness_id.upper().startswith("CWE-") else f"CWE-{weakness_id}"
        mitigations_value = _pick(entry, "potential_mitigations", "mitigations")
        mitigations_container = _mapping(mitigations_value)
        if mitigations_container is not None:
            mitigations_value = _pick(mitigations_container, "mitigation", "items")
        mitigation_values = _sequence(mitigations_value)
        if mitigation_values is None and _mapping(mitigations_value) is not None:
            mitigation_values = [mitigations_value]
        if mitigation_values is None:
            stats["invalid_skipped"] += 1
            continue

        for mitigation_value in mitigation_values:
            stats["candidate_count"] += 1
            mitigation = _mapping(mitigation_value)
            if mitigation is None:
                stats["invalid_skipped"] += 1
                continue
            description = _source_text(_pick(mitigation, "description", "guidance", "requirement"))
            phase = _source_text(_pick(mitigation, "phase", "lifecycle_phase"))
            mitigation_id = _source_text(_pick(mitigation, "mitigation_id", "id"))
            published_at = _first_event_time(
                _pick(entry, "published_at", "created", "date"), catalog_date
            )
            modified_at = _first_event_time(
                _pick(entry, "modified_at", "modified", "last_modified"), published_at
            )
            if not description or not published_at:
                stats["unsafe_skipped"] += 1
                continue
            phase_label = phase or "the applicable development lifecycle phase"
            stable_mitigation_id = (
                mitigation_id
                or hashlib.sha256(f"{phase_label}\n{description}".encode()).hexdigest()[:16]
            )
            question = (
                f"What source-defined mitigation should defenders apply for "
                f"{cwe_id} ({weakness_name}) during {phase_label}?"
            )
            answer = f"{phase_label}: {description}"
            record_key = f"{cwe_id}:mitigation:{stable_mitigation_id}"
            records.append(
                _build_record(
                    source=source,
                    upstream_revision=upstream_revision,
                    retrieved_at=retrieved_at,
                    record_key=record_key,
                    record_id=f"{source.id}:{_safe_identifier(record_key)}",
                    split_group=f"{source.id}:{_safe_identifier(cwe_id)}",
                    task=TaskType.SECURE_CODE_REVIEW,
                    question=question,
                    answer=answer,
                    labels={
                        "record_kind": "cwe_mitigation",
                        "cwe_id": cwe_id,
                        "mitigation_id": stable_mitigation_id,
                        "phase": phase_label,
                        "published_at": published_at,
                        "modified_at": modified_at or published_at,
                        "source_excerpt_sha256": hashlib.sha256(description.encode()).hexdigest(),
                    },
                )
            )
            if len(records) > _MAX_OUTPUT_RECORDS:
                raise ValueError("canonical output record count exceeds limit")
    return records, stats


def _adapt_cwe_json(
    payload: Any,
    source: DataSource,
    upstream_revision: str,
    retrieved_at: datetime,
) -> tuple[list[SecurityRecord], dict[str, int]]:
    catalog = _mapping(payload)
    if catalog is None:
        values = _sequence(payload)
        if values is None:
            raise ValueError("CWE JSON must be an object or array")
        entries = [entry for value in values if (entry := _mapping(value)) is not None]
        catalog_date = None
    else:
        entries_value = _pick(catalog, "weaknesses", "items", "records")
        entries_container = _mapping(entries_value)
        if entries_container is not None:
            entries_value = _pick(entries_container, "weakness", "items")
        values = _sequence(entries_value)
        if values is None:
            raise ValueError("CWE JSON does not contain a weakness array")
        entries = [entry for value in values if (entry := _mapping(value)) is not None]
        catalog_date = _pick(catalog, "date", "release_date", "published_at")
    return _adapt_cwe_entries(entries, source, upstream_revision, retrieved_at, catalog_date)


def _adapt_cwe_xml(
    content: bytes,
    source: DataSource,
    upstream_revision: str,
    retrieved_at: datetime,
) -> tuple[list[SecurityRecord], dict[str, int]]:
    try:
        xml_text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("canonicalization input must be UTF-8") from exc
    if contains_executable_binary_payload(xml_text):
        raise ValueError("raw executable or archive payload detected in CWE XML")
    if re.search(rb"(?i)<!DOCTYPE|<!ENTITY", content):
        raise ValueError("DTD and entity declarations are prohibited in CWE XML")
    try:
        root = ET.fromstring(content)  # noqa: S314 - bounded input; DTD/entities rejected above.
    except ET.ParseError as exc:
        raise ValueError("canonicalization input is not valid CWE XML") from exc
    elements = list(root.iter())
    if len(elements) > _MAX_OBJECTS:
        raise ValueError("CWE XML element count exceeds limit")
    for element in elements:
        for value in (*element.attrib.values(), element.text or "", element.tail or ""):
            if contains_executable_binary_payload(value):
                raise ValueError("raw executable or archive payload detected in CWE XML")

    catalog_date = root.attrib.get("Date") or root.attrib.get("date")
    entries: list[Mapping[str, Any]] = []
    for weakness in (element for element in elements if _local_name(element.tag) == "Weakness"):
        mitigations: list[dict[str, Any]] = []
        potential = next(
            (child for child in weakness if _local_name(child.tag) == "Potential_Mitigations"),
            None,
        )
        if potential is not None:
            for mitigation in (
                child for child in potential if _local_name(child.tag) == "Mitigation"
            ):
                description_element = next(
                    (
                        child
                        for child in mitigation.iter()
                        if _local_name(child.tag) == "Description"
                    ),
                    None,
                )
                phases = [
                    _element_text(child)
                    for child in mitigation.iter()
                    if _local_name(child.tag) == "Phase"
                ]
                mitigations.append(
                    {
                        "mitigation_id": mitigation.attrib.get("Mitigation_ID"),
                        "phase": ", ".join(phase for phase in phases if phase),
                        "description": _element_text(description_element),
                    }
                )
        modification_dates = [
            _element_text(child)
            for child in _descendants(weakness, "Modification_Date")
            if _parse_event_time(_element_text(child)) is not None
        ]
        entries.append(
            {
                "id": weakness.attrib.get("ID"),
                "name": weakness.attrib.get("Name"),
                "status": weakness.attrib.get("Status"),
                "modified_at": max(modification_dates, default=catalog_date or ""),
                "potential_mitigations": mitigations,
            }
        )
    return _adapt_cwe_entries(entries, source, upstream_revision, retrieved_at, catalog_date)


def _adapt_defensive_catalog_json(
    payload: Any,
    source: DataSource,
    upstream_revision: str,
    retrieved_at: datetime,
) -> tuple[list[SecurityRecord], dict[str, int]]:
    if source.safety.content_kind != ContentKind.DEFENSIVE_GUIDANCE:
        raise ValueError("defensive catalog adapter requires defensive_guidance source content")
    catalog = _mapping(payload)
    if catalog is None:
        values = _sequence(payload)
    else:
        values = _sequence(_pick(catalog, "requirements", "items", "records"))
    if values is None:
        raise ValueError("defensive catalog JSON must contain a requirements array")
    if len(values) > _MAX_OBJECTS:
        raise ValueError("defensive catalog record count exceeds limit")

    records: list[SecurityRecord] = []
    stats = {"candidate_count": len(values), "unsafe_skipped": 0, "invalid_skipped": 0}
    for value in values:
        item = _mapping(value)
        if item is None:
            stats["invalid_skipped"] += 1
            continue
        if _pick(item, "model_generated") is not False or _pick(item, "human_reviewed") is not True:
            stats["invalid_skipped"] += 1
            continue
        requirement_id = _source_text(_pick(item, "id", "requirement_id", "control_id"))
        title = _source_text(_pick(item, "title", "name", "category"))
        guidance = _source_text(_pick(item, "requirement", "guidance", "description"))
        published_at = _first_event_time(_pick(item, "published_at", "created", "date"))
        modified_at = _first_event_time(
            _pick(item, "modified_at", "modified", "last_modified"), published_at
        )
        if not requirement_id or not title or not guidance or not published_at:
            stats["unsafe_skipped"] += 1
            continue
        question = (
            f"What source-defined defensive requirement applies to {title} ({requirement_id})?"
        )
        answer = f"{title} ({requirement_id}): {guidance}"
        records.append(
            _build_record(
                source=source,
                upstream_revision=upstream_revision,
                retrieved_at=retrieved_at,
                record_key=requirement_id,
                record_id=f"{source.id}:{_safe_identifier(requirement_id)}",
                split_group=f"{source.id}:{_safe_identifier(requirement_id)}",
                task=TaskType.SECURE_CODE_REVIEW,
                question=question,
                answer=answer,
                labels={
                    "record_kind": "defensive_requirement",
                    "requirement_id": requirement_id,
                    "source_human_reviewed": True,
                    "published_at": published_at,
                    "modified_at": modified_at or published_at,
                    "source_excerpt_sha256": hashlib.sha256(guidance.encode()).hexdigest(),
                },
            )
        )
        if len(records) > _MAX_OUTPUT_RECORDS:
            raise ValueError("canonical output record count exceeds limit")
    return records, stats


def _resolve_adapter(
    source: DataSource, input_path: Path, adapter: AdapterKind | None
) -> AdapterKind:
    if adapter is not None:
        return adapter
    if source.id.startswith("mitre-attack-"):
        return AdapterKind.ATTACK_STIX
    if source.id == "mitre-cwe":
        return (
            AdapterKind.CWE_XML if input_path.suffix.casefold() == ".xml" else AdapterKind.CWE_JSON
        )
    if source.safety.content_kind == ContentKind.DEFENSIVE_GUIDANCE:
        return AdapterKind.DEFENSIVE_CATALOG_JSON
    raise ValueError(f"no bounded canonical adapter is registered for source {source.id}")


def _assert_unique_output(records: list[SecurityRecord]) -> None:
    by_id: dict[str, str] = {}
    for record in records:
        digest = record.canonical_hash()
        previous = by_id.get(record.record_id)
        if previous is None:
            by_id[record.record_id] = digest
        elif previous != digest:
            raise ValueError(f"adapter produced conflicting record id: {record.record_id}")
        else:
            raise ValueError(f"adapter produced duplicate record id: {record.record_id}")


def canonicalize_downloaded_source(
    input_path: Path,
    output_path: Path,
    *,
    source_id: str,
    upstream_revision: str,
    retrieved_at: datetime,
    registry_path: Path = Path("configs/data/sources.yaml"),
    adapter: AdapterKind | None = None,
    max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES,
) -> dict[str, Any]:
    """Canonicalize one local source snapshot without network or model generation."""

    registry: SourceRegistry = load_registry(registry_path)
    # This is deliberately the first source-specific operation: all broader source
    # classes fail before parsing or adapter selection.
    source = registry.require_purpose(source_id, Purpose.TRAIN)
    if not upstream_revision.strip():
        raise ValueError("upstream_revision must not be blank")
    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware")
    selected_adapter = _resolve_adapter(source, input_path, adapter)
    content = _read_bounded(input_path, max_input_bytes)

    if selected_adapter == AdapterKind.CWE_XML:
        if source.id != "mitre-cwe":
            raise ValueError("CWE XML adapter is restricted to the registered MITRE CWE source")
        records, stats = _adapt_cwe_xml(content, source, upstream_revision, retrieved_at)
    else:
        payload = _load_json(content)
        if selected_adapter == AdapterKind.ATTACK_STIX:
            if not source.id.startswith("mitre-attack-"):
                raise ValueError("ATT&CK adapter is restricted to registered ATT&CK sources")
            records, stats = _adapt_attack_stix(payload, source, upstream_revision, retrieved_at)
        elif selected_adapter == AdapterKind.CWE_JSON:
            if source.id != "mitre-cwe":
                raise ValueError(
                    "CWE JSON adapter is restricted to the registered MITRE CWE source"
                )
            records, stats = _adapt_cwe_json(payload, source, upstream_revision, retrieved_at)
        elif selected_adapter == AdapterKind.DEFENSIVE_CATALOG_JSON:
            records, stats = _adapt_defensive_catalog_json(
                payload, source, upstream_revision, retrieved_at
            )
        else:
            raise AssertionError(f"unsupported adapter: {selected_adapter}")

    records.sort(key=lambda record: record.record_id)
    if not records:
        raise ValueError("adapter produced no safe source-grounded records")
    _assert_unique_output(records)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite canonical artifacts: {existing}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as writer:
        for record in records:
            writer.write(record.model_dump_json() + "\n")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "adapter": selected_adapter,
        "source_id": source.id,
        "source_policy_class": source.policy_class,
        "license_id": source.license.id,
        "upstream_revision": upstream_revision,
        "retrieved_at_utc": retrieved_at.astimezone(UTC).isoformat(),
        "registry_sha256": registry.canonical_sha256(),
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "size": input_path.stat().st_size,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size": output_path.stat().st_size,
            "record_count": len(records),
        },
        "statistics": stats,
        "controls": {
            "source_grounded_only": True,
            "model_generated": False,
            "generic_input_requires_human_reviewed_non_model_declaration": True,
            "human_review_required": True,
            "raw_binary_rejection": True,
            "actionable_payload_filter": True,
        },
    }
    write_json_exclusive(manifest_path, manifest)
    return manifest
