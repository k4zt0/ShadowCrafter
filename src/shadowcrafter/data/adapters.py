"""Bounded, deterministic adapters from approved defensive sources to SecurityRecord."""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import stat
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

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
_MAX_ARCHIVE_ENTRIES = 64
_MAX_ARCHIVE_COMPRESSION_RATIO = 100
_MAX_RULE_FILE_BYTES = 1024 * 1024
_MAX_RULE_TREE_BYTES = 256 * 1024 * 1024
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
    CAPEC_XML = "capec_xml"
    CWE_XML = "cwe_xml"
    CWE_JSON = "cwe_json"
    DEFENSIVE_CATALOG_JSON = "defensive_catalog_json"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError("YAML mapping key must be hashable") from exc
        if duplicate:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


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


def _read_single_xml_archive(content: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > _MAX_ARCHIVE_ENTRIES:
                raise ValueError("source archive has an invalid entry count")
            candidates: list[zipfile.ZipInfo] = []
            for entry in entries:
                path = Path(entry.filename)
                if (
                    entry.is_dir()
                    or path.is_absolute()
                    or ".." in path.parts
                    or len(path.parts) != 1
                    or entry.flag_bits & 0x1
                ):
                    raise ValueError("source archive contains an unsafe entry")
                if entry.file_size > _DEFAULT_MAX_INPUT_BYTES:
                    raise ValueError("source archive XML exceeds the input size limit")
                if entry.file_size > max(1, entry.compress_size) * _MAX_ARCHIVE_COMPRESSION_RATIO:
                    raise ValueError("source archive exceeds the compression-ratio limit")
                if path.suffix.casefold() == ".xml":
                    candidates.append(entry)
            if len(candidates) != 1:
                raise ValueError("source archive must contain exactly one XML catalog")
            extracted = archive.read(candidates[0])
            if len(extracted) != candidates[0].file_size:
                raise ValueError("source archive XML size changed during extraction")
            return extracted
    except zipfile.BadZipFile as exc:
        raise ValueError("source archive is not a valid ZIP file") from exc


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
        if item.get("type") == "relationship" and item.get("relationship_type") in {
            "detects",
            "mitigates",
            "uses",
        }:
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
        source_object = objects.get(str(source_ref))
        technique = objects.get(str(target_ref))
        if source_object is None or technique is None:
            stats["invalid_skipped"] += 1
            continue
        if technique.get("type") != "attack-pattern":
            stats["invalid_skipped"] += 1
            continue
        if any(
            item.get("revoked") is True or item.get("x_mitre_deprecated") is True
            for item in (relationship, source_object, technique)
        ):
            stats["invalid_skipped"] += 1
            continue

        technique_name = _source_text(technique.get("name"))
        published_at = _first_event_time(
            relationship.get("created"), source_object.get("created"), technique.get("created")
        )
        modified_at = _first_event_time(
            relationship.get("modified"), source_object.get("modified"), technique.get("modified")
        )
        if not technique_name or not published_at:
            stats["unsafe_skipped"] += 1
            continue

        technique_id = _external_id(technique, "T") or str(target_ref)
        relationship_type = relationship.get("relationship_type")
        if relationship_type == "mitigates":
            if source_object.get("type") != "course-of-action":
                stats["invalid_skipped"] += 1
                continue
            mitigation_name = _source_text(source_object.get("name"))
            description = _source_text(source_object.get("description"))
            if not mitigation_name or not description:
                stats["unsafe_skipped"] += 1
                continue
            mitigation_id = _external_id(source_object, "M") or str(source_ref)
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
        elif relationship_type == "detects":
            if source_object.get("type") != "x-mitre-detection-strategy":
                stats["invalid_skipped"] += 1
                continue
            strategy_name = _source_text(source_object.get("name"))
            strategy_id = _external_id(source_object, "DET") or str(source_ref)
            analytic_refs = _sequence(source_object.get("x_mitre_analytic_refs")) or []
            emitted = 0
            for analytic_ref in sorted(str(value) for value in analytic_refs):
                analytic = objects.get(analytic_ref)
                if analytic is None or analytic.get("type") != "x-mitre-analytic":
                    continue
                if analytic.get("revoked") is True or analytic.get("x_mitre_deprecated") is True:
                    continue
                analytic_name = _source_text(analytic.get("name"))
                description = _source_text(analytic.get("description"))
                if not strategy_name or not analytic_name or not description:
                    continue
                analytic_id = _external_id(analytic, "AN") or analytic_ref
                log_sources: list[str] = []
                for log_source_value in (
                    _sequence(analytic.get("x_mitre_log_source_references")) or []
                ):
                    log_source = _mapping(log_source_value)
                    if log_source is None:
                        continue
                    name = _source_text(log_source.get("name"))
                    channel = _source_text(log_source.get("channel"))
                    label = ": ".join(part for part in (name, channel) if part)
                    if label:
                        log_sources.append(label)
                log_suffix = (
                    f" Log sources: {'; '.join(sorted(set(log_sources)))}." if log_sources else ""
                )
                question = (
                    f"What MITRE ATT&CK detection analytic should defenders use for "
                    f"{technique_name} ({technique_id})?"
                )
                answer = (
                    f"{strategy_name} ({strategy_id}), {analytic_name} ({analytic_id}): "
                    f"{description}{log_suffix}"
                )
                record_key = f"{relationship_id}:{analytic_ref}"
                records.append(
                    _build_record(
                        source=source,
                        upstream_revision=upstream_revision,
                        retrieved_at=retrieved_at,
                        record_key=record_key,
                        record_id=f"{source.id}:{_safe_identifier(record_key)}",
                        split_group=(f"mitre-attack:technique:{_safe_identifier(technique_id)}"),
                        task=TaskType.DETECTION_ENGINEERING,
                        question=question,
                        answer=answer,
                        labels={
                            "record_kind": "attack_detection_analytic",
                            "technique_id": technique_id,
                            "strategy_id": strategy_id,
                            "analytic_id": analytic_id,
                            "relationship_id": relationship_id,
                            "published_at": published_at,
                            "modified_at": modified_at or published_at,
                            "source_excerpt_sha256": hashlib.sha256(
                                description.encode()
                            ).hexdigest(),
                        },
                    )
                )
                emitted += 1
            if emitted == 0:
                stats["unsafe_skipped"] += 1
        elif relationship_type == "uses":
            if source_object.get("type") not in {
                "campaign",
                "intrusion-set",
                "malware",
                "tool",
            }:
                stats["invalid_skipped"] += 1
                continue
            actor_name = _source_text(source_object.get("name"))
            procedure = _source_text(relationship.get("description"))
            if not actor_name or not procedure:
                stats["unsafe_skipped"] += 1
                continue
            question = (
                "Map this source-grounded MITRE ATT&CK procedure to the correct technique "
                f"for defensive threat intelligence: {actor_name} — {procedure}"
            )
            answer = f"{technique_name} ({technique_id})"
            records.append(
                _build_record(
                    source=source,
                    upstream_revision=upstream_revision,
                    retrieved_at=retrieved_at,
                    record_key=str(relationship_id),
                    record_id=f"{source.id}:{_safe_identifier(str(relationship_id))}",
                    split_group=f"mitre-attack:technique:{_safe_identifier(technique_id)}",
                    task=TaskType.THREAT_INTELLIGENCE,
                    question=question,
                    answer=answer,
                    labels={
                        "record_kind": "attack_procedure_mapping",
                        "technique_id": technique_id,
                        "source_object_type": source_object.get("type"),
                        "relationship_id": relationship_id,
                        "published_at": published_at,
                        "modified_at": modified_at or published_at,
                        "source_excerpt_sha256": hashlib.sha256(procedure.encode()).hexdigest(),
                    },
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

        emitted_record_keys: set[str] = set()
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
            if record_key in emitted_record_keys:
                stats["invalid_skipped"] += 1
                continue
            emitted_record_keys.add(record_key)
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


def _adapt_capec_xml(
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
        raise ValueError("raw executable or archive payload detected in CAPEC XML")
    if re.search(rb"(?i)<!DOCTYPE|<!ENTITY", content):
        raise ValueError("DTD and entity declarations are prohibited in CAPEC XML")
    try:
        root = ET.fromstring(content)  # noqa: S314 - bounded input; DTD/entities rejected above.
    except ET.ParseError as exc:
        raise ValueError("canonicalization input is not valid CAPEC XML") from exc
    elements = list(root.iter())
    if len(elements) > _MAX_OBJECTS:
        raise ValueError("CAPEC XML element count exceeds limit")
    for element in elements:
        for value in (*element.attrib.values(), element.text or "", element.tail or ""):
            if contains_executable_binary_payload(value):
                raise ValueError("raw executable or archive payload detected in CAPEC XML")

    catalog_date = root.attrib.get("Date") or root.attrib.get("date")
    records: list[SecurityRecord] = []
    stats = {"candidate_count": 0, "unsafe_skipped": 0, "invalid_skipped": 0}
    for pattern in (
        element for element in elements if _local_name(element.tag) == "Attack_Pattern"
    ):
        pattern_id_value = pattern.attrib.get("ID")
        pattern_name = _source_text(pattern.attrib.get("Name"))
        if (
            not pattern_id_value
            or not pattern_name
            or pattern.attrib.get("Status", "").casefold() == "deprecated"
        ):
            stats["invalid_skipped"] += 1
            continue
        capec_id = (
            pattern_id_value
            if pattern_id_value.upper().startswith("CAPEC-")
            else f"CAPEC-{pattern_id_value}"
        )
        description_element = next(
            (child for child in pattern if _local_name(child.tag) == "Description"),
            None,
        )
        description = _source_text(_element_text(description_element))
        submission_dates = [
            _element_text(child)
            for child in _descendants(pattern, "Submission_Date")
            if _parse_event_time(_element_text(child)) is not None
        ]
        modification_dates = [
            _element_text(child)
            for child in _descendants(pattern, "Modification_Date")
            if _parse_event_time(_element_text(child)) is not None
        ]
        published_at = _first_event_time(min(submission_dates, default=""), catalog_date)
        modified_at = _first_event_time(max(modification_dates, default=""), published_at)
        if not published_at:
            stats["invalid_skipped"] += 1
            continue

        mitigations_container = next(
            (child for child in pattern if _local_name(child.tag) == "Mitigations"),
            None,
        )
        mitigation_elements = (
            [child for child in mitigations_container if _local_name(child.tag) == "Mitigation"]
            if mitigations_container is not None
            else []
        )
        for index, mitigation_element in enumerate(mitigation_elements, start=1):
            stats["candidate_count"] += 1
            mitigation = _source_text(_element_text(mitigation_element))
            if not mitigation:
                stats["unsafe_skipped"] += 1
                continue
            mitigation_digest = hashlib.sha256(mitigation.encode()).hexdigest()
            record_key = f"{capec_id}:mitigation:{index}:{mitigation_digest[:16]}"
            records.append(
                _build_record(
                    source=source,
                    upstream_revision=upstream_revision,
                    retrieved_at=retrieved_at,
                    record_key=record_key,
                    record_id=f"{source.id}:{_safe_identifier(record_key)}",
                    split_group=f"{source.id}:{_safe_identifier(capec_id)}",
                    task=TaskType.SECURE_CODE_REVIEW,
                    question=(
                        f"What source-defined defensive mitigation applies to "
                        f"{capec_id} ({pattern_name})?"
                    ),
                    answer=mitigation,
                    labels={
                        "record_kind": "capec_mitigation",
                        "capec_id": capec_id,
                        "published_at": published_at,
                        "modified_at": modified_at or published_at,
                        "source_excerpt_sha256": mitigation_digest,
                    },
                )
            )

        weakness_ids = sorted(
            {
                f"CWE-{weakness_id}"
                for weakness in _descendants(pattern, "Related_Weakness")
                if (weakness_id := weakness.attrib.get("CWE_ID"))
            }
        )
        if weakness_ids:
            stats["candidate_count"] += 1
            if not description:
                stats["unsafe_skipped"] += 1
            else:
                record_key = f"{capec_id}:cwe-mapping"
                records.append(
                    _build_record(
                        source=source,
                        upstream_revision=upstream_revision,
                        retrieved_at=retrieved_at,
                        record_key=record_key,
                        record_id=f"{source.id}:{_safe_identifier(record_key)}",
                        split_group=f"{source.id}:{_safe_identifier(capec_id)}",
                        task=TaskType.CWE_MAPPING,
                        question=(
                            f"Map the source-grounded attack pattern {capec_id} "
                            f"({pattern_name}) to its related CWE IDs: {description}"
                        ),
                        answer=", ".join(weakness_ids),
                        labels={
                            "record_kind": "capec_cwe_mapping",
                            "capec_id": capec_id,
                            "cwe_ids": weakness_ids,
                            "published_at": published_at,
                            "modified_at": modified_at or published_at,
                            "source_excerpt_sha256": hashlib.sha256(
                                description.encode()
                            ).hexdigest(),
                        },
                    )
                )
        if len(records) > _MAX_OUTPUT_RECORDS:
            raise ValueError("canonical output record count exceeds limit")
    return records, stats


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
        declared_task = _source_text(_pick(item, "task"))
        if declared_task is None:
            task = TaskType.SECURE_CODE_REVIEW
            record_kind = "defensive_requirement"
        elif declared_task == TaskType.BLACK_BOX_ASSESSMENT:
            task = TaskType.BLACK_BOX_ASSESSMENT
            record_kind = "blackbox_defensive_requirement"
        else:
            stats["invalid_skipped"] += 1
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
                task=task,
                question=question,
                answer=answer,
                labels={
                    "record_kind": record_kind,
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
    if source.id == "mitre-capec":
        return AdapterKind.CAPEC_XML
    if source.id == "mitre-cwe":
        return (
            AdapterKind.CWE_XML
            if input_path.suffix.casefold() in {".xml", ".zip"}
            else AdapterKind.CWE_JSON
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


def canonicalize_splunk_security_content(
    repository_path: Path,
    output_path: Path,
    *,
    upstream_revision: str,
    retrieved_at: datetime,
    registry_path: Path = Path("configs/data/sources.yaml"),
) -> dict[str, Any]:
    """Convert production Splunk detections without loading simulation or test data."""

    registry = load_registry(registry_path)
    source = registry.require_purpose("splunk-security-content", Purpose.TRAIN)
    if not upstream_revision.strip():
        raise ValueError("upstream_revision must not be blank")
    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware")
    detections_path = repository_path / "detections"
    if not detections_path.is_dir() or detections_path.is_symlink():
        raise ValueError("Splunk source must contain a non-linked detections directory")

    rule_paths = sorted(detections_path.rglob("*.yml"), key=lambda path: path.as_posix())
    if not rule_paths or len(rule_paths) > _MAX_OBJECTS:
        raise ValueError("Splunk detection file count is outside the allowed range")
    inventory: list[dict[str, Any]] = []
    total_size = 0
    records: list[SecurityRecord] = []
    stats = {"candidate_count": len(rule_paths), "unsafe_skipped": 0, "invalid_skipped": 0}
    for rule_path in rule_paths:
        metadata = rule_path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"Splunk detection is not a non-linked regular file: {rule_path}")
        if metadata.st_size <= 0 or metadata.st_size > _MAX_RULE_FILE_BYTES:
            raise ValueError(f"Splunk detection exceeds its file size limit: {rule_path}")
        total_size += metadata.st_size
        if total_size > _MAX_RULE_TREE_BYTES:
            raise ValueError("Splunk detection tree exceeds its total size limit")
        relative_path = rule_path.relative_to(repository_path).as_posix()
        rule_sha256 = sha256_file(rule_path)
        inventory.append({"path": relative_path, "sha256": rule_sha256, "size": metadata.st_size})
        try:
            text = rule_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Splunk detection is not UTF-8: {relative_path}") from exc
        if contains_executable_binary_payload(text):
            raise ValueError(f"raw executable or archive payload in Splunk rule: {relative_path}")
        try:
            payload = yaml.load(text, Loader=_UniqueKeySafeLoader)  # noqa: S506
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid Splunk detection YAML: {relative_path}") from exc
        rule = _mapping(payload)
        if rule is None or rule.get("status") != "production":
            stats["invalid_skipped"] += 1
            continue

        rule_id = _source_text(rule.get("id"))
        name = _source_text(rule.get("name"))
        description = _source_text(rule.get("description"))
        search = _source_text(rule.get("search"))
        implementation = _source_text(rule.get("how_to_implement"))
        false_positives = _source_text(rule.get("known_false_positives"))
        detection_type = _source_text(rule.get("type"))
        security_domain = _source_text(rule.get("security_domain"))
        data_sources = _source_text(rule.get("data_source"))
        attack_ids = sorted(
            {
                value
                for item in (_sequence(rule.get("mitre_attack_id")) or [])
                if (value := _source_text(item)) is not None
            }
        )
        published_at = _first_event_time(str(rule.get("creation_date", "")))
        modified_at = _first_event_time(
            str(rule.get("modification_date", "")),
            published_at,
        )
        if (
            rule_id is None
            or name is None
            or description is None
            or search is None
            or implementation is None
            or false_positives is None
            or detection_type is None
            or security_domain is None
            or published_at is None
        ):
            stats["unsafe_skipped"] += 1
            continue

        attack_suffix = f" ATT&CK mappings: {', '.join(attack_ids)}." if attack_ids else ""
        data_source_suffix = f" Data sources: {data_sources}." if data_sources else ""
        question = (
            f"Provide the source-defined production Splunk detection for {name} "
            f"({rule_id}) in the {security_domain} security domain.{attack_suffix}"
        )
        answer = (
            f"Purpose: {description} SPL: {search} Implementation: {implementation} "
            f"Known false positives: {false_positives}{data_source_suffix}"
        )
        try:
            record = _build_record(
                source=source,
                upstream_revision=upstream_revision,
                retrieved_at=retrieved_at,
                record_key=rule_id,
                record_id=f"{source.id}:{_safe_identifier(rule_id)}",
                split_group=f"{source.id}:{_safe_identifier(rule_id)}",
                task=TaskType.SIEM_QUERY,
                question=question,
                answer=answer,
                labels={
                    "record_kind": "splunk_production_detection",
                    "rule_id": rule_id,
                    "rule_type": detection_type,
                    "security_domain": security_domain,
                    "mitre_attack_ids": attack_ids,
                    "source_path": relative_path,
                    "source_file_sha256": rule_sha256,
                    "published_at": published_at,
                    "modified_at": modified_at or published_at,
                    "source_excerpt_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                },
            )
        except ValueError:
            stats["unsafe_skipped"] += 1
            continue
        records.append(record)

    records.sort(key=lambda record: record.record_id)
    if not records:
        raise ValueError("Splunk adapter produced no safe production detections")
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
        "adapter": "splunk_security_content_production_v1",
        "source_id": source.id,
        "source_policy_class": source.policy_class,
        "license_id": source.license.id,
        "upstream_revision": upstream_revision,
        "retrieved_at_utc": retrieved_at.astimezone(UTC).isoformat(),
        "registry_sha256": registry.canonical_sha256(),
        "input": {
            "path": str(repository_path),
            "inventory_sha256": hashlib.sha256(
                json.dumps(
                    inventory,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "file_count": len(inventory),
            "size": total_size,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size": output_path.stat().st_size,
            "record_count": len(records),
        },
        "statistics": stats,
        "controls": {
            "production_status_only": True,
            "detections_directory_only": True,
            "attack_simulation_excluded": True,
            "tests_not_loaded": True,
            "duplicate_yaml_keys_rejected": True,
            "raw_binary_rejection": True,
            "actionable_payload_filter": True,
        },
    }
    write_json_exclusive(manifest_path, manifest)
    return manifest


def canonicalize_ocsf_schema(
    repository_path: Path,
    output_path: Path,
    *,
    upstream_revision: str,
    upstream_committed_at: datetime,
    retrieved_at: datetime,
    registry_path: Path = Path("configs/data/sources.yaml"),
) -> dict[str, Any]:
    """Convert OCSF dictionary and core schema definitions into source-grounded records."""

    registry = load_registry(registry_path)
    source = registry.require_purpose("ocsf-schema", Purpose.TRAIN)
    if not upstream_revision.strip():
        raise ValueError("upstream_revision must not be blank")
    if upstream_committed_at.tzinfo is None or retrieved_at.tzinfo is None:
        raise ValueError("OCSF commit and retrieval times must be timezone-aware")
    if not repository_path.is_dir() or repository_path.is_symlink():
        raise ValueError("OCSF source must be a non-linked repository directory")

    required_files = [repository_path / name for name in ("categories.json", "dictionary.json")]
    version_path = repository_path / "version.json"
    required_files.append(version_path)
    selected_dirs = [
        repository_path / name for name in ("events", "extensions", "objects", "profiles")
    ]
    if any(not path.is_file() or path.is_symlink() for path in required_files):
        raise ValueError("OCSF source is missing a required non-linked core file")
    candidates = set(required_files)
    for directory in selected_dirs:
        if directory.exists():
            if not directory.is_dir() or directory.is_symlink():
                raise ValueError("OCSF schema directory must not be linked")
            candidates.update(directory.rglob("*.json"))
    schema_paths = sorted(candidates, key=lambda path: path.relative_to(repository_path).as_posix())
    if len(schema_paths) > _MAX_OBJECTS:
        raise ValueError("OCSF schema file count exceeds limit")

    inventory: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    total_size = 0
    for schema_path in schema_paths:
        metadata = schema_path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"OCSF schema is not a non-linked regular file: {schema_path}")
        if metadata.st_size <= 0 or metadata.st_size > _MAX_RULE_FILE_BYTES:
            raise ValueError(f"OCSF schema exceeds its file size limit: {schema_path}")
        total_size += metadata.st_size
        if total_size > _MAX_RULE_TREE_BYTES:
            raise ValueError("OCSF schema tree exceeds its total size limit")
        relative_path = schema_path.relative_to(repository_path).as_posix()
        content = _read_bounded(schema_path, _MAX_RULE_FILE_BYTES)
        payloads[relative_path] = _load_json(content)
        inventory.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": metadata.st_size,
            }
        )

    version_payload = _mapping(payloads["version.json"])
    version = _source_text(version_payload.get("version")) if version_payload else None
    if version is None:
        raise ValueError("OCSF version.json lacks a safe version identifier")
    published_at = upstream_committed_at.astimezone(UTC).isoformat()
    records: list[SecurityRecord] = []
    stats = {"candidate_count": 0, "unsafe_skipped": 0, "invalid_skipped": 0}

    def add_record(
        *,
        record_key: str,
        schema_kind: str,
        caption: str,
        description: str,
        answer_suffix: str,
        source_path: str,
        source_file_sha256: str,
    ) -> None:
        stats["candidate_count"] += 1
        question = (
            f"In OCSF {version}, what does the {schema_kind} {record_key} "
            "represent for security event normalization?"
        )
        answer = f"{caption}: {description}{answer_suffix}"
        try:
            records.append(
                _build_record(
                    source=source,
                    upstream_revision=upstream_revision,
                    retrieved_at=retrieved_at,
                    record_key=f"{schema_kind}:{record_key}",
                    record_id=(
                        f"{source.id}:{_safe_identifier(schema_kind)}:"
                        f"{_safe_identifier(record_key)}"
                    ),
                    split_group=(
                        f"{source.id}:{_safe_identifier(schema_kind)}:"
                        f"{_safe_identifier(record_key)}"
                    ),
                    task=TaskType.SIEM_QUERY,
                    question=question,
                    answer=answer,
                    labels={
                        "record_kind": "ocsf_schema_definition",
                        "ocsf_version": version,
                        "schema_kind": schema_kind,
                        "schema_key": record_key,
                        "source_path": source_path,
                        "source_file_sha256": source_file_sha256,
                        "published_at": published_at,
                        "modified_at": published_at,
                        "source_excerpt_sha256": hashlib.sha256(description.encode()).hexdigest(),
                    },
                )
            )
        except ValueError:
            stats["unsafe_skipped"] += 1

    inventory_by_path = {entry["path"]: entry for entry in inventory}
    dictionary = _mapping(payloads["dictionary.json"])
    dictionary_attributes = _mapping(dictionary.get("attributes")) if dictionary else None
    if dictionary_attributes is None:
        raise ValueError("OCSF dictionary lacks an attributes mapping")
    for attribute_name in sorted(str(key) for key in dictionary_attributes):
        attribute = _mapping(dictionary_attributes.get(attribute_name))
        if attribute is None:
            stats["invalid_skipped"] += 1
            continue
        caption = _source_text(attribute.get("caption")) or attribute_name
        description = _source_text(attribute.get("description"))
        attribute_type = _source_text(attribute.get("type"))
        if description is None or attribute_type is None:
            stats["unsafe_skipped"] += 1
            continue
        cardinality = "array" if attribute.get("is_array") is True else "scalar"
        add_record(
            record_key=attribute_name,
            schema_kind="attribute",
            caption=caption,
            description=description,
            answer_suffix=f" Type: {attribute_type}; cardinality: {cardinality}.",
            source_path="dictionary.json",
            source_file_sha256=str(inventory_by_path["dictionary.json"]["sha256"]),
        )

    categories = _mapping(payloads["categories.json"])
    category_attributes = _mapping(categories.get("attributes")) if categories else None
    if category_attributes is None:
        raise ValueError("OCSF categories file lacks an attributes mapping")
    for category_name in sorted(str(key) for key in category_attributes):
        category = _mapping(category_attributes.get(category_name))
        if category is None:
            stats["invalid_skipped"] += 1
            continue
        caption = _source_text(category.get("caption")) or category_name
        description = _source_text(category.get("description"))
        uid = _source_text(category.get("uid"))
        if description is None or uid is None:
            stats["unsafe_skipped"] += 1
            continue
        add_record(
            record_key=category_name,
            schema_kind="category",
            caption=caption,
            description=description,
            answer_suffix=f" Category UID: {uid}.",
            source_path="categories.json",
            source_file_sha256=str(inventory_by_path["categories.json"]["sha256"]),
        )

    for relative_path in sorted(payloads):
        if relative_path in {"categories.json", "dictionary.json", "version.json"}:
            continue
        payload = _mapping(payloads[relative_path])
        if payload is None:
            stats["invalid_skipped"] += 1
            continue
        schema_name = _source_text(payload.get("name"))
        schema_caption = _source_text(payload.get("caption"))
        schema_description = _source_text(payload.get("description"))
        attributes = _mapping(payload.get("attributes"))
        if (
            schema_name is None
            or schema_caption is None
            or schema_description is None
            or attributes is None
        ):
            stats["unsafe_skipped"] += 1
            continue
        requirements: dict[str, list[str]] = {"required": [], "recommended": []}
        for attribute_name, attribute_value in attributes.items():
            attribute = _mapping(attribute_value)
            requirement = _source_text(attribute.get("requirement")) if attribute else None
            if requirement in requirements:
                requirements[requirement].append(str(attribute_name))
        requirement_parts = [
            f" {level.title()} attributes: {', '.join(sorted(names))}."
            for level, names in requirements.items()
            if names
        ]
        schema_kind = relative_path.split("/", 1)[0].rstrip("s") or "schema"
        add_record(
            record_key=schema_name,
            schema_kind=schema_kind,
            caption=schema_caption,
            description=schema_description,
            answer_suffix="".join(requirement_parts),
            source_path=relative_path,
            source_file_sha256=str(inventory_by_path[relative_path]["sha256"]),
        )

    records.sort(key=lambda record: record.record_id)
    if not records:
        raise ValueError("OCSF adapter produced no safe schema definitions")
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
        "adapter": "ocsf_core_schema_v1",
        "source_id": source.id,
        "source_policy_class": source.policy_class,
        "license_id": source.license.id,
        "upstream_revision": upstream_revision,
        "upstream_committed_at_utc": published_at,
        "retrieved_at_utc": retrieved_at.astimezone(UTC).isoformat(),
        "registry_sha256": registry.canonical_sha256(),
        "input": {
            "path": str(repository_path),
            "inventory_sha256": hashlib.sha256(
                json.dumps(
                    inventory,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "file_count": len(inventory),
            "size": total_size,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size": output_path.stat().st_size,
            "record_count": len(records),
        },
        "statistics": stats,
        "controls": {
            "schema_directories_only": True,
            "examples_and_logs_excluded": True,
            "raw_binary_rejection": True,
            "actionable_payload_filter": True,
        },
    }
    write_json_exclusive(manifest_path, manifest)
    return manifest


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
        xml_content = (
            _read_single_xml_archive(content) if input_path.suffix.casefold() == ".zip" else content
        )
        records, stats = _adapt_cwe_xml(xml_content, source, upstream_revision, retrieved_at)
    elif selected_adapter == AdapterKind.CAPEC_XML:
        if source.id != "mitre-capec":
            raise ValueError("CAPEC XML adapter is restricted to the registered MITRE CAPEC source")
        records, stats = _adapt_capec_xml(content, source, upstream_revision, retrieved_at)
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
