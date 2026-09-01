"""Validate, sanitize, deduplicate, and group-isolate canonical security JSONL."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from shadowcrafter.data.hygiene import (
    assert_split_isolation,
    deduplicate,
    deduplicate_near,
    normalized_content_hash,
    record_contains_raw_binary,
    redact_secrets,
    split_isolation_report,
    stable_split,
)
from shadowcrafter.data.manifest import canonical_json_sha256, sha256_file, write_json_exclusive
from shadowcrafter.data.registry import (
    DataSource,
    PolicyClass,
    Purpose,
    SourceRegistry,
    load_registry,
)
from shadowcrafter.schemas import SecurityRecord

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_OUTPUT_SPLITS = ("train", "validation", "test", "evaluation")


class SplitMode(StrEnum):
    """Explicit preparation modes; the isolated holdout remains the default."""

    ISOLATED_HOLDOUT = "isolated_holdout"
    TRAIN_ONLY = "train_only"


@dataclass(frozen=True)
class TemporalSplit:
    """Cutoffs for a conservative time-plus-group holdout."""

    validation_after: datetime
    test_after: datetime

    def __post_init__(self) -> None:
        if self.validation_after.tzinfo is None or self.test_after.tzinfo is None:
            raise ValueError("temporal split cutoffs must be timezone-aware")
        if self.validation_after >= self.test_after:
            raise ValueError("validation_after must be earlier than test_after")


def _sanitize_record(record: SecurityRecord) -> tuple[SecurityRecord, int]:
    payload = record.model_dump()
    total_redactions = 0
    for message in payload["messages"]:
        message["content"], count = redact_secrets(message["content"])
        total_redactions += count
    sanitized = SecurityRecord.model_validate(payload)
    sanitized.provenance.content_sha256 = sanitized.canonical_hash()
    return sanitized, total_redactions


def _validate_provenance(record: SecurityRecord, registry: SourceRegistry) -> DataSource:
    source = registry.source(record.provenance.source_id)
    if record.provenance.license != source.license.id:
        raise ValueError(
            f"{record.record_id}: provenance license {record.provenance.license!r} does not "
            f"match registry license {source.license.id!r}"
        )
    if not _SHA256.fullmatch(record.provenance.content_sha256):
        raise ValueError(f"{record.record_id}: provenance content_sha256 must be 64 hex digits")
    if record.provenance.content_sha256.lower() != record.canonical_hash():
        raise ValueError(f"{record.record_id}: provenance content_sha256 does not match record")
    if not record.provenance.record_key.strip():
        raise ValueError(f"{record.record_id}: provenance record_key must not be blank")
    if not record.split_group.strip():
        raise ValueError(f"{record.record_id}: split_group must not be blank")
    if record.provenance.retrieved_at.tzinfo is None:
        raise ValueError(f"{record.record_id}: retrieved_at must be timezone-aware")
    if source.mutable and not record.provenance.upstream_revision:
        raise ValueError(f"{record.record_id}: mutable source requires upstream_revision")
    return source


def _dotted_value(payload: Any, path: str) -> Any | None:
    current = payload
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _record_event_time(record: SecurityRecord, source: DataSource) -> datetime | None:
    payload = record.model_dump(mode="python")
    canonical_fields = (
        "labels.published_at",
        "labels.modified_at",
        "labels.observed_at",
        "labels.first_seen",
    )
    for field in (*canonical_fields, *source.time_fields):
        parsed = _parse_datetime(_dotted_value(payload, field))
        if parsed is not None:
            return parsed
    return None


def _assert_parent_groups(records: list[SecurityRecord]) -> None:
    by_id = {record.record_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("record_id values must be unique before deduplication")
    for record in records:
        parent_id = record.labels.get("parent_record_id")
        if not isinstance(parent_id, str) or parent_id not in by_id:
            continue
        if by_id[parent_id].split_group != record.split_group:
            raise ValueError(f"{record.record_id}: derived record must inherit parent split_group")


def _assert_no_train_eval_duplicates(
    records: list[SecurityRecord], sources: dict[str, DataSource]
) -> None:
    exact_classes: dict[str, set[PolicyClass]] = defaultdict(set)
    normalized_classes: dict[str, set[PolicyClass]] = defaultdict(set)
    for record in records:
        policy_class = sources[record.record_id].policy_class
        exact_classes[record.canonical_hash()].add(policy_class)
        normalized_classes[normalized_content_hash(record)].add(policy_class)
    conflicting = {PolicyClass.ALLOW_TRAIN, PolicyClass.EVAL_ONLY}
    if any(classes >= conflicting for classes in exact_classes.values()):
        raise ValueError("exact duplicate appears in both trainable and evaluation-only sources")
    if any(classes >= conflicting for classes in normalized_classes.values()):
        raise ValueError("normalized duplicate appears in trainable and evaluation-only sources")


def _assign_splits(
    records: list[SecurityRecord],
    sources: dict[str, DataSource],
    temporal_split: TemporalSplit | None,
) -> dict[str, list[SecurityRecord]]:
    grouped: dict[str, list[SecurityRecord]] = defaultdict(list)
    for record in records:
        grouped[record.split_group].append(record)

    splits: dict[str, list[SecurityRecord]] = {name: [] for name in _OUTPUT_SPLITS}
    for group_name in sorted(grouped):
        group = grouped[group_name]
        classes = {sources[record.record_id].policy_class for record in group}
        if PolicyClass.EVAL_ONLY in classes and PolicyClass.ALLOW_TRAIN in classes:
            raise ValueError(f"split group {group_name!r} mixes train and evaluation sources")
        if classes == {PolicyClass.EVAL_ONLY}:
            split = "evaluation"
        elif classes != {PolicyClass.ALLOW_TRAIN}:
            raise ValueError(f"split group {group_name!r} contains an ineligible policy class")
        elif any(record.benchmark_holdout for record in group):
            split = "test"
        elif temporal_split is None:
            split = stable_split(group_name)
        else:
            timestamps = [_record_event_time(record, sources[record.record_id]) for record in group]
            if any(timestamp is None for timestamp in timestamps):
                raise ValueError(f"split group {group_name!r} lacks a valid event timestamp")
            # Assign the whole lineage by its latest member. This may reduce train volume,
            # but it prevents older variants entering train while a newer variant enters test.
            latest = max(timestamp for timestamp in timestamps if timestamp is not None)
            if latest >= temporal_split.test_after.astimezone(UTC):
                split = "test"
            elif latest >= temporal_split.validation_after.astimezone(UTC):
                split = "validation"
            else:
                split = "train"
        splits[split].extend(sorted(group, key=lambda record: record.record_id))
    return splits


def _source_manifest(
    records: list[SecurityRecord], sources: dict[str, DataSource]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[SecurityRecord]] = defaultdict(list)
    for record in records:
        revision = record.provenance.upstream_revision or "immutable"
        grouped[(record.provenance.source_id, revision)].append(record)

    entries: list[dict[str, Any]] = []
    for (source_id, revision), source_records in sorted(grouped.items()):
        source = sources[source_records[0].record_id]
        timestamps = [record.provenance.retrieved_at.astimezone(UTC) for record in source_records]
        entries.append(
            {
                "source_id": source_id,
                "upstream_revision": revision,
                "license_id": source.license.id,
                "license_url": str(source.license.url),
                "policy_class": source.policy_class,
                "allowed_purposes": sorted(source.allowed_purposes),
                "record_count": len(source_records),
                "retrieved_at_min": min(timestamps).isoformat(),
                "retrieved_at_max": max(timestamps).isoformat(),
            }
        )
    return entries


def _ordered_input_paths(input_paths: Sequence[Path]) -> list[Path]:
    if not input_paths:
        raise ValueError("at least one canonical JSONL input is required")

    ordered: list[Path] = []
    identities: dict[tuple[int, int], Path] = {}
    resolved_paths: dict[Path, Path] = {}
    for input_path in input_paths:
        path = Path(input_path)
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"canonical JSONL input does not exist: {path}") from exc
        if not resolved.is_file():
            raise ValueError(f"canonical JSONL input is not a regular file: {path}")

        stat = resolved.stat()
        identity = (stat.st_dev, stat.st_ino)
        previous = identities.get(identity)
        if previous is not None:
            raise ValueError(
                f"duplicate canonical JSONL input path: {path} resolves to the same file as "
                f"{previous}"
            )
        if resolved in resolved_paths:
            raise ValueError(f"duplicate canonical JSONL input path: {path} resolves to {resolved}")
        identities[identity] = path
        resolved_paths[resolved] = path
        ordered.append(resolved)
    return sorted(ordered, key=lambda path: path.as_posix())


def prepare_jsonl_many(
    input_paths: Sequence[Path],
    output_dir: Path,
    *,
    registry_path: Path = Path("configs/data/sources.yaml"),
    temporal_split: TemporalSplit | None = None,
    split_mode: SplitMode = SplitMode.ISOLATED_HOLDOUT,
) -> dict[str, Any]:
    """Prepare one or more canonical JSONL files under global isolation controls."""

    selected_split_mode = SplitMode(split_mode)
    if selected_split_mode == SplitMode.TRAIN_ONLY and temporal_split is not None:
        raise ValueError("train-only split mode cannot be combined with a temporal split")

    registry = load_registry(registry_path)
    ordered_inputs = _ordered_input_paths(input_paths)
    records: list[SecurityRecord] = []
    sources: dict[str, DataSource] = {}
    record_locations: dict[str, tuple[Path, int]] = {}
    input_entries: list[dict[str, Any]] = []
    redactions = 0
    for input_path in ordered_inputs:
        initial_size = input_path.stat().st_size
        initial_sha256 = sha256_file(input_path)
        file_record_count = 0
        with input_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = SecurityRecord.model_validate_json(line)
                    source = _validate_provenance(record, registry)
                except Exception as exc:
                    raise ValueError(
                        f"invalid record at {input_path}:{line_number}: {exc}"
                    ) from exc

                previous = record_locations.get(record.record_id)
                if previous is not None:
                    previous_path, previous_line = previous
                    raise ValueError(
                        f"duplicate record_id {record.record_id!r} at "
                        f"{input_path}:{line_number}; first seen at "
                        f"{previous_path}:{previous_line}"
                    )
                record_locations[record.record_id] = (input_path, line_number)

                if record_contains_raw_binary(record):
                    raise ValueError(
                        f"prohibited raw executable payload at {input_path}:{line_number}"
                    )
                if source.policy_class == PolicyClass.ALLOW_TRAIN:
                    registry.require_purpose(source.id, Purpose.TRAIN)
                elif source.policy_class == PolicyClass.EVAL_ONLY:
                    registry.require_purpose(source.id, Purpose.EVALUATE)
                    if selected_split_mode == SplitMode.TRAIN_ONLY:
                        raise ValueError(
                            f"train-only split mode rejects evaluation-only source {source.id}; "
                            "prepare external evaluation separately"
                        )
                else:
                    raise ValueError(
                        f"source {source.id} is {source.policy_class}; "
                        "supervised preparation denied"
                    )
                if selected_split_mode == SplitMode.TRAIN_ONLY and record.benchmark_holdout:
                    raise ValueError(
                        f"train-only split mode rejects benchmark_holdout record "
                        f"{record.record_id!r}; prepare holdouts separately"
                    )

                sanitized, count = _sanitize_record(record)
                redactions += count
                records.append(sanitized)
                sources[sanitized.record_id] = source
                file_record_count += 1

        final_size = input_path.stat().st_size
        final_sha256 = sha256_file(input_path)
        if final_size != initial_size or final_sha256 != initial_sha256:
            raise ValueError(f"canonical JSONL input changed while reading: {input_path}")
        input_entries.append(
            {
                "path": str(input_path),
                "sha256": initial_sha256,
                "size": initial_size,
                "record_count": file_record_count,
            }
        )

    _assert_parent_groups(records)
    _assert_no_train_eval_duplicates(records, sources)
    input_record_count = len(records)
    # Restrictive records win if normalization finds duplicate holdout variants.
    records.sort(key=lambda record: (not record.benchmark_holdout, record.record_id))
    records, exact_duplicates = deduplicate(records)
    records, normalized_duplicates = deduplicate_near(records)
    if selected_split_mode == SplitMode.TRAIN_ONLY:
        splits: dict[str, list[SecurityRecord]] = {name: [] for name in _OUTPUT_SPLITS}
        splits["train"] = sorted(records, key=lambda record: record.record_id)
    else:
        splits = _assign_splits(records, sources, temporal_split)
    assert_split_isolation(splits)

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: output_dir / f"{name}.jsonl" for name in _OUTPUT_SPLITS}
    targets["manifest"] = output_dir / "manifest.json"
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite prepared artifacts: {existing}")

    for split_name in _OUTPUT_SPLITS:
        with targets[split_name].open("x", encoding="utf-8") as writer:
            for record in splits[split_name]:
                writer.write(record.model_dump_json() + "\n")

    artifacts = {
        split_name: {
            "path": str(targets[split_name]),
            "sha256": sha256_file(targets[split_name]),
            "size": targets[split_name].stat().st_size,
            "record_count": len(splits[split_name]),
        }
        for split_name in _OUTPUT_SPLITS
    }
    isolation = split_isolation_report(splits)
    legacy_input = {
        "path": input_entries[0]["path"],
        "sha256": input_entries[0]["sha256"],
        "size": input_entries[0]["size"],
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        # Preserve the original single-input field for callers of prepare_jsonl.
        "input": legacy_input if len(input_entries) == 1 else None,
        "inputs": input_entries,
        "input_set_sha256": canonical_json_sha256(input_entries),
        "registry": {
            "path": str(registry_path),
            "sha256": registry.canonical_sha256(),
        },
        "record_count": len(records),
        "input_record_count": input_record_count,
        "exact_duplicate_count": exact_duplicates,
        "normalized_duplicate_count": normalized_duplicates,
        "deduplication": {
            "exact_method": "SecurityRecord.canonical_hash(task,messages,labels)",
            "normalized_method": "NFKC + normalized line endings + trailing-space trim",
            "exact_duplicate_count": exact_duplicates,
            "normalized_duplicate_count": normalized_duplicates,
            "applied_before_split": True,
        },
        "secret_redactions": redactions,
        "split_counts": {name: len(splits[name]) for name in _OUTPUT_SPLITS},
        "split_policy": {
            "mode": selected_split_mode,
            "strategy": (
                "train_only_no_internal_evaluation"
                if selected_split_mode == SplitMode.TRAIN_ONLY
                else "time_plus_group"
                if temporal_split
                else "stable_group_hash"
            ),
            "benchmark_holdout": (
                "rejected"
                if selected_split_mode == SplitMode.TRAIN_ONLY
                else "entire group forced to test"
            ),
            "eligible_for_release_evaluation": selected_split_mode != SplitMode.TRAIN_ONLY,
            "separate_evaluation_required": selected_split_mode == SplitMode.TRAIN_ONLY,
            "evaluation_requirement": (
                "Use a separately governed eval-only benchmark or a future temporal holdout; "
                "these train-only artifacts provide no internal validation or test evidence."
                if selected_split_mode == SplitMode.TRAIN_ONLY
                else None
            ),
            "validation_after": (
                temporal_split.validation_after.astimezone(UTC).isoformat()
                if temporal_split
                else None
            ),
            "test_after": (
                temporal_split.test_after.astimezone(UTC).isoformat() if temporal_split else None
            ),
        },
        "isolation": isolation,
        "sources": _source_manifest(records, sources),
        "artifacts": artifacts,
    }
    manifest["dataset_sha256"] = canonical_json_sha256(
        {name: artifact["sha256"] for name, artifact in artifacts.items()}
    )
    write_json_exclusive(targets["manifest"], manifest)
    return manifest


def prepare_jsonl(
    input_path: Path,
    output_dir: Path,
    *,
    registry_path: Path = Path("configs/data/sources.yaml"),
    temporal_split: TemporalSplit | None = None,
    split_mode: SplitMode = SplitMode.ISOLATED_HOLDOUT,
) -> dict[str, Any]:
    """Prepare one canonical JSONL file; retained for API compatibility."""

    return prepare_jsonl_many(
        [input_path],
        output_dir,
        registry_path=registry_path,
        temporal_split=temporal_split,
        split_mode=split_mode,
    )
