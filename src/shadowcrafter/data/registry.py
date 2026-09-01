"""Fail-closed source registry for auditable security-data use."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class PolicyClass(StrEnum):
    """Mutually exclusive ways a source may enter the system."""

    ALLOW_TRAIN = "allow_train"
    RAG_ONLY = "rag_only"
    EVAL_ONLY = "eval_only"
    QUARANTINE = "quarantine"
    PROHIBITED = "prohibited"


class Purpose(StrEnum):
    TRAIN = "train"
    RAG = "rag"
    EVALUATE = "evaluate"
    MANUAL_REVIEW = "manual_review"


class LicenseStatus(StrEnum):
    VERIFIED = "verified"
    REVIEW_REQUIRED = "review_required"
    PROHIBITED = "prohibited"


class Redistribution(StrEnum):
    ALLOWED = "allowed"
    ATTRIBUTION = "attribution_required"
    NOTICE = "notice_required"
    TERMS_LIMITED = "terms_limited"
    NOT_VERIFIED = "not_verified"
    PROHIBITED = "prohibited"


class SourceType(StrEnum):
    HTTP_JSON = "http_json"
    HTTP_XML = "http_xml"
    HTTP_API = "http_api"
    HTTP_ARCHIVE = "http_archive"
    HTTP_GZIP_CSV = "http_gzip_csv"
    GIT = "git"
    HUGGINGFACE_DATASET = "huggingface_dataset"
    WEBSITE = "website"
    LOCAL_PROHIBITED = "local_prohibited"


class ContentKind(StrEnum):
    STRUCTURED_RECORDS = "structured_records"
    TAXONOMY = "taxonomy"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    DETECTION_RULES = "detection_rules"
    DEFENSIVE_GUIDANCE = "defensive_guidance"
    SECURE_CODE = "secure_code"
    STATIC_FEATURES = "static_features"
    SANITIZED_REPORTS = "sanitized_reports"
    SANITIZED_TELEMETRY = "sanitized_telemetry"
    SECURITY_SCHEMA = "security_schema"
    EVALUATION_BENCHMARK = "evaluation_benchmark"
    RAW_EXECUTABLE_BINARY = "raw_executable_binary"


class RegistryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unknown_source: Literal["prohibited"]
    missing_or_ambiguous_license: Literal["quarantine"]
    raw_malware: Literal["prohibited"]
    eval_to_train: Literal["prohibited"]
    split_before_instruction_generation: Literal[True]
    require_content_sha256: Literal[True]
    require_group_split: Literal[True]
    temporal_holdout_requires_cutoff: Literal[True]
    max_snapshot_bytes: int = Field(gt=0, le=1_073_741_824)


class LicenseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=2)
    url: HttpUrl
    status: LicenseStatus
    redistribution: Redistribution
    attribution_required: bool = False


class SafetyMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_kind: ContentKind
    raw_malware_binaries: bool = False
    executable_content: bool = False
    required_filters: list[str] = Field(default_factory=list)


class SnapshotPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_bytes: int | None = Field(default=None, gt=0, le=1_073_741_824)
    allowed_media_types: list[str] = Field(default_factory=list)


_EXPECTED_PURPOSES: dict[PolicyClass, frozenset[Purpose]] = {
    PolicyClass.ALLOW_TRAIN: frozenset({Purpose.TRAIN, Purpose.RAG}),
    PolicyClass.RAG_ONLY: frozenset({Purpose.RAG}),
    PolicyClass.EVAL_ONLY: frozenset({Purpose.EVALUATE}),
    PolicyClass.QUARANTINE: frozenset({Purpose.MANUAL_REVIEW}),
    PolicyClass.PROHIBITED: frozenset(),
}


class DataSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    title: str = Field(min_length=2)
    provider: str = Field(min_length=2)
    type: SourceType
    policy_class: PolicyClass
    allowed_purposes: set[Purpose]
    url: HttpUrl | None = None
    repo_id: str | None = None
    license: LicenseMetadata
    attribution: str | None = None
    mutable: bool
    temporal_snapshot: bool
    snapshot: SnapshotPolicy = Field(default_factory=SnapshotPolicy)
    safety: SafetyMetadata
    split_group_keys: list[str] = Field(default_factory=list)
    time_fields: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def enforce_policy_contract(self) -> DataSource:
        expected = _EXPECTED_PURPOSES[self.policy_class]
        if self.allowed_purposes != expected:
            raise ValueError(f"{self.id}: {self.policy_class} requires purposes {sorted(expected)}")

        if self.type == SourceType.HUGGINGFACE_DATASET:
            if not self.repo_id or self.url is not None:
                raise ValueError(f"{self.id}: Hugging Face sources require only repo_id")
        elif self.type == SourceType.LOCAL_PROHIBITED:
            if self.url is not None or self.repo_id is not None:
                raise ValueError(f"{self.id}: prohibited local source cannot have a locator")
        elif self.url is None or self.repo_id is not None:
            raise ValueError(f"{self.id}: source type {self.type} requires only url")

        usable = {
            PolicyClass.ALLOW_TRAIN,
            PolicyClass.RAG_ONLY,
            PolicyClass.EVAL_ONLY,
        }
        if self.policy_class in usable and self.license.status != LicenseStatus.VERIFIED:
            raise ValueError(f"{self.id}: usable sources require a verified license or terms")
        if (
            self.policy_class == PolicyClass.QUARANTINE
            and self.license.status == LicenseStatus.PROHIBITED
        ):
            raise ValueError(f"{self.id}: prohibited license cannot be marked quarantine")
        if self.policy_class == PolicyClass.PROHIBITED:
            if self.license.status != LicenseStatus.PROHIBITED:
                raise ValueError(
                    f"{self.id}: prohibited source must have prohibited license status"
                )
            if self.license.redistribution != Redistribution.PROHIBITED:
                raise ValueError(f"{self.id}: prohibited source must prohibit redistribution")

        is_raw = (
            self.safety.raw_malware_binaries
            or self.safety.content_kind == ContentKind.RAW_EXECUTABLE_BINARY
        )
        if is_raw and self.policy_class != PolicyClass.PROHIBITED:
            raise ValueError(f"{self.id}: raw executable binaries are prohibited")
        if self.policy_class == PolicyClass.ALLOW_TRAIN and not self.split_group_keys:
            raise ValueError(f"{self.id}: trainable sources require split_group_keys")
        if self.policy_class == PolicyClass.EVAL_ONLY and not self.split_group_keys:
            raise ValueError(f"{self.id}: evaluation sources require split_group_keys")
        if self.mutable and self.policy_class in usable and not self.temporal_snapshot:
            raise ValueError(f"{self.id}: mutable usable sources require immutable snapshots")

        if self.license.attribution_required and not self.attribution:
            raise ValueError(f"{self.id}: attribution text is required")
        if self.snapshot.enabled:
            if (
                self.type
                not in {
                    SourceType.HTTP_ARCHIVE,
                    SourceType.HTTP_JSON,
                    SourceType.HTTP_XML,
                }
                or self.url is None
            ):
                raise ValueError(
                    f"{self.id}: automatic snapshots support only bounded HTTP documents"
                )
            if not self.temporal_snapshot:
                raise ValueError(f"{self.id}: automatic snapshots must be temporal")
            if not self.snapshot.allowed_media_types:
                raise ValueError(f"{self.id}: snapshot media-type allowlist is required")
        return self

    def permits(self, purpose: Purpose) -> bool:
        return purpose in self.allowed_purposes


class SourceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[2]
    policy: RegistryPolicy
    sources: list[DataSource] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_source_ids(self) -> SourceRegistry:
        identifiers = [source.id for source in self.sources]
        duplicates = sorted(
            {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
        )
        if duplicates:
            raise ValueError(f"duplicate source ids: {duplicates}")
        return self

    @property
    def by_id(self) -> dict[str, DataSource]:
        return {source.id: source for source in self.sources}

    def source(self, source_id: str) -> DataSource:
        try:
            return self.by_id[source_id]
        except KeyError as exc:
            raise ValueError(f"unregistered source is prohibited: {source_id}") from exc

    def require_purpose(self, source_id: str, purpose: Purpose) -> DataSource:
        source = self.source(source_id)
        if not source.permits(purpose):
            raise ValueError(
                f"source {source_id} is {source.policy_class} and does not permit {purpose}"
            )
        return source

    def canonical_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload["sources"] = sorted(payload["sources"], key=lambda source: source["id"])
        for source in payload["sources"]:
            source["allowed_purposes"] = sorted(source["allowed_purposes"])
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_registry(path: Path) -> SourceRegistry:
    """Load and strictly validate a source registry."""

    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"registry must contain a mapping: {path}")
    return SourceRegistry.model_validate(payload)
