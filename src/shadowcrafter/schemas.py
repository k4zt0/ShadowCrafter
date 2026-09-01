"""Canonical, provenance-preserving schemas used across the project."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(StrEnum):
    CVE_TRIAGE = "cve_triage"
    CWE_MAPPING = "cwe_mapping"
    MALWARE_ANALYSIS = "malware_analysis"
    DETECTION_ENGINEERING = "detection_engineering"
    INCIDENT_RESPONSE = "incident_response"
    THREAT_INTELLIGENCE = "threat_intelligence"
    SECURE_CODE_REVIEW = "secure_code_review"
    SECURITY_REPORT = "security_report"
    SIEM_QUERY = "siem_query"
    SOAR_PLAN = "soar_plan"
    SAFETY_ALIGNMENT = "safety_alignment"


class RiskTier(StrEnum):
    DEFENSIVE = "defensive"
    DUAL_USE_CONTROLLED = "dual_use_controlled"
    DISALLOWED = "disallowed"


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant|tool)$")
    content: str = Field(min_length=1)


class Provenance(BaseModel):
    source_id: str
    source_url: str | None = None
    license: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    upstream_revision: str | None = None
    record_key: str
    content_sha256: str


class SecurityRecord(BaseModel):
    """One auditable supervised or preference-learning example."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    task: TaskType
    risk_tier: RiskTier = RiskTier.DEFENSIVE
    language: str = "en"
    messages: list[Message] = Field(min_length=2)
    labels: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance
    split_group: str
    benchmark_holdout: bool = False
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("messages")
    @classmethod
    def require_user_and_assistant(cls, messages: list[Message]) -> list[Message]:
        roles = {message.role for message in messages}
        if "user" not in roles or "assistant" not in roles:
            raise ValueError("messages must contain at least one user and assistant turn")
        return messages

    def canonical_hash(self) -> str:
        payload = {
            "task": self.task,
            "messages": [message.model_dump() for message in self.messages],
            "labels": self.labels,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()


class ActionRequest(BaseModel):
    intent: str
    target: str | None = None
    authorization_evidence: str | None = None
    destructive: bool = False
    requests_credentials: bool = False
    requests_evasion: bool = False
    sandboxed: bool = False


class ActionDecision(BaseModel):
    allowed: bool
    risk_tier: RiskTier
    reason: str
    requires_human_approval: bool = False


class EvalMetric(BaseModel):
    name: str
    value: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)
    task: str
    split_hash: str
    contamination_rate: float = Field(default=0.0, ge=0.0, le=1.0)
