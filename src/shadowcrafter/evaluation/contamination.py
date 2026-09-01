"""Immutable split fingerprints and exact-content contamination checks."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path

_SHA256_LENGTH = 64


class SplitIntegrityError(ValueError):
    """Raised when split membership no longer matches its frozen fingerprint."""


class SplitContaminationError(ValueError):
    """Raised when evaluation content overlaps a forbidden split."""


def _json_default(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        canonical_items = [canonical_json(item) for item in value]
        return [json.loads(item) for item in sorted(canonical_items)]
    raise TypeError(f"unsupported value in canonical record: {type(value).__name__}")


def canonical_json(record: object) -> str:
    """Serialize a record deterministically, rejecting non-standard JSON numbers."""

    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def canonical_record_hash(record: object) -> str:
    """Return a SHA-256 content hash independent of mapping key order."""

    payload = record if isinstance(record, bytes) else canonical_json(record).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_hash(digest: str) -> str:
    normalized = digest.lower()
    if len(normalized) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("record hashes must be 64-character SHA-256 hex digests")
    return normalized


def immutable_split_hash(record_hashes: Iterable[str]) -> str:
    """Hash complete split membership in an order-independent form.

    Duplicate membership is deliberately retained: accidentally duplicating a
    sample changes the split fingerprint even though contamination checks use
    unique content hashes.
    """

    normalized = sorted(_validate_hash(digest) for digest in record_hashes)
    payload = canonical_json(
        {"algorithm": "sha256", "record_hashes": normalized, "version": 1}
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SplitFingerprint:
    """Frozen membership proof for one train, validation, or test split."""

    name: str
    record_hashes: tuple[str, ...]
    digest: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("split name must not be empty")
        normalized = tuple(sorted(_validate_hash(item) for item in self.record_hashes))
        if normalized != self.record_hashes:
            raise SplitIntegrityError("record_hashes must be normalized and sorted")
        expected = immutable_split_hash(normalized)
        if not hmac.compare_digest(expected, _validate_hash(self.digest)):
            raise SplitIntegrityError("split digest does not match record membership")

    @property
    def sample_count(self) -> int:
        return len(self.record_hashes)

    @property
    def unique_sample_count(self) -> int:
        return len(set(self.record_hashes))

    @classmethod
    def from_records(cls, name: str, records: Iterable[object]) -> SplitFingerprint:
        hashes = tuple(sorted(canonical_record_hash(record) for record in records))
        return cls(name=name, record_hashes=hashes, digest=immutable_split_hash(hashes))

    @classmethod
    def from_hashes(cls, name: str, record_hashes: Iterable[str]) -> SplitFingerprint:
        hashes = tuple(sorted(_validate_hash(digest) for digest in record_hashes))
        return cls(name=name, record_hashes=hashes, digest=immutable_split_hash(hashes))

    def verify_records(self, records: Iterable[object]) -> bool:
        candidate = type(self).from_records(self.name, records)
        return hmac.compare_digest(self.digest, candidate.digest)


@dataclass(frozen=True, slots=True)
class ContaminationFinding:
    """One exact-content overlap between two named splits."""

    left_split: str
    right_split: str
    overlapping_hashes: tuple[str, ...]
    right_contamination_rate: float

    @property
    def overlap_count(self) -> int:
        return len(self.overlapping_hashes)


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    """Pairwise overlap report for an immutable split collection."""

    split_digests: tuple[tuple[str, str], ...]
    findings: tuple[ContaminationFinding, ...]

    @property
    def contaminated(self) -> bool:
        return any(finding.overlap_count for finding in self.findings)

    @property
    def overlap_count(self) -> int:
        return sum(finding.overlap_count for finding in self.findings)


def build_split_fingerprint(name: str, records: Iterable[object]) -> SplitFingerprint:
    """Build a frozen fingerprint from canonicalized record content."""

    return SplitFingerprint.from_records(name, records)


def verify_split_fingerprint(fingerprint: SplitFingerprint, records: Iterable[object]) -> None:
    """Raise if the supplied records differ from the frozen split."""

    if not fingerprint.verify_records(records):
        raise SplitIntegrityError(f"split {fingerprint.name!r} no longer matches its digest")


def split_hash(records: Iterable[object]) -> str:
    """Return the immutable membership digest for raw records."""

    hashes = (canonical_record_hash(record) for record in records)
    return immutable_split_hash(hashes)


def contamination_rate(reference_hashes: Iterable[str], evaluation_hashes: Iterable[str]) -> float:
    """Return the fraction of unique evaluation content present in a reference split."""

    reference = {_validate_hash(digest) for digest in reference_hashes}
    evaluation = {_validate_hash(digest) for digest in evaluation_hashes}
    return len(reference & evaluation) / len(evaluation) if evaluation else 0.0


def assert_splits_disjoint(left: SplitFingerprint, right: SplitFingerprint) -> None:
    """Enforce disjointness for a named split pair."""

    if left.name == right.name:
        raise ValueError("split names must be distinct for a disjointness check")
    assert_no_contamination({left.name: left, right.name: right})


def check_split_contamination(
    splits: Mapping[str, SplitFingerprint],
) -> ContaminationReport:
    """Check every split pair for exact-content overlap."""

    names = tuple(sorted(splits))
    findings: list[ContaminationFinding] = []
    for index, left_name in enumerate(names):
        left_hashes = set(splits[left_name].record_hashes)
        for right_name in names[index + 1 :]:
            right_hashes = set(splits[right_name].record_hashes)
            overlap = tuple(sorted(left_hashes & right_hashes))
            rate = len(overlap) / len(right_hashes) if right_hashes else 0.0
            if overlap:
                findings.append(
                    ContaminationFinding(
                        left_split=left_name,
                        right_split=right_name,
                        overlapping_hashes=overlap,
                        right_contamination_rate=rate,
                    )
                )
    return ContaminationReport(
        split_digests=tuple((name, splits[name].digest) for name in names),
        findings=tuple(findings),
    )


def assert_no_contamination(splits: Mapping[str, SplitFingerprint]) -> None:
    """Enforce completely disjoint content across all supplied splits."""

    report = check_split_contamination(splits)
    if report.contaminated:
        pairs = ", ".join(
            f"{finding.left_split}/{finding.right_split}={finding.overlap_count}"
            for finding in report.findings
        )
        raise SplitContaminationError(f"cross-split contamination detected: {pairs}")


# Convenient aliases for evaluation scripts that use hash/snapshot terminology.
calculate_split_hash = immutable_split_hash
fingerprint_split = build_split_fingerprint
