"""Deterministic data hygiene helpers for secrets, duplicates, and split isolation."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any

from shadowcrafter.schemas import SecurityRecord

_SECRET_PATTERNS = (
    re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)

_BASE64_BLOB = re.compile(
    r"(?<![A-Za-z0-9+/_-])(?:[A-Za-z0-9+/_-]{4}){64,}"
    r"(?:[A-Za-z0-9+/_-]{2}==|[A-Za-z0-9+/_-]{3}=)?"
)
_HEX_BLOB = re.compile(
    r"(?i)(?<![0-9a-f])(?:4d5a|7f454c46|504b0304|52617221|377abcaf271c)[0-9a-f]{252,}"
)
_EXECUTABLE_MAGICS = (
    b"MZ",
    b"\x7fELF",
    b"PK\x03\x04",
    b"Rar!\x1a\x07",
    b"7z\xbc\xaf\x27\x1c",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
)


def redact_secrets(text: str) -> tuple[str, int]:
    redactions = 0
    for pattern in _SECRET_PATTERNS:
        text, count = pattern.subn("<REDACTED_SECRET>", text)
        redactions += count
    return text, redactions


def stable_split(group: str, train: int = 90, validation: int = 5) -> str:
    if not group.strip():
        raise ValueError("split group must not be blank")
    if train < 0 or validation < 0 or train + validation >= 100:
        raise ValueError("train and validation must be non-negative and sum to less than 100")
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def deduplicate(records: Iterable[SecurityRecord]) -> tuple[list[SecurityRecord], int]:
    seen: set[str] = set()
    kept: list[SecurityRecord] = []
    dropped = 0
    for record in records:
        digest = record.canonical_hash()
        if digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        kept.append(record)
    return kept, dropped


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+$", "", line) for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def normalized_content_hash(record: SecurityRecord) -> str:
    """Hash task and normalized conversation, intentionally excluding source metadata."""

    payload = {
        "task": str(record.task),
        "messages": [
            {"role": message.role, "content": _normalize_text(message.content)}
            for message in record.messages
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def deduplicate_near(records: Iterable[SecurityRecord]) -> tuple[list[SecurityRecord], int]:
    """Drop normalization-only duplicates without using nondeterministic embeddings."""

    seen: set[str] = set()
    kept: list[SecurityRecord] = []
    dropped = 0
    for record in records:
        digest = normalized_content_hash(record)
        if digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        kept.append(record)
    return kept, dropped


def _role_content_hashes(records: Iterable[SecurityRecord], role: str) -> set[str]:
    return {
        hashlib.sha256(_normalize_text(message.content).encode()).hexdigest()
        for record in records
        for message in record.messages
        if message.role == role
    }


def split_isolation_report(
    splits: Mapping[str, Iterable[SecurityRecord]],
) -> dict[str, dict[str, int]]:
    """Return pairwise record, role-content, and lineage overlap counts."""

    materialized = {name: list(records) for name, records in splits.items()}
    report: dict[str, dict[str, int]] = {}
    for left_name, right_name in combinations(sorted(materialized), 2):
        left = materialized[left_name]
        right = materialized[right_name]
        exact_left = {record.canonical_hash() for record in left}
        exact_right = {record.canonical_hash() for record in right}
        normalized_left = {normalized_content_hash(record) for record in left}
        normalized_right = {normalized_content_hash(record) for record in right}
        user_left = _role_content_hashes(left, "user")
        user_right = _role_content_hashes(right, "user")
        assistant_left = _role_content_hashes(left, "assistant")
        assistant_right = _role_content_hashes(right, "assistant")
        groups_left = {record.split_group for record in left}
        groups_right = {record.split_group for record in right}
        report[f"{left_name}:{right_name}"] = {
            "exact_hash_overlap": len(exact_left & exact_right),
            "normalized_hash_overlap": len(normalized_left & normalized_right),
            "user_content_overlap": len(user_left & user_right),
            "assistant_content_overlap": len(assistant_left & assistant_right),
            "split_group_overlap": len(groups_left & groups_right),
        }
    return report


def assert_split_isolation(splits: Mapping[str, Iterable[SecurityRecord]]) -> None:
    report = split_isolation_report(splits)
    contaminated = {pair: counts for pair, counts in report.items() if any(counts.values())}
    if contaminated:
        raise ValueError(f"cross-split contamination detected: {contaminated}")


def _labels_indicate_raw_binary(labels: dict[str, Any]) -> bool:
    if labels.get("raw_malware_binary") is True:
        return True
    artifact_type = str(labels.get("artifact_type", "")).casefold()
    return artifact_type in {"malware_binary", "raw_binary", "executable_binary"}


def contains_executable_binary_payload(text: str) -> bool:
    """Detect embedded executable bytes; ordinary hashes and short signatures remain allowed."""

    if "\x00" in text or _HEX_BLOB.search(text):
        return True
    for match in _BASE64_BLOB.finditer(text):
        candidate = match.group(0)
        try:
            decoded_prefix = base64.b64decode(
                candidate[:256],
                altchars=b"-_",
                validate=False,
            )
        except (binascii.Error, ValueError):
            continue
        if any(decoded_prefix.startswith(magic) for magic in _EXECUTABLE_MAGICS):
            return True
    return False


def record_contains_raw_binary(record: SecurityRecord) -> bool:
    if _labels_indicate_raw_binary(record.labels):
        return True
    return any(contains_executable_binary_payload(message.content) for message in record.messages)


def contamination_rate(train_hashes: set[str], evaluation_hashes: set[str]) -> float:
    if not evaluation_hashes:
        return 0.0
    return len(train_hashes & evaluation_hashes) / len(evaluation_hashes)
