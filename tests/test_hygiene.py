import base64
from datetime import UTC, datetime

import pytest

from shadowcrafter.data.hygiene import (
    contains_executable_binary_payload,
    contamination_rate,
    normalized_content_hash,
    redact_secrets,
    split_isolation_report,
    stable_split,
)
from shadowcrafter.schemas import Message, Provenance, SecurityRecord, TaskType


def _record(record_id: str, group: str, content: str) -> SecurityRecord:
    record = SecurityRecord(
        record_id=record_id,
        task=TaskType.CVE_TRIAGE,
        messages=[
            Message(role="user", content=content),
            Message(role="assistant", content="Apply the vendor patch."),
        ],
        provenance=Provenance(
            source_id="odytssey-curated-security-instructions",
            license="Apache-2.0",
            retrieved_at=datetime.now(UTC),
            upstream_revision="test-revision",
            record_key=record_id,
            content_sha256="0" * 64,
        ),
        split_group=group,
    )
    record.provenance.content_sha256 = record.canonical_hash()
    return record


def test_redacts_known_token_shapes() -> None:
    fake_token = "hf_" + "abcdefghijklmnopqrstuvwxyz123456"
    text, count = redact_secrets(f"token={fake_token}")
    assert count == 1
    assert "<REDACTED_SECRET>" in text


def test_split_is_stable() -> None:
    assert stable_split("CVE-2026-1234") == stable_split("CVE-2026-1234")


def test_split_rejects_invalid_percentages_and_blank_group() -> None:
    with pytest.raises(ValueError):
        stable_split("", train=90, validation=5)
    with pytest.raises(ValueError):
        stable_split("group", train=-1, validation=5)


def test_contamination_rate() -> None:
    assert contamination_rate({"a", "b"}, {"b", "c"}) == 0.5


def test_normalized_hash_ignores_line_endings_and_trailing_space() -> None:
    left = _record("left", "group", "Review this advisory.  \r\n")
    right = _record("right", "group", "Review this advisory.\n")
    assert normalized_content_hash(left) == normalized_content_hash(right)


def test_split_report_finds_group_and_content_overlap() -> None:
    left = _record("left", "same-lineage", "same prompt")
    right = _record("right", "same-lineage", "same prompt")
    report = split_isolation_report({"train": [left], "test": [right]})["test:train"]
    assert report["normalized_hash_overlap"] == 1
    assert report["user_content_overlap"] == 1
    assert report["assistant_content_overlap"] == 1
    assert report["split_group_overlap"] == 1


def test_split_report_finds_repeated_completion_with_distinct_prompts() -> None:
    left = _record("left", "left-lineage", "first prompt")
    right = _record("right", "right-lineage", "second prompt")
    report = split_isolation_report({"train": [left], "validation": [right]})["train:validation"]
    assert report["normalized_hash_overlap"] == 0
    assert report["user_content_overlap"] == 0
    assert report["assistant_content_overlap"] == 1
    assert report["split_group_overlap"] == 0


def test_detects_embedded_pe_payload_but_not_short_signature() -> None:
    encoded = base64.b64encode(b"MZ" + (b"\x00" * 2048)).decode()
    encoded_archive = base64.b64encode(b"PK\x03\x04" + (b"x" * 512)).decode()
    assert contains_executable_binary_payload(encoded)
    assert contains_executable_binary_payload(encoded_archive)
    assert not contains_executable_binary_payload("MZ header and hash deadbeef")
