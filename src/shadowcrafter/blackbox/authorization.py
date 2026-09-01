"""Verification of immutable, scope-bound black-box authorization artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from shadowcrafter.blackbox.models import AuthorizationArtifact
from shadowcrafter.integrations.contracts import BlackBoxScope

MAX_AUTHORIZATION_ARTIFACT_BYTES = 65_536


class AuthorizationError(ValueError):
    """The supplied authorization proof is missing, invalid, expired, or mismatched."""


class _StrictJSONError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> object:
    raise _StrictJSONError(f"non-standard JSON constant: {value}")


def read_authorization_artifact(path: Path) -> bytes:
    """Read at most the artifact limit plus one byte from a local approval file."""

    with path.open("rb") as stream:
        return stream.read(MAX_AUTHORIZATION_ARTIFACT_BYTES + 1)


def verify_authorization_artifact(
    artifact_bytes: bytes,
    scope: BlackBoxScope,
    *,
    now: datetime | None = None,
) -> AuthorizationArtifact:
    """Verify content hash, validity window, identity, targets, and methods.

    The SHA-256 digest in ``scope.authorization`` must cover the exact bytes
    supplied here. This makes the approved document immutable without placing
    signing keys or approval bypasses in the assessment runtime.
    """

    if not artifact_bytes:
        raise AuthorizationError("an explicit authorization artifact is required")
    if len(artifact_bytes) > MAX_AUTHORIZATION_ARTIFACT_BYTES:
        raise AuthorizationError("authorization artifact exceeds the 64 KiB limit")

    expected_digest = scope.authorization.evidence_sha256
    actual_digest = hashlib.sha256(artifact_bytes).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise AuthorizationError("authorization artifact SHA-256 does not match the scope")

    try:
        raw = json.loads(
            artifact_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
        artifact = AuthorizationArtifact.model_validate(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, _StrictJSONError) as exc:
        raise AuthorizationError("authorization artifact is not valid strict JSON") from exc

    proof = scope.authorization
    if artifact.authorization_id != proof.authorization_id:
        raise AuthorizationError("authorization_id does not match the scope evidence")
    if artifact.scope_id != scope.scope_id:
        raise AuthorizationError("authorization artifact is bound to another scope")
    if artifact.approved_by != proof.approved_by:
        raise AuthorizationError("authorization approver does not match the scope evidence")
    if artifact.valid_from != proof.valid_from or artifact.valid_until != proof.valid_until:
        raise AuthorizationError("authorization validity window does not match the scope evidence")
    if artifact.allowed_targets != scope.allowed_targets:
        raise AuthorizationError("authorization targets do not exactly match the runtime scope")
    if artifact.safe_methods != scope.safe_methods:
        raise AuthorizationError("authorization methods do not exactly match the runtime scope")

    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise AuthorizationError("authorization verification time must be timezone-aware")
    if not proof.valid_from <= current < proof.valid_until:
        raise AuthorizationError("authorization is not currently valid")
    return artifact
