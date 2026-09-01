"""Passive response evidence capture, redaction, and candidate findings."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Literal, cast
from urllib.parse import urlsplit, urlunsplit

from shadowcrafter.blackbox.models import EvidenceRecord, ObservedHeader, SafetyLimits
from shadowcrafter.blackbox.network import TransportResponse, ValidatedTarget
from shadowcrafter.integrations.contracts import (
    BlackBoxFinding,
    BlackBoxScope,
    Confidence,
    EvidenceReference,
    Severity,
)
from shadowcrafter.integrations.validators import target_is_allowlisted

_CAPTURED_HEADERS = frozenset(
    {
        "access-control-allow-credentials",
        "access-control-allow-methods",
        "access-control-allow-origin",
        "allow",
        "content-security-policy",
        "content-type",
        "cross-origin-embedder-policy",
        "cross-origin-opener-policy",
        "cross-origin-resource-policy",
        "location",
        "permissions-policy",
        "referrer-policy",
        "server",
        "set-cookie",
        "strict-transport-security",
        "x-content-type-options",
        "x-frame-options",
        "x-powered-by",
    }
)
_CSP_SECRET = re.compile(r"(?i)(?:nonce|sha256|sha384|sha512)-[^\s;'\"]+")
_COOKIE_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_KEY_VALUE_SECRET = re.compile(r"(?i)\b(token|secret|password|passwd|api[-_]?key)=([^\s;,]+)")
_AUTHORIZATION_VALUE = re.compile(r"(?i)\b(?:basic|bearer)\s+[A-Za-z0-9._~+/=-]+")
_JWT_VALUE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")


def _clean_header_text(value: str, *, limit: int = 2000) -> str:
    cleaned = " ".join(value.replace("\x00", "").split()) or "[empty]"
    cleaned = _KEY_VALUE_SECRET.sub(r"\1=[redacted]", cleaned)
    cleaned = _AUTHORIZATION_VALUE.sub("authorization-value-[redacted]", cleaned)
    cleaned = _JWT_VALUE.sub("jwt-[redacted]", cleaned)
    return cleaned if len(cleaned) <= limit else f"{cleaned[: limit - 12]} [truncated]"


def _redact_cookie(value: str) -> str:
    parts = [part.strip() for part in value.split(";") if part.strip()]
    first = parts[0] if parts else "cookie"
    name, separator, _cookie_value = first.partition("=")
    safe_name = name if separator and _COOKIE_NAME.fullmatch(name) else "cookie"
    attributes: list[str] = []
    for part in parts[1:]:
        attribute, separator, attribute_value = part.partition("=")
        normalized = attribute.strip().lower()
        if normalized in {"secure", "httponly", "partitioned"}:
            attributes.append(normalized.title())
        elif normalized == "samesite" and separator:
            same_site = attribute_value.strip().lower()
            attributes.append(
                f"SameSite={same_site.title()}"
                if same_site in {"strict", "lax", "none"}
                else "SameSite=[redacted]"
            )
        elif normalized in {"domain", "expires", "max-age", "path", "priority"}:
            attributes.append(f"{attribute.strip()}=[redacted]")
    suffix = "; " + "; ".join(attributes) if attributes else ""
    return f"{safe_name}=[redacted]{suffix}"


def _redact_location(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
            return "[redacted non-HTTP location]"
        hostname = parsed.hostname
        if parsed.scheme and not hostname:
            return "[redacted invalid location]"
        if hostname:
            host = hostname.lower().rstrip(".")
            netloc = f"[{host}]" if ":" in host else host
            return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", "", ""))
        return parsed.path or "/"
    except ValueError:
        return "[redacted invalid location]"


def redact_response_headers(headers: tuple[tuple[str, str], ...]) -> tuple[ObservedHeader, ...]:
    """Retain only security-relevant headers and remove credential-like values."""

    captured: list[ObservedHeader] = []
    for raw_name, raw_value in headers:
        name = raw_name.strip().lower()
        if name not in _CAPTURED_HEADERS:
            continue
        if name == "set-cookie":
            value = _redact_cookie(raw_value)
        elif name == "location":
            value = _redact_location(raw_value)
        elif name == "content-security-policy":
            value = _CSP_SECRET.sub("nonce-or-hash-[redacted]", _clean_header_text(raw_value))
        else:
            value = _clean_header_text(raw_value)
        captured.append(ObservedHeader(name=name, value=value))
    return tuple(captured)


def _header_map(headers: tuple[tuple[str, str], ...]) -> dict[str, tuple[str, ...]]:
    collected: dict[str, list[str]] = {}
    for raw_name, raw_value in headers:
        name = raw_name.strip().lower()
        if name in _CAPTURED_HEADERS:
            collected.setdefault(name, []).append(_clean_header_text(raw_value, limit=4000))
    return {name: tuple(values) for name, values in collected.items()}


def build_evidence_record(
    *,
    scope: BlackBoxScope,
    target: ValidatedTarget,
    method: str,
    response: TransportResponse,
    limits: SafetyLimits,
    captured_at: datetime | None = None,
) -> EvidenceRecord:
    """Create bounded metadata evidence without persisting response body content."""

    body_prefix = response.body[: limits.max_response_bytes]
    body_truncated = len(response.body) > limits.max_response_bytes
    identity = "\x00".join(
        (scope.scope_id, target.canonical_url, method, str(response.status_code), response.peer_ip)
    )
    evidence_id = f"bb-ev-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    return EvidenceRecord(
        evidence_id=evidence_id,
        target=target.canonical_url,
        method=cast(Literal["GET", "HEAD", "OPTIONS"], method),
        status_code=response.status_code,
        peer_ip=response.peer_ip,
        response_headers=redact_response_headers(response.headers),
        body_prefix_sha256=hashlib.sha256(body_prefix).hexdigest(),
        body_bytes_captured=len(body_prefix),
        body_truncated=body_truncated,
        elapsed_ms=response.elapsed_ms,
        tls=response.tls,
        captured_at=captured_at or datetime.now(UTC),
    )


def _evidence_reference(record: EvidenceRecord) -> EvidenceReference:
    canonical = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return EvidenceReference(
        evidence_id=record.evidence_id,
        source_uri=f"evidence://black-box/{record.evidence_id}",
        description=(
            f"Passive {record.method} response metadata was captured; "
            "response body content and credentials were not retained."
        ),
        sha256=hashlib.sha256(canonical).hexdigest(),
        observed_at=record.captured_at,
    )


def _finding(
    *,
    rule_id: str,
    scope: BlackBoxScope,
    record: EvidenceRecord,
    title: str,
    observation: str,
    severity: Severity,
    confidence: Confidence,
    remediation: str,
    cwes: tuple[str, ...] = (),
) -> BlackBoxFinding:
    identifier = hashlib.sha256(
        f"{scope.scope_id}\x00{record.evidence_id}\x00{rule_id}".encode()
    ).hexdigest()[:20]
    finding = BlackBoxFinding(
        finding_id=f"bb-{identifier}",
        scope_id=scope.scope_id,
        target=record.target,
        assessed_methods=(record.method,),
        title=title,
        observation=observation,
        severity=severity,
        confidence=confidence,
        cwe_candidates=cwes,
        evidence=(_evidence_reference(record),),
        remediation=(remediation,),
    )
    finding.validate_against_scope(scope)
    return finding


def derive_passive_findings(
    *,
    scope: BlackBoxScope,
    target: ValidatedTarget,
    response: TransportResponse,
    record: EvidenceRecord,
    now: datetime | None = None,
) -> tuple[BlackBoxFinding, ...]:
    """Derive conservative candidates from already-observed response metadata."""

    headers = _header_map(response.headers)
    findings: list[BlackBoxFinding] = []

    def add(
        rule_id: str,
        title: str,
        observation: str,
        severity: Severity,
        confidence: Confidence,
        remediation: str,
        cwes: tuple[str, ...] = (),
    ) -> None:
        findings.append(
            _finding(
                rule_id=rule_id,
                scope=scope,
                record=record,
                title=title,
                observation=observation,
                severity=severity,
                confidence=confidence,
                remediation=remediation,
                cwes=cwes,
            )
        )

    if target.scheme == "http":
        add(
            "cleartext-http",
            "Cleartext HTTP transport observed",
            "The authorized endpoint was observed over HTTP without transport encryption.",
            Severity.MEDIUM,
            Confidence.HIGH,
            "Prefer HTTPS and redirect users to the authenticated TLS endpoint at the boundary.",
            ("CWE-319",),
        )
    elif "strict-transport-security" not in headers:
        add(
            "missing-hsts",
            "Transport policy header not observed",
            "The HTTPS response did not contain a Strict-Transport-Security header.",
            Severity.LOW,
            Confidence.MEDIUM,
            "Review deployment requirements and add a bounded transport policy when appropriate.",
            ("CWE-319",),
        )

    if "content-security-policy" not in headers:
        add(
            "missing-csp",
            "Content security policy not observed",
            (
                "The response did not contain a Content-Security-Policy header; "
                "applicability depends on whether the endpoint serves browser content."
            ),
            Severity.LOW,
            Confidence.LOW,
            "For browser-rendered content, define and test a restrictive content security policy.",
            ("CWE-693",),
        )
    if "x-content-type-options" not in headers:
        add(
            "missing-nosniff",
            "Content type sniffing protection not observed",
            "The response did not contain an X-Content-Type-Options nosniff directive.",
            Severity.LOW,
            Confidence.MEDIUM,
            (
                "Set an explicit content type and enable nosniff protection where browser "
                "clients consume the response."
            ),
            ("CWE-693",),
        )
    policy_values = " ".join(headers.get("content-security-policy", ())).lower()
    if "x-frame-options" not in headers and "frame-ancestors" not in policy_values:
        add(
            "missing-frame-policy",
            "Frame embedding policy not observed",
            (
                "Neither X-Frame-Options nor a frame-ancestors content policy was observed; "
                "relevance depends on browser rendering."
            ),
            Severity.LOW,
            Confidence.LOW,
            "For browser-rendered pages, define an explicit frame embedding policy.",
            ("CWE-1021",),
        )

    cookie_headers = headers.get("set-cookie", ())
    if cookie_headers:
        cookie_attributes = tuple(
            {
                part.partition("=")[0].strip().lower()
                for part in value.split(";")[1:]
                if part.strip()
            }
            for value in cookie_headers
        )
        if target.scheme == "https" and any(
            "secure" not in attributes for attributes in cookie_attributes
        ):
            add(
                "cookie-missing-secure",
                "Cookie without Secure attribute observed",
                (
                    "At least one HTTPS response cookie did not include the Secure attribute; "
                    "cookie values were redacted."
                ),
                Severity.MEDIUM,
                Confidence.HIGH,
                "Mark security-sensitive cookies Secure and validate their transport behavior.",
                ("CWE-614",),
            )
        if any("httponly" not in attributes for attributes in cookie_attributes):
            add(
                "cookie-missing-httponly",
                "Cookie without HttpOnly attribute observed",
                (
                    "At least one response cookie did not include the HttpOnly attribute; "
                    "cookie values were redacted."
                ),
                Severity.LOW,
                Confidence.MEDIUM,
                "Mark cookies that do not require script access as HttpOnly.",
                ("CWE-1004",),
            )
        if any("samesite" not in attributes for attributes in cookie_attributes):
            add(
                "cookie-missing-samesite",
                "Cookie without SameSite attribute observed",
                (
                    "At least one response cookie did not include an explicit SameSite attribute; "
                    "cookie values were redacted."
                ),
                Severity.LOW,
                Confidence.MEDIUM,
                "Set a SameSite policy appropriate for the application's cross-site requirements.",
                ("CWE-1275",),
            )

    origins = {value.strip().lower() for value in headers.get("access-control-allow-origin", ())}
    credentials = {
        value.strip().lower() for value in headers.get("access-control-allow-credentials", ())
    }
    if "*" in origins:
        add(
            "cors-wildcard",
            "Wildcard cross-origin policy observed",
            "The response advertised a wildcard Access-Control-Allow-Origin policy.",
            Severity.LOW,
            Confidence.MEDIUM,
            (
                "Confirm that all returned resources are intended for any web origin and "
                "otherwise restrict the origin policy."
            ),
            ("CWE-942",),
        )
        if "true" in credentials:
            add(
                "cors-wildcard-credentials",
                "Conflicting cross-origin credential policy observed",
                (
                    "Wildcard origin and credential allowance headers were observed together; "
                    "browser handling and endpoint intent require manual validation."
                ),
                Severity.MEDIUM,
                Confidence.MEDIUM,
                (
                    "Use an explicit allowlist of trusted origins and review whether credentials "
                    "are required."
                ),
                ("CWE-942",),
            )

    if "server" in headers or "x-powered-by" in headers:
        add(
            "technology-disclosure",
            "Technology identification header observed",
            (
                "The response exposed a Server or X-Powered-By header that may reveal "
                "implementation details."
            ),
            Severity.INFORMATIONAL,
            Confidence.HIGH,
            "Minimize unnecessary product and version disclosure at the application boundary.",
            ("CWE-200",),
        )

    if record.method == "OPTIONS":
        advertised = ",".join(
            headers.get("allow", ()) + headers.get("access-control-allow-methods", ())
        )
        methods = {item.strip().upper() for item in advertised.split(",") if item.strip()}
        state_changing = sorted(methods - {"GET", "HEAD", "OPTIONS"})
        if state_changing:
            add(
                "advertised-state-changing-methods",
                "State-changing HTTP methods advertised",
                (
                    "The passive OPTIONS response advertised additional state-changing methods; "
                    "none were invoked."
                ),
                Severity.INFORMATIONAL,
                Confidence.MEDIUM,
                (
                    "Confirm that advertised methods require appropriate authentication, "
                    "authorization, and request validation."
                ),
                ("CWE-749",),
            )

    if 300 <= response.status_code < 400:
        locations = headers.get("location", ())
        if locations:
            location = _redact_location(locations[0])
            absolute_location = location
            if location.startswith("/"):
                absolute_location = f"{target.scheme}://{target.host}{location}"
            off_scope = not target_is_allowlisted(absolute_location, scope.allowed_targets)
            add(
                "off-scope-redirect" if off_scope else "redirect-observed",
                "Off-scope redirect blocked" if off_scope else "Redirect observed and not followed",
                (
                    (
                        "The response pointed outside the authorized host allowlist; "
                        "the redirect was not followed."
                    )
                    if off_scope
                    else (
                        "The response returned a redirect within the host allowlist; "
                        "redirect following remained disabled."
                    )
                ),
                Severity.LOW if off_scope else Severity.INFORMATIONAL,
                Confidence.HIGH,
                (
                    "Review redirect intent and keep downstream targets independently authorized "
                    "before assessment."
                ),
                ("CWE-601",) if off_scope else (),
            )

    if target.scheme == "https" and record.tls is not None:
        if record.tls.protocol in {"TLSv1", "TLSv1.1", "SSLv2", "SSLv3"}:
            add(
                "legacy-tls",
                "Legacy TLS protocol observed",
                "The TLS connection negotiated a legacy protocol version.",
                Severity.MEDIUM,
                Confidence.HIGH,
                "Disable legacy protocol versions and require a currently supported TLS baseline.",
                ("CWE-326",),
            )
        current = now or datetime.now(UTC)
        not_after = record.tls.certificate_not_after
        if not_after is not None and not_after <= current + timedelta(days=30):
            add(
                "certificate-expiry",
                "TLS certificate expiry is near",
                "The observed TLS certificate expiry is within thirty days or has already passed.",
                Severity.MEDIUM,
                Confidence.HIGH,
                (
                    "Renew and deploy the certificate through the approved certificate lifecycle "
                    "before expiry."
                ),
                ("CWE-324",),
            )

    return tuple(findings)
