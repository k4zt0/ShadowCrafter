"""Shared validation helpers for defensive integration output contracts."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping, Sequence
from urllib.parse import urlparse

_CVE_PATTERN = re.compile(r"^CVE-(?:19|20)\d{2}-\d{4,}$", re.IGNORECASE)
_CWE_PATTERN = re.compile(r"^CWE-\d{1,5}$", re.IGNORECASE)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_STIX_ID_PATTERN = re.compile(
    r"^[a-z][a-z0-9-]*--[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HOSTNAME_PATTERN = re.compile(
    r"^(?=.{1,253}\.?$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.?$",
    re.IGNORECASE,
)

# These patterns identify executable instructions rather than ordinary prose
# such as "the issue could permit command execution".
_EXECUTABLE_PATTERNS = (
    re.compile(r"(?:^|\n)\s*(?:\$|#|>)\s+\S"),
    re.compile(r"\$\([^\n]+\)"),
    re.compile(r"`[^`\n]+`"),
    re.compile(
        r"\b(?:bash|zsh|sh|cmd(?:\.exe)?|powershell|pwsh|python\d*|perl|ruby)"
        r"\s+(?:-c|/c|-e|-enc|--command)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:curl|wget)\b[^\n]{0,240}\|\s*(?:bash|sh|zsh)\b", re.IGNORECASE),
    re.compile(
        r"(?:;|&&|\|\|)\s*(?:rm|curl|wget|bash|sh|python\d*|powershell|cmd)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:msfvenom|meterpreter|shellcode)\b", re.IGNORECASE),
)

_STATE_CHANGING_QUERY_PATTERNS = (
    re.compile(r"(?:^|[;|])\s*(?:delete|drop|update|insert|alter|truncate|create)\b", re.I),
    re.compile(r"\b(?:exec(?:ute)?|outputlookup|sendalert)\b", re.I),
)

SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def validate_cve_id(value: str) -> str:
    normalized = value.strip().upper()
    if not _CVE_PATTERN.fullmatch(normalized):
        raise ValueError("CVE identifiers must use the form CVE-YYYY-NNNN")
    return normalized


def validate_cwe_id(value: str) -> str:
    normalized = value.strip().upper()
    if not _CWE_PATTERN.fullmatch(normalized):
        raise ValueError("CWE identifiers must use the form CWE-NNN")
    return normalized


def validate_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("expected a 64-character SHA-256 hex digest")
    return normalized


def validate_stix_id(value: str) -> str:
    normalized = value.strip().lower()
    if not _STIX_ID_PATTERN.fullmatch(normalized):
        raise ValueError("invalid STIX 2.x identifier")
    return normalized


def reject_executable_content(value: str) -> str:
    """Reject command-shaped content while allowing defensive narrative text."""

    normalized = value.strip()
    if not normalized:
        raise ValueError("text must not be empty")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in normalized):
        raise ValueError("control characters are not permitted")
    if any(pattern.search(normalized) for pattern in _EXECUTABLE_PATTERNS):
        raise ValueError("executable commands and offensive payloads are not permitted")
    return normalized


def validate_detection_query(value: str) -> str:
    """Allow read-only SIEM searches while rejecting mutation/action syntax."""

    normalized = reject_executable_content(value)
    if ";" in normalized or any(
        pattern.search(normalized) for pattern in _STATE_CHANGING_QUERY_PATTERNS
    ):
        raise ValueError("SIEM candidates must be read-only searches")
    return normalized


def validate_safe_http_method(value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in SAFE_HTTP_METHODS:
        raise ValueError("black-box assessment permits only GET, HEAD, and OPTIONS")
    return normalized


def validate_allowlist_entry(value: str) -> str:
    """Validate one exact host/IP or bounded CIDR without wildcard expansion."""

    normalized = value.strip().lower().rstrip(".")
    invalid_entry = (
        not normalized
        or "*" in normalized
        or "://" in normalized
        or ("/" in normalized and " " in normalized)
    )
    if invalid_entry:
        raise ValueError("allowlist entries must be exact hosts, IPs, or CIDRs")
    try:
        network = ipaddress.ip_network(normalized, strict=False)
    except ValueError:
        if "/" in normalized or not _HOSTNAME_PATTERN.fullmatch(normalized):
            raise ValueError("invalid host or CIDR allowlist entry") from None
        return normalized
    if network.prefixlen == 0:
        raise ValueError("internet-wide CIDRs are never valid assessment scope")
    return str(network)


def target_is_allowlisted(target: str, allowlist: Sequence[str]) -> bool:
    """Return whether a hostname/IP/URL resolves syntactically to the fixed scope.

    This function performs no DNS lookup.  Hostname entries require exact matches
    (or an explicit parent domain entry prefixed by no wildcard); IP addresses are
    checked against exact IP/CIDR membership.
    """

    parsed = urlparse(target if "://" in target else f"//{target}")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return any("/" not in entry and host == entry.rstrip(".") for entry in allowlist)
    for entry in allowlist:
        try:
            if address in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


def reject_payload_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Reject opaque execution-bearing keys in extension metadata."""

    forbidden_keys = {
        "binary",
        "bytes",
        "command",
        "cmd",
        "executable",
        "exploit",
        "payload",
        "raw_sample",
        "script",
        "shellcode",
    }
    for key, item in value.items():
        if key.lower().replace("-", "_") in forbidden_keys:
            raise ValueError(f"execution-bearing metadata key is forbidden: {key}")
        if isinstance(item, (bytes, bytearray, memoryview)):
            raise ValueError("binary content is not permitted in output contracts")
        if isinstance(item, Mapping):
            reject_payload_mapping(item)
    return value
