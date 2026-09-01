"""DNS-pinned, redirect-free transport for passive HTTP/TLS observations."""

from __future__ import annotations

import asyncio
import hashlib
import http.client
import ipaddress
import socket
import ssl
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import SplitResult, urlsplit, urlunsplit

from shadowcrafter.blackbox.models import SafetyLimits, TLSMetadata, validate_passive_path
from shadowcrafter.integrations.contracts import BlackBoxScope
from shadowcrafter.integrations.validators import target_is_allowlisted


class NetworkSafetyError(ValueError):
    """A target, DNS answer, peer, or response violated a fail-closed boundary."""


@dataclass(frozen=True, slots=True)
class ValidatedTarget:
    """A syntactically safe exact URL before DNS resolution."""

    canonical_url: str
    scheme: str
    host: str
    port: int
    path: str


@dataclass(frozen=True, slots=True)
class PinnedRequest:
    """A bodyless request bound to one already-validated IP address."""

    target: ValidatedTarget
    method: str
    pinned_ip: str
    timeout_seconds: float
    max_response_bytes: int
    max_header_bytes: int


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """Internal bounded response; body bytes never leave the assessor as evidence."""

    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    peer_ip: str
    elapsed_ms: float
    tls: TLSMetadata | None = None


class Resolver(Protocol):
    async def resolve(self, host: str, port: int) -> tuple[str, ...]:
        """Resolve an exact hostname to a bounded tuple of textual IP addresses."""


class PassiveTransport(Protocol):
    async def send(self, request: PinnedRequest) -> TransportResponse:
        """Send only the method/path represented by a bodyless pinned request."""


class SystemResolver:
    """System DNS resolver with deduplicated address output."""

    async def resolve(self, host: str, port: int) -> tuple[str, ...]:
        return await asyncio.to_thread(self._resolve, host, port)

    @staticmethod
    def _resolve(host: str, port: int) -> tuple[str, ...]:
        try:
            answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise NetworkSafetyError("target DNS resolution failed") from exc
        return tuple(dict.fromkeys(str(answer[4][0]) for answer in answers))


def parse_target(target: str, limits: SafetyLimits) -> ValidatedTarget:
    """Accept only query-free HTTP(S) URLs on their standard ports."""

    try:
        parsed = urlsplit(target)
        port = parsed.port
    except ValueError as exc:
        raise NetworkSafetyError("target URL contains an invalid port") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise NetworkSafetyError("only http and https targets are supported")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise NetworkSafetyError("target must contain a host and no user information")
    if parsed.query or parsed.fragment:
        raise NetworkSafetyError("query strings and fragments are outside passive scope")

    host = parsed.hostname.lower().rstrip(".")
    if not host or any(ord(character) > 127 or ord(character) < 33 for character in host):
        raise NetworkSafetyError("target hostname must be printable ASCII")
    expected_port = 443 if scheme == "https" else 80
    effective_port = port or expected_port
    if effective_port != expected_port:
        raise NetworkSafetyError("non-standard ports require a separate connector and are denied")

    path = parsed.path or "/"
    try:
        validate_passive_path(path, max_length=limits.max_path_length)
    except ValueError as exc:
        raise NetworkSafetyError("target path is not an unambiguous passive path") from exc

    netloc = f"[{host}]" if ":" in host else host
    canonical = urlunsplit(SplitResult(scheme, netloc, path, "", ""))
    return ValidatedTarget(
        canonical_url=canonical,
        scheme=scheme,
        host=host,
        port=effective_port,
        path=path,
    )


def _numeric_scope_allows(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address, scope: BlackBoxScope
) -> bool:
    for entry in scope.allowed_targets:
        try:
            if address in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


def _validate_resolved_address(
    value: str,
    *,
    direct_ip_target: bool,
    scope: BlackBoxScope,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise NetworkSafetyError("DNS returned a non-IP address") from exc

    always_denied = (
        address.is_unspecified
        or address.is_multicast
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
    )
    if always_denied:
        raise NetworkSafetyError("DNS returned an address class that is always denied")

    numeric_permission = _numeric_scope_allows(address, scope)
    if direct_ip_target and not numeric_permission:
        raise NetworkSafetyError("direct IP target is outside the exact IP/CIDR allowlist")
    if not address.is_global and not numeric_permission:
        raise NetworkSafetyError(
            "hostname resolved to a non-global address not separately authorized by IP/CIDR"
        )
    return address


async def resolve_and_pin_target(
    target: ValidatedTarget,
    scope: BlackBoxScope,
    resolver: Resolver,
    limits: SafetyLimits,
) -> str:
    """Resolve immediately before use and choose only a fully validated address.

    Every answer is checked, not just the selected one. A mixed public/private
    response therefore fails closed instead of enabling DNS rebinding or SSRF.
    """

    if not target_is_allowlisted(target.canonical_url, scope.allowed_targets):
        raise NetworkSafetyError("target host is outside the exact allowlist")
    try:
        literal_address = ipaddress.ip_address(target.host)
    except ValueError:
        literal_address = None

    answers: tuple[str, ...]
    if literal_address is not None:
        answers = (str(literal_address),)
    else:
        answers = await resolver.resolve(target.host, target.port)
    if not answers:
        raise NetworkSafetyError("target resolution produced no addresses")
    if len(answers) > limits.max_dns_answers:
        raise NetworkSafetyError("target returned too many DNS addresses")

    validated = {
        _validate_resolved_address(
            answer,
            direct_ip_target=literal_address is not None,
            scope=scope,
        )
        for answer in answers
    }
    # A deterministic choice makes audit logs reproducible. The transport is
    # given this address directly, eliminating a second hostname lookup.
    return str(sorted(validated, key=lambda item: (item.version, int(item)))[0])


def validate_peer_ip(peer_ip: str, pinned_ip: str) -> None:
    """Reject a transport that did not connect to the exact validated address."""

    try:
        peer = ipaddress.ip_address(peer_ip)
        pinned = ipaddress.ip_address(pinned_ip)
    except ValueError as exc:
        raise NetworkSafetyError("transport returned an invalid peer address") from exc
    if peer != pinned:
        raise NetworkSafetyError("connected peer differs from the DNS-pinned address")


class PinnedStdlibTransport:
    """Minimal HTTP/1.1 transport with TLS SNI and no redirect implementation."""

    async def send(self, request: PinnedRequest) -> TransportResponse:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._send_blocking, request),
                timeout=request.timeout_seconds,
            )
        except TimeoutError as exc:
            raise NetworkSafetyError("passive request exceeded the wall-clock timeout") from exc

    @staticmethod
    def _send_blocking(request: PinnedRequest) -> TransportResponse:
        started = time.monotonic()
        connection: socket.socket | ssl.SSLSocket | None = None
        response: http.client.HTTPResponse | None = None
        tls_metadata: TLSMetadata | None = None
        try:
            pinned_address = ipaddress.ip_address(request.pinned_ip)
            family = socket.AF_INET6 if pinned_address.version == 6 else socket.AF_INET
            connection = socket.socket(family, socket.SOCK_STREAM)
            connection.settimeout(request.timeout_seconds)
            destination: tuple[str, int] | tuple[str, int, int, int]
            destination = (
                (request.pinned_ip, request.target.port, 0, 0)
                if pinned_address.version == 6
                else (request.pinned_ip, request.target.port)
            )
            connection.connect(destination)
            if request.target.scheme == "https":
                context = ssl.create_default_context()
                connection = context.wrap_socket(connection, server_hostname=request.target.host)
                certificate = connection.getpeercert(binary_form=True)
                parsed_certificate = connection.getpeercert() or {}
                not_after_text = parsed_certificate.get("notAfter")
                not_after = (
                    datetime.fromtimestamp(ssl.cert_time_to_seconds(not_after_text), tz=UTC)
                    if isinstance(not_after_text, str)
                    else None
                )
                cipher_details = connection.cipher()
                tls_metadata = TLSMetadata(
                    protocol=connection.version() or "unknown",
                    cipher=cipher_details[0] if cipher_details else None,
                    certificate_sha256=(
                        hashlib.sha256(certificate).hexdigest() if certificate else None
                    ),
                    certificate_not_after=not_after,
                )

            peer_ip = str(connection.getpeername()[0])
            host_header = (
                f"[{request.target.host}]" if ":" in request.target.host else request.target.host
            )
            request_text = (
                f"{request.method} {request.target.path} HTTP/1.1\r\n"
                f"Host: {host_header}\r\n"
                "User-Agent: ShadowCrafter-Passive-Assessment/0.1\r\n"
                "Accept: */*\r\n"
                "Accept-Encoding: identity\r\n"
                "Connection: close\r\n\r\n"
            )
            connection.sendall(request_text.encode("ascii"))
            response = http.client.HTTPResponse(connection)
            response.begin()
            headers = tuple((name, value) for name, value in response.getheaders())
            header_bytes = sum(
                len(name.encode()) + len(value.encode()) + 4 for name, value in headers
            )
            if header_bytes > request.max_header_bytes:
                raise NetworkSafetyError("response headers exceed the configured limit")
            body = response.read(request.max_response_bytes + 1)
            elapsed_ms = (time.monotonic() - started) * 1000
            return TransportResponse(
                status_code=response.status,
                headers=headers,
                body=body,
                peer_ip=peer_ip,
                elapsed_ms=elapsed_ms,
                tls=tls_metadata,
            )
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            raise NetworkSafetyError("passive HTTP/TLS observation failed") from exc
        finally:
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()
