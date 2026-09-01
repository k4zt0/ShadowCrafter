import asyncio
import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from shadowcrafter.blackbox.assessor import (
    AuthorizedBlackBoxAssessor,
    SlidingWindowRateLimiter,
)
from shadowcrafter.blackbox.authorization import AuthorizationError, read_blackbox_scope
from shadowcrafter.blackbox.models import AuthorizationArtifact, SafetyLimits, TLSMetadata
from shadowcrafter.blackbox.network import (
    NetworkSafetyError,
    PinnedRequest,
    TransportResponse,
)
from shadowcrafter.integrations.contracts import AuthorizationEvidence, BlackBoxScope

NOW = datetime.now(UTC)
PUBLIC_TEST_IP = "93.184.216.34"


class FakeResolver:
    def __init__(self, answers: Mapping[str, tuple[str, ...]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, int]] = []

    async def resolve(self, host: str, port: int) -> tuple[str, ...]:
        self.calls.append((host, port))
        return self.answers[host]


class FakeTransport:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: tuple[tuple[str, str], ...] = (),
        body: bytes = b"",
        peer_ip: str | None = None,
        delay: float = 0.0,
        tls_protocol: str = "TLSv1.3",
        tls_cipher: str = "TLS_AES_256_GCM_SHA384",
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.body = body
        self.peer_ip = peer_ip
        self.delay = delay
        self.tls_protocol = tls_protocol
        self.tls_cipher = tls_cipher
        self.calls: list[PinnedRequest] = []
        self.active = 0
        self.max_active = 0

    async def send(self, request: PinnedRequest) -> TransportResponse:
        self.calls.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            tls = (
                TLSMetadata(
                    protocol=self.tls_protocol,
                    cipher=self.tls_cipher,
                    certificate_sha256="c" * 64,
                    certificate_not_after=NOW + timedelta(days=90),
                )
                if request.target.scheme == "https"
                else None
            )
            return TransportResponse(
                status_code=self.status_code,
                headers=self.headers,
                body=self.body,
                peer_ip=self.peer_ip or request.pinned_ip,
                elapsed_ms=max(self.delay * 1000, 1.0),
                tls=tls,
            )
        finally:
            self.active -= 1


def make_scope_and_artifact(
    *,
    targets: tuple[str, ...] = ("app.example.test",),
    paths: tuple[str, ...] = ("/",),
    methods: tuple[str, ...] = ("GET", "HEAD", "OPTIONS"),
    requests_per_minute: int = 30,
    max_concurrency: int = 2,
    valid_from: datetime = NOW - timedelta(hours=1),
    valid_until: datetime = NOW + timedelta(hours=1),
) -> tuple[BlackBoxScope, bytes]:
    artifact = AuthorizationArtifact(
        authorization_id="AUTH-BB-001",
        scope_id="scope-bb-001",
        approved_by="asset-owner@example.test",
        purpose="Passive HTTP and TLS configuration review for an owned test service.",
        allowed_targets=targets,
        allowed_paths=paths,
        safe_methods=methods,
        valid_from=valid_from,
        valid_until=valid_until,
        passive_read_only_only=True,
        payloads_allowed=False,
        redirects_allowed=False,
        brute_force_allowed=False,
        credential_testing_allowed=False,
        exploit_execution_allowed=False,
        denial_of_service_allowed=False,
        state_changing_requests_allowed=False,
    )
    artifact_bytes = artifact.model_dump_json().encode()
    scope = BlackBoxScope(
        scope_id=artifact.scope_id,
        authorization=AuthorizationEvidence(
            authorization_id=artifact.authorization_id,
            approved_by=artifact.approved_by,
            evidence_uri="vault://authorization/AUTH-BB-001",
            evidence_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
            valid_from=valid_from,
            valid_until=valid_until,
        ),
        allowed_targets=targets,
        safe_methods=methods,
        requests_per_minute=requests_per_minute,
        max_concurrency=max_concurrency,
    )
    return scope, artifact_bytes


def make_assessor(
    *,
    scope: BlackBoxScope,
    artifact: bytes,
    resolver: FakeResolver,
    transport: FakeTransport,
    limits: SafetyLimits | None = None,
) -> AuthorizedBlackBoxAssessor:
    return AuthorizedBlackBoxAssessor(
        scope=scope,
        authorization_artifact=artifact,
        resolver=resolver,
        transport=transport,
        limits=limits,
    )


def test_authorization_artifact_is_required_hash_bound_and_current() -> None:
    scope, artifact = make_scope_and_artifact()
    resolver = FakeResolver({"app.example.test": (PUBLIC_TEST_IP,)})
    transport = FakeTransport()

    with pytest.raises(AuthorizationError, match="SHA-256"):
        make_assessor(
            scope=scope,
            artifact=artifact + b" ",
            resolver=resolver,
            transport=transport,
        )

    expired_scope, expired_artifact = make_scope_and_artifact(
        valid_from=NOW - timedelta(days=2),
        valid_until=NOW - timedelta(days=1),
    )
    with pytest.raises(AuthorizationError, match="currently valid"):
        make_assessor(
            scope=expired_scope,
            artifact=expired_artifact,
            resolver=resolver,
            transport=transport,
        )

    duplicate_key_artifact = artifact.replace(
        b'{"schema_version":"1.0"',
        b'{"schema_version":"1.0","schema_version":"1.0"',
        1,
    )
    duplicate_proof = scope.authorization.model_copy(
        update={"evidence_sha256": hashlib.sha256(duplicate_key_artifact).hexdigest()}
    )
    duplicate_scope = scope.model_copy(update={"authorization": duplicate_proof})
    with pytest.raises(AuthorizationError, match="strict JSON"):
        make_assessor(
            scope=duplicate_scope,
            artifact=duplicate_key_artifact,
            resolver=resolver,
            transport=transport,
        )


def test_runtime_scope_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    scope, _artifact = make_scope_and_artifact()
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(scope.model_dump_json())
    assert read_blackbox_scope(scope_path) == scope

    duplicate = scope.model_dump_json().replace(
        '{"scope_id":',
        '{"scope_id":"duplicate","scope_id":',
        1,
    )
    scope_path.write_text(duplicate)
    with pytest.raises(AuthorizationError, match="strict JSON"):
        read_blackbox_scope(scope_path)


def test_passive_observation_redacts_secrets_and_returns_typed_findings() -> None:
    scope, artifact = make_scope_and_artifact(methods=("GET",))
    secret_cookie = "session=TOP-SECRET-TOKEN; Path=/; SameSite=Lax"  # noqa: S105
    transport = FakeTransport(
        headers=(
            ("Server", "Example/1.0"),
            ("Set-Cookie", secret_cookie),
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Credentials", "true"),
            ("Content-Security-Policy", "script-src 'nonce-SECRET-NONCE'"),
            ("Location", "https://app.example.test/next?token=LOCATION-SECRET"),
            ("X-Internal-Token", "HEADER-SECRET"),
        ),
        body=b"BODY-SECRET",
    )
    assessor = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=FakeResolver({"app.example.test": (PUBLIC_TEST_IP,)}),
        transport=transport,
    )

    result = asyncio.run(assessor.assess(("https://app.example.test/",), methods=("GET",)))

    serialized = result.model_dump_json()
    assert "TOP-SECRET-TOKEN" not in serialized
    assert "SECRET-NONCE" not in serialized
    assert "LOCATION-SECRET" not in serialized
    assert "HEADER-SECRET" not in serialized
    assert "BODY-SECRET" not in serialized
    assert "session=[redacted]" in serialized
    assert not result.payloads_sent
    assert not result.credentials_sent
    assert not result.redirects_followed
    assert result.evidence[0].response_body_included is False
    assert {finding.title for finding in result.findings} >= {
        "Cookie without Secure attribute observed",
        "Conflicting cross-origin credential policy observed",
        "Technology identification header observed",
    }
    assert len(transport.calls) == 1
    assert transport.calls[0].method == "GET"
    assert transport.calls[0].pinned_ip == PUBLIC_TEST_IP


def test_exact_host_path_and_query_boundaries_fail_before_transport() -> None:
    scope, artifact = make_scope_and_artifact(paths=("/health",), methods=("HEAD",))
    resolver = FakeResolver(
        {
            "app.example.test": (PUBLIC_TEST_IP,),
            "child.app.example.test": (PUBLIC_TEST_IP,),
        }
    )
    transport = FakeTransport()
    assessor = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=resolver,
        transport=transport,
    )

    with pytest.raises(AuthorizationError, match="path"):
        asyncio.run(assessor.assess(("https://app.example.test/admin",)))
    with pytest.raises(NetworkSafetyError, match="query"):
        asyncio.run(assessor.assess(("https://app.example.test/health?probe=1",)))
    with pytest.raises(NetworkSafetyError, match="passive path"):
        asyncio.run(assessor.assess(("https://app.example.test/%2e%2e/health",)))
    with pytest.raises(NetworkSafetyError, match="non-standard ports"):
        asyncio.run(assessor.assess(("https://app.example.test:8443/health",)))
    with pytest.raises(NetworkSafetyError, match="allowlist"):
        asyncio.run(assessor.assess(("https://child.app.example.test/health",)))
    assert transport.calls == []


@pytest.mark.parametrize(
    "answers",
    [
        ("127.0.0.1",),
        (PUBLIC_TEST_IP, "10.10.0.5"),
        ("169.254.169.254",),
    ],
)
def test_private_or_mixed_dns_answers_require_explicit_numeric_scope(
    answers: tuple[str, ...],
) -> None:
    scope, artifact = make_scope_and_artifact(methods=("HEAD",))
    transport = FakeTransport()
    assessor = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=FakeResolver({"app.example.test": answers}),
        transport=transport,
    )

    with pytest.raises(NetworkSafetyError, match="non-global|always denied"):
        asyncio.run(assessor.assess(("https://app.example.test/",)))
    assert transport.calls == []


def test_explicit_private_cidr_is_allowed_but_peer_rebinding_is_rejected() -> None:
    targets = ("app.example.test", "10.10.0.0/24")
    scope, artifact = make_scope_and_artifact(targets=targets, methods=("HEAD",))
    resolver = FakeResolver({"app.example.test": ("10.10.0.5",)})
    accepted = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=resolver,
        transport=FakeTransport(),
    )
    result = asyncio.run(accepted.assess(("https://app.example.test/",)))
    assert result.evidence[0].peer_ip == "10.10.0.5"

    changed_peer = FakeTransport(peer_ip="10.10.0.6")
    rejected = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=resolver,
        transport=changed_peer,
    )
    with pytest.raises(NetworkSafetyError, match="peer differs"):
        asyncio.run(rejected.assess(("https://app.example.test/",)))

    link_local_targets = ("app.example.test", "169.254.0.0/16")
    link_local_scope, link_local_artifact = make_scope_and_artifact(
        targets=link_local_targets,
        methods=("HEAD",),
    )
    link_local = make_assessor(
        scope=link_local_scope,
        artifact=link_local_artifact,
        resolver=FakeResolver({"app.example.test": ("169.254.169.254",)}),
        transport=FakeTransport(),
    )
    with pytest.raises(NetworkSafetyError, match="always denied"):
        asyncio.run(link_local.assess(("https://app.example.test/",)))


def test_redirect_is_evidence_only_and_never_followed() -> None:
    scope, artifact = make_scope_and_artifact(methods=("HEAD",))
    transport = FakeTransport(
        status_code=302,
        headers=(("Location", "https://outside.example.test/path?secret=value"),),
    )
    assessor = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=FakeResolver({"app.example.test": (PUBLIC_TEST_IP,)}),
        transport=transport,
    )

    result = asyncio.run(assessor.assess(("https://app.example.test/",)))

    assert len(transport.calls) == 1
    assert "secret" not in result.model_dump_json()
    assert "Off-scope redirect blocked" in {finding.title for finding in result.findings}
    assert not result.redirects_followed


def test_body_header_and_request_limits_fail_closed() -> None:
    scope, artifact = make_scope_and_artifact(
        paths=("/a", "/b"),
        methods=("GET", "HEAD"),
    )
    resolver = FakeResolver({"app.example.test": (PUBLIC_TEST_IP,)})
    small_limits = SafetyLimits(max_response_bytes=8, max_requests=2)
    bounded_transport = FakeTransport(body=b"123456789")
    assessor = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=resolver,
        transport=bounded_transport,
        limits=small_limits,
    )
    result = asyncio.run(assessor.assess(("https://app.example.test/a",), methods=("GET",)))
    assert result.evidence[0].body_bytes_captured == 8
    assert result.evidence[0].body_truncated

    oversized_transport = FakeTransport(body=b"1234567890")
    oversized = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=resolver,
        transport=oversized_transport,
        limits=small_limits,
    )
    with pytest.raises(NetworkSafetyError, match="bounded response"):
        asyncio.run(oversized.assess(("https://app.example.test/a",), methods=("GET",)))

    large_headers = FakeTransport(headers=(("Server", "x" * 1100),))
    header_limited = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=resolver,
        transport=large_headers,
        limits=small_limits.model_copy(update={"max_header_bytes": 1024}),
    )
    with pytest.raises(NetworkSafetyError, match="headers exceed"):
        asyncio.run(header_limited.assess(("https://app.example.test/a",), methods=("GET",)))

    with pytest.raises(NetworkSafetyError, match="request count"):
        asyncio.run(
            assessor.assess(
                ("https://app.example.test/a", "https://app.example.test/b"),
                methods=("GET", "HEAD"),
            )
        )


def test_concurrency_and_sliding_window_limits_are_enforced() -> None:
    paths = ("/a", "/b", "/c", "/d")
    scope, artifact = make_scope_and_artifact(
        paths=paths,
        methods=("HEAD",),
        max_concurrency=2,
    )
    transport = FakeTransport(delay=0.01)
    resolver = FakeResolver({"app.example.test": (PUBLIC_TEST_IP,)})
    assessor = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=resolver,
        transport=transport,
    )
    targets = tuple(f"https://app.example.test{path}" for path in paths)
    asyncio.run(assessor.assess(targets))
    assert transport.max_active == 2
    assert len(resolver.calls) == 1

    current = [0.0]
    sleeps: list[float] = []

    async def advance(delay: float) -> None:
        sleeps.append(delay)
        current[0] += delay

    limiter = SlidingWindowRateLimiter(
        2,
        window_seconds=60.0,
        clock=lambda: current[0],
        sleep=advance,
    )

    async def consume() -> None:
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()

    asyncio.run(consume())
    assert sleeps == [60.0]


def test_only_get_head_options_are_accepted() -> None:
    scope, artifact = make_scope_and_artifact(methods=("GET",))
    assessor = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=FakeResolver({"app.example.test": (PUBLIC_TEST_IP,)}),
        transport=FakeTransport(),
    )

    with pytest.raises(ValueError, match="GET, HEAD, and OPTIONS"):
        asyncio.run(assessor.assess(("https://app.example.test/",), methods=("POST",)))


def test_extended_blackbox_rules_are_evidence_grounded_and_non_executing() -> None:
    scope, artifact = make_scope_and_artifact(methods=("GET", "OPTIONS"))
    sensitive_marker = "do-not-retain-this-path"
    transport = FakeTransport(
        status_code=500,
        headers=(
            ("Strict-Transport-Security", "max-age=0"),
            ("Content-Security-Policy-Report-Only", "script-src 'nonce-PRIVATE'"),
            ("X-Content-Type-Options", "off"),
            ("X-Frame-Options", "ALLOW-FROM https://legacy.example.test"),
            ("Referrer-Policy", "unsafe-url"),
            ("Permissions-Policy", "camera=*, geolocation=(self), microphone=()"),
            ("Set-Cookie", "session=value; SameSite=None"),
            ("Cache-Control", "public, max-age=60"),
            ("Access-Control-Allow-Origin", "null"),
            ("Access-Control-Allow-Credentials", "true"),
            ("Allow", "GET, HEAD, OPTIONS, TRACE"),
            ("Server", "ExampleServer/1.2.3"),
            ("Content-Type", "text/html; charset=utf-8"),
        ),
        body=(
            "Traceback (most recent call last):\n"
            f' File \"/{sensitive_marker}/app.py\", line 7, in handler\n'
        ).encode(),
        tls_cipher="ECDHE-RSA-RC4-SHA",
    )
    assessor = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=FakeResolver({"app.example.test": (PUBLIC_TEST_IP,)}),
        transport=transport,
    )

    result = asyncio.run(
        assessor.assess(
            ("https://app.example.test/",),
            methods=("GET", "OPTIONS"),
        )
    )

    titles = {finding.title for finding in result.findings}
    assert titles >= {
        "Transport policy explicitly disabled",
        "Content security policy is report-only",
        "Invalid content type protection observed",
        "Invalid legacy frame policy observed",
        "Unsafe referrer disclosure policy observed",
        "Broad browser feature policy observed",
        "Cross-site cookie lacks Secure protection",
        "Public caching allowed on a cookie-setting response",
        "Null origin accepted with credentials",
        "Technology version disclosure observed",
        "Diagnostic or tunneling HTTP method advertised",
        "Weak TLS cipher observed",
        "Verbose error details observed",
    }
    get_evidence = next(item for item in result.evidence if item.method == "GET")
    assert [signal.signal_id for signal in get_evidence.body_signals] == [
        "verbose-error-detail"
    ]
    serialized = result.model_dump_json()
    assert sensitive_marker not in serialized
    assert "PRIVATE" not in serialized
    assert not result.payloads_sent
    assert not result.credentials_sent

    invalid = get_evidence.model_dump(mode="json")
    invalid["method"] = "HEAD"
    with pytest.raises(ValidationError, match="authorized GET"):
        type(get_evidence).model_validate(invalid)


@pytest.mark.parametrize(
    ("body", "expected_signal", "expected_title"),
    [
        (
            b"<html><title>Index of /public</title><a href='../'>Parent Directory</a></html>",
            "directory-listing",
            "Directory listing content observed",
        ),
        (
            b"<html><title>phpinfo()</title><h1>PHP Version 8</h1><p>PHP Credits</p></html>",
            "runtime-diagnostic-page",
            "Runtime diagnostic page observed",
        ),
    ],
)
def test_bounded_get_body_signatures_never_return_body_content(
    body: bytes,
    expected_signal: str,
    expected_title: str,
) -> None:
    scope, artifact = make_scope_and_artifact(methods=("GET",))
    transport = FakeTransport(
        headers=(("Content-Type", "text/html"),),
        body=body,
    )
    assessor = make_assessor(
        scope=scope,
        artifact=artifact,
        resolver=FakeResolver({"app.example.test": (PUBLIC_TEST_IP,)}),
        transport=transport,
    )

    result = asyncio.run(assessor.assess(("https://app.example.test/",), methods=("GET",)))

    assert expected_signal in {item.signal_id for item in result.evidence[0].body_signals}
    assert expected_title in {finding.title for finding in result.findings}
    assert body.decode() not in result.model_dump_json()
