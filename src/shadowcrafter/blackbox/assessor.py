"""Fail-closed orchestration for authorized passive black-box observations."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from shadowcrafter.blackbox.authorization import (
    AuthorizationError,
    read_authorization_artifact,
    verify_authorization_artifact,
)
from shadowcrafter.blackbox.models import (
    AuthorizationArtifact,
    BlackBoxAssessmentResult,
    EvidenceRecord,
    SafetyLimits,
)
from shadowcrafter.blackbox.network import (
    NetworkSafetyError,
    PassiveTransport,
    PinnedRequest,
    PinnedStdlibTransport,
    Resolver,
    SystemResolver,
    TransportResponse,
    ValidatedTarget,
    parse_target,
    resolve_and_pin_target,
    validate_peer_ip,
)
from shadowcrafter.blackbox.observations import (
    build_evidence_record,
    derive_passive_findings,
)
from shadowcrafter.integrations.contracts import BlackBoxFinding, BlackBoxScope
from shadowcrafter.integrations.validators import validate_safe_http_method


class SlidingWindowRateLimiter:
    """Concurrency-safe, strict sliding-window request limiter."""

    def __init__(
        self,
        requests_per_window: int,
        *,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if requests_per_window < 1 or requests_per_window > 60:
            raise ValueError("rate limit must remain between 1 and 60 requests per window")
        if window_seconds <= 0:
            raise ValueError("rate-limit window must be positive")
        self._limit = requests_per_window
        self._window = window_seconds
        self._clock = clock
        self._sleep = sleep
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            delay = 0.0
            async with self._lock:
                current = self._clock()
                while self._timestamps and self._timestamps[0] <= current - self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._limit:
                    self._timestamps.append(current)
                    return
                delay = max(0.0, self._window - (current - self._timestamps[0]))
            await self._sleep(delay)


class AuthorizedBlackBoxAssessor:
    """Passive assessor that requires immutable approval before any request."""

    def __init__(
        self,
        *,
        scope: BlackBoxScope,
        authorization_artifact: bytes,
        resolver: Resolver | None = None,
        transport: PassiveTransport | None = None,
        limits: SafetyLimits | None = None,
    ) -> None:
        self.scope = scope
        self.limits = limits or SafetyLimits()
        self.authorization: AuthorizationArtifact = verify_authorization_artifact(
            authorization_artifact,
            scope,
        )
        self._resolver = resolver or SystemResolver()
        self._transport = transport or PinnedStdlibTransport()
        self._semaphore = asyncio.Semaphore(scope.max_concurrency)
        self._rate_limiter = SlidingWindowRateLimiter(
            scope.requests_per_minute,
        )
        self._dns_pins: dict[tuple[str, int], str] = {}
        self._dns_pin_lock = asyncio.Lock()

    def _require_current_authorization(self) -> None:
        current = datetime.now(UTC)
        if not self.authorization.valid_from <= current < self.authorization.valid_until:
            raise AuthorizationError("authorization expired before the passive request")

    async def _resolve_once(self, target: ValidatedTarget) -> str:
        """Pin one validated address per host for the complete assessment instance."""

        key = (target.host, target.port)
        async with self._dns_pin_lock:
            existing = self._dns_pins.get(key)
            if existing is not None:
                return existing
            pinned = await resolve_and_pin_target(
                target,
                self.scope,
                self._resolver,
                self.limits,
            )
            self._dns_pins[key] = pinned
            return pinned

    def _prepare_targets(self, targets: Sequence[str]) -> tuple[ValidatedTarget, ...]:
        if not targets:
            raise NetworkSafetyError("at least one exact target URL is required")
        if len(targets) > self.limits.max_targets:
            raise NetworkSafetyError("target count exceeds the assessment limit")
        parsed = tuple(parse_target(target, self.limits) for target in targets)
        canonical = tuple(target.canonical_url for target in parsed)
        if len(set(canonical)) != len(canonical):
            raise NetworkSafetyError("target URLs must be unique after normalization")
        allowed_paths = set(self.authorization.allowed_paths)
        if any(target.path not in allowed_paths for target in parsed):
            raise AuthorizationError(
                "target path is not exactly listed in the authorization artifact"
            )
        return parsed

    def _prepare_methods(self, methods: Sequence[str]) -> tuple[str, ...]:
        if not methods:
            raise NetworkSafetyError("at least one safe HTTP method is required")
        normalized = tuple(validate_safe_http_method(method) for method in methods)
        if len(set(normalized)) != len(normalized):
            raise NetworkSafetyError("assessment methods must be unique")
        if not set(normalized).issubset(self.scope.safe_methods):
            raise AuthorizationError("requested method is not authorized by the scope")
        return normalized

    def _validate_transport_response(
        self,
        target: ValidatedTarget,
        pinned_ip: str,
        response: TransportResponse,
    ) -> None:
        validate_peer_ip(response.peer_ip, pinned_ip)
        if response.status_code < 100 or response.status_code > 599:
            raise NetworkSafetyError("transport returned an invalid HTTP status")
        if (
            response.elapsed_ms < 0
            or response.elapsed_ms > self.limits.request_timeout_seconds * 1000
        ):
            raise NetworkSafetyError("transport response exceeded the configured time limit")
        if len(response.body) > self.limits.max_response_bytes + 1:
            raise NetworkSafetyError("transport exceeded the bounded response-body read")
        header_bytes = 0
        for name, value in response.headers:
            if not name or any(character in name for character in "\r\n"):
                raise NetworkSafetyError("transport returned an invalid response header name")
            if any(character in value for character in "\r\n\x00"):
                raise NetworkSafetyError("transport returned an unsafe response header value")
            header_bytes += len(name.encode()) + len(value.encode()) + 4
        if header_bytes > self.limits.max_header_bytes:
            raise NetworkSafetyError("response headers exceed the configured limit")
        if target.scheme == "https" and response.tls is None:
            raise NetworkSafetyError("HTTPS observation did not provide verified TLS metadata")
        if target.scheme == "http" and response.tls is not None:
            raise NetworkSafetyError("HTTP observation returned inconsistent TLS metadata")

    async def _observe_one(
        self,
        target: ValidatedTarget,
        method: str,
    ) -> tuple[EvidenceRecord, tuple[BlackBoxFinding, ...]]:
        async with self._semaphore:
            await self._rate_limiter.acquire()
            self._require_current_authorization()

            async def resolve_and_send() -> tuple[str, TransportResponse]:
                pinned_ip = await self._resolve_once(target)
                self._require_current_authorization()
                request = PinnedRequest(
                    target=target,
                    method=method,
                    pinned_ip=pinned_ip,
                    timeout_seconds=self.limits.request_timeout_seconds,
                    max_response_bytes=self.limits.max_response_bytes,
                    max_header_bytes=self.limits.max_header_bytes,
                )
                return pinned_ip, await self._transport.send(request)

            try:
                pinned_ip, response = await asyncio.wait_for(
                    resolve_and_send(),
                    timeout=self.limits.request_timeout_seconds,
                )
            except TimeoutError as exc:
                raise NetworkSafetyError("passive transport timed out") from exc
            self._validate_transport_response(target, pinned_ip, response)
            captured_at = datetime.now(UTC)
            record = build_evidence_record(
                scope=self.scope,
                target=target,
                method=method,
                response=response,
                limits=self.limits,
                captured_at=captured_at,
            )
            findings = derive_passive_findings(
                scope=self.scope,
                target=target,
                response=response,
                record=record,
                now=captured_at,
            )
            return record, findings

    async def assess(
        self,
        targets: Sequence[str],
        *,
        methods: Sequence[str] = ("HEAD",),
    ) -> BlackBoxAssessmentResult:
        """Assess exact approved URLs; any safety failure cancels the batch."""

        prepared_targets = self._prepare_targets(targets)
        prepared_methods = self._prepare_methods(methods)
        request_count = len(prepared_targets) * len(prepared_methods)
        if request_count > self.limits.max_requests:
            raise NetworkSafetyError("request count exceeds the assessment limit")
        self._require_current_authorization()
        started_at = datetime.now(UTC)

        tasks = [
            asyncio.create_task(self._observe_one(target, method))
            for target in prepared_targets
            for method in prepared_methods
        ]
        try:
            observations = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        evidence = tuple(
            sorted((item[0] for item in observations), key=lambda item: item.evidence_id)
        )
        findings = tuple(
            sorted(
                (finding for _record, candidates in observations for finding in candidates),
                key=lambda finding: finding.finding_id,
            )
        )
        return BlackBoxAssessmentResult(
            scope_id=self.scope.scope_id,
            authorization_id=self.authorization.authorization_id,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            findings=findings,
            evidence=evidence,
        )


async def assess_authorized_targets(
    *,
    scope: BlackBoxScope,
    authorization_artifact: bytes,
    targets: Sequence[str],
    methods: Sequence[str] = ("HEAD",),
    resolver: Resolver | None = None,
    transport: PassiveTransport | None = None,
    limits: SafetyLimits | None = None,
) -> BlackBoxAssessmentResult:
    """Async library entry point for broker-controlled integrations."""

    assessor = AuthorizedBlackBoxAssessor(
        scope=scope,
        authorization_artifact=authorization_artifact,
        resolver=resolver,
        transport=transport,
        limits=limits,
    )
    return await assessor.assess(targets, methods=methods)


def run_authorized_assessment(
    *,
    scope: BlackBoxScope,
    authorization_artifact: bytes | Path,
    targets: Sequence[str],
    methods: Sequence[str] = ("HEAD",),
    resolver: Resolver | None = None,
    transport: PassiveTransport | None = None,
    limits: SafetyLimits | None = None,
) -> BlackBoxAssessmentResult:
    """Synchronous library entry point; use the async entry inside event loops."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("use assess_authorized_targets inside an active event loop")
    artifact_bytes = (
        read_authorization_artifact(authorization_artifact)
        if isinstance(authorization_artifact, Path)
        else authorization_artifact
    )
    return asyncio.run(
        assess_authorized_targets(
            scope=scope,
            authorization_artifact=artifact_bytes,
            targets=targets,
            methods=methods,
            resolver=resolver,
            transport=transport,
            limits=limits,
        )
    )
