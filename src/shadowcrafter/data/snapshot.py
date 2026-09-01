"""Immutable, hash-addressed snapshots for approved HTTPS JSON feeds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from shadowcrafter.data.manifest import sha256_bytes, write_json_exclusive
from shadowcrafter.data.registry import DataSource, SourceType, load_registry

_SNAPSHOT_HOST_ALLOWLIST = {
    "raw.githubusercontent.com",
    "www.cisa.gov",
}
_MAX_REDIRECTS = 3


@dataclass(frozen=True)
class FetchResult:
    content: bytes
    final_url: str
    media_type: str
    etag: str | None
    last_modified: str | None


def _validate_snapshot_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"snapshot URL must use HTTPS: {url}")
    if parsed.username or parsed.password:
        raise ValueError("snapshot URL must not contain credentials")
    if parsed.port not in (None, 443):
        raise ValueError(f"snapshot URL uses an unapproved port: {parsed.port}")
    if parsed.hostname not in _SNAPSHOT_HOST_ALLOWLIST:
        raise ValueError(f"unapproved snapshot host: {parsed.hostname}")


def _read_bounded(response: httpx.Response, max_bytes: int) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length from snapshot source") from exc
        if declared_size > max_bytes:
            raise ValueError(
                f"snapshot exceeds configured size limit: {declared_size} > {max_bytes}"
            )

    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > max_bytes:
            raise ValueError(f"snapshot exceeds configured size limit: > {max_bytes}")
        chunks.append(chunk)
    return b"".join(chunks)


def _fetch_json_source(
    client: httpx.Client,
    source: DataSource,
    global_max_bytes: int,
) -> FetchResult:
    if source.url is None or source.type != SourceType.HTTP_JSON:
        raise ValueError(f"source {source.id} is not an HTTPS JSON snapshot source")
    if not source.snapshot.enabled:
        raise ValueError(f"automatic snapshot is disabled for source {source.id}")

    current_url = str(source.url)
    max_bytes = min(source.snapshot.max_bytes or global_max_bytes, global_max_bytes)
    for redirect_number in range(_MAX_REDIRECTS + 1):
        _validate_snapshot_url(current_url)
        with client.stream(
            "GET",
            current_url,
            follow_redirects=False,
            headers={
                "Accept": "application/json",
                "User-Agent": "ShadowCrafter-data-snapshot/1",
            },
        ) as response:
            if response.is_redirect:
                if redirect_number == _MAX_REDIRECTS:
                    raise ValueError("snapshot source exceeded redirect limit")
                location = response.headers.get("location")
                if not location:
                    raise ValueError("snapshot redirect omitted Location header")
                current_url = urljoin(current_url, location)
                # Validate before the next request, preventing redirect-based SSRF.
                _validate_snapshot_url(current_url)
                continue

            response.raise_for_status()
            media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if media_type not in source.snapshot.allowed_media_types:
                raise ValueError(
                    f"source {source.id} returned unapproved media type {media_type!r}"
                )
            content = _read_bounded(response, max_bytes)
            return FetchResult(
                content=content,
                final_url=current_url,
                media_type=media_type,
                etag=response.headers.get("etag"),
                last_modified=response.headers.get("last-modified"),
            )
    raise AssertionError("redirect loop exited unexpectedly")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _validate_json(content: bytes) -> None:
    try:
        payload = json.loads(content, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("snapshot response is not valid UTF-8 JSON") from exc
    if not isinstance(payload, (dict, list)):
        raise ValueError("snapshot JSON must contain an object or array")


def snapshot_http_sources(
    config_path: Path,
    output_dir: Path,
    *,
    source_ids: set[str] | None = None,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Snapshot approved registry feeds without following unvalidated redirects."""

    registry = load_registry(config_path)
    if source_ids:
        selected = [registry.source(source_id) for source_id in sorted(source_ids)]
    else:
        selected = [source for source in registry.sources if source.snapshot.enabled]

    owned_client = client is None
    active_client = client or httpx.Client(timeout=httpx.Timeout(120.0))
    retrieved_at = datetime.now(UTC)
    manifests: list[dict[str, Any]] = []
    try:
        for source in selected:
            result = _fetch_json_source(
                active_client,
                source,
                registry.policy.max_snapshot_bytes,
            )
            _validate_json(result.content)
            digest = sha256_bytes(result.content)
            snapshot_id = f"{source.id}-{retrieved_at.strftime('%Y%m%dT%H%M%SZ')}-{digest[:12]}"
            target = output_dir / source.id / snapshot_id
            target.mkdir(parents=True, exist_ok=False)
            snapshot_path = target / "snapshot.json"
            with snapshot_path.open("xb") as handle:
                handle.write(result.content)

            upstream_revision = result.etag or result.last_modified or f"sha256:{digest}"
            manifest: dict[str, Any] = {
                "schema_version": 2,
                "snapshot_id": snapshot_id,
                "source": {
                    "id": source.id,
                    "provider": source.provider,
                    "type": source.type,
                    "requested_url": str(source.url),
                    "resolved_url": result.final_url,
                    "policy_class": source.policy_class,
                    "allowed_purposes": sorted(source.allowed_purposes),
                    "license": {
                        "id": source.license.id,
                        "url": str(source.license.url),
                        "status": source.license.status,
                        "redistribution": source.license.redistribution,
                        "attribution": source.attribution,
                    },
                    "content_kind": source.safety.content_kind,
                },
                "registry_sha256": registry.canonical_sha256(),
                "retrieved_at_utc": retrieved_at.isoformat(),
                "upstream_revision": upstream_revision,
                "http": {
                    "etag": result.etag,
                    "last_modified": result.last_modified,
                    "media_type": result.media_type,
                },
                "artifact": {
                    "path": "snapshot.json",
                    "sha256": digest,
                    "size": len(result.content),
                },
                "validation": {
                    "https_allowlist": "passed",
                    "bounded_download": "passed",
                    "json_parse": "passed",
                    "raw_malware_binaries": "prohibited_by_registry",
                },
            }
            write_json_exclusive(target / "manifest.json", manifest)
            manifests.append(manifest)
    finally:
        if owned_client:
            active_client.close()
    return manifests
