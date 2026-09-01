import json
from pathlib import Path

import httpx
import pytest

from shadowcrafter.data.manifest import sha256_bytes
from shadowcrafter.data.snapshot import snapshot_http_sources

REGISTRY = Path("configs/data/sources.yaml")
CISA_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def test_snapshot_writes_hash_addressed_artifact_and_manifest(tmp_path: Path) -> None:
    content = b'{"catalogVersion":"test","vulnerabilities":[]}'

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CISA_URL
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json; charset=utf-8",
                "etag": '"revision-one"',
            },
            content=content,
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        manifests = snapshot_http_sources(
            REGISTRY,
            tmp_path,
            source_ids={"cisa-kev"},
            client=client,
        )

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest["artifact"]["sha256"] == sha256_bytes(content)
    assert manifest["source"]["policy_class"] == "rag_only"
    target = tmp_path / "cisa-kev" / manifest["snapshot_id"]
    assert (target / "snapshot.json").read_bytes() == content
    assert json.loads((target / "manifest.json").read_text())["upstream_revision"] == (
        '"revision-one"'
    )


def test_snapshot_blocks_redirect_to_unapproved_host_before_request(tmp_path: Path) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            302,
            headers={"location": "https://127.0.0.1/internal"},
            request=request,
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValueError, match="unapproved snapshot host"),
    ):
        snapshot_http_sources(
            REGISTRY,
            tmp_path,
            source_ids={"cisa-kev"},
            client=client,
        )
    assert requests == [CISA_URL]


def test_snapshot_rejects_oversize_response_before_retaining_it(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "content-length": "999999999",
            },
            content=b"{}",
            request=request,
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValueError, match="exceeds configured size limit"),
    ):
        snapshot_http_sources(
            REGISTRY,
            tmp_path,
            source_ids={"cisa-kev"},
            client=client,
        )
    assert not (tmp_path / "cisa-kev").exists()


def test_snapshot_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b'{"version":1,"version":2}',
            request=request,
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValueError, match="duplicate JSON object key"),
    ):
        snapshot_http_sources(
            REGISTRY,
            tmp_path,
            source_ids={"cisa-kev"},
            client=client,
        )
