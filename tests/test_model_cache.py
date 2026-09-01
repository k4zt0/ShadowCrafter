from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shadowcrafter.training.model_cache import (
    ModelCacheError,
    load_pinned_cache_manifest,
    verify_hf_cache_snapshot,
    write_cache_manifest,
)

MODEL_ID = "org/model"
REVISION = "1" * 40


def _git_blob(content: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(content)}\0".encode())
    digest.update(content)
    return digest.hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, SimpleNamespace]:
    cache = tmp_path / "models--org--model"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    config = b'{"model_type":"verified"}\n'
    weights = b"safe tensor bytes"
    (blobs / "config").write_bytes(config)
    (blobs / "weights").write_bytes(weights)
    (snapshot / "config.json").symlink_to(Path("../../blobs/config"))
    (snapshot / "model.safetensors").symlink_to(Path("../../blobs/weights"))
    info = SimpleNamespace(
        sha=REVISION,
        siblings=[
            SimpleNamespace(
                rfilename="config.json",
                size=len(config),
                blob_id=_git_blob(config),
                lfs=None,
            ),
            SimpleNamespace(
                rfilename="model.safetensors",
                size=len(weights),
                blob_id="2" * 40,
                lfs=SimpleNamespace(
                    size=len(weights),
                    sha256=hashlib.sha256(weights).hexdigest(),
                ),
            ),
        ],
    )
    return snapshot, info


def test_stream_verifies_complete_snapshot_with_injected_metadata(tmp_path: Path) -> None:
    snapshot, info = _fixture(tmp_path)
    calls: list[tuple[str, str]] = []

    def loader(model_id: str, revision: str) -> SimpleNamespace:
        calls.append((model_id, revision))
        return info

    manifest = verify_hf_cache_snapshot(
        snapshot,
        model_id=MODEL_ID,
        revision=REVISION,
        model_info_loader=loader,
    )
    assert calls == [(MODEL_ID, REVISION)]
    assert manifest["complete_repository_snapshot"] is True
    assert manifest["file_count"] == 2
    algorithms = {item["identity"]["algorithm"] for item in manifest["files"]}
    assert algorithms == {"git-blob-sha1", "sha256"}


def test_cache_refuses_mutated_or_partial_weights(tmp_path: Path) -> None:
    snapshot, info = _fixture(tmp_path)
    (snapshot.parent.parent / "blobs" / "weights").write_bytes(b"mutated")
    with pytest.raises(ModelCacheError, match="size mismatch"):
        verify_hf_cache_snapshot(
            snapshot,
            model_id=MODEL_ID,
            revision=REVISION,
            model_info=info,
        )

    (snapshot / "config.json").unlink()
    with pytest.raises(ModelCacheError, match="incomplete"):
        verify_hf_cache_snapshot(
            snapshot,
            model_id=MODEL_ID,
            revision=REVISION,
            model_info=info,
        )


def test_cache_refuses_escape_and_unhashed_model_artifact(tmp_path: Path) -> None:
    snapshot, info = _fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("not cache owned", encoding="utf-8")
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(outside)
    with pytest.raises(ModelCacheError, match="escapes"):
        verify_hf_cache_snapshot(
            snapshot,
            model_id=MODEL_ID,
            revision=REVISION,
            model_info=info,
        )

    snapshot, info = _fixture(tmp_path / "second")
    info.siblings[1].lfs = None
    with pytest.raises(ModelCacheError, match="no authoritative LFS"):
        verify_hf_cache_snapshot(
            snapshot,
            model_id=MODEL_ID,
            revision=REVISION,
            model_info=info,
        )


def test_manifest_pin_is_exact_and_never_overwritten(tmp_path: Path) -> None:
    snapshot, info = _fixture(tmp_path)
    payload = verify_hf_cache_snapshot(
        snapshot,
        model_id=MODEL_ID,
        revision=REVISION,
        model_info=info,
    )
    path = tmp_path / "cache-manifest.json"
    sha256 = write_cache_manifest(path, payload)
    pinned = load_pinned_cache_manifest(
        path,
        sha256,
        model_id=MODEL_ID,
        revision=REVISION,
    )
    assert pinned.payload == payload
    with pytest.raises(ModelCacheError, match="overwrite"):
        write_cache_manifest(path, payload)
    with pytest.raises(ModelCacheError, match="SHA-256 mismatch"):
        load_pinned_cache_manifest(
            path,
            "f" * 64,
            model_id=MODEL_ID,
            revision=REVISION,
        )


def test_manifest_rejects_inventory_hash_tampering(tmp_path: Path) -> None:
    snapshot, info = _fixture(tmp_path)
    payload = verify_hf_cache_snapshot(
        snapshot,
        model_id=MODEL_ID,
        revision=REVISION,
        model_info=info,
    )
    payload["files_sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ModelCacheError, match="inventory hash"):
        load_pinned_cache_manifest(
            path,
            sha256,
            model_id=MODEL_ID,
            revision=REVISION,
        )
