from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

import shadowcrafter.release.remote_huggingface as publisher
from shadowcrafter.cli import app
from shadowcrafter.evaluation.gate import GateResult
from shadowcrafter.release.remote_huggingface import (
    PublishResult,
    RemoteReleaseManifest,
    build_ssh_reader_argv,
    load_remote_release_manifest,
    publish_remote_official_release,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@pytest.fixture(autouse=True)
def _local_hub_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(publisher, "_local_hub_token", lambda: "test-credential")


class FakeApi:
    def __init__(self, *, parent: str, commit: str, private_before: bool = True) -> None:
        self.parent = parent
        self.commit = commit
        self.private_before = private_before
        self.commit_calls: list[dict[str, Any]] = []

    def model_info(self, _repo_id: str, **kwargs: Any) -> object:
        revision = kwargs.get("revision")
        if revision is None:
            return SimpleNamespace(private=self.private_before, sha=self.parent)
        return SimpleNamespace(private=True, sha=revision)

    def create_commit(self, _repo_id: str, operations: object, **kwargs: Any) -> object:
        self.commit_calls.append({"operations": list(operations), **kwargs})
        return SimpleNamespace(oid=self.commit)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _approval_payload(
    name: str,
    checkpoint: str,
    inventory_sha256: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "review": name,
        "passed": True,
        "repo_id": "KaztoRay/ShadowCrafter-9B",
        "release_id": "exp-v1",
        "candidate_checkpoint_sha256": checkpoint,
        "remote_inventory_sha256": inventory_sha256,
        "private_official_release_authorized": True,
        "public_release_authorized": False,
    }
    if name == "license":
        payload.update(
            {
                "commercial_release_authorized": False,
                "benchmark_material_sharing_authorized": False,
            }
        )
    return payload


def _model_card(
    *,
    evaluation_status: str,
    reason: str | None = None,
    report: dict[str, Any] | None = None,
) -> bytes:
    evaluation: dict[str, Any]
    if report is None:
        evaluation = {
            "status": "not-yet-evaluated",
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "quality_target_met": None,
        }
    else:
        evaluation = {
            "status": "measured",
            "benchmark": "AI4Sec/cti-bench",
            "revision": report["benchmark"]["upstream_revision"],
            "dataset_sha256": report["benchmark"]["dataset_sha256"],
            "sample_count": report["benchmark"]["sample_count"],
            **report["overall"]["metrics"],
            "quality_target_met": report["quality_target_met"],
        }
    metadata = {
        "license": "other",
        "shadowcrafter_release": {
            "status": "Official Release",
            "visibility": "private",
            "commercial_use": False,
            "release_id": "exp-v1",
            "repository": "KaztoRay/ShadowCrafter-9B",
            "candidate_checkpoint_sha256": "b" * 64,
        },
        "shadowcrafter_evaluation": evaluation,
    }
    body = "# ShadowCrafter Official Release\n"
    if evaluation_status == "not-yet-evaluated":
        body += f"\nEvaluation status: not yet evaluated. {reason}\n"
    return ("---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n" + body).encode()


def _bundle(
    root: Path,
    *,
    measured_report: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, bytes]]:
    release_id = "exp-v1"
    parent = "a" * 40
    checkpoint = "b" * 64
    reason = "Evaluation is scheduled after this engineering-only private upload."
    card = _model_card(
        evaluation_status="measured" if measured_report is not None else "not-yet-evaluated",
        reason=reason,
        report=measured_report,
    )
    remote_files = {
        "README.md": card,
        f"releases/{release_id}/adapter_config.json": b'{"peft_type":"LORA"}\n',
        f"releases/{release_id}/adapter_model.safetensors": b"SAFE-FAKE-TENSOR-DATA",
    }

    entries = [
        {"path": path, "size": len(content), "sha256": _sha256(content)}
        for path, content in sorted(remote_files.items())
    ]
    remote_root = "/root/ShadowCrafter/artifacts/releases/shadowcrafter-9b/exp-v1"
    total_bytes = sum(len(content) for content in remote_files.values())
    inventory = {
        "remote_root": remote_root,
        "files": entries,
        "total_bytes": total_bytes,
    }
    inventory_sha256 = _sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    )
    approval_refs: dict[str, dict[str, str]] = {}
    for name in ("artifact_integrity", "provenance", "license", "privacy", "safety"):
        path = root / f"{name}.json"
        _write_json(path, _approval_payload(name, checkpoint, inventory_sha256))
        approval_refs[name] = {"path": path.name, "sha256": _sha256(path.read_bytes())}

    evidence_path = root / "release-evidence.json"
    if measured_report is not None:
        evidence_path.write_bytes(b'{"frozen":"measured"}\n')
        evaluation = {
            "status": "measured",
            "evidence_manifest_sha256": _sha256(evidence_path.read_bytes()),
            "reason": None,
        }
    else:
        evaluation = {
            "status": "not-yet-evaluated",
            "evidence_manifest_sha256": None,
            "reason": reason,
        }

    manifest = {
        "schema_version": 1,
        "release_id": release_id,
        "repo_id": "KaztoRay/ShadowCrafter-9B",
        "release_tier": "Official Release",
        "visibility": "private",
        "commercial_release": False,
        "parent_commit": parent,
        "candidate_checkpoint_sha256": checkpoint,
        "remote_root": remote_root,
        "ssh": {"host": "capella.cloud.vessl.ai", "port": 31044, "user": "root"},
        "files": entries,
        "total_bytes": total_bytes,
        "evaluation": evaluation,
        "approvals": approval_refs,
    }
    manifest_path = root / "remote-release-manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, evidence_path, remote_files


def _key(root: Path) -> Path:
    path = root / "test-key.pem"
    path.write_text("not-a-real-key")
    path.chmod(0o600)
    return path


def _manifest_pin(path: Path) -> str:
    return _sha256(path.read_bytes())


def _report(*, quality_target_met: bool) -> dict[str, Any]:
    checkpoint = "b" * 64
    return {
        "passed": True,
        "evidence_manifest_sha256": _sha256(b'{"frozen":"measured"}\n'),
        "candidate": {
            "model_id": "KaztoRay/ShadowCrafter-9B",
            "checkpoint_sha256": checkpoint,
        },
        "benchmark": {
            "repository_id": "AI4Sec/cti-bench",
            "upstream_revision": "9" * 40,
            "dataset_sha256": "8" * 64,
            "sample_count": 5533,
        },
        "overall": {
            "metrics": {
                "accuracy": 0.51,
                "balanced_accuracy": 0.49,
                "macro_f1": 0.48,
            }
        },
        "quality_target_met": quality_target_met,
        "authorization": {
            "model_publication_authorized": True,
            "required_visibility": "private",
            "public_publication_authorized": False,
            "release_tier": "Official Release",
            "commercial_use_permitted": False,
        },
    }


def test_not_yet_evaluated_private_release_is_one_verified_memory_commit(tmp_path: Path) -> None:
    manifest_path, _evidence_path, remote_files = _bundle(tmp_path)
    key_path = _key(tmp_path)
    manifest_pin = _manifest_pin(manifest_path)
    manifest, _ = load_remote_release_manifest(manifest_path, manifest_pin)
    api = FakeApi(parent=manifest.parent_commit, commit="c" * 40)
    remote_calls: list[tuple[str, str]] = []

    def remote_reader(
        connection: publisher.SSHConnection,
        root: str,
        entry: publisher.RemoteFileEntry,
        key: Path,
    ) -> bytes:
        assert "HF_TOKEN" not in str(connection)
        assert key == key_path
        path = entry.path
        remote_calls.append((root, path))
        return remote_files[path]

    def hub_stream(_repo: str, _revision: str, path: str, credential: str) -> list[bytes]:
        assert credential
        return [remote_files[path][:3], remote_files[path][3:]]

    before_files = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    result = publish_remote_official_release(
        manifest_path,
        manifest_sha256=manifest_pin,
        ssh_key=key_path,
        api=api,
        remote_reader=remote_reader,
        hub_streamer=hub_stream,
        operation_factory=lambda path, content: (path, content),
    )
    after_files = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}

    assert result.commit_sha == "c" * 40
    assert result.evaluation_status == "not-yet-evaluated"
    assert result.quality_target_met is None
    assert len(api.commit_calls) == 1
    call = api.commit_calls[0]
    assert call["parent_commit"] == "a" * 40
    assert call["revision"] == "main"
    assert call["num_threads"] == 1
    assert dict(call["operations"]) == remote_files
    assert {path for _, path in remote_calls} == set(remote_files)
    assert before_files == after_files


def test_measured_accuracy_shortfall_does_not_block_private_official_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(quality_target_met=False)
    manifest_path, evidence_path, remote_files = _bundle(tmp_path, measured_report=report)
    manifest_pin = _manifest_pin(manifest_path)
    manifest, _ = load_remote_release_manifest(manifest_path, manifest_pin)
    monkeypatch.setattr(
        publisher,
        "load_and_evaluate",
        lambda _evidence, _config: GateResult(passed=True, failures=(), report=report),
    )
    api = FakeApi(parent=manifest.parent_commit, commit="d" * 40)

    result = publish_remote_official_release(
        manifest_path,
        manifest_sha256=manifest_pin,
        ssh_key=_key(tmp_path),
        evidence_path=evidence_path,
        api=api,
        remote_reader=lambda _connection, _root, entry, _key_path: remote_files[entry.path],
        hub_streamer=lambda _repo, _revision, path, _token: [remote_files[path]],
        operation_factory=lambda path, content: (path, content),
    )

    assert result.quality_target_met is False
    assert len(api.commit_calls) == 1


def test_public_repo_or_parent_race_blocks_before_remote_bytes(tmp_path: Path) -> None:
    manifest_path, _evidence_path, _remote_files = _bundle(tmp_path)
    manifest_pin = _manifest_pin(manifest_path)
    manifest, _ = load_remote_release_manifest(manifest_path, manifest_pin)
    calls = 0

    def reader(*_args: object) -> bytes:
        nonlocal calls
        calls += 1
        return b""

    with pytest.raises(ValueError, match="not private"):
        publish_remote_official_release(
            manifest_path,
            manifest_sha256=manifest_pin,
            ssh_key=_key(tmp_path),
            api=FakeApi(parent=manifest.parent_commit, commit="c" * 40, private_before=False),
            remote_reader=reader,
        )
    assert calls == 0

    with pytest.raises(ValueError, match="parent commit changed"):
        publish_remote_official_release(
            manifest_path,
            manifest_sha256=manifest_pin,
            ssh_key=_key(tmp_path),
            api=FakeApi(parent="e" * 40, commit="c" * 40),
            remote_reader=reader,
        )
    assert calls == 0


def test_post_upload_stream_checksum_mismatch_fails(tmp_path: Path) -> None:
    manifest_path, _evidence_path, remote_files = _bundle(tmp_path)
    manifest_pin = _manifest_pin(manifest_path)
    manifest, _ = load_remote_release_manifest(manifest_path, manifest_pin)
    api = FakeApi(parent=manifest.parent_commit, commit="c" * 40)

    with pytest.raises(RuntimeError, match="post-upload Hub checksum mismatch"):
        publish_remote_official_release(
            manifest_path,
            manifest_sha256=manifest_pin,
            ssh_key=_key(tmp_path),
            api=api,
            remote_reader=lambda _connection, _root, entry, _key_path: remote_files[entry.path],
            hub_streamer=lambda _repo, _revision, _path, _token: [b"tampered"],
            operation_factory=lambda path, content: (path, content),
        )


def test_manifest_rejects_shell_paths_and_unapproved_repositories(tmp_path: Path) -> None:
    manifest_path, _evidence_path, _remote_files = _bundle(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["files"][0]["path"] = "README.md;touch-pwned"
    with pytest.raises(ValidationError):
        RemoteReleaseManifest.model_validate(payload)

    payload = json.loads(manifest_path.read_text())
    payload["repo_id"] = "attacker/public"
    with pytest.raises(ValidationError):
        RemoteReleaseManifest.model_validate(payload)


def test_ssh_argv_is_batch_only_and_contains_no_hf_token_or_remote_path(tmp_path: Path) -> None:
    manifest_path, _evidence_path, _remote_files = _bundle(tmp_path)
    manifest, _ = load_remote_release_manifest(manifest_path, _manifest_pin(manifest_path))
    key_path = _key(tmp_path)

    argv = build_ssh_reader_argv(manifest.ssh, key_path)

    assert argv[0] == "ssh"
    assert argv[1:3] == ("-F", "/dev/null")
    assert "BatchMode=yes" in argv
    assert "StrictHostKeyChecking=yes" in argv
    assert "PermitLocalCommand=no" in argv
    assert "ProxyCommand=none" in argv
    assert "HF_TOKEN" not in " ".join(argv)
    assert manifest.remote_root not in " ".join(argv)
    assert all(entry.path not in " ".join(argv) for entry in manifest.files)


def test_subprocess_reader_sends_only_bounded_json_on_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _evidence_path, remote_files = _bundle(tmp_path)
    manifest, _ = load_remote_release_manifest(manifest_path, _manifest_pin(manifest_path))
    entry = next(item for item in manifest.files if item.path.endswith(".safetensors"))

    class RecordingInput:
        def __init__(self) -> None:
            self.data = bytearray()
            self.closed = False

        def write(self, content: bytes) -> int:
            self.data.extend(content)
            return len(content)

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self, content: bytes) -> None:
            self.stdin = RecordingInput()
            stdout_read, stdout_write = os.pipe()
            stderr_read, stderr_write = os.pipe()
            os.write(stdout_write, content)
            os.close(stdout_write)
            os.close(stderr_write)
            self.stdout = os.fdopen(stdout_read, "rb", buffering=0)
            self.stderr = os.fdopen(stderr_read, "rb", buffering=0)
            self.killed = False

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def kill(self) -> None:
            self.killed = True

    process = FakeProcess(remote_files[entry.path])
    captured: dict[str, object] = {}

    def fake_popen(argv: tuple[str, ...], **kwargs: object) -> FakeProcess:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(publisher.subprocess, "Popen", fake_popen)
    observed = publisher.read_remote_file_via_ssh(
        manifest.ssh,
        manifest.remote_root,
        entry,
        _key(tmp_path),
    )

    request = json.loads(process.stdin.data)
    assert observed == remote_files[entry.path]
    assert request == {
        "expected_size": entry.size,
        "maximum_size": publisher._MAX_FILE_BYTES,
        "path": entry.path,
        "root": manifest.remote_root,
    }
    argv_text = " ".join(captured["argv"])
    assert manifest.remote_root not in argv_text
    assert entry.path not in argv_text
    assert "HF_TOKEN" not in argv_text
    assert captured["kwargs"] == {
        "stdin": publisher.subprocess.PIPE,
        "stdout": publisher.subprocess.PIPE,
        "stderr": publisher.subprocess.PIPE,
        "bufsize": 0,
    }


def test_manifest_pin_and_release_bound_approvals_fail_closed(tmp_path: Path) -> None:
    manifest_path, _evidence_path, _remote_files = _bundle(tmp_path)
    with pytest.raises(ValueError, match="immutable pin"):
        load_remote_release_manifest(manifest_path, "0" * 64)

    manifest_pin = _manifest_pin(manifest_path)
    approval_path = tmp_path / "safety.json"
    approval = json.loads(approval_path.read_text())
    approval["release_id"] = "different-release"
    _write_json(approval_path, approval)
    with pytest.raises(ValueError, match="checksum mismatch"):
        publish_remote_official_release(
            manifest_path,
            manifest_sha256=manifest_pin,
            ssh_key=_key(tmp_path),
        )


def test_unknown_repository_is_rejected(tmp_path: Path) -> None:
    manifest_path, _evidence_path, _remote_files = _bundle(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["repo_id"] = "KaztoRay/ShadowCrafter-Other"
    with pytest.raises(ValidationError, match="repo_id"):
        RemoteReleaseManifest.model_validate(payload)


def test_cli_contract_emits_receipt_without_accepting_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_publish(manifest: Path, **kwargs: object) -> PublishResult:
        captured["manifest"] = manifest
        captured.update(kwargs)
        return PublishResult(
            repo_id="KaztoRay/ShadowCrafter-9B",
            commit_sha="c" * 40,
            release_id="exp-v1",
            manifest_sha256="d" * 64,
            evaluation_status="not-yet-evaluated",
            quality_target_met=None,
            total_bytes=123,
            file_count=3,
        )

    monkeypatch.setattr(publisher, "publish_remote_official_release", fake_publish)
    result = CliRunner().invoke(
        app,
        [
            "release",
            "publish-remote-official",
            "--manifest",
            "bundle.json",
            "--manifest-sha256",
            "d" * 64,
            "--ssh-key",
            "identity.pem",
        ],
    )

    assert result.exit_code == 0
    assert '"commit_sha": "cccccccccccccccccccccccccccccccccccccccc"' in result.stdout
    assert captured["manifest_sha256"] == "d" * 64
    assert "token" not in captured


def test_cli_withholds_external_exception_details(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_publish(_manifest: Path, **_kwargs: object) -> PublishResult:
        raise RuntimeError("credential-that-must-not-appear")

    monkeypatch.setattr(publisher, "publish_remote_official_release", fail_publish)
    result = CliRunner().invoke(
        app,
        [
            "release",
            "publish-remote-official",
            "--manifest",
            "bundle.json",
            "--manifest-sha256",
            "d" * 64,
            "--ssh-key",
            "identity.pem",
        ],
    )

    assert result.exit_code == 1
    assert "failed closed" in result.output
    assert "credential-that-must-not-appear" not in result.output
