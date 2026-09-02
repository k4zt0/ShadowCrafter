from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shadowcrafter.cli import app
from shadowcrafter.data.manifest import sha256_file
from shadowcrafter.release.colab_export import ColabExportError, import_colab_export


def _candidate_archive(root: Path, *, valid_hashes: bool = True) -> Path:
    candidate = root / "v2.0-test"
    adapter = candidate / "adapter"
    adapter.mkdir(parents=True)
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    config.write_text('{"peft_type":"LORA"}\n')
    weights.write_bytes(b"SAFE-FAKE-LORA")
    manifest = {
        "schema_version": 2,
        "version": "v2.0-colab-candidate",
        "base_model": {
            "id": "ornith-ai/Ornith-1.5-9B",
            "revision": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
        },
        "environment": {"git_revision": "a" * 40},
        "effective_training_invariants": {"checkpoint_storage": "ephemeral"},
        "adapter": {
            "lora_only": True,
            "finite": True,
            "safe_serialization": True,
            "adapter_config_sha256": sha256_file(config),
            "adapter_weights_sha256": sha256_file(weights) if valid_hashes else "b" * 64,
        },
    }
    (candidate / "run-manifest.json").write_text(json.dumps(manifest) + "\n")
    archive = root / "ShadowCrafter-V2-candidate-test.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(candidate, arcname=candidate.name)
    return archive


def test_import_colab_candidate_verifies_and_is_idempotent(tmp_path: Path) -> None:
    archive = _candidate_archive(tmp_path)
    destination = tmp_path / "mirror"

    first = import_colab_export(archive, destination)
    second = import_colab_export(archive, destination)

    assert first == second
    assert first.kind == "candidate"
    assert first.source_revision == "a" * 40
    assert first.adapter_sha256 == sha256_file(
        first.root / "adapter/adapter_model.safetensors"
    )
    receipt = json.loads(first.receipt.read_text())
    assert receipt["archive_sha256"] == sha256_file(archive)
    assert not any(path.is_symlink() for path in first.root.rglob("*"))


def test_import_colab_candidate_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    archive = _candidate_archive(tmp_path, valid_hashes=False)

    with pytest.raises(ColabExportError, match="differ"):
        import_colab_export(archive, tmp_path / "mirror")


@pytest.mark.parametrize(
    ("name", "member_type", "linkname"),
    [
        ("../../escaped", tarfile.REGTYPE, ""),
        ("v2.0-test/adapter/link", tarfile.SYMTYPE, "target"),
    ],
)
def test_import_colab_export_rejects_unsafe_tar_members(
    tmp_path: Path,
    name: str,
    member_type: bytes,
    linkname: str,
) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo(name)
        member.type = member_type
        member.linkname = linkname
        if member_type == tarfile.REGTYPE:
            member.size = 1
            handle.addfile(member, io.BytesIO(b"x"))
        else:
            handle.addfile(member)

    with pytest.raises(ColabExportError):
        import_colab_export(archive, tmp_path / "mirror")


def test_import_colab_export_cli(tmp_path: Path) -> None:
    archive = _candidate_archive(tmp_path)
    destination = tmp_path / "mirror"

    result = CliRunner().invoke(
        app,
        [
            "release",
            "import-colab-export",
            "--archive",
            str(archive),
            "--destination-root",
            str(destination),
        ],
    )

    assert result.exit_code == 0
    assert '"kind": "candidate"' in result.stdout
