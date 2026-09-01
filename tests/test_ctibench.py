import base64
import csv
import hashlib
import io
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

import shadowcrafter.data.ctibench as ctibench
import shadowcrafter.data.hf_snapshot as hf_snapshot
from shadowcrafter.data.ctibench import (
    CTIBenchTask,
    assert_no_ctibench_training_contamination,
    canonicalize_ctibench_evaluation,
    load_ctibench_eval_cases,
    render_ctibench_input,
)
from shadowcrafter.data.hf_snapshot import snapshot_ctibench
from shadowcrafter.data.manifest import canonical_json_sha256, sha256_file
from shadowcrafter.data.prepare import prepare_jsonl
from shadowcrafter.data.registry import load_registry
from shadowcrafter.schemas import Message, Provenance, SecurityRecord, TaskType

REGISTRY = Path("configs/data/sources.yaml")
RETRIEVED_AT = datetime(2026, 9, 1, tzinfo=UTC)
UPSTREAM_PROMPT = "UPSTREAM-DO-NOT-PRESERVE reverse shell instructions"


def _tsv(headers: tuple[str, ...], rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _snapshot_files(*, taa_prompt: str = "Unscored upstream prompt") -> dict[str, str]:
    return {
        ".gitattributes": "*.zip binary\n",
        "README.md": "---\nlicense: cc-by-nc-sa-4.0\n---\n# CTIBench\n",
        "cti-ate.tsv": _tsv(
            ctibench._CONFIGS["cti-ate.tsv"].headers,
            [
                {
                    "URL": "https://attack.mitre.org/software/S0001/",
                    "Platform": "Enterprise",
                    "Description": "The report describes HTTP command-and-control behavior.",
                    "Prompt": UPSTREAM_PROMPT,
                    "GT": "T1071, T1083",
                }
            ],
        ),
        "cti-mcq.tsv": _tsv(
            ctibench._CONFIGS["cti-mcq.tsv"].headers,
            [
                {
                    "URL": "Manual",
                    "Question": "Which control prevents unapproved software execution?",
                    "Option A": "Audit only",
                    "Option B": "Application allowlisting",
                    "Option C": "Password rotation",
                    "Option D": "Network segmentation",
                    "Prompt": UPSTREAM_PROMPT,
                    "GT": "b",
                }
            ],
        ),
        "cti-rcm-2021.tsv": _tsv(
            ctibench._CONFIGS["cti-rcm-2021.tsv"].headers,
            [
                {
                    "URL": "https://nvd.nist.gov/vuln/detail/CVE-2021-0001",
                    "Description": "A resource is used after it has been freed.",
                    "Prompt": UPSTREAM_PROMPT,
                    "GT": "CWE-416",
                }
            ],
        ),
        "cti-rcm.tsv": _tsv(
            ctibench._CONFIGS["cti-rcm.tsv"].headers,
            [
                {
                    "URL": "https://nvd.nist.gov/vuln/detail/CVE-2024-0001",
                    "Description": "The application does not neutralize SQL query elements.",
                    "Prompt": UPSTREAM_PROMPT,
                    "GT": "CWE-89",
                }
            ],
        ),
        "cti-taa.tsv": _tsv(
            ctibench._CONFIGS["cti-taa.tsv"].headers,
            [
                {
                    "URL": "https://example.org/report",
                    "Text": "Unscored threat report text must not enter runner cases.",
                    "Prompt": taa_prompt,
                }
            ],
        ),
        "cti-vsp.tsv": _tsv(
            ctibench._CONFIGS["cti-vsp.tsv"].headers,
            [
                {
                    "URL": "https://nvd.nist.gov/vuln/detail/CVE-2024-0001",
                    "Description": "A remotely reachable flaw affects confidentiality.",
                    "Prompt": UPSTREAM_PROMPT,
                    "GT": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
                }
            ],
        ),
    }


def _hf_card() -> dict[str, object]:
    return {
        "license": "cc-by-nc-sa-4.0",
        "configs": [
            {
                "config_name": config_name,
                "data_files": [{"split": "test", "path": source_file}],
                "sep": "\t",
            }
            for config_name, source_file in sorted(
                {
                    "cti-ate": "cti-ate.tsv",
                    "cti-mcq": "cti-mcq.tsv",
                    "cti-rcm": "cti-rcm.tsv",
                    "cti-rcm-2021": "cti-rcm-2021.tsv",
                    "cti-taa": "cti-taa.tsv",
                    "cti-vsp": "cti-vsp.tsv",
                }.items()
            )
        ],
    }


def _write_snapshot(
    root: Path,
    *,
    taa_prompt: str = "Unscored upstream prompt",
    revision: str = ctibench.CTIBENCH_REVIEWED_REVISION,
) -> Path:
    snapshot_dir = root / "snapshot"
    snapshot_dir.mkdir()
    files = _snapshot_files(taa_prompt=taa_prompt)
    inventory = []
    for relative_path, content in files.items():
        path = snapshot_dir / relative_path
        path.write_text(content)
        inventory.append(
            {
                "path": relative_path,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "git_blob_id": hashlib.sha1(  # noqa: S324 - test fixture Git object ID.
                    f"blob {path.stat().st_size}\0".encode() + path.read_bytes(),
                    usedforsecurity=False,
                ).hexdigest(),
                "lfs": None,
            }
        )
    inventory.sort(key=lambda item: item["path"])
    manifest = {
        "schema_version": 2,
        "snapshot_id": f"ctibench-{revision}",
        "source": {
            "allowed_purposes": ["evaluate"],
            "id": "ctibench",
            "provider": "AI4Sec",
            "repo_id": "AI4Sec/cti-bench",
            "type": "huggingface_dataset",
            "policy_class": "eval_only",
            "license": {
                "attribution": "CTIBench authors and cited source datasets",
                "id": "CC-BY-NC-SA-4.0",
                "redistribution": "terms_limited",
                "status": "verified",
                "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode",
            },
        },
        "registry_sha256": load_registry(REGISTRY).canonical_sha256(),
        "upstream_revision": revision,
        "retrieved_at_utc": RETRIEVED_AT.isoformat(),
        "files": inventory,
        "snapshot_sha256": canonical_json_sha256(
            [
                {"path": item["path"], "sha256": item["sha256"], "size": item["size"]}
                for item in inventory
            ]
        ),
        "snapshot_sha256_algorithm": "sha256-canonical-json-v1(path,size,sha256)",
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return snapshot_dir


@pytest.fixture(autouse=True)
def _small_reviewed_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ctibench,
        "_CONFIGS",
        {path: replace(spec, expected_rows=1) for path, spec in ctibench._CONFIGS.items()},
    )


def _training_record(
    record_id: str, content: str, *, source_id: str = "mitre-cwe"
) -> SecurityRecord:
    record = SecurityRecord(
        record_id=record_id,
        task=TaskType.CWE_MAPPING,
        messages=[
            Message(role="user", content=content),
            Message(role="assistant", content="Return a defensive classification."),
        ],
        provenance=Provenance(
            source_id=source_id,
            license="MITRE-CWE-Terms-of-Use",
            retrieved_at=RETRIEVED_AT,
            upstream_revision="test-revision",
            record_key=record_id,
            content_sha256="0" * 64,
        ),
        split_group=record_id,
    )
    record.provenance.content_sha256 = record.canonical_hash()
    return record


def test_ctibench_adapter_emits_only_structured_eval_cases(tmp_path: Path) -> None:
    snapshot_dir = _write_snapshot(tmp_path)
    output_path = tmp_path / "ctibench.eval.jsonl"

    manifest = canonicalize_ctibench_evaluation(snapshot_dir, output_path, registry_path=REGISTRY)
    cases = load_ctibench_eval_cases(output_path)

    assert len(cases) == 5
    assert manifest["statistics"] == {
        "candidate_count": 6,
        "emitted_count": 5,
        "missing_ground_truth_skipped": 1,
        "unsafe_or_invalid_skipped": 0,
        "by_task": {
            "cti-ate": 1,
            "cti-mcq": 1,
            "cti-rcm": 1,
            "cti-rcm-2021": 1,
            "cti-vsp": 1,
        },
    }
    mcq = next(case for case in cases if case.task == CTIBenchTask.MULTIPLE_CHOICE)
    assert mcq.choices == {
        "A": "Audit only",
        "B": "Application allowlisting",
        "C": "Password rotation",
        "D": "Network segmentation",
    }
    assert mcq.answer == "B"
    assert mcq.provenance.source_reference == "Manual"
    assert render_ctibench_input(mcq).endswith("D. Network segmentation")
    assert all(case.eval_only and case.benchmark_holdout for case in cases)
    assert all(not case.prompt_training_eligible for case in cases)
    assert all(case.content_sha256 == case.canonical_hash() for case in cases)

    serialized = output_path.read_text()
    assert UPSTREAM_PROMPT not in serialized
    assert '"Prompt"' not in serialized
    assert '"messages"' not in serialized
    assert manifest["controls"]["upstream_prompt_preserved"] is False
    assert manifest["controls"]["commercial_use_permitted"] is False
    assert manifest["license"] == {
        "id": "CC-BY-NC-SA-4.0",
        "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode",
        "attribution": "CTIBench authors and cited source datasets",
        "noncommercial_only": True,
        "attribution_required_when_shared": True,
        "share_alike_required_when_adapted_material_is_shared": True,
        "no_additional_downstream_restrictions": True,
    }


def test_ctibench_eval_schema_cannot_enter_security_record_preparation(tmp_path: Path) -> None:
    snapshot_dir = _write_snapshot(tmp_path)
    output_path = tmp_path / "ctibench.eval.jsonl"
    canonicalize_ctibench_evaluation(snapshot_dir, output_path, registry_path=REGISTRY)

    payload = json.loads(output_path.read_text().splitlines()[0])
    with pytest.raises(ValidationError):
        SecurityRecord.model_validate(payload)
    with pytest.raises(ValueError, match="invalid record"):
        prepare_jsonl(output_path, tmp_path / "prepared", registry_path=REGISTRY)
    assert not (tmp_path / "prepared").exists()


def test_ctibench_contamination_scan_uses_hash_without_retaining_prompt(tmp_path: Path) -> None:
    snapshot_dir = _write_snapshot(tmp_path)
    output_path = tmp_path / "ctibench.eval.jsonl"
    canonicalize_ctibench_evaluation(snapshot_dir, output_path, registry_path=REGISTRY)
    cases = load_ctibench_eval_cases(output_path)

    assert_no_ctibench_training_contamination(
        [_training_record("clean", "A separately authored defensive question")], cases
    )
    with pytest.raises(ValueError, match="benchmark-content"):
        assert_no_ctibench_training_contamination(
            [_training_record("leaked", UPSTREAM_PROMPT)], cases
        )
    with pytest.raises(ValueError, match="ctibench-provenance"):
        assert_no_ctibench_training_contamination(
            [_training_record("wrong-source", "Unrelated", source_id="ctibench")], cases
        )
    rendered = render_ctibench_input(
        next(case for case in cases if case.task == CTIBenchTask.MULTIPLE_CHOICE)
    )
    with pytest.raises(ValueError, match="benchmark-content"):
        assert_no_ctibench_training_contamination(
            [_training_record("embedded", f"Trusted wrapper:\n{rendered}\nReturn JSON.")], cases
        )


def test_ctibench_adapter_rejects_binary_in_unscored_source_field(tmp_path: Path) -> None:
    encoded = base64.b64encode(b"MZ" + (b"\x00" * 2_048)).decode()
    snapshot_dir = _write_snapshot(tmp_path, taa_prompt=encoded)

    with pytest.raises(ValueError, match="raw executable or archive payload"):
        canonicalize_ctibench_evaluation(
            snapshot_dir, tmp_path / "ctibench.eval.jsonl", registry_path=REGISTRY
        )


def test_ctibench_adapter_rejects_unreviewed_revision(tmp_path: Path) -> None:
    snapshot_dir = _write_snapshot(tmp_path, revision="a" * 40)
    with pytest.raises(ValueError, match="has not completed repository review"):
        canonicalize_ctibench_evaluation(
            snapshot_dir, tmp_path / "ctibench.eval.jsonl", registry_path=REGISTRY
        )


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_ctibench_adapter_rejects_linked_snapshot_files(tmp_path: Path, link_kind: str) -> None:
    snapshot_dir = _write_snapshot(tmp_path)
    victim = snapshot_dir / "cti-ate.tsv"
    outside = tmp_path / "outside.tsv"
    shutil.copyfile(victim, outside)
    victim.unlink()
    if link_kind == "symlink":
        victim.symlink_to(outside)
    else:
        victim.hardlink_to(outside)

    with pytest.raises(ValueError, match="non-linked regular file"):
        canonicalize_ctibench_evaluation(
            snapshot_dir, tmp_path / "ctibench.eval.jsonl", registry_path=REGISTRY
        )


def test_ctibench_adapter_rejects_manifest_inventory_tampering(tmp_path: Path) -> None:
    snapshot_dir = _write_snapshot(tmp_path)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["snapshot_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="inventory checksum mismatch"):
        canonicalize_ctibench_evaluation(
            snapshot_dir, tmp_path / "ctibench.eval.jsonl", registry_path=REGISTRY
        )


def test_ctibench_download_reader_rejects_paths_outside_private_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    outside = tmp_path / "outside.tsv"
    outside.write_bytes(b"reviewed")

    with pytest.raises(ValueError, match="outside its private cache"):
        hf_snapshot._read_bounded_file(
            outside,
            expected_size=len(b"reviewed"),
            max_bytes=1024,
            cache_dir=cache_dir,
        )


def test_ctibench_tsv_parser_preserves_literal_unescaped_quotes() -> None:
    spec = ctibench._CONFIGS["cti-rcm-2021.tsv"]
    description = '"Clear History and Website Data" did not clear the history.'
    text = "\t".join(spec.headers) + "\n"
    text += "\t".join(
        (
            "https://nvd.nist.gov/vuln/detail/CVE-2019-8768",
            description,
            f"Classify this description: {description}",
            "CWE-459",
        )
    )
    text += "\n"

    rows = ctibench._read_tsv(text, "cti-rcm-2021.tsv", spec)
    assert rows[0]["Description"] == description


def test_ctibench_registry_records_eval_and_license_restrictions() -> None:
    source = load_registry(REGISTRY).source("ctibench")
    assert source.policy_class == "eval_only"
    assert source.allowed_purposes == {"evaluate"}
    assert source.license.id == "CC-BY-NC-SA-4.0"
    assert {
        "evaluation-only",
        "never-in-prompts-or-training",
        "contamination-scan",
        "drop-upstream-prompt",
        "answer-key-isolation",
        "noncommercial-use-only",
    } <= set(source.safety.required_filters)
    assert source.notes is not None and "noncommercial evaluation only" in source.notes


def test_ctibench_hf_snapshot_is_pinned_bounded_and_credential_free(tmp_path: Path) -> None:
    source_dir = tmp_path / "hub-files"
    source_dir.mkdir()
    files = _snapshot_files()
    files["README.md"] = "---\n" + yaml.safe_dump(_hf_card(), sort_keys=False) + "---\n# CTIBench\n"
    siblings = []
    for relative_path, content in files.items():
        path = source_dir / relative_path
        path.write_text(content)
        blob_id = hashlib.sha1(  # noqa: S324 - Git object identity, not security hashing.
            f"blob {path.stat().st_size}\0".encode() + path.read_bytes(),
            usedforsecurity=False,
        ).hexdigest()
        siblings.append(
            SimpleNamespace(
                rfilename=relative_path,
                size=path.stat().st_size,
                blob_id=blob_id,
                lfs=None,
            )
        )
    info = SimpleNamespace(
        id="AI4Sec/cti-bench",
        sha=ctibench.CTIBENCH_REVIEWED_REVISION,
        private=False,
        gated=False,
        disabled=False,
        card_data=_hf_card(),
        siblings=siblings,
    )

    class FakeApi:
        def dataset_info(self, *_args: object, **kwargs: object) -> object:
            assert kwargs == {
                "revision": ctibench.CTIBENCH_REVIEWED_REVISION,
                "files_metadata": True,
                "token": False,
            }
            return info

    downloads: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> Path:
        downloads.append(kwargs)
        assert kwargs["repo_id"] == "AI4Sec/cti-bench"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == ctibench.CTIBENCH_REVIEWED_REVISION
        assert kwargs["token"] is False
        cache_path = Path(str(kwargs["cache_dir"])) / str(kwargs["filename"])
        shutil.copyfile(source_dir / str(kwargs["filename"]), cache_path)
        return cache_path

    output_dir = tmp_path / "raw" / "snapshots"
    manifest = snapshot_ctibench(
        REGISTRY,
        output_dir,
        api=FakeApi(),
        downloader=fake_download,
        retrieved_at=RETRIEVED_AT,
    )
    target = output_dir / "ctibench" / f"ctibench-{ctibench.CTIBENCH_REVIEWED_REVISION}"

    assert target.is_dir()
    assert len(downloads) == 8
    assert {str(call["filename"]) for call in downloads} == set(files)
    assert {entry["path"] for entry in manifest["files"]} == set(files)
    assert manifest["source"]["policy_class"] == "eval_only"
    assert manifest["source"]["license"]["id"] == "CC-BY-NC-SA-4.0"
    assert manifest["upstream_revision"] == ctibench.CTIBENCH_REVIEWED_REVISION
    assert (
        json.loads((target / "manifest.json").read_text())["snapshot_sha256"]
        == (manifest["snapshot_sha256"])
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        snapshot_ctibench(
            REGISTRY,
            output_dir,
            api=FakeApi(),
            downloader=fake_download,
            retrieved_at=RETRIEVED_AT,
        )
