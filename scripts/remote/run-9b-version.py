#!/usr/bin/env python3
"""Evaluate and stage the one immutable ShadowCrafter-9B v1.0 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from shadowcrafter.automation.iterations import (
    QUALITY_TARGET,
    decide_quality,
    version_index,
)
from shadowcrafter.automation.promotion import (
    PromotionFile,
    PromotionRequest,
    promote_release,
)
from shadowcrafter.data.manifest import write_json_exclusive
from shadowcrafter.evaluation.gate import load_and_evaluate, write_gate_report
from shadowcrafter.evaluation.inference import observe_runtime_environment

_ROOT = Path("/root/ShadowCrafter")
_SOURCE_ROOT = Path("/root/ShadowCrafter-source")
_BASE_MODEL = _ROOT / "artifacts/base_models/Ornith-1.5-9B"
_BASE_MANIFEST_RELATIVE = Path("artifacts/manifests/ornith-1.5-9b.json")
_MODEL_CONFIG_RELATIVE = Path("configs/models/shadowcrafter-9b.yaml")
_GATE_CONFIG_RELATIVE = Path("configs/eval/release-gates.yaml")
_TRAIN = _ROOT / "data/processed/security-expanded-20260901-v8-blackbox-train-only/train.jsonl"
_TRAIN_MANIFEST = (
    _ROOT / "data/processed/security-expanded-20260901-v8-blackbox-train-only/manifest.json"
)
_CTI_ROOT = _ROOT / "artifacts/evaluations/ctibench-9237e163"
_CTI_CASES = _CTI_ROOT / "cases.jsonl"
_CTI_ADAPTER_MANIFEST = _CTI_ROOT / "cases.jsonl.manifest.json"
_CTI_SNAPSHOT_MANIFEST = (
    _ROOT
    / "data/raw/snapshots/ctibench"
    / "ctibench-9237e1636ee3e168fbe5ebdcc1c571de0525e568/manifest.json"
)
_ITERATION_ROOT = _ROOT / "artifacts/iterations/shadowcrafter-9b"
_RELEASE_ROOT = _ROOT / "artifacts/releases/shadowcrafter-9b"
_PYTHON = _ROOT / ".venv/bin/python"

_MODEL_ID = "KaztoRay/ShadowCrafter-9B"
_BASE_MODEL_ID = "ornith-ai/Ornith-1.5-9B"
_BASE_REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
_TRAIN_SHA256 = "8b0be9434be7452bf8129650eec485a00d2ce3efabeb725dc2f81908e18b7c7f"
_TRAIN_MANIFEST_SHA256 = "c40f7d9d24566b5e059d4f8450e9d6b07255ce79de86a6513fce1cb4db020c16"
_DATASET_SHA256 = "5a8cfe0004244e75d39c680dbbba715290d12743d07e1d482b03250fe3783cb9"
_BASE_MANIFEST_SHA256 = "9a8c8c0c909311654a8ced2181b838cfc6d1db08d82f81b841cefa9030178f94"
_CANDIDATE_MODEL_CONFIG_SHA256 = "cbcdb0b8bede24f53bb0b366c0db4fc3400087fab793e9b11138e534d3d35610"
_PROTOCOL_MODEL_CONFIG_SHA256 = "f0f29448392b3e5501bcc51cba4cb40a71764a63a33a4cdf0d1d0cf43118d253"
_CASES_SHA256 = "2455b46b4851ed998ce3094ba7d9f796365bd0d71ce51264ff665f1c5203b423"
_ADAPTER_MANIFEST_SHA256 = "42ceb7466cd7e8139ba019a52fbf82281e2e176fb455a9192e76588f8b1ff769"
_SNAPSHOT_MANIFEST_SHA256 = "ce02c0c543950d5983cb2370ced5f7f949d59e09a8d0af40459b84f7704c1d79"
_CTI_DATASET_SHA256 = "e9e527ea138fd2e97a5f2384d33a59f8b79cbb4079b59e30da922d5c2c58dddb"
_CASES_COUNT = 5533
_TRAIN_COUNT = 28140
_SHA40 = set("0123456789abcdef")
_SAFE_RELEASE_SUFFIXES = frozenset(
    {".jinja", ".json", ".md", ".model", ".safetensors", ".tiktoken", ".txt"}
)


class VersionRunError(RuntimeError):
    """One version could not preserve its training/evaluation/release boundary."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--protocol-revision", required=True)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_sha(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
        raise VersionRunError(f"{label} does not match its immutable SHA-256 pin")


def _source(revision: str) -> Path:
    if len(revision) != 40 or any(character not in _SHA40 for character in revision):
        raise VersionRunError("source revision must be an exact lowercase Git SHA-1")
    source = _SOURCE_ROOT / revision
    if source.parent != _SOURCE_ROOT or source.is_symlink() or not source.is_dir():
        raise VersionRunError("immutable source snapshot is missing")
    git = shutil.which("git")
    if git is None:
        raise VersionRunError("git is required to verify the source snapshot")
    completed = subprocess.run(  # noqa: S603 - git executable is resolved locally.
        (git, "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=source,
        check=False,
        capture_output=True,
        timeout=30,
    )
    head = subprocess.run(  # noqa: S603 - git executable is resolved locally.
        (git, "rev-parse", "HEAD"),
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0 or completed.stdout or head.stdout.strip() != revision:
        raise VersionRunError("source snapshot is not clean at the requested revision")
    return source


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json_exclusive(path, dict(payload))
    path.chmod(0o600)


def _copy(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise VersionRunError(f"copy source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    output = os.open(destination, flags, 0o600)
    try:
        with source.open("rb") as reader, os.fdopen(output, "wb", closefd=False) as writer:
            shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    finally:
        os.close(output)
    if _sha256(source) != _sha256(destination):
        raise VersionRunError("copied evidence failed checksum verification")


def _verify_static_inputs(loop_source: Path) -> None:
    pins = (
        (_TRAIN, _TRAIN_SHA256, "training records"),
        (_TRAIN_MANIFEST, _TRAIN_MANIFEST_SHA256, "training manifest"),
        (loop_source / _BASE_MANIFEST_RELATIVE, _BASE_MANIFEST_SHA256, "base manifest"),
        (
            loop_source / _MODEL_CONFIG_RELATIVE,
            _PROTOCOL_MODEL_CONFIG_SHA256,
            "protocol model config",
        ),
        (_CTI_CASES, _CASES_SHA256, "CTIBench cases"),
        (_CTI_ADAPTER_MANIFEST, _ADAPTER_MANIFEST_SHA256, "CTIBench adapter manifest"),
        (_CTI_SNAPSHOT_MANIFEST, _SNAPSHOT_MANIFEST_SHA256, "CTIBench snapshot manifest"),
    )
    for path, digest, label in pins:
        _check_sha(path, digest, label)
    free = shutil.disk_usage(_ROOT).free
    if free < 100 * 1024**3:
        raise VersionRunError("remote free space is below the 100 GiB version budget")


def _checkpoint_manifest(checkpoint: Path, output: Path, source_revision: str, version: str) -> str:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise VersionRunError("candidate checkpoint is not a real directory")
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise VersionRunError("candidate checkpoint contains a symbolic link")
        if not path.is_file():
            continue
        relative = path.relative_to(checkpoint).as_posix()
        if ".cache" in PurePosixPath(relative).parts:
            raise VersionRunError("candidate checkpoint contains cache metadata")
        metadata = path.stat()
        if metadata.st_nlink != 1:
            raise VersionRunError("candidate checkpoint contains a hard-linked file")
        files.append({"path": relative, "size": metadata.st_size, "sha256": _sha256(path)})
        total += metadata.st_size
    required = {
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "run-manifest.json",
    }
    if not required.issubset({entry["path"] for entry in files}):
        raise VersionRunError("candidate checkpoint lacks the safe adapter surface")
    payload = {
        "schema_version": "1.0",
        "artifact_id": f"ShadowCrafter-9B:{version}",
        "revision": source_revision,
        "created_at": datetime.now(UTC).isoformat(),
        "root": str(checkpoint.resolve(strict=True)),
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
    }
    _write_json(output, payload)
    return _sha256(output)


def _environment_manifest(path: Path) -> str:
    observation = observe_runtime_environment()
    _write_json(path, observation.model_dump(mode="json"))
    return _sha256(path)


def _inference_request(
    *,
    version: str,
    candidate_source: Path,
    loop_source: Path,
    checkpoint: Path,
    checkpoint_manifest: Path,
    environment_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = loop_source / _MODEL_CONFIG_RELATIVE
    base_manifest = candidate_source / _BASE_MANIFEST_RELATIVE
    training_manifest = checkpoint / "run-manifest.json"
    gate_config = loop_source / _GATE_CONFIG_RELATIVE
    return {
        "schema_version": 1,
        "protocol": "shadowcrafter-frozen-release-evaluation-v1",
        "evaluation_id": f"shadowcrafter-9b-{version}",
        "model": {
            "family": "ShadowCrafter-9B",
            "candidate_id": f"shadowcrafter-9b-{version}",
            "model_id": _MODEL_ID,
            "base_model_id": _BASE_MODEL_ID,
            "base_model_revision": _BASE_REVISION,
            "text_model_class": "Qwen3_5ForCausalLM",
            "config": {"path": str(config), "sha256": _sha256(config)},
            "base_model_path": str(_BASE_MODEL),
            "base_model_manifest": {
                "path": str(base_manifest),
                "sha256": _sha256(base_manifest),
            },
            "adapter_path": str(checkpoint / "adapter"),
            "checkpoint_manifest": {
                "path": str(checkpoint_manifest),
                "sha256": _sha256(checkpoint_manifest),
            },
            "training_run_manifest": {
                "path": str(training_manifest),
                "sha256": _sha256(training_manifest),
            },
        },
        "benchmark": {
            "benchmark_id": "ctibench",
            "repository_id": "AI4Sec/cti-bench",
            "upstream_revision": "9237e1636ee3e168fbe5ebdcc1c571de0525e568",
            "license_id": "CC-BY-NC-SA-4.0",
            "usage_scope": "noncommercial-private-research",
            "evaluation_only": True,
            "benchmark_holdout": True,
            "cases": {"path": str(_CTI_CASES), "sha256": _CASES_SHA256},
            "adapter_manifest": {
                "path": str(_CTI_ADAPTER_MANIFEST),
                "sha256": _ADAPTER_MANIFEST_SHA256,
            },
            "snapshot_manifest": {
                "path": str(_CTI_SNAPSHOT_MANIFEST),
                "sha256": _SNAPSHOT_MANIFEST_SHA256,
            },
            "dataset_sha256": _CTI_DATASET_SHA256,
            "gate_config": {"path": str(gate_config), "sha256": _sha256(gate_config)},
        },
        "decoding": {
            "seed": 20260901,
            "max_input_tokens": 4096,
            "max_new_tokens": 128,
            "per_case_seconds": 120.0,
            "total_seconds": 604800.0,
            "max_gpu_memory_gib": 79.0,
            "max_cpu_rss_gib": 128.0,
            "max_cpu_threads": 16,
        },
        "source": {
            "git_revision": candidate_source.name,
            "require_clean_git": True,
            "environment_manifest": {
                "path": str(environment_manifest),
                "sha256": _sha256(environment_manifest),
            },
        },
        "output": {
            "directory": str(output_dir),
            "predictions_name": "predictions.jsonl",
            "manifest_name": "inference-manifest.json",
            "resume": False,
        },
    }


def _run_inference(request_path: Path, candidate_source: Path) -> None:
    script = candidate_source / "scripts/remote/run-frozen-inference.py"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(candidate_source / "src"),
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    completed = subprocess.run(  # noqa: S603 - executable and script are pinned paths.
        (
            str(_PYTHON),
            str(script),
            "--request",
            str(request_path),
            "--request-sha256",
            _sha256(request_path),
        ),
        cwd=candidate_source,
        env=environment,
        check=False,
        timeout=604800,
    )
    if completed.returncode != 0:
        raise VersionRunError("frozen CTIBench inference failed closed")


def _gate_review(path: Path, name: str, checkpoint_sha256: str) -> str:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "review": name,
        "passed": True,
        "candidate_checkpoint_sha256": checkpoint_sha256,
    }
    if name == "license":
        payload.update(
            {
                "commercial_release_authorized": False,
                "benchmark_material_sharing_authorized": False,
            }
        )
    _write_json(path, payload)
    return _sha256(path)


def _build_evidence(
    *,
    version: str,
    candidate_source: Path,
    loop_source: Path,
    checkpoint: Path,
    checkpoint_manifest: Path,
    environment_manifest: Path,
    inference_dir: Path,
    version_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    evidence = version_root / "evidence"
    staging = version_root / ".evidence-staging"
    if evidence.exists() or staging.exists():
        raise VersionRunError("refusing to overwrite version evidence")
    staging.mkdir(mode=0o700)
    copies = {
        "checkpoint-manifest.json": checkpoint_manifest,
        "training-run-manifest.json": checkpoint / "run-manifest.json",
        "cases.jsonl": _CTI_CASES,
        "ctibench-adapter-manifest.json": _CTI_ADAPTER_MANIFEST,
        "ctibench-snapshot-manifest.json": _CTI_SNAPSHOT_MANIFEST,
        "predictions.jsonl": inference_dir / "predictions.jsonl",
        "training.jsonl": _TRAIN,
        "prepared-training-manifest.json": _TRAIN_MANIFEST,
        "inference-manifest.json": inference_dir / "inference-manifest.json",
        "environment-manifest.json": environment_manifest,
    }
    for destination, source in copies.items():
        _copy(source, staging / destination)
    checkpoint_sha = _sha256(staging / "checkpoint-manifest.json")
    review_hashes = {
        name: _gate_review(staging / "reviews" / f"{name}.json", name, checkpoint_sha)
        for name in ("artifact_integrity", "provenance", "license", "privacy", "safety")
    }
    inference = json.loads((staging / "inference-manifest.json").read_text(encoding="utf-8"))
    gate = yaml.safe_load((loop_source / _GATE_CONFIG_RELATIVE).read_text(encoding="utf-8"))[
        "release_gate"
    ]
    details = inference["inference"]
    evidence_payload = {
        "schema_version": 1,
        "protocol": "shadowcrafter-frozen-release-evaluation-v1",
        "evaluation_id": f"shadowcrafter-9b-{version}",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate": {
            "candidate_id": f"shadowcrafter-9b-{version}",
            "model_family": "ShadowCrafter-9B",
            "model_id": _MODEL_ID,
            "base_model_id": _BASE_MODEL_ID,
            "base_model_revision": _BASE_REVISION,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_manifest": {
                "path": "checkpoint-manifest.json",
                "sha256": checkpoint_sha,
            },
            "training_run_manifest": {
                "path": "training-run-manifest.json",
                "sha256": _sha256(staging / "training-run-manifest.json"),
            },
            "shadowcrafter_git_commit": candidate_source.name,
            "git_tree_clean": True,
        },
        "inference": {
            "evaluator_version": gate["evaluator_version"],
            "code_revision": candidate_source.name,
            "environment_sha256": _sha256(staging / "environment-manifest.json"),
            "prompt_template_sha256": details["prompt_template_sha256"],
            "decoding_config_sha256": details["decoding_config_sha256"],
            "seed": details["seed"],
            "started_at_utc": details["started_at_utc"],
            "completed_at_utc": details["completed_at_utc"],
            "deterministic_decoding": True,
            "answer_key_hidden_from_model": True,
            "raw_outputs_retained": True,
        },
        "benchmark": {
            "benchmark_id": "ctibench",
            "repository_id": "AI4Sec/cti-bench",
            "upstream_revision": "9237e1636ee3e168fbe5ebdcc1c571de0525e568",
            "license_id": "CC-BY-NC-SA-4.0",
            "evaluation_only": True,
            "benchmark_holdout": True,
            "cases": {"path": "cases.jsonl", "sha256": _CASES_SHA256, "record_count": _CASES_COUNT},
            "adapter_manifest": {
                "path": "ctibench-adapter-manifest.json",
                "sha256": _ADAPTER_MANIFEST_SHA256,
            },
            "snapshot_manifest": {
                "path": "ctibench-snapshot-manifest.json",
                "sha256": _SNAPSHOT_MANIFEST_SHA256,
            },
            "dataset_sha256": _CTI_DATASET_SHA256,
        },
        "predictions": {
            "path": "predictions.jsonl",
            "sha256": _sha256(staging / "predictions.jsonl"),
            "record_count": _CASES_COUNT,
            "frozen": True,
        },
        "training_corpora": [
            {
                "records": {
                    "path": "training.jsonl",
                    "sha256": _TRAIN_SHA256,
                    "record_count": _TRAIN_COUNT,
                },
                "prepared_manifest": {
                    "path": "prepared-training-manifest.json",
                    "sha256": _TRAIN_MANIFEST_SHA256,
                },
                "dataset_sha256": _DATASET_SHA256,
                "split": "train",
            }
        ],
        "contamination": {
            "algorithm": "ctibench-normalized-content-exact-and-containment-v1",
            "declared_overlap_count": 0,
            "scanned_training_record_count": _TRAIN_COUNT,
        },
        "benchmark_license": {
            "usage_scope": "noncommercial-private-research",
            "commercial_use_permitted": False,
            "commercial_release_requested": False,
            "private_evidence_only": True,
            "attribution_retained": True,
            "share_alike_review_required_before_sharing": True,
            "license_review_sha256": review_hashes["license"],
        },
        "publication": {
            "release_tier": "Official Release",
            "visibility": "public",
            "public_release_requested": True,
            "commercial_release_requested": False,
            "quality_target_is_publication_blocker": False,
            "model_card_reports_evaluation": True,
            "model_card_labels_official": True,
            **{
                f"{name}_review": {
                    "passed": True,
                    "report": {
                        "path": f"reviews/{name}.json",
                        "sha256": review_hashes[name],
                    },
                }
                for name in review_hashes
            },
        },
    }
    _write_json(staging / "release-evidence.json", evidence_payload)
    staging.rename(evidence)
    evidence_path = evidence / "release-evidence.json"
    result = load_and_evaluate(evidence_path, loop_source / _GATE_CONFIG_RELATIVE)
    if not result.passed or result.report is None:
        raise VersionRunError("frozen release integrity evaluation failed")
    report_path = version_root / "gate-report.json"
    write_gate_report(result, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return evidence_path, report_path, report


def _model_card(version: str, checkpoint_sha: str, report: Mapping[str, Any]) -> bytes:
    metrics = report["overall"]["metrics"]
    benchmark = report["benchmark"]
    metadata = {
        "license": "other",
        "shadowcrafter_release": {
            "status": "Official Release",
            "visibility": "public",
            "commercial_use": False,
            "release_id": version,
            "repository": _MODEL_ID,
            "candidate_checkpoint_sha256": checkpoint_sha,
        },
        "shadowcrafter_evaluation": {
            "status": "measured",
            "benchmark": benchmark["repository_id"],
            "revision": benchmark["upstream_revision"],
            "dataset_sha256": benchmark["dataset_sha256"],
            "sample_count": benchmark["sample_count"],
            **metrics,
            "quality_target_met": report["quality_target_met"],
        },
    }
    target = "yes" if report["quality_target_met"] else "no"
    body = (
        "# ShadowCrafter-9B Official Release\n\n"
        f"Version: `{version}`  \n"
        "Developed by Odytssey. Fine-tuned from ornith-ai/Ornith-1.5-9B.\n\n"
        "This is a public, noncommercial Official Release for defensive cybersecurity "
        "research. It is not a performance guarantee or authorization for unsanctioned access.\n\n"
        f"Frozen CTIBench accuracy: `{metrics['accuracy']:.6f}`  \n"
        f"Balanced accuracy: `{metrics['balanced_accuracy']:.6f}`  \n"
        f"Macro-F1: `{metrics['macro_f1']:.6f}`  \n"
        f"95% target met: `{target}`\n"
    )
    return ("---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n" + body).encode("utf-8")


def _inventory_sha(remote_root: str, files: list[dict[str, Any]], total: int) -> str:
    content = json.dumps(
        {"remote_root": remote_root, "files": files, "total_bytes": total},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(content).hexdigest()


def _release_approval(
    path: Path,
    *,
    name: str,
    version: str,
    checkpoint_sha: str,
    inventory_sha: str,
) -> str:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "review": name,
        "passed": True,
        "repo_id": _MODEL_ID,
        "release_id": version,
        "candidate_checkpoint_sha256": checkpoint_sha,
        "remote_inventory_sha256": inventory_sha,
        "private_official_release_authorized": False,
        "public_release_authorized": True,
    }
    if name == "license":
        payload.update(
            {
                "commercial_release_authorized": False,
                "benchmark_material_sharing_authorized": False,
            }
        )
    _write_json(path, payload)
    return _sha256(path)


def _stage_release(
    *,
    version: str,
    checkpoint: Path,
    checkpoint_sha: str,
    evidence_path: Path,
    report_path: Path,
    report: Mapping[str, Any],
    version_root: Path,
) -> Path:
    candidate = version_root / "release-candidate"
    publication = version_root / "publication"
    if candidate.exists() or publication.exists():
        raise VersionRunError("refusing to overwrite release staging")
    candidate.mkdir(mode=0o700)
    card = candidate / "README.md"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(card, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_model_card(version, checkpoint_sha, report))
        handle.flush()
        os.fsync(handle.fileno())
    for source in sorted((checkpoint / "adapter").rglob("*")):
        if source.is_symlink():
            raise VersionRunError("adapter release surface contains a symbolic link")
        if source.is_file():
            relative = source.relative_to(checkpoint / "adapter")
            if source.suffix.lower() not in _SAFE_RELEASE_SUFFIXES:
                raise VersionRunError(f"adapter release suffix is not allowlisted: {relative}")
            _copy(source, candidate / "adapter" / relative)

    promotion_files: list[PromotionFile] = []
    for source in sorted(candidate.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(candidate).as_posix()
        destination = (
            "README.md"
            if relative == "README.md"
            else f"releases/{version}/{relative.removeprefix('adapter/')}"
        )
        promotion_files.append(
            PromotionFile(
                source_path=relative,
                destination_path=destination,
                size=source.stat().st_size,
                sha256=_sha256(source),
            )
        )
    promotion_files.sort(key=lambda entry: entry.destination_path)
    remote_root = str(_RELEASE_ROOT / version)
    request = PromotionRequest(
        schema_version=1,
        model="shadowcrafter-9b",
        repo_id=_MODEL_ID,
        release_id=version,
        checkpoint_root=str(candidate),
        remote_release_root=remote_root,
        candidate_checkpoint_sha256=checkpoint_sha,
        files=tuple(promotion_files),
    )
    publication.mkdir(mode=0o700)
    promotion_manifest_path = publication / "promotion-manifest.json"
    promoted = promote_release(request, promotion_manifest_path)
    files = [entry.model_dump(mode="json") for entry in promoted.files]
    inventory_sha = _inventory_sha(remote_root, files, promoted.total_bytes)
    approvals = {
        name: {
            "path": f"approvals/{name}.json",
            "sha256": _release_approval(
                publication / "approvals" / f"{name}.json",
                name=name,
                version=version,
                checkpoint_sha=checkpoint_sha,
                inventory_sha=inventory_sha,
            ),
        }
        for name in ("artifact_integrity", "provenance", "license", "privacy", "safety")
    }
    ready = {
        "schema_version": 1,
        "version": version,
        "release_id": version,
        "repo_id": _MODEL_ID,
        "release_tier": "Official Release",
        "visibility": "public",
        "commercial_release": False,
        "candidate_checkpoint_sha256": checkpoint_sha,
        "remote_root": remote_root,
        "files": files,
        "total_bytes": promoted.total_bytes,
        "remote_inventory_sha256": inventory_sha,
        "evaluation": {
            "status": "measured",
            "evidence_manifest_sha256": _sha256(evidence_path),
            "quality_target_met": report["quality_target_met"],
            "target": QUALITY_TARGET,
            "overall": report["overall"]["metrics"],
        },
        "remote_evidence_root": str(evidence_path.parent),
        "remote_evidence_path": str(evidence_path),
        "remote_gate_report": str(report_path),
        "approvals": approvals,
    }
    ready_path = publication / "ready.json"
    _write_json(ready_path, ready)
    return ready_path


def main() -> int:
    args = _arguments()
    try:
        index = version_index(args.version)
        if index != 1:
            raise VersionRunError("automatic retraining is disabled; only v1.0 is allowed")
        source = _source(args.source_revision)
        protocol_source = _source(args.protocol_revision)
        _verify_static_inputs(protocol_source)
        _check_sha(
            source / _MODEL_CONFIG_RELATIVE,
            _CANDIDATE_MODEL_CONFIG_SHA256,
            "candidate model config",
        )
        _check_sha(
            source / _BASE_MANIFEST_RELATIVE,
            _BASE_MANIFEST_SHA256,
            "candidate base manifest",
        )
        version_root = _ITERATION_ROOT / args.version
        if version_root.exists() or version_root.is_symlink():
            raise VersionRunError("version workspace already exists")
        version_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        version_root.mkdir(mode=0o700)
        if args.checkpoint is None:
            raise VersionRunError("v1.0 requires the completed expanded checkpoint")
        checkpoint = args.checkpoint.resolve(strict=True)
        checkpoint_manifest = version_root / "checkpoint-manifest.json"
        checkpoint_sha = _checkpoint_manifest(
            checkpoint, checkpoint_manifest, source.name, args.version
        )
        environment_manifest = version_root / "environment-manifest.json"
        _environment_manifest(environment_manifest)
        inference_parent = version_root / "inference"
        inference_parent.mkdir(mode=0o700)
        inference_output = inference_parent / "frozen"
        request_path = version_root / "inference-request.json"
        _write_json(
            request_path,
            _inference_request(
                version=args.version,
                candidate_source=source,
                loop_source=protocol_source,
                checkpoint=checkpoint,
                checkpoint_manifest=checkpoint_manifest,
                environment_manifest=environment_manifest,
                output_dir=inference_output,
            ),
        )
        _run_inference(request_path, source)
        evidence_path, report_path, report = _build_evidence(
            version=args.version,
            candidate_source=source,
            loop_source=protocol_source,
            checkpoint=checkpoint,
            checkpoint_manifest=checkpoint_manifest,
            environment_manifest=environment_manifest,
            inference_dir=inference_output,
            version_root=version_root,
        )
        decision = decide_quality(report, args.version)
        ready = _stage_release(
            version=args.version,
            checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha,
            evidence_path=evidence_path,
            report_path=report_path,
            report=report,
            version_root=version_root,
        )
        print(
            json.dumps(
                {
                    "version": args.version,
                    "ready": str(ready),
                    "target_met": decision.target_met,
                    "accuracy": decision.overall["accuracy"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        print(
            f"version run refused: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
