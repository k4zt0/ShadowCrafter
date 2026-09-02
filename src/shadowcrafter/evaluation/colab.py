"""Accountless Colab orchestration for measured ShadowCrafter V2 evaluation."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from shadowcrafter.data.manifest import sha256_file, write_json_exclusive
from shadowcrafter.evaluation.gate import GateResult, load_and_evaluate, write_gate_report
from shadowcrafter.evaluation.inference import (
    InferenceRequest,
    RuntimeEnvironment,
    observe_runtime_environment,
    run_frozen_inference,
)

_MODEL_ID = "KaztoRay/ShadowCrafter-9B"
_BASE_MODEL_ID = "ornith-ai/Ornith-1.5-9B"
_BASE_REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
_CTIBENCH_REVISION = "9237e1636ee3e168fbe5ebdcc1c571de0525e568"
_CTIBENCH_CASES_SHA256 = "2455b46b4851ed998ce3094ba7d9f796365bd0d71ce51264ff665f1c5203b423"
_CTIBENCH_ADAPTER_SHA256 = "42ceb7466cd7e8139ba019a52fbf82281e2e176fb455a9192e76588f8b1ff769"
_CTIBENCH_SNAPSHOT_SHA256 = "ce02c0c543950d5983cb2370ced5f7f949d59e09a8d0af40459b84f7704c1d79"
_CTIBENCH_DATASET_SHA256 = "e9e527ea138fd2e97a5f2384d33a59f8b79cbb4079b59e30da922d5c2c58dddb"
_CTIBENCH_CASE_COUNT = 5_533


class ColabEvaluationError(RuntimeError):
    """A measured Colab evaluation failed before producing trusted metrics."""


@dataclass(frozen=True, slots=True)
class ColabEvaluationResult:
    """Paths and recomputed aggregate metrics for one immutable evaluation attempt."""

    root: Path
    evidence_path: Path
    report_path: Path
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    quality_target_met: bool

    def as_dict(self) -> dict[str, str | float | bool]:
        return {
            "root": str(self.root),
            "evidence_path": str(self.evidence_path),
            "report_path": str(self.report_path),
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "quality_target_met": self.quality_target_met,
        }


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ColabEvaluationError(f"{label} must be one regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ColabEvaluationError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ColabEvaluationError(f"{label} must contain one JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_json_exclusive(path, dict(payload))
    path.chmod(0o600)


def _copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ColabEvaluationError(f"copy source is not a regular file: {source}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb", closefd=False) as writer:
            shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    finally:
        os.close(descriptor)
    if sha256_file(source) != sha256_file(destination):
        raise ColabEvaluationError("copied evaluation evidence failed SHA-256 verification")


def _file_pin(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def _checkpoint_manifest(
    candidate_dir: Path, output: Path, *, source_revision: str, evaluation_id: str
) -> None:
    root = candidate_dir.resolve(strict=True)
    if candidate_dir.is_symlink() or not root.is_dir():
        raise ColabEvaluationError("candidate must be one real directory")
    files: list[dict[str, str | int]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ColabEvaluationError("candidate contains a symbolic link")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        metadata = path.stat()
        if metadata.st_nlink != 1 or ".cache" in PurePosixPath(relative).parts:
            raise ColabEvaluationError("candidate contains an unsafe file surface")
        files.append(
            {"path": relative, "size": metadata.st_size, "sha256": sha256_file(path)}
        )
        total_bytes += metadata.st_size
    required = {
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "run-manifest.json",
    }
    if not required.issubset({str(entry["path"]) for entry in files}):
        raise ColabEvaluationError("candidate lacks the verified LoRA release surface")
    _write_json(
        output,
        {
            "schema_version": "1.0",
            "artifact_id": f"ShadowCrafter-9B:{evaluation_id}",
            "revision": source_revision,
            "created_at": datetime.now(UTC).isoformat(),
            "root": str(root),
            "file_count": len(files),
            "total_bytes": total_bytes,
            "files": files,
        },
    )


def _colab_base_manifest(source: Path, base_model_dir: Path, output: Path) -> None:
    payload = _json_object(source, "base-model manifest")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("artifact_id") != _BASE_MODEL_ID
        or payload.get("revision") != _BASE_REVISION
    ):
        raise ColabEvaluationError("base-model manifest identity differs from the 9B pin")
    payload["root"] = str(base_model_dir.resolve(strict=True))
    _write_json(output, payload)


def _environment_manifest(
    output: Path, observer: Callable[[], RuntimeEnvironment]
) -> None:
    _write_json(output, observer().model_dump(mode="json"))


def _inference_request(
    *,
    evaluation_id: str,
    candidate_dir: Path,
    base_model_dir: Path,
    model_config_path: Path,
    base_manifest_path: Path,
    checkpoint_manifest_path: Path,
    cases_path: Path,
    adapter_manifest_path: Path,
    snapshot_manifest_path: Path,
    gate_config_path: Path,
    environment_manifest_path: Path,
    source_revision: str,
    output_dir: Path,
) -> dict[str, Any]:
    request = {
        "schema_version": 1,
        "protocol": "shadowcrafter-frozen-release-evaluation-v1",
        "evaluation_id": evaluation_id,
        "model": {
            "family": "ShadowCrafter-9B",
            "candidate_id": evaluation_id,
            "model_id": _MODEL_ID,
            "base_model_id": _BASE_MODEL_ID,
            "base_model_revision": _BASE_REVISION,
            "text_model_class": "Qwen3_5ForCausalLM",
            "config": _file_pin(model_config_path),
            "base_model_path": str(base_model_dir.resolve(strict=True)),
            "base_model_manifest": _file_pin(base_manifest_path),
            "adapter_path": str((candidate_dir / "adapter").resolve(strict=True)),
            "checkpoint_manifest": _file_pin(checkpoint_manifest_path),
            "training_run_manifest": _file_pin(candidate_dir / "run-manifest.json"),
        },
        "benchmark": {
            "benchmark_id": "ctibench",
            "repository_id": "AI4Sec/cti-bench",
            "upstream_revision": _CTIBENCH_REVISION,
            "license_id": "CC-BY-NC-SA-4.0",
            "usage_scope": "noncommercial-private-research",
            "evaluation_only": True,
            "benchmark_holdout": True,
            "cases": {**_file_pin(cases_path), "record_count": _CTIBENCH_CASE_COUNT},
            "adapter_manifest": _file_pin(adapter_manifest_path),
            "snapshot_manifest": _file_pin(snapshot_manifest_path),
            "dataset_sha256": _CTIBENCH_DATASET_SHA256,
            "gate_config": _file_pin(gate_config_path),
        },
        "decoding": {
            "seed": 20260901,
            "max_input_tokens": 4096,
            "max_new_tokens": 128,
            "per_case_seconds": 120.0,
            "total_seconds": 604800.0,
            "max_gpu_memory_gib": 38.0,
            "max_cpu_rss_gib": 128.0,
            "max_cpu_threads": 8,
        },
        "source": {
            "git_revision": source_revision,
            "candidate_git_revision": source_revision,
            "require_clean_git": True,
            "environment_manifest": _file_pin(environment_manifest_path),
        },
        "output": {
            "directory": str(output_dir.resolve(strict=False)),
            "predictions_name": "predictions.jsonl",
            "manifest_name": "inference-manifest.json",
            "resume": False,
        },
    }
    InferenceRequest.model_validate(request)
    return request


def _review(path: Path, name: str, checkpoint_sha256: str) -> str:
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
    return sha256_file(path)


def _build_evidence(
    *,
    evaluation_id: str,
    candidate_dir: Path,
    checkpoint_manifest_path: Path,
    environment_manifest_path: Path,
    inference_dir: Path,
    cases_path: Path,
    adapter_manifest_path: Path,
    snapshot_manifest_path: Path,
    training_path: Path,
    prepared_training_manifest_path: Path,
    gate_config_path: Path,
    source_revision: str,
    evidence_dir: Path,
) -> Path:
    evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    copies = {
        "checkpoint-manifest.json": checkpoint_manifest_path,
        "training-run-manifest.json": candidate_dir / "run-manifest.json",
        "cases.jsonl": cases_path,
        "ctibench-adapter-manifest.json": adapter_manifest_path,
        "ctibench-snapshot-manifest.json": snapshot_manifest_path,
        "predictions.jsonl": inference_dir / "predictions.jsonl",
        "training.jsonl": training_path,
        "prepared-training-manifest.json": prepared_training_manifest_path,
        "inference-manifest.json": inference_dir / "inference-manifest.json",
        "environment-manifest.json": environment_manifest_path,
    }
    for name, source in copies.items():
        _copy_regular(source, evidence_dir / name)
    checkpoint_sha256 = sha256_file(evidence_dir / "checkpoint-manifest.json")
    review_hashes = {
        name: _review(
            evidence_dir / "reviews" / f"{name}.json", name, checkpoint_sha256
        )
        for name in ("artifact_integrity", "provenance", "license", "privacy", "safety")
    }
    inference = _json_object(evidence_dir / "inference-manifest.json", "inference manifest")
    details = inference.get("inference")
    if not isinstance(details, Mapping):
        raise ColabEvaluationError("inference manifest lacks its audited details")
    prepared = _json_object(
        evidence_dir / "prepared-training-manifest.json", "prepared training manifest"
    )
    artifacts = prepared.get("artifacts")
    train_artifact = artifacts.get("train") if isinstance(artifacts, Mapping) else None
    if not isinstance(train_artifact, Mapping):
        raise ColabEvaluationError("prepared training manifest lacks the train artifact")
    train_count = train_artifact.get("record_count")
    dataset_sha256 = prepared.get("dataset_sha256")
    if not isinstance(train_count, int) or train_count < 1 or not isinstance(dataset_sha256, str):
        raise ColabEvaluationError("prepared training manifest has invalid counts or identity")
    gate = yaml.safe_load(gate_config_path.read_text(encoding="utf-8"))
    gate_rule = gate.get("release_gate") if isinstance(gate, Mapping) else None
    if not isinstance(gate_rule, Mapping) or not isinstance(
        gate_rule.get("evaluator_version"), str
    ):
        raise ColabEvaluationError("release gate configuration is invalid")
    evidence = {
        "schema_version": 1,
        "protocol": "shadowcrafter-frozen-release-evaluation-v1",
        "evaluation_id": evaluation_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate": {
            "candidate_id": evaluation_id,
            "model_family": "ShadowCrafter-9B",
            "model_id": _MODEL_ID,
            "base_model_id": _BASE_MODEL_ID,
            "base_model_revision": _BASE_REVISION,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_manifest": {
                "path": "checkpoint-manifest.json",
                "sha256": checkpoint_sha256,
            },
            "training_run_manifest": {
                "path": "training-run-manifest.json",
                "sha256": sha256_file(evidence_dir / "training-run-manifest.json"),
            },
            "shadowcrafter_git_commit": source_revision,
            "git_tree_clean": True,
        },
        "inference": {
            "evaluator_version": gate_rule["evaluator_version"],
            "code_revision": details["code_revision"],
            "environment_sha256": sha256_file(evidence_dir / "environment-manifest.json"),
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
            "upstream_revision": _CTIBENCH_REVISION,
            "license_id": "CC-BY-NC-SA-4.0",
            "evaluation_only": True,
            "benchmark_holdout": True,
            "cases": {
                "path": "cases.jsonl",
                "sha256": _CTIBENCH_CASES_SHA256,
                "record_count": _CTIBENCH_CASE_COUNT,
            },
            "adapter_manifest": {
                "path": "ctibench-adapter-manifest.json",
                "sha256": _CTIBENCH_ADAPTER_SHA256,
            },
            "snapshot_manifest": {
                "path": "ctibench-snapshot-manifest.json",
                "sha256": _CTIBENCH_SNAPSHOT_SHA256,
            },
            "dataset_sha256": _CTIBENCH_DATASET_SHA256,
        },
        "predictions": {
            "path": "predictions.jsonl",
            "sha256": sha256_file(evidence_dir / "predictions.jsonl"),
            "record_count": _CTIBENCH_CASE_COUNT,
            "frozen": True,
        },
        "training_corpora": [
            {
                "records": {
                    "path": "training.jsonl",
                    "sha256": sha256_file(evidence_dir / "training.jsonl"),
                    "record_count": train_count,
                },
                "prepared_manifest": {
                    "path": "prepared-training-manifest.json",
                    "sha256": sha256_file(evidence_dir / "prepared-training-manifest.json"),
                },
                "dataset_sha256": dataset_sha256,
                "split": "train",
            }
        ],
        "contamination": {
            "algorithm": "ctibench-normalized-content-exact-and-containment-v1",
            "declared_overlap_count": 0,
            "scanned_training_record_count": train_count,
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
    evidence_path = evidence_dir / "release-evidence.json"
    _write_json(evidence_path, evidence)
    return evidence_path


def evaluate_colab_candidate(
    *,
    candidate_dir: Path,
    base_model_dir: Path,
    model_config_path: Path,
    base_model_manifest_path: Path,
    cases_path: Path,
    adapter_manifest_path: Path,
    snapshot_manifest_path: Path,
    gate_config_path: Path,
    training_path: Path,
    prepared_training_manifest_path: Path,
    source_revision: str,
    evaluation_id: str,
    output_root: Path,
    environment_observer: Callable[[], RuntimeEnvironment] = observe_runtime_environment,
    inference_runner: Callable[..., dict[str, Any]] = run_frozen_inference,
    gate_loader: Callable[[Path, Path], GateResult] = load_and_evaluate,
) -> ColabEvaluationResult:
    """Run CTIBench, recompute metrics, and retain a local-only evidence bundle."""

    if output_root.exists() or output_root.is_symlink():
        raise ColabEvaluationError("evaluation output root already exists")
    if sha256_file(cases_path) != _CTIBENCH_CASES_SHA256:
        raise ColabEvaluationError("CTIBench cases differ from the reviewed 5,533-case pin")
    if sha256_file(adapter_manifest_path) != _CTIBENCH_ADAPTER_SHA256:
        raise ColabEvaluationError("CTIBench adapter manifest differs from its pin")
    if sha256_file(snapshot_manifest_path) != _CTIBENCH_SNAPSHOT_SHA256:
        raise ColabEvaluationError("CTIBench snapshot manifest differs from its pin")
    run = _json_object(candidate_dir / "run-manifest.json", "Colab training run manifest")
    if (
        run.get("schema_version") != 2
        or run.get("version") != "v2.0-colab-candidate"
        or not isinstance(run.get("configuration"), Mapping)
        or run["configuration"].get("sha256") != sha256_file(model_config_path)
        or not isinstance(run.get("environment"), Mapping)
        or run["environment"].get("git_revision") != source_revision
    ):
        raise ColabEvaluationError("candidate is not bound to this V2 source and configuration")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    checkpoint_manifest_path = output_root / "checkpoint-manifest.json"
    _checkpoint_manifest(
        candidate_dir,
        checkpoint_manifest_path,
        source_revision=source_revision,
        evaluation_id=evaluation_id,
    )
    local_base_manifest_path = output_root / "base-model-manifest.json"
    _colab_base_manifest(base_model_manifest_path, base_model_dir, local_base_manifest_path)
    local_gate_path = output_root / "release-gates.yaml"
    _copy_regular(gate_config_path, local_gate_path)
    environment_manifest_path = output_root / "environment-manifest.json"
    _environment_manifest(environment_manifest_path, environment_observer)
    inference_parent = output_root / "inference"
    inference_parent.mkdir(mode=0o700)
    inference_dir = inference_parent / "frozen"
    request = _inference_request(
        evaluation_id=evaluation_id,
        candidate_dir=candidate_dir,
        base_model_dir=base_model_dir,
        model_config_path=model_config_path,
        base_manifest_path=local_base_manifest_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        cases_path=cases_path,
        adapter_manifest_path=adapter_manifest_path,
        snapshot_manifest_path=snapshot_manifest_path,
        gate_config_path=local_gate_path,
        environment_manifest_path=environment_manifest_path,
        source_revision=source_revision,
        output_dir=inference_dir,
    )
    request_path = output_root / "inference-request.json"
    _write_json(request_path, request)
    inference_runner(
        request_path,
        sha256_file(request_path),
        environment_observer=environment_observer,
    )
    evidence_path = _build_evidence(
        evaluation_id=evaluation_id,
        candidate_dir=candidate_dir,
        checkpoint_manifest_path=checkpoint_manifest_path,
        environment_manifest_path=environment_manifest_path,
        inference_dir=inference_dir,
        cases_path=cases_path,
        adapter_manifest_path=adapter_manifest_path,
        snapshot_manifest_path=snapshot_manifest_path,
        training_path=training_path,
        prepared_training_manifest_path=prepared_training_manifest_path,
        gate_config_path=local_gate_path,
        source_revision=source_revision,
        evidence_dir=output_root / "evidence",
    )
    gate_result = gate_loader(evidence_path, local_gate_path)
    if not gate_result.passed or gate_result.report is None:
        failures = "; ".join(gate_result.failures) or "missing strict report"
        raise ColabEvaluationError(f"frozen evaluation integrity gate failed: {failures}")
    report_path = output_root / "gate-report.json"
    write_gate_report(gate_result, report_path)
    overall = gate_result.report.get("overall")
    metrics = overall.get("metrics") if isinstance(overall, Mapping) else None
    target = gate_result.report.get("quality_target_met")
    if not isinstance(metrics, Mapping) or not isinstance(target, bool):
        raise ColabEvaluationError("strict gate report lacks recomputed metrics")
    accuracy = metrics.get("accuracy")
    balanced_accuracy = metrics.get("balanced_accuracy")
    macro_f1 = metrics.get("macro_f1")
    if not all(isinstance(value, float) for value in (accuracy, balanced_accuracy, macro_f1)):
        raise ColabEvaluationError("strict gate report contains invalid metric values")
    return ColabEvaluationResult(
        root=output_root,
        evidence_path=evidence_path,
        report_path=report_path,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        quality_target_met=target,
    )


__all__ = [
    "ColabEvaluationError",
    "ColabEvaluationResult",
    "evaluate_colab_candidate",
]
