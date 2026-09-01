from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from shadowcrafter.data.ctibench import CTIBenchEvalCase, CTIBenchProvenance, CTIBenchTask
from shadowcrafter.evaluation.gate import FrozenPrediction
from shadowcrafter.evaluation.inference import (
    GenerationResult,
    InferenceError,
    InferenceRequest,
    RuntimeEnvironment,
    _messages,
    _offline_process_environment,
    prompt_template_sha256,
    run_frozen_inference,
)

REVISION = "9237e1636ee3e168fbe5ebdcc1c571de0525e568"
GIT_REVISION = "5" * 40
BASE_REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file_pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path.read_bytes())}


def _tree_manifest(root: Path, artifact_id: str, revision: str) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            content = path.read_bytes()
            total += len(content)
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": len(content),
                    "sha256": _sha(content),
                }
            )
    return {
        "schema_version": "1.0",
        "artifact_id": artifact_id,
        "revision": revision,
        "root": str(root),
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
    }


def _case(index: int, answer: str) -> CTIBenchEvalCase:
    case = CTIBenchEvalCase(
        case_id=f"ctibench:cti-mcq:{index:06d}",
        task=CTIBenchTask.MULTIPLE_CHOICE,
        input_text=f"Defensive benchmark question {index}?",
        choices={"A": "alpha", "B": "bravo", "C": "charlie", "D": "delta"},
        answer=answer,
        provenance=CTIBenchProvenance(
            upstream_revision=REVISION,
            retrieved_at="2026-09-01T00:00:00+00:00",
            source_file="cti-mcq.tsv",
            source_row=index,
            source_reference="fixture",
            source_file_sha256="1" * 64,
            source_prompt_sha256="2" * 64,
            source_prompt_normalized_sha256="3" * 64,
            input_normalized_sha256="4" * 64,
            rendered_input_normalized_sha256="5" * 64,
        ),
        content_sha256="0" * 64,
    )
    case.content_sha256 = case.canonical_hash()
    return case


class FakeBackend:
    def __init__(self, outputs: list[str], mutate: Any = None) -> None:
        self.outputs = iter(outputs)
        self.messages: list[list[dict[str, str]]] = []
        self.closed = False
        self.mutate = mutate

    def generate(
        self,
        messages: Any,
        *,
        max_input_tokens: int,
        max_new_tokens: int,
        max_seconds: float,
    ) -> GenerationResult:
        assert max_input_tokens == 1024
        assert max_new_tokens == 64
        assert max_seconds <= 30
        self.messages.append([dict(message) for message in messages])
        if self.mutate is not None:
            mutate, self.mutate = self.mutate, None
            mutate()
        return GenerationResult(next(self.outputs), 32, 1, 1024, 2048)

    def close(self) -> None:
        self.closed = True


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        schema_version=1,
        python="3.12.3",
        packages={
            "bitsandbytes": "0.50.2",
            "peft": "0.19.1",
            "safetensors": "0.6.2",
            "torch": "2.10.0",
            "transformers": "5.12.1",
        },
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_name="Fake H100",
        cuda_capability=(9, 0),
        torch_cuda_version="12.8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, str, Path, Path]:
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "ShadowCrafter-9B"},
                "base_model": {
                    "id": "ornith-ai/Ornith-1.5-9B",
                    "revision": BASE_REVISION,
                    "text_model_class": "Qwen3_5ForCausalLM",
                    "trust_remote_code": False,
                },
                "training": {
                    "backend": "transformers_peft",
                    "load_in_4bit": True,
                    "quant_type": "nf4",
                    "double_quant": True,
                    "compute_dtype": "bfloat16",
                },
                "release": {"hf_repo": "KaztoRay/ShadowCrafter-9B", "private": True},
            },
            sort_keys=False,
        )
    )

    base_root = tmp_path / "base"
    base_root.mkdir()
    (base_root / "config.json").write_text("{}\n")
    base_manifest_path = tmp_path / "base-manifest.json"
    _write_json(
        base_manifest_path,
        _tree_manifest(base_root, "ornith-ai/Ornith-1.5-9B", BASE_REVISION),
    )

    checkpoint_root = tmp_path / "checkpoint"
    adapter_root = checkpoint_root / "adapter"
    adapter_root.mkdir(parents=True)
    adapter_config = {
        "base_model_name_or_path": "ornith-ai/Ornith-1.5-9B",
        "revision": BASE_REVISION,
        "peft_type": "LORA",
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
    }
    _write_json(adapter_root / "adapter_config.json", adapter_config)
    (adapter_root / "adapter_model.safetensors").write_bytes(b"safe fixture tensor bytes")
    run_manifest_path = checkpoint_root / "run-manifest.json"
    _write_json(
        run_manifest_path,
        {
            "schema_version": 1,
            "project": {"name": "ShadowCrafter-9B"},
            "base_model": {
                "id": "ornith-ai/Ornith-1.5-9B",
                "revision": BASE_REVISION,
            },
            "configuration": {"sha256": _sha(model_config_path.read_bytes())},
            "effective_training_invariants": {
                "push_to_hub": False,
                "resume_from_checkpoint": False,
            },
            "training_observation": {"lora_parameters_changed": True},
            "adapter": {
                "path": str(adapter_root),
                "adapter_config_sha256": _sha((adapter_root / "adapter_config.json").read_bytes()),
                "adapter_weights_sha256": _sha(
                    (adapter_root / "adapter_model.safetensors").read_bytes()
                ),
                "safe_serialization": True,
                "lora_only": True,
                "finite": True,
            },
            "environment": {"git_revision": GIT_REVISION},
        },
    )
    checkpoint_manifest_path = tmp_path / "checkpoint-manifest.json"
    _write_json(
        checkpoint_manifest_path,
        _tree_manifest(checkpoint_root, "KaztoRay/ShadowCrafter-9B", GIT_REVISION),
    )

    cases = [_case(0, "A"), _case(1, "B")]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("".join(case.model_dump_json() + "\n" for case in cases))
    cases_sha = _sha(cases_path.read_bytes())
    snapshot_path = tmp_path / "snapshot-manifest.json"
    _write_json(
        snapshot_path,
        {
            "source": {"repo_id": "AI4Sec/cti-bench", "policy_class": "eval_only"},
            "license": {"id": "CC-BY-NC-SA-4.0"},
            "upstream_revision": REVISION,
        },
    )
    dataset_sha = "6" * 64
    adapter_manifest_path = tmp_path / "ctibench-adapter-manifest.json"
    _write_json(
        adapter_manifest_path,
        {
            "upstream_revision": REVISION,
            "license_id": "CC-BY-NC-SA-4.0",
            "dataset_sha256": dataset_sha,
            "output": {"sha256": cases_sha, "record_count": 2},
            "controls": {
                "evaluation_only": True,
                "answer_key_isolated": True,
                "trusted_runner_template_required": True,
                "commercial_use_permitted": False,
            },
        },
    )
    gate_path = tmp_path / "gate.yaml"
    gate = {
        "schema_version": 2,
        "protocol": "shadowcrafter-frozen-release-evaluation-v1",
        "claim": "Fixture noncommercial private evaluation.",
        "metric_thresholds": {"accuracy": 0.95, "balanced_accuracy": 0.95, "macro_f1": 0.95},
        "require_per_task_metrics": True,
        "max_contamination_overlap_count": 0,
        "contamination_algorithm": "ctibench-normalized-content-exact-and-containment-v1",
        "evaluator_version": "shadowcrafter-ctibench-evaluator-v1",
        "require_clean_git": True,
        "quality_target_is_publication_blocker": False,
        "authorization_scope": "noncommercial-private-experimental-release",
        "commercial_use_permitted": False,
        "model_publication_authorized": True,
        "required_visibility": "private",
        "public_publication_authorized": False,
        "release_tier": "Experimental Release",
        "benchmark": {
            "benchmark_id": "ctibench",
            "repository_id": "AI4Sec/cti-bench",
            "upstream_revision": REVISION,
            "license_id": "CC-BY-NC-SA-4.0",
            "snapshot_manifest_sha256": _sha(snapshot_path.read_bytes()),
            "adapter_manifest_sha256": _sha(adapter_manifest_path.read_bytes()),
            "cases_sha256": cases_sha,
            "dataset_sha256": dataset_sha,
            "expected_sample_count": 2,
            "tasks": {"cti-mcq": {"sample_count": 2, "minimum_reference_classes": 2}},
        },
        "allowed_candidates": {
            "ShadowCrafter-9B": {
                "model_id": "KaztoRay/ShadowCrafter-9B",
                "base_model_id": "ornith-ai/Ornith-1.5-9B",
                "base_model_revision": BASE_REVISION,
            },
        },
    }
    gate_path.write_text(yaml.safe_dump({"release_gate": gate}, sort_keys=False))
    environment_path = tmp_path / "environment.json"
    _write_json(environment_path, _environment().model_dump(mode="json"))

    output_dir = tmp_path / "outputs" / "evaluation-v1"
    output_dir.parent.mkdir()
    request_path = tmp_path / "request.json"
    request: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "shadowcrafter-frozen-release-evaluation-v1",
        "evaluation_id": "evaluation-v1",
        "model": {
            "family": "ShadowCrafter-9B",
            "candidate_id": "candidate-v1",
            "model_id": "KaztoRay/ShadowCrafter-9B",
            "base_model_id": "ornith-ai/Ornith-1.5-9B",
            "base_model_revision": BASE_REVISION,
            "text_model_class": "Qwen3_5ForCausalLM",
            "config": _file_pin(model_config_path),
            "base_model_path": str(base_root),
            "base_model_manifest": _file_pin(base_manifest_path),
            "adapter_path": str(adapter_root),
            "checkpoint_manifest": _file_pin(checkpoint_manifest_path),
            "training_run_manifest": _file_pin(run_manifest_path),
        },
        "benchmark": {
            "benchmark_id": "ctibench",
            "repository_id": "AI4Sec/cti-bench",
            "upstream_revision": REVISION,
            "license_id": "CC-BY-NC-SA-4.0",
            "usage_scope": "noncommercial-private-research",
            "evaluation_only": True,
            "benchmark_holdout": True,
            "cases": {**_file_pin(cases_path), "record_count": 2},
            "adapter_manifest": _file_pin(adapter_manifest_path),
            "snapshot_manifest": _file_pin(snapshot_path),
            "dataset_sha256": dataset_sha,
            "gate_config": _file_pin(gate_path),
        },
        "decoding": {
            "seed": 20260901,
            "max_input_tokens": 1024,
            "max_new_tokens": 64,
            "per_case_seconds": 30,
            "total_seconds": 600,
            "max_gpu_memory_gib": 72,
            "max_cpu_rss_gib": 128,
            "max_cpu_threads": 4,
        },
        "source": {
            "git_revision": GIT_REVISION,
            "require_clean_git": True,
            "environment_manifest": _file_pin(environment_path),
        },
        "output": {
            "directory": str(output_dir),
            "predictions_name": "predictions.jsonl",
            "manifest_name": "inference-manifest.json",
            "resume": False,
        },
    }
    _write_json(request_path, request)
    return request_path, _sha(request_path.read_bytes()), output_dir, cases_path


def test_prompt_renderer_never_passes_answer_to_model() -> None:
    case = CTIBenchEvalCase(
        case_id="ctibench:cti-rcm:000000",
        task=CTIBenchTask.CWE_MAPPING,
        input_text="Map this defensive description to a weakness.",
        answer="CWE-987654",
        provenance=CTIBenchProvenance(
            upstream_revision=REVISION,
            retrieved_at="2026-09-01T00:00:00+00:00",
            source_file="cti-rcm.tsv",
            source_row=0,
            source_reference="fixture",
            source_file_sha256="1" * 64,
            source_prompt_sha256="2" * 64,
            source_prompt_normalized_sha256="3" * 64,
            input_normalized_sha256="4" * 64,
            rendered_input_normalized_sha256="5" * 64,
        ),
        content_sha256="0" * 64,
    )
    case.content_sha256 = case.canonical_hash()
    rendered = json.dumps(_messages(case))
    assert "CWE-987654" not in rendered
    assert "source_prompt" not in rendered
    assert len(prompt_template_sha256()) == 64


def test_frozen_inference_writes_gate_compatible_hash_only_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, request_sha, output_dir, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        "shadowcrafter.evaluation.inference._verify_git", lambda _revision: tmp_path
    )
    backend = FakeBackend(["A", "B"])
    manifest = run_frozen_inference(
        request_path,
        request_sha,
        backend_factory=lambda _request: backend,
        environment_observer=_environment,
    )

    predictions = (output_dir / "predictions.jsonl").read_text().splitlines()
    parsed = [FrozenPrediction.model_validate_json(line) for line in predictions]
    assert [prediction.case_id for prediction in parsed] == [
        "ctibench:cti-mcq:000000",
        "ctibench:cti-mcq:000001",
    ]
    assert [prediction.raw_output for prediction in parsed] == ["A", "B"]
    assert manifest["predictions"]["sha256"] == _sha(
        (output_dir / "predictions.jsonl").read_bytes()
    )
    assert manifest["inference"]["scores_computed"] is False
    assert manifest["benchmark"]["model_publication_authorized_by_benchmark"] is False
    assert manifest["benchmark"]["raw_examples_logged"] is False
    assert backend.closed
    assert all("answer" not in message for batch in backend.messages for message in batch)
    assert (output_dir.stat().st_mode & 0o777) == 0o500
    assert ((output_dir / "predictions.jsonl").stat().st_mode & 0o777) == 0o400


def test_existing_output_refuses_resume_before_backend_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, request_sha, output_dir, _ = _fixture(tmp_path)
    output_dir.mkdir()
    monkeypatch.setattr(
        "shadowcrafter.evaluation.inference._verify_git", lambda _revision: tmp_path
    )
    called = False

    def factory(_request: InferenceRequest) -> FakeBackend:
        nonlocal called
        called = True
        return FakeBackend(["A", "B"])

    with pytest.raises(InferenceError, match="overwrite or resume"):
        run_frozen_inference(
            request_path,
            request_sha,
            backend_factory=factory,
            environment_observer=_environment,
        )
    assert not called


def test_input_mutation_after_generation_fails_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, request_sha, output_dir, cases_path = _fixture(tmp_path)
    monkeypatch.setattr(
        "shadowcrafter.evaluation.inference._verify_git", lambda _revision: tmp_path
    )
    backend = FakeBackend(["A", "B"], mutate=lambda: cases_path.write_bytes(b"tampered\n"))
    with pytest.raises(InferenceError, match="SHA-256 mismatch"):
        run_frozen_inference(
            request_path,
            request_sha,
            backend_factory=lambda _request: backend,
            environment_observer=_environment,
        )
    assert not output_dir.exists()
    assert backend.closed


def test_request_rejects_quality_or_identity_scope_changes(tmp_path: Path) -> None:
    request_path, _, _, _ = _fixture(tmp_path)
    payload = json.loads(request_path.read_text())
    payload["model"]["model_id"] = "attacker/replacement"
    with pytest.raises(ValueError, match="audited ShadowCrafter contract"):
        InferenceRequest.model_validate(payload)
    payload = json.loads(request_path.read_text())
    payload["output"]["resume"] = True
    with pytest.raises(ValueError):
        InferenceRequest.model_validate(payload)


def test_offline_environment_blocks_python_network_and_restores_socket() -> None:
    original = __import__("socket").create_connection
    with (
        _offline_process_environment(2),
        pytest.raises(OSError, match="network access is disabled"),
    ):
        __import__("socket").create_connection(("example.invalid", 443))
    assert __import__("socket").create_connection is original
