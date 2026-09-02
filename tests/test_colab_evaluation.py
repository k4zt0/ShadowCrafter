from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

import shadowcrafter.evaluation.colab as colab_evaluation
from shadowcrafter.data.manifest import sha256_file
from shadowcrafter.evaluation.gate import GateResult
from shadowcrafter.evaluation.inference import RuntimeEnvironment


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        schema_version=1,
        python="3.12.3",
        packages={
            "bitsandbytes": "0.50.2",
            "peft": "0.19.1",
            "safetensors": "0.8.0",
            "torch": "2.10.0",
            "transformers": "5.12.1",
        },
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_name="Fake A100",
        cuda_capability=(8, 0),
        torch_cuda_version="12.8",
    )


def test_colab_evaluation_exports_recomputed_metrics_and_evidence(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source_revision = "a" * 40
    model_config = tmp_path / "shadowcrafter-9b.yaml"
    model_config.write_text("project:\n  name: ShadowCrafter-9B\n")
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "config.json").write_text("{}\n")
    base_manifest = tmp_path / "base-manifest.json"
    _write_json(
        base_manifest,
        {
            "schema_version": "1.0",
            "artifact_id": "ornith-ai/Ornith-1.5-9B",
            "revision": colab_evaluation._BASE_REVISION,
            "root": "/remote/original",
            "file_count": 1,
            "total_bytes": 3,
            "files": [
                {
                    "path": "config.json",
                    "size": 3,
                    "sha256": sha256_file(base_model / "config.json"),
                }
            ],
        },
    )
    candidate = tmp_path / "candidate"
    adapter = candidate / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n")
    (adapter / "adapter_model.safetensors").write_bytes(b"safe-adapter")
    _write_json(
        candidate / "run-manifest.json",
        {
            "schema_version": 2,
            "version": "v2.0-colab-candidate",
            "configuration": {"sha256": sha256_file(model_config)},
            "environment": {"git_revision": source_revision},
        },
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"case_id":"one"}\n{"case_id":"two"}\n')
    adapter_manifest = tmp_path / "adapter-manifest.json"
    _write_json(adapter_manifest, {"schema_version": 1})
    snapshot_manifest = tmp_path / "snapshot-manifest.json"
    _write_json(snapshot_manifest, {"schema_version": 2})
    training = tmp_path / "train.jsonl"
    training.write_text('{"record":"one"}\n')
    prepared = tmp_path / "prepared-manifest.json"
    _write_json(
        prepared,
        {
            "dataset_sha256": "d" * 64,
            "artifacts": {"train": {"record_count": 1}},
        },
    )
    gate_config = tmp_path / "release-gates.yaml"
    gate_config.write_text(
        yaml.safe_dump(
            {"release_gate": {"evaluator_version": "shadowcrafter-ctibench-evaluator-v1"}}
        )
    )
    monkeypatch.setattr(colab_evaluation, "_CTIBENCH_CASES_SHA256", sha256_file(cases))
    monkeypatch.setattr(
        colab_evaluation, "_CTIBENCH_ADAPTER_SHA256", sha256_file(adapter_manifest)
    )
    monkeypatch.setattr(
        colab_evaluation, "_CTIBENCH_SNAPSHOT_SHA256", sha256_file(snapshot_manifest)
    )
    monkeypatch.setattr(colab_evaluation, "_CTIBENCH_CASE_COUNT", 2)

    def inference_runner(request_path: Path, _sha256: str, **_kwargs: Any) -> dict[str, Any]:
        request = json.loads(request_path.read_text())
        output = Path(request["output"]["directory"])
        output.mkdir()
        (output / "predictions.jsonl").write_text('{"prediction":"A"}\n' * 2)
        manifest = {
            "inference": {
                "code_revision": source_revision,
                "prompt_template_sha256": "1" * 64,
                "decoding_config_sha256": "2" * 64,
                "seed": 20260901,
                "started_at_utc": "2026-09-02T00:00:00+00:00",
                "completed_at_utc": "2026-09-02T01:00:00+00:00",
            }
        }
        _write_json(output / "inference-manifest.json", manifest)
        return manifest

    def gate_loader(_evidence: Path, _config: Path) -> GateResult:
        return GateResult(
            passed=True,
            failures=(),
            report={
                "overall": {
                    "metrics": {
                        "accuracy": 0.91,
                        "balanced_accuracy": 0.89,
                        "macro_f1": 0.88,
                    }
                },
                "quality_target_met": False,
            },
        )

    result = colab_evaluation.evaluate_colab_candidate(
        candidate_dir=candidate,
        base_model_dir=base_model,
        model_config_path=model_config,
        base_model_manifest_path=base_manifest,
        cases_path=cases,
        adapter_manifest_path=adapter_manifest,
        snapshot_manifest_path=snapshot_manifest,
        gate_config_path=gate_config,
        training_path=training,
        prepared_training_manifest_path=prepared,
        source_revision=source_revision,
        evaluation_id="shadowcrafter-9b-v2-test",
        output_root=tmp_path / "evaluation",
        environment_observer=_environment,
        inference_runner=inference_runner,
        gate_loader=gate_loader,
    )

    assert result.accuracy == 0.91
    assert result.balanced_accuracy == 0.89
    assert result.macro_f1 == 0.88
    assert result.quality_target_met is False
    assert result.report_path.is_file()
    assert result.evidence_path.is_file()
    evidence = json.loads(result.evidence_path.read_text())
    assert evidence["predictions"]["record_count"] == 2
    assert evidence["benchmark_license"]["private_evidence_only"] is True
