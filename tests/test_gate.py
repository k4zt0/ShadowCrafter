from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from shadowcrafter.data.ctibench import CTIBenchEvalCase, CTIBenchProvenance, CTIBenchTask
from shadowcrafter.evaluation.gate import (
    evaluate_gate,
    load_and_evaluate,
    wilson_lower_bound,
    write_gate_report,
)
from shadowcrafter.release.huggingface import publish_private_model
from shadowcrafter.schemas import (
    EvalMetric,
    Message,
    Provenance,
    SecurityRecord,
    TaskType,
)

REVISION = "9237e1636ee3e168fbe5ebdcc1c571de0525e568"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CONTAMINATION_ALGORITHM = "ctibench-normalized-content-exact-and-containment-v1"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return _sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _case(index: int, answer: str, source_sha256: str) -> CTIBenchEvalCase:
    input_text = f"Defensive multiple-choice question number {index:04d}?"
    choices = {
        "A": f"Approved defensive answer {index:04d}",
        "B": f"Alternative defensive answer {index:04d}",
        "C": f"Insufficient evidence {index:04d}",
        "D": f"Escalate for review {index:04d}",
    }
    rendered = "\n".join(
        (input_text, *(f"{label}. {choices[label]}" for label in ("A", "B", "C", "D")))
    )
    source_prompt = f"Reviewed upstream prompt {index:04d}"
    case = CTIBenchEvalCase(
        case_id=f"ctibench:cti-mcq:{index:06d}",
        task=CTIBenchTask.MULTIPLE_CHOICE,
        input_text=input_text,
        choices=choices,
        answer=answer,
        provenance=CTIBenchProvenance(
            upstream_revision=REVISION,
            retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
            source_file="cti-mcq.tsv",
            source_row=index,
            source_reference="Manual",
            source_file_sha256=source_sha256,
            source_prompt_sha256=_sha256(source_prompt.encode()),
            source_prompt_normalized_sha256=_sha256(source_prompt.encode()),
            input_normalized_sha256=_sha256(input_text.encode()),
            rendered_input_normalized_sha256=_sha256(rendered.encode()),
        ),
        content_sha256="0" * 64,
    )
    case.content_sha256 = case.canonical_hash()
    return case


def _training_record(content: str) -> SecurityRecord:
    record = SecurityRecord(
        record_id="mitre-attack:training:000001",
        task=TaskType.THREAT_INTELLIGENCE,
        messages=[
            Message(role="user", content=content),
            Message(role="assistant", content="Use read-only defensive analysis and human review."),
        ],
        provenance=Provenance(
            source_id="mitre-attack-enterprise",
            source_url="https://attack.mitre.org/",
            license="MITRE-ATTACK-Terms-of-Use",
            retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
            upstream_revision="reviewed-fixture",
            record_key="training:000001",
            content_sha256="0" * 64,
        ),
        split_group="mitre-attack:training:000001",
    )
    record.provenance.content_sha256 = record.canonical_hash()
    return record


def _write_bundle(
    root: Path,
    *,
    answers: list[str],
    outputs: list[str] | None = None,
    training_content: str = "A separately authored, defensive ATT&CK training question.",
) -> tuple[Path, Path]:
    source_sha256 = "1" * 64
    cases = [_case(index, answer, source_sha256) for index, answer in enumerate(answers)]
    cases_path = root / "ctibench.eval.jsonl"
    cases_path.write_text("".join(case.model_dump_json() + "\n" for case in cases))
    cases_sha256 = _sha256(cases_path.read_bytes())

    snapshot_manifest_path = root / "snapshot-manifest.json"
    snapshot_manifest = {
        "schema_version": 1,
        "source": {
            "id": "ctibench",
            "repo_id": "AI4Sec/cti-bench",
            "policy_class": "eval_only",
            "license": {"id": "CC-BY-NC-SA-4.0"},
        },
        "upstream_revision": REVISION,
        "files": [{"path": "cti-mcq.tsv", "sha256": source_sha256}],
    }
    _write_json(snapshot_manifest_path, snapshot_manifest)
    snapshot_manifest_sha256 = _sha256(snapshot_manifest_path.read_bytes())
    dataset_sha256 = _canonical_sha256(
        {
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "output_sha256": cases_sha256,
        }
    )

    adapter_manifest_path = root / "adapter-manifest.json"
    adapter_manifest = {
        "schema_version": 1,
        "adapter": "ctibench_eval_only_v1",
        "source_id": "ctibench",
        "repo_id": "AI4Sec/cti-bench",
        "source_policy_class": "eval_only",
        "license_id": "CC-BY-NC-SA-4.0",
        "license": {
            "id": "CC-BY-NC-SA-4.0",
            "noncommercial_only": True,
            "attribution_required_when_shared": True,
            "share_alike_required_when_adapted_material_is_shared": True,
        },
        "upstream_revision": REVISION,
        "snapshot_manifest": {"sha256": snapshot_manifest_sha256},
        "output": {"sha256": cases_sha256, "record_count": len(cases)},
        "statistics": {
            "emitted_count": len(cases),
            "by_task": {"cti-mcq": len(cases)},
        },
        "controls": {
            "evaluation_only": True,
            "benchmark_holdout": True,
            "prompt_training_eligible": False,
            "security_record_schema_used": False,
            "upstream_prompt_preserved": False,
            "answer_key_isolated": True,
            "training_contamination_check_required": True,
            "commercial_use_permitted": False,
            "attribution_required": True,
            "share_alike_required_when_adapted_material_is_shared": True,
        },
        "dataset_sha256": dataset_sha256,
    }
    _write_json(adapter_manifest_path, adapter_manifest)

    resolved_outputs = answers if outputs is None else outputs
    predictions_path = root / "predictions.jsonl"
    prediction_lines = []
    for case, raw_output in zip(cases, resolved_outputs, strict=True):
        prediction_lines.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "case_id": case.case_id,
                    "raw_output": raw_output,
                    "raw_output_sha256": _sha256(raw_output.encode()),
                },
                sort_keys=True,
            )
            + "\n"
        )
    predictions_path.write_text("".join(prediction_lines))

    training_path = root / "train.jsonl"
    training_record = _training_record(training_content)
    training_path.write_text(training_record.model_dump_json() + "\n")
    training_sha256 = _sha256(training_path.read_bytes())
    artifact_hashes = {
        "evaluation": EMPTY_SHA256,
        "test": EMPTY_SHA256,
        "train": training_sha256,
        "validation": EMPTY_SHA256,
    }
    training_dataset_sha256 = _canonical_sha256(artifact_hashes)
    training_manifest_path = root / "training-manifest.json"
    training_manifest = {
        "schema_version": 2,
        "dataset_sha256": training_dataset_sha256,
        "artifacts": {
            name: {
                "sha256": digest,
                "record_count": 1 if name == "train" else 0,
            }
            for name, digest in artifact_hashes.items()
        },
        "split_counts": {"evaluation": 0, "test": 0, "train": 1, "validation": 0},
        "sources": [
            {
                "source_id": "mitre-attack-enterprise",
                "policy_class": "allow_train",
                "allowed_purposes": ["rag", "train"],
            }
        ],
    }
    _write_json(training_manifest_path, training_manifest)

    checkpoint_manifest_path = root / "checkpoint-manifest.json"
    _write_json(
        checkpoint_manifest_path,
        {"schema_version": 1, "checkpoint_sha256": "2" * 64},
    )
    training_run_manifest_path = root / "training-run-manifest.json"
    _write_json(
        training_run_manifest_path,
        {
            "schema_version": 1,
            "base_model_id": "ornith-ai/Ornith-1.5-9B",
            "base_model_revision": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
        },
    )
    checkpoint_sha256 = _sha256(checkpoint_manifest_path.read_bytes())
    review_paths: dict[str, Path] = {}
    for review_name in ("artifact_integrity", "provenance", "license", "privacy", "safety"):
        review_path = root / f"{review_name}-review.json"
        review_payload: dict[str, Any] = {
            "schema_version": 1,
            "review": review_name,
            "passed": True,
            "candidate_checkpoint_sha256": checkpoint_sha256,
        }
        if review_name == "license":
            review_payload.update(
                {
                    "commercial_release_authorized": False,
                    "benchmark_material_sharing_authorized": False,
                }
            )
        _write_json(review_path, review_payload)
        review_paths[review_name] = review_path

    config_path = root / "gate.yaml"
    config: dict[str, Any] = {
        "release_gate": {
            "schema_version": 2,
            "protocol": "shadowcrafter-frozen-release-evaluation-v1",
            "claim": "Fixture-only noncommercial frozen evaluation protocol.",
            "metric_thresholds": {
                "accuracy": 0.95,
                "balanced_accuracy": 0.95,
                "macro_f1": 0.95,
            },
            "require_per_task_metrics": True,
            "max_contamination_overlap_count": 0,
            "contamination_algorithm": CONTAMINATION_ALGORITHM,
            "evaluator_version": "shadowcrafter-ctibench-evaluator-v1",
            "require_clean_git": True,
            "quality_target_is_publication_blocker": False,
            "authorization_scope": "noncommercial-public-official-release",
            "commercial_use_permitted": False,
            "model_publication_authorized": True,
            "required_visibility": "public",
            "public_publication_authorized": True,
            "release_tier": "Official Release",
            "benchmark": {
                "benchmark_id": "ctibench",
                "repository_id": "AI4Sec/cti-bench",
                "upstream_revision": REVISION,
                "license_id": "CC-BY-NC-SA-4.0",
                "snapshot_manifest_sha256": snapshot_manifest_sha256,
                "adapter_manifest_sha256": _sha256(adapter_manifest_path.read_bytes()),
                "cases_sha256": cases_sha256,
                "dataset_sha256": dataset_sha256,
                "expected_sample_count": len(cases),
                "tasks": {
                    "cti-mcq": {
                        "sample_count": len(cases),
                        "minimum_reference_classes": 2,
                    }
                },
            },
            "allowed_candidates": {
                "ShadowCrafter-9B": {
                    "model_id": "KaztoRay/ShadowCrafter-9B",
                    "base_model_id": "ornith-ai/Ornith-1.5-9B",
                    "base_model_revision": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
                },
            },
        }
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    evidence_path = root / "release-evidence.json"
    evidence = {
        "schema_version": 1,
        "protocol": "shadowcrafter-frozen-release-evaluation-v1",
        "evaluation_id": "fixture-evaluation-v1",
        "created_at_utc": "2026-09-01T00:02:00+00:00",
        "candidate": {
            "candidate_id": "fixture-candidate-v1",
            "model_family": "ShadowCrafter-9B",
            "model_id": "KaztoRay/ShadowCrafter-9B",
            "base_model_id": "ornith-ai/Ornith-1.5-9B",
            "base_model_revision": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_manifest": {
                "path": checkpoint_manifest_path.name,
                "sha256": _sha256(checkpoint_manifest_path.read_bytes()),
            },
            "training_run_manifest": {
                "path": training_run_manifest_path.name,
                "sha256": _sha256(training_run_manifest_path.read_bytes()),
            },
            "shadowcrafter_git_commit": "5" * 40,
            "git_tree_clean": True,
        },
        "inference": {
            "evaluator_version": "shadowcrafter-ctibench-evaluator-v1",
            "code_revision": "5" * 40,
            "environment_sha256": "6" * 64,
            "prompt_template_sha256": "7" * 64,
            "decoding_config_sha256": "8" * 64,
            "seed": 20260901,
            "started_at_utc": "2026-09-01T00:00:00+00:00",
            "completed_at_utc": "2026-09-01T00:01:00+00:00",
            "deterministic_decoding": True,
            "answer_key_hidden_from_model": True,
            "raw_outputs_retained": True,
        },
        "benchmark": {
            "benchmark_id": "ctibench",
            "repository_id": "AI4Sec/cti-bench",
            "upstream_revision": REVISION,
            "license_id": "CC-BY-NC-SA-4.0",
            "evaluation_only": True,
            "benchmark_holdout": True,
            "cases": {
                "path": cases_path.name,
                "sha256": cases_sha256,
                "record_count": len(cases),
            },
            "adapter_manifest": {
                "path": adapter_manifest_path.name,
                "sha256": _sha256(adapter_manifest_path.read_bytes()),
            },
            "snapshot_manifest": {
                "path": snapshot_manifest_path.name,
                "sha256": snapshot_manifest_sha256,
            },
            "dataset_sha256": dataset_sha256,
        },
        "predictions": {
            "path": predictions_path.name,
            "sha256": _sha256(predictions_path.read_bytes()),
            "record_count": len(cases),
            "frozen": True,
        },
        "training_corpora": [
            {
                "records": {
                    "path": training_path.name,
                    "sha256": training_sha256,
                    "record_count": 1,
                },
                "prepared_manifest": {
                    "path": training_manifest_path.name,
                    "sha256": _sha256(training_manifest_path.read_bytes()),
                },
                "dataset_sha256": training_dataset_sha256,
                "split": "train",
            }
        ],
        "contamination": {
            "algorithm": CONTAMINATION_ALGORITHM,
            "declared_overlap_count": 0,
            "scanned_training_record_count": 1,
        },
        "benchmark_license": {
            "usage_scope": "noncommercial-private-research",
            "commercial_use_permitted": False,
            "commercial_release_requested": False,
            "private_evidence_only": True,
            "attribution_retained": True,
            "share_alike_review_required_before_sharing": True,
            "license_review_sha256": "9" * 64,
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
                f"{review_name}_review": {
                    "passed": True,
                    "report": {
                        "path": review_path.name,
                        "sha256": _sha256(review_path.read_bytes()),
                    },
                }
                for review_name, review_path in review_paths.items()
            },
        },
    }
    _write_json(evidence_path, evidence)
    return evidence_path, config_path


def test_wilson_bound_is_conservative() -> None:
    assert wilson_lower_bound(0.95, 1000) < 0.95


def test_aggregate_helper_rejects_contamination_but_is_not_release_evidence() -> None:
    metric = EvalMetric(
        name="accuracy",
        value=0.99,
        sample_count=1000,
        task="fixture",
        split_hash="fixture-only",
        contamination_rate=0.01,
    )
    result = evaluate_gate(
        [metric],
        {
            "max_contamination_rate": 0.0,
            "metrics": {"accuracy": {"threshold": 0.94, "minimum_samples": 10}},
        },
    )
    assert not result.passed
    assert result.report is None


def test_frozen_release_gate_recomputes_metrics_and_class_results(tmp_path: Path) -> None:
    evidence_path, config_path = _write_bundle(tmp_path, answers=["A", "B", "A", "B"])

    result = load_and_evaluate(evidence_path, config_path)

    assert result.passed, result.failures
    assert result.report is not None
    assert result.report["overall"]["metrics"] == {
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "macro_f1": 1.0,
    }
    task = result.report["tasks"]["cti-mcq"]
    assert task["sample_count"] == 4
    assert {row["label"]: row["support"] for row in task["classes"]} == {"A": 2, "B": 2}
    assert result.report["contamination"]["overlap_count"] == 0
    assert result.report["quality_target_met"] is True
    assert result.report["target_95_met"] is True
    assert result.report["authorization"]["commercial_use_permitted"] is False
    assert result.report["authorization"]["model_publication_authorized"] is True
    assert result.report["authorization"]["required_visibility"] == "public"
    assert result.report["authorization"]["public_publication_authorized"] is True


def test_quality_shortfall_is_reported_but_does_not_block_official_release(
    tmp_path: Path,
) -> None:
    answers = ["A"] * 95 + ["B"] * 5
    outputs = ["A"] * 100
    evidence_path, config_path = _write_bundle(tmp_path, answers=answers, outputs=outputs)

    result = load_and_evaluate(evidence_path, config_path)

    assert result.passed
    assert result.failures == ()
    assert result.report is not None
    assert result.report["overall"]["metrics"]["accuracy"] == 0.95
    assert result.report["overall"]["metrics"]["balanced_accuracy"] < 0.95
    assert result.report["overall"]["metrics"]["macro_f1"] < 0.95
    assert result.report["quality_target_met"] is False
    assert result.report["target_95_met"] is False
    assert any(
        "balanced_accuracy" in shortfall for shortfall in result.report["quality_shortfalls"]
    )
    assert any("macro_f1" in shortfall for shortfall in result.report["quality_shortfalls"])
    assert result.report["authorization"]["model_publication_authorized"] is True


def test_gate_rejects_prediction_file_tampering(tmp_path: Path) -> None:
    evidence_path, config_path = _write_bundle(tmp_path, answers=["A", "B", "A", "B"])
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        predictions.read_text().replace('"raw_output": "A"', '"raw_output": "B"', 1)
    )

    result = load_and_evaluate(evidence_path, config_path)

    assert not result.passed
    assert result.report is None
    assert "checksum mismatch" in result.failures[0]


def test_gate_rejects_legacy_hand_written_aggregate(tmp_path: Path) -> None:
    config_path = tmp_path / "gate.yaml"
    config_path.write_text(Path("configs/eval/release-gates.yaml").read_text())
    aggregate_path = tmp_path / "metrics.json"
    _write_json(
        aggregate_path,
        {
            "metrics": [
                {
                    "name": "accuracy",
                    "value": 1.0,
                    "sample_count": 999999,
                    "contamination_rate": 0.0,
                }
            ]
        },
    )

    result = load_and_evaluate(aggregate_path, config_path)

    assert not result.passed
    assert result.report is None
    assert "invalid release evidence" in result.failures[0]


def test_gate_recomputes_and_rejects_training_contamination(tmp_path: Path) -> None:
    leaked = "Defensive multiple-choice question number 0000?"
    evidence_path, config_path = _write_bundle(
        tmp_path,
        answers=["A", "B", "A", "B"],
        training_content=f"Copied benchmark: {leaked}",
    )

    result = load_and_evaluate(evidence_path, config_path)

    assert not result.passed
    assert result.report is not None
    assert result.report["contamination"]["overlap_count"] == 1
    assert "contamination overlap_count 1" in result.failures[0]


def test_gate_rejects_commercial_or_publication_authorization(tmp_path: Path) -> None:
    evidence_path, config_path = _write_bundle(tmp_path, answers=["A", "B", "A", "B"])
    evidence = json.loads(evidence_path.read_text())
    evidence["benchmark_license"]["commercial_use_permitted"] = True
    _write_json(evidence_path, evidence)

    result = load_and_evaluate(evidence_path, config_path)

    assert not result.passed
    assert result.report is None
    assert "commercial_use_permitted" in result.failures[0]


def test_failed_safety_review_blocks_official_publication(tmp_path: Path) -> None:
    evidence_path, config_path = _write_bundle(tmp_path, answers=["A", "B", "A", "B"])
    safety_path = tmp_path / "safety-review.json"
    safety = json.loads(safety_path.read_text())
    safety["passed"] = False
    _write_json(safety_path, safety)
    evidence = json.loads(evidence_path.read_text())
    evidence["publication"]["safety_review"]["report"]["sha256"] = _sha256(safety_path.read_bytes())
    _write_json(evidence_path, evidence)

    result = load_and_evaluate(evidence_path, config_path)

    assert not result.passed
    assert result.report is None
    assert "safety review" in result.failures[0]


def test_gate_rejects_threshold_configuration_below_94_percent(tmp_path: Path) -> None:
    evidence_path, config_path = _write_bundle(tmp_path, answers=["A", "B", "A", "B"])
    config = yaml.safe_load(config_path.read_text())
    config["release_gate"]["metric_thresholds"]["macro_f1"] = 0.93
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    result = load_and_evaluate(evidence_path, config_path)

    assert not result.passed
    assert result.report is None
    assert "greater than or equal to 0.95" in result.failures[0]


def test_local_directory_publisher_is_disabled_by_immutable_release_policy(
    tmp_path: Path,
) -> None:
    evidence_path, config_path = _write_bundle(tmp_path, answers=["A", "B", "A", "B"])
    model_dir = tmp_path / "remote-model-reference-only"
    model_dir.mkdir()

    with pytest.raises(RuntimeError, match="local directory publication is disabled"):
        publish_private_model(
            "KaztoRay/ShadowCrafter-9B",
            model_dir,
            evidence_path,
            config_path,
        )


def test_gate_report_is_created_once_and_never_overwritten(tmp_path: Path) -> None:
    evidence_path, config_path = _write_bundle(tmp_path, answers=["A", "B", "A", "B"])
    result = load_and_evaluate(evidence_path, config_path)
    report_path = tmp_path / "gate-report.json"

    write_gate_report(result, report_path)

    assert json.loads(report_path.read_text())["passed"] is True
    with pytest.raises(FileExistsError):
        write_gate_report(result, report_path)
