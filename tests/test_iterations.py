from __future__ import annotations

import pytest

from shadowcrafter.automation.iterations import (
    IterationPolicyError,
    decide_quality,
    training_overrides,
    version_for,
    version_index,
)
from shadowcrafter.automation.models import RemoteEvidence


def _report(value: float, *, decision: bool) -> dict[str, object]:
    metrics = {"accuracy": value, "balanced_accuracy": value, "macro_f1": value}
    return {
        "passed": True,
        "failures": [],
        "quality_target_met": decision,
        "target_95_met": decision,
        "overall": {"metrics": metrics},
        "tasks": {"cti-mcq": {"metrics": metrics}},
    }


def test_major_versions_are_strict_and_sequential() -> None:
    assert version_for(1) == "v1.0"
    assert version_for(12) == "v12.0"
    assert version_index("v12.0") == 12
    with pytest.raises(IterationPolicyError):
        version_index("v1.1")


def test_quality_requires_every_metric_overall_and_per_task() -> None:
    assert decide_quality(_report(0.95, decision=True), "v1.0").target_met
    decision = decide_quality(_report(0.9499, decision=False), "v2.0")
    assert not decision.target_met
    assert len(decision.shortfalls) == 6


def test_quality_rejects_caller_decision_drift() -> None:
    with pytest.raises(IterationPolicyError, match="differs"):
        decide_quality(_report(0.95, decision=False), "v1.0")


def test_training_search_is_bounded_and_does_not_accept_benchmark_feedback() -> None:
    assert training_overrides(1) == {}
    second = training_overrides(2)
    assert second == {
        "epochs": 3.0,
        "learning_rate": 0.00007,
        "lora_rank": 32,
        "lora_alpha": 64,
        "seed": 20260902,
    }
    assert set(second) == {"epochs", "learning_rate", "lora_rank", "lora_alpha", "seed"}


def test_version_control_evidence_may_use_the_dedicated_remote_namespace() -> None:
    evidence = RemoteEvidence(
        remote_path=(
            "/root/ShadowCrafter/artifacts/iterations/"
            "shadowcrafter-9b/v1.0/publication/ready.json"
        ),
        local_path="reports/private/version-loop/shadowcrafter-9b/v1.0/ready.json",
    )
    assert evidence.remote_path.endswith("ready.json")
