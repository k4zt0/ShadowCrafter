"""Fail-closed version and quality policy for repeated ShadowCrafter-9B runs."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

QUALITY_TARGET = 0.95
_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
_VERSION = re.compile(r"^v([1-9][0-9]*)\.0$")


class IterationPolicyError(ValueError):
    """An iteration report or version violates the registered loop policy."""


@dataclass(frozen=True, slots=True)
class QualityDecision:
    version: str
    target: float
    target_met: bool
    overall: dict[str, float]
    task_metrics: dict[str, dict[str, float]]
    shortfalls: tuple[str, ...]


def version_for(index: int) -> str:
    """Return the immutable major-only release label for a one-based iteration."""

    if not isinstance(index, int) or isinstance(index, bool) or index < 1:
        raise IterationPolicyError("iteration index must be a positive integer")
    return f"v{index}.0"


def version_index(version: str) -> int:
    match = _VERSION.fullmatch(version)
    if match is None:
        raise IterationPolicyError("release version must use v<positive-major>.0")
    return int(match.group(1))


def training_overrides(index: int) -> dict[str, int | float]:
    """Return a bounded, deterministic search schedule without benchmark-derived data."""

    version_for(index)
    schedules: tuple[dict[str, int | float], ...] = (
        {"epochs": 3.0, "learning_rate": 0.00007, "lora_rank": 32, "lora_alpha": 64},
        {"epochs": 2.0, "learning_rate": 0.00007, "lora_rank": 64, "lora_alpha": 128},
        {"epochs": 3.0, "learning_rate": 0.00005, "lora_rank": 64, "lora_alpha": 128},
        {"epochs": 2.0, "learning_rate": 0.00005, "lora_rank": 96, "lora_alpha": 192},
        {"epochs": 3.0, "learning_rate": 0.00003, "lora_rank": 96, "lora_alpha": 192},
    )
    selected = dict(schedules[(index - 2) % len(schedules)]) if index >= 2 else {}
    if selected:
        selected["seed"] = 20260901 + index - 1
    return selected


def _metrics(scope: Mapping[str, Any], label: str) -> dict[str, float]:
    raw = scope.get("metrics")
    if not isinstance(raw, Mapping) or set(raw) != set(_METRICS):
        raise IterationPolicyError(f"{label} has an invalid metric surface")
    result: dict[str, float] = {}
    for name in _METRICS:
        value = raw.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise IterationPolicyError(f"{label}.{name} is not a finite probability")
        result[name] = float(value)
    return result


def decide_quality(report: Mapping[str, Any], version: str) -> QualityDecision:
    """Recompute the 95% stop decision from a frozen gate report."""

    version_index(version)
    if report.get("passed") is not True or report.get("failures") not in ([], ()):
        raise IterationPolicyError("integrity-failed evaluation cannot drive retraining")
    overall_scope = report.get("overall")
    task_scopes = report.get("tasks")
    if not isinstance(overall_scope, Mapping) or not isinstance(task_scopes, Mapping):
        raise IterationPolicyError("evaluation report lacks overall or per-task metrics")
    overall = _metrics(overall_scope, "overall")
    tasks: dict[str, dict[str, float]] = {}
    for task, scope in sorted(task_scopes.items()):
        if not isinstance(task, str) or not task or not isinstance(scope, Mapping):
            raise IterationPolicyError("evaluation report has an invalid task entry")
        tasks[task] = _metrics(scope, task)
    if not tasks:
        raise IterationPolicyError("evaluation report contains no task metrics")

    shortfalls = [
        f"overall {name} {value:.6f} < target {QUALITY_TARGET:.6f}"
        for name, value in overall.items()
        if value < QUALITY_TARGET
    ]
    for task, values in tasks.items():
        shortfalls.extend(
            f"{task} {name} {value:.6f} < target {QUALITY_TARGET:.6f}"
            for name, value in values.items()
            if value < QUALITY_TARGET
        )
    target_met = not shortfalls
    if report.get("quality_target_met") is not target_met:
        raise IterationPolicyError("gate quality decision differs from the 95% loop decision")
    if report.get("target_95_met") is not target_met:
        raise IterationPolicyError("gate report lacks the exact target_95_met decision")
    return QualityDecision(
        version=version,
        target=QUALITY_TARGET,
        target_met=target_met,
        overall=overall,
        task_metrics=tasks,
        shortfalls=tuple(shortfalls),
    )


__all__ = [
    "IterationPolicyError",
    "QUALITY_TARGET",
    "QualityDecision",
    "decide_quality",
    "training_overrides",
    "version_for",
    "version_index",
]
