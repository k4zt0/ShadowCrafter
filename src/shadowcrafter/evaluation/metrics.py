"""Dependency-free classification metrics for defensive security tasks.

The implementations intentionally mirror the common definitions used by
scikit-learn while keeping the core evaluation path usable without the optional
training dependencies.  All metrics are bounded to ``[-1, 1]`` (MCC) or
``[0, 1]`` (the remaining scores).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

Label = Hashable


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """A reproducible bundle of task-level classification measurements."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    matthews_correlation_coefficient: float
    sample_count: int
    class_count: int

    @property
    def mcc(self) -> float:
        """Short name for :attr:`matthews_correlation_coefficient`."""

        return self.matthews_correlation_coefficient

    def as_dict(self) -> dict[str, float | int]:
        """Return names suitable for evaluation manifests and release gates."""

        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "mcc": self.matthews_correlation_coefficient,
            "sample_count": self.sample_count,
            "class_count": self.class_count,
        }


def _validated_inputs(
    y_true: Sequence[Label], y_pred: Sequence[Label]
) -> tuple[tuple[Label, ...], tuple[Label, ...]]:
    truth = tuple(y_true)
    predictions = tuple(y_pred)
    if len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must contain the same number of samples")
    if not truth:
        raise ValueError("classification metrics require at least one sample")
    for label in (*truth, *predictions):
        try:
            hash(label)
        except TypeError as exc:
            raise TypeError("classification labels must be hashable") from exc
    return truth, predictions


def _ordered_unique(values: Sequence[Label]) -> tuple[Label, ...]:
    # A dict preserves observation order and works for mixed hashable label types,
    # whereas sorting could fail for combinations such as integers and strings.
    return tuple(dict.fromkeys(values))


def accuracy_score(y_true: Sequence[Label], y_pred: Sequence[Label]) -> float:
    """Return the fraction of exactly correct labels."""

    truth, predictions = _validated_inputs(y_true, y_pred)
    return sum(
        expected == observed for expected, observed in zip(truth, predictions, strict=True)
    ) / len(truth)


def balanced_accuracy_score(y_true: Sequence[Label], y_pred: Sequence[Label]) -> float:
    """Return mean recall across classes represented in the reference labels.

    This prevents a high-volume benign class from hiding poor recall on rare
    vulnerability, malware, or incident classes.
    """

    truth, predictions = _validated_inputs(y_true, y_pred)
    recalls: list[float] = []
    for label in _ordered_unique(truth):
        support = sum(expected == label for expected in truth)
        true_positives = sum(
            expected == label and observed == label
            for expected, observed in zip(truth, predictions, strict=True)
        )
        recalls.append(true_positives / support)
    return sum(recalls) / len(recalls)


def macro_f1_score(y_true: Sequence[Label], y_pred: Sequence[Label]) -> float:
    """Return the unweighted mean one-vs-rest F1 across all observed labels."""

    truth, predictions = _validated_inputs(y_true, y_pred)
    labels = _ordered_unique((*truth, *predictions))
    scores: list[float] = []
    for label in labels:
        true_positives = sum(
            expected == label and observed == label
            for expected, observed in zip(truth, predictions, strict=True)
        )
        false_positives = sum(
            expected != label and observed == label
            for expected, observed in zip(truth, predictions, strict=True)
        )
        false_negatives = sum(
            expected == label and observed != label
            for expected, observed in zip(truth, predictions, strict=True)
        )
        denominator = 2 * true_positives + false_positives + false_negatives
        scores.append(0.0 if denominator == 0 else 2 * true_positives / denominator)
    return sum(scores) / len(scores)


def matthews_correlation_coefficient(y_true: Sequence[Label], y_pred: Sequence[Label]) -> float:
    """Return the generalized multiclass Matthews correlation coefficient.

    A degenerate single-class comparison has an undefined denominator.  As in
    common ML tooling, it is reported as ``0.0`` rather than as NaN so release
    manifests remain valid JSON.
    """

    truth, predictions = _validated_inputs(y_true, y_pred)
    labels = _ordered_unique((*truth, *predictions))
    true_totals = {label: 0 for label in labels}
    predicted_totals = {label: 0 for label in labels}
    correct = 0

    for expected, observed in zip(truth, predictions, strict=True):
        true_totals[expected] += 1
        predicted_totals[observed] += 1
        correct += expected == observed

    sample_count = len(truth)
    covariance = correct * sample_count - sum(
        predicted_totals[label] * true_totals[label] for label in labels
    )
    predicted_variance = sample_count**2 - sum(count**2 for count in predicted_totals.values())
    true_variance = sample_count**2 - sum(count**2 for count in true_totals.values())
    denominator = math.sqrt(predicted_variance * true_variance)
    if denominator == 0:
        return 0.0
    # Floating-point roundoff can otherwise yield values a few ulps outside the
    # mathematical interval.
    return max(-1.0, min(1.0, covariance / denominator))


def compute_classification_metrics(
    y_true: Sequence[Label], y_pred: Sequence[Label]
) -> ClassificationMetrics:
    """Compute the standard defensive classification metric bundle."""

    truth, predictions = _validated_inputs(y_true, y_pred)
    return ClassificationMetrics(
        accuracy=accuracy_score(truth, predictions),
        balanced_accuracy=balanced_accuracy_score(truth, predictions),
        macro_f1=macro_f1_score(truth, predictions),
        matthews_correlation_coefficient=matthews_correlation_coefficient(truth, predictions),
        sample_count=len(truth),
        class_count=len(_ordered_unique((*truth, *predictions))),
    )


# Descriptive alias retained for callers that organize evaluation by task type.
evaluate_defensive_classification = compute_classification_metrics
macro_f1 = macro_f1_score
matthews_corrcoef = matthews_correlation_coefficient
mcc_score = matthews_correlation_coefficient
