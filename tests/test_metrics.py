import pytest

from shadowcrafter.evaluation.contamination import (
    SplitContaminationError,
    SplitIntegrityError,
    assert_no_contamination,
    assert_splits_disjoint,
    build_split_fingerprint,
    check_split_contamination,
    contamination_rate,
    split_hash,
    verify_split_fingerprint,
)
from shadowcrafter.evaluation.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    compute_classification_metrics,
    macro_f1_score,
    matthews_correlation_coefficient,
)


def test_perfect_classification_scores_one() -> None:
    metrics = compute_classification_metrics(
        ["benign", "vulnerable", "malware", "benign"],
        ["benign", "vulnerable", "malware", "benign"],
    )

    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.mcc == 1.0
    assert metrics.sample_count == 4
    assert metrics.class_count == 3


def test_balanced_metrics_expose_majority_class_failure() -> None:
    truth = ["benign"] * 9 + ["malware"]
    predictions = ["benign"] * 10

    assert accuracy_score(truth, predictions) == 0.9
    assert balanced_accuracy_score(truth, predictions) == 0.5
    assert macro_f1_score(truth, predictions) < 0.5
    assert matthews_correlation_coefficient(truth, predictions) == 0.0


def test_multiclass_mcc_handles_total_disagreement() -> None:
    assert matthews_correlation_coefficient([0, 0, 1, 1], [1, 1, 0, 0]) == -1.0


@pytest.mark.parametrize(
    ("truth", "prediction"),
    [([], []), (["safe"], []), ([], ["safe"])],
)
def test_metric_inputs_must_be_nonempty_and_aligned(
    truth: list[str], prediction: list[str]
) -> None:
    with pytest.raises(ValueError):
        compute_classification_metrics(truth, prediction)


def test_split_fingerprint_is_canonical_and_order_independent() -> None:
    first = build_split_fingerprint("test", [{"label": 1, "text": "a"}, {"text": "b"}])
    second = build_split_fingerprint("test", [{"text": "b"}, {"text": "a", "label": 1}])

    assert first.digest == second.digest
    assert first.record_hashes == second.record_hashes
    assert first.digest == split_hash([{"text": "b"}, {"label": 1, "text": "a"}])
    verify_split_fingerprint(first, [{"text": "b"}, {"label": 1, "text": "a"}])


def test_split_fingerprint_rejects_membership_change() -> None:
    fingerprint = build_split_fingerprint("blind-test", [{"id": "one"}])

    with pytest.raises(SplitIntegrityError):
        verify_split_fingerprint(fingerprint, [{"id": "one"}, {"id": "two"}])


def test_cross_split_contamination_is_reported_and_enforced() -> None:
    train = build_split_fingerprint("train", [{"id": "shared"}, {"id": "train-only"}])
    test = build_split_fingerprint("test", [{"id": "shared"}, {"id": "test-only"}])
    splits = {"train": train, "test": test}

    report = check_split_contamination(splits)
    assert report.contaminated
    assert report.overlap_count == 1
    assert report.findings[0].right_contamination_rate == 0.5
    assert contamination_rate(train.record_hashes, test.record_hashes) == 0.5
    with pytest.raises(SplitContaminationError):
        assert_no_contamination(splits)
    with pytest.raises(SplitContaminationError):
        assert_splits_disjoint(train, test)
