"""Fail-closed numerical and frozen-evidence release gates.

``evaluate_gate`` is retained for small, in-process metric checks. It is not a
release authorization boundary. ``load_and_evaluate`` is the release boundary:
it recomputes every configured classification metric from immutable benchmark
cases and raw predictions, verifies the referenced manifests and training data,
and independently scans the supplied training records for CTIBench leakage.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from shadowcrafter.data.ctibench import CTIBenchEvalCase, CTIBenchTask
from shadowcrafter.evaluation.metrics import compute_classification_metrics
from shadowcrafter.schemas import EvalMetric, SecurityRecord

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_SHA_PATTERN = r"^[0-9a-f]{40}$"
_SAFE_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,199}$"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_EVALUATION_BYTES = 512 * 1024 * 1024
_MAX_TRAINING_BYTES = 4 * 1024 * 1024 * 1024
_MAX_JSONL_LINE_BYTES = 2 * 1024 * 1024
_INVALID_OUTPUT = "<INVALID_OUTPUT>"
_RELEASE_PROTOCOL = "shadowcrafter-frozen-release-evaluation-v1"
_CONTAMINATION_ALGORITHM = "ctibench-normalized-content-exact-and-containment-v1"


def wilson_lower_bound(success_rate: float, sample_count: int, z: float = 1.96) -> float:
    """Return the two-sided Wilson interval's lower bound for a proportion."""

    if sample_count <= 0:
        return 0.0
    denominator = 1 + z * z / sample_count
    centre = success_rate + z * z / (2 * sample_count)
    margin = z * math.sqrt(
        success_rate * (1 - success_rate) / sample_count + z * z / (4 * sample_count**2)
    )
    return max(0.0, (centre - margin) / denominator)


@dataclass(frozen=True)
class GateResult:
    """Release decision plus a non-sensitive, reproducible aggregate report."""

    passed: bool
    failures: tuple[str, ...]
    report: Mapping[str, Any] | None = None


def evaluate_gate(metrics: list[EvalMetric], gate_config: dict[str, Any]) -> GateResult:
    """Evaluate caller-supplied aggregates for development dashboards only."""

    by_name = {metric.name: metric for metric in metrics}
    failures: list[str] = []
    max_contamination = float(gate_config.get("max_contamination_rate", 0.0))
    for name, rule in gate_config["metrics"].items():
        metric = by_name.get(name)
        if metric is None:
            failures.append(f"missing metric: {name}")
            continue
        minimum_samples = int(rule.get("minimum_samples", 1))
        if metric.sample_count < minimum_samples:
            failures.append(
                f"{name}: sample_count {metric.sample_count} < required {minimum_samples}"
            )
        if metric.contamination_rate > max_contamination:
            failures.append(
                f"{name}: contamination {metric.contamination_rate:.4f} > {max_contamination:.4f}"
            )
        threshold = float(rule["threshold"])
        observed = (
            wilson_lower_bound(metric.value, metric.sample_count)
            if rule.get("use_wilson_lower_bound", False)
            else metric.value
        )
        if observed < threshold:
            failures.append(f"{name}: gated value {observed:.4f} < threshold {threshold:.4f}")
    return GateResult(passed=not failures, failures=tuple(failures))


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class FileReference(_StrictModel):
    """A bundle-relative immutable file reference."""

    path: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=_SHA256_PATTERN)


class RecordFileReference(FileReference):
    record_count: int = Field(ge=1, le=20_000_000)


class TrainingCorpusReference(_StrictModel):
    records: RecordFileReference
    prepared_manifest: FileReference
    dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    split: Literal["train"] = "train"


class CandidateEvidence(_StrictModel):
    candidate_id: str = Field(pattern=_SAFE_IDENTIFIER_PATTERN)
    model_family: Literal["ShadowCrafter-9B"]
    model_id: str = Field(pattern=_SAFE_IDENTIFIER_PATTERN)
    base_model_id: str = Field(pattern=_SAFE_IDENTIFIER_PATTERN)
    base_model_revision: str = Field(pattern=_GIT_SHA_PATTERN)
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint_manifest: FileReference
    training_run_manifest: FileReference
    shadowcrafter_git_commit: str = Field(pattern=_GIT_SHA_PATTERN)
    git_tree_clean: Literal[True]

    @model_validator(mode="after")
    def bind_checkpoint_identity(self) -> CandidateEvidence:
        if not hmac.compare_digest(self.checkpoint_sha256, self.checkpoint_manifest.sha256):
            raise ValueError("checkpoint_sha256 must equal the checkpoint manifest digest")
        return self


class InferenceEvidence(_StrictModel):
    evaluator_version: str = Field(pattern=_SAFE_IDENTIFIER_PATTERN)
    code_revision: str = Field(pattern=_GIT_SHA_PATTERN)
    environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_template_sha256: str = Field(pattern=_SHA256_PATTERN)
    decoding_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    seed: int = Field(ge=0, le=2**63 - 1)
    started_at_utc: datetime
    completed_at_utc: datetime
    deterministic_decoding: Literal[True]
    answer_key_hidden_from_model: Literal[True]
    raw_outputs_retained: Literal[True]

    @model_validator(mode="after")
    def validate_times(self) -> InferenceEvidence:
        if self.started_at_utc.tzinfo is None or self.completed_at_utc.tzinfo is None:
            raise ValueError("inference timestamps must be timezone-aware")
        if self.completed_at_utc < self.started_at_utc:
            raise ValueError("inference completed_at_utc precedes started_at_utc")
        return self


class BenchmarkEvidence(_StrictModel):
    benchmark_id: Literal["ctibench"]
    repository_id: Literal["AI4Sec/cti-bench"]
    upstream_revision: str = Field(pattern=_GIT_SHA_PATTERN)
    license_id: Literal["CC-BY-NC-SA-4.0"]
    evaluation_only: Literal[True]
    benchmark_holdout: Literal[True]
    cases: RecordFileReference
    adapter_manifest: FileReference
    snapshot_manifest: FileReference
    dataset_sha256: str = Field(pattern=_SHA256_PATTERN)


class PredictionsEvidence(RecordFileReference):
    frozen: Literal[True]


class ContaminationEvidence(_StrictModel):
    algorithm: Literal["ctibench-normalized-content-exact-and-containment-v1"]
    declared_overlap_count: Literal[0]
    scanned_training_record_count: int = Field(ge=1, le=20_000_000)


class BenchmarkLicenseEvidence(_StrictModel):
    """Explicitly prevent an NC benchmark from silently authorizing distribution."""

    usage_scope: Literal["noncommercial-private-research"]
    commercial_use_permitted: Literal[False]
    commercial_release_requested: Literal[False]
    private_evidence_only: Literal[True]
    attribution_retained: Literal[True]
    share_alike_review_required_before_sharing: Literal[True]
    license_review_sha256: str = Field(pattern=_SHA256_PATTERN)


class BlockingReviewEvidence(_StrictModel):
    """One mandatory non-quality approval, bound to its immutable report."""

    passed: Literal[True]
    report: FileReference


class ExperimentalPublicationEvidence(_StrictModel):
    """Private experimental publication controls; quality is deliberately non-blocking."""

    release_tier: Literal["Experimental Release"]
    visibility: Literal["private"]
    public_release_requested: Literal[False]
    commercial_release_requested: Literal[False]
    quality_target_is_publication_blocker: Literal[False]
    model_card_reports_evaluation: Literal[True]
    model_card_labels_experimental: Literal[True]
    artifact_integrity_review: BlockingReviewEvidence
    provenance_review: BlockingReviewEvidence
    license_review: BlockingReviewEvidence
    privacy_review: BlockingReviewEvidence
    safety_review: BlockingReviewEvidence


class FrozenReleaseEvidence(_StrictModel):
    schema_version: Literal[1]
    protocol: Literal["shadowcrafter-frozen-release-evaluation-v1"]
    evaluation_id: str = Field(pattern=_SAFE_IDENTIFIER_PATTERN)
    created_at_utc: datetime
    candidate: CandidateEvidence
    inference: InferenceEvidence
    benchmark: BenchmarkEvidence
    predictions: PredictionsEvidence
    training_corpora: list[TrainingCorpusReference] = Field(min_length=1, max_length=128)
    contamination: ContaminationEvidence
    benchmark_license: BenchmarkLicenseEvidence
    publication: ExperimentalPublicationEvidence

    @model_validator(mode="after")
    def validate_bundle(self) -> FrozenReleaseEvidence:
        if self.created_at_utc.tzinfo is None:
            raise ValueError("created_at_utc must be timezone-aware")
        if self.created_at_utc < self.inference.completed_at_utc:
            raise ValueError("evidence cannot be created before inference completed")
        paths = [
            self.candidate.checkpoint_manifest.path,
            self.candidate.training_run_manifest.path,
            self.benchmark.cases.path,
            self.benchmark.adapter_manifest.path,
            self.benchmark.snapshot_manifest.path,
            self.predictions.path,
        ]
        for corpus in self.training_corpora:
            paths.extend((corpus.records.path, corpus.prepared_manifest.path))
        paths.extend(
            (
                self.publication.artifact_integrity_review.report.path,
                self.publication.provenance_review.report.path,
                self.publication.license_review.report.path,
                self.publication.privacy_review.report.path,
                self.publication.safety_review.report.path,
            )
        )
        if len(paths) != len(set(paths)):
            raise ValueError("every evidence file reference must use a distinct path")
        return self


class TaskGateRule(_StrictModel):
    sample_count: int = Field(ge=2, le=20_000_000)
    minimum_reference_classes: int = Field(default=2, ge=2, le=1_000_000)


class MetricThresholds(_StrictModel):
    accuracy: float = Field(ge=0.94, le=1.0)
    balanced_accuracy: float = Field(ge=0.94, le=1.0)
    macro_f1: float = Field(ge=0.94, le=1.0)


class CandidateRule(_StrictModel):
    model_id: str = Field(pattern=_SAFE_IDENTIFIER_PATTERN)
    base_model_id: str = Field(pattern=_SAFE_IDENTIFIER_PATTERN)
    base_model_revision: str = Field(pattern=_GIT_SHA_PATTERN)


class BenchmarkRule(_StrictModel):
    benchmark_id: Literal["ctibench"]
    repository_id: Literal["AI4Sec/cti-bench"]
    upstream_revision: str = Field(pattern=_GIT_SHA_PATTERN)
    license_id: Literal["CC-BY-NC-SA-4.0"]
    snapshot_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    adapter_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    cases_sha256: str = Field(pattern=_SHA256_PATTERN)
    dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_sample_count: int = Field(ge=2, le=20_000_000)
    tasks: dict[CTIBenchTask, TaskGateRule] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts(self) -> BenchmarkRule:
        configured_total = sum(rule.sample_count for rule in self.tasks.values())
        if configured_total != self.expected_sample_count:
            raise ValueError("benchmark task sample counts do not equal expected_sample_count")
        return self


class StrictReleaseGateConfig(_StrictModel):
    schema_version: Literal[2]
    protocol: Literal["shadowcrafter-frozen-release-evaluation-v1"]
    claim: str = Field(min_length=1, max_length=500)
    metric_thresholds: MetricThresholds
    require_per_task_metrics: Literal[True]
    max_contamination_overlap_count: Literal[0]
    contamination_algorithm: Literal["ctibench-normalized-content-exact-and-containment-v1"]
    evaluator_version: str = Field(pattern=_SAFE_IDENTIFIER_PATTERN)
    require_clean_git: Literal[True]
    quality_target_is_publication_blocker: Literal[False]
    authorization_scope: Literal["noncommercial-private-experimental-release"]
    commercial_use_permitted: Literal[False]
    model_publication_authorized: Literal[True]
    required_visibility: Literal["private"]
    public_publication_authorized: Literal[False]
    release_tier: Literal["Experimental Release"]
    benchmark: BenchmarkRule
    allowed_candidates: dict[Literal["ShadowCrafter-9B"], CandidateRule] = Field(
        min_length=1,
        max_length=1,
    )


class FrozenPrediction(_StrictModel):
    """One raw model output; answer keys and caller-computed scores are forbidden."""

    # Preserve raw model bytes semantically; release evidence must not normalize
    # an output before verifying its per-record digest.
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    case_id: str = Field(pattern=r"^ctibench:[a-z0-9-]+:[0-9]{6}$")
    raw_output: str = Field(min_length=1, max_length=4_096)
    raw_output_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_raw_hash(self) -> FrozenPrediction:
        expected = hashlib.sha256(self.raw_output.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(expected, self.raw_output_sha256):
            raise ValueError("raw_output_sha256 does not match raw_output")
        return self


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_json_object(content: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(content, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _safe_bundle_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe evidence path: {relative!r}")
    candidate = root.joinpath(*pure.parts)
    current = root
    for component in pure.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"evidence paths may not contain symlinks: {relative!r}")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"evidence file does not exist: {relative!r}") from exc
    if resolved.parent != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"evidence path escapes its bundle: {relative!r}")
    if not resolved.is_file():
        raise ValueError(f"evidence reference is not a regular file: {relative!r}")
    return resolved


def _read_verified_file(
    root: Path,
    reference: FileReference,
    *,
    maximum_bytes: int,
) -> bytes:
    path = _safe_bundle_path(root, reference.path)
    with path.open("rb") as handle:
        initial = os.fstat(handle.fileno())
        if initial.st_size > maximum_bytes:
            raise ValueError(f"evidence file exceeds size bound: {reference.path}")
        content = handle.read(maximum_bytes + 1)
        final = os.fstat(handle.fileno())
    if len(content) > maximum_bytes:
        raise ValueError(f"evidence file exceeds size bound: {reference.path}")
    identity_before = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
    identity_after = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
    if identity_before != identity_after or len(content) != initial.st_size:
        raise ValueError(f"evidence file changed while it was read: {reference.path}")
    actual_sha256 = _sha256_bytes(content)
    if not hmac.compare_digest(actual_sha256, reference.sha256):
        raise ValueError(f"evidence file checksum mismatch: {reference.path}")
    return content


def _load_jsonl_models(
    content: bytes,
    *,
    reference: RecordFileReference,
    model: type[BaseModel],
    description: str,
) -> list[BaseModel]:
    records: list[BaseModel] = []
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"{description} contains a blank line at {line_number}")
        if len(raw_line) > _MAX_JSONL_LINE_BYTES:
            raise ValueError(f"{description} line {line_number} exceeds its size bound")
        try:
            payload = json.loads(raw_line, object_pairs_hook=_reject_duplicate_json_keys)
            records.append(model.model_validate(payload))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, ValidationError) as exc:
            raise ValueError(f"invalid {description} record at line {line_number}: {exc}") from exc
    if len(records) != reference.record_count:
        raise ValueError(
            f"{description} record count {len(records)} does not match frozen count "
            f"{reference.record_count}"
        )
    return records


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _normalized_sha256(value: str) -> str:
    return hashlib.sha256(_normalize_text(value).encode("utf-8")).hexdigest()


def _validate_config(raw: object) -> StrictReleaseGateConfig:
    if not isinstance(raw, dict) or set(raw) != {"release_gate"}:
        raise ValueError("gate config must contain only the release_gate object")
    return StrictReleaseGateConfig.model_validate(raw["release_gate"])


def _verify_candidate(evidence: FrozenReleaseEvidence, config: StrictReleaseGateConfig) -> None:
    rule = config.allowed_candidates.get(evidence.candidate.model_family)
    if rule is None:
        raise ValueError("candidate model family is not allowed by the release protocol")
    observed = (
        evidence.candidate.model_id,
        evidence.candidate.base_model_id,
        evidence.candidate.base_model_revision,
    )
    expected = (rule.model_id, rule.base_model_id, rule.base_model_revision)
    if observed != expected:
        raise ValueError(
            "candidate model identity or pinned upstream revision does not match config"
        )
    if config.require_clean_git and not evidence.candidate.git_tree_clean:
        raise ValueError("release evaluation requires a clean Git tree")
    if evidence.inference.evaluator_version != config.evaluator_version:
        raise ValueError("evaluator version does not match the frozen gate config")


def _verify_blocking_reviews(
    root: Path,
    evidence: FrozenReleaseEvidence,
) -> dict[str, str]:
    reviews = {
        "artifact_integrity": evidence.publication.artifact_integrity_review,
        "provenance": evidence.publication.provenance_review,
        "license": evidence.publication.license_review,
        "privacy": evidence.publication.privacy_review,
        "safety": evidence.publication.safety_review,
    }
    hashes: dict[str, str] = {}
    for name, review in reviews.items():
        content = _read_verified_file(
            root,
            review.report,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
        payload = _load_json_object(content, f"{name} review report")
        if (
            payload.get("schema_version") != 1
            or payload.get("review") != name
            or payload.get("passed") is not True
            or payload.get("candidate_checkpoint_sha256") != evidence.candidate.checkpoint_sha256
        ):
            raise ValueError(f"{name} review is missing, failed, or bound to another candidate")
        if name == "license" and (
            payload.get("commercial_release_authorized") is not False
            or payload.get("benchmark_material_sharing_authorized") is not False
        ):
            raise ValueError("license review must preserve CTIBench NC and no-sharing limits")
        hashes[name] = review.report.sha256
    return hashes


def _verify_snapshot_manifest(
    content: bytes,
    evidence: FrozenReleaseEvidence,
    config: StrictReleaseGateConfig,
) -> dict[str, dict[str, Any]]:
    manifest = _load_json_object(content, "CTIBench snapshot manifest")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("CTIBench snapshot manifest lacks source provenance")
    license_info = source.get("license")
    if not isinstance(license_info, dict):
        raise ValueError("CTIBench snapshot manifest lacks license provenance")
    expected = config.benchmark
    if (
        source.get("id") != expected.benchmark_id
        or source.get("repo_id") != expected.repository_id
        or source.get("policy_class") != "eval_only"
        or license_info.get("id") != expected.license_id
        or manifest.get("upstream_revision") != expected.upstream_revision
    ):
        raise ValueError("CTIBench snapshot identity, license, policy, or revision mismatch")
    files_raw = manifest.get("files")
    if not isinstance(files_raw, list) or not files_raw:
        raise ValueError("CTIBench snapshot manifest lacks its file inventory")
    files: dict[str, dict[str, Any]] = {}
    for item in files_raw:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("CTIBench snapshot contains an invalid file entry")
        path = item["path"]
        digest = item.get("sha256")
        if (
            path in files
            or not isinstance(digest, str)
            or re.fullmatch(_SHA256_PATTERN, digest) is None
        ):
            raise ValueError("CTIBench snapshot file inventory is duplicated or unhashed")
        files[path] = item
    return files


def _verify_adapter_manifest(
    content: bytes,
    evidence: FrozenReleaseEvidence,
    config: StrictReleaseGateConfig,
) -> dict[str, Any]:
    manifest = _load_json_object(content, "CTIBench adapter manifest")
    output = manifest.get("output")
    snapshot = manifest.get("snapshot_manifest")
    statistics = manifest.get("statistics")
    controls = manifest.get("controls")
    license_info = manifest.get("license")
    if not all(
        isinstance(value, dict) for value in (license_info, output, snapshot, statistics, controls)
    ):
        raise ValueError("CTIBench adapter manifest is structurally incomplete")
    assert isinstance(license_info, dict)
    assert isinstance(output, dict)
    assert isinstance(snapshot, dict)
    assert isinstance(statistics, dict)
    assert isinstance(controls, dict)
    expected = config.benchmark
    if (
        manifest.get("source_id") != expected.benchmark_id
        or manifest.get("repo_id") != expected.repository_id
        or manifest.get("source_policy_class") != "eval_only"
        or manifest.get("license_id") != expected.license_id
        or license_info.get("id") != expected.license_id
        or license_info.get("noncommercial_only") is not True
        or license_info.get("attribution_required_when_shared") is not True
        or license_info.get("share_alike_required_when_adapted_material_is_shared") is not True
        or manifest.get("upstream_revision") != expected.upstream_revision
    ):
        raise ValueError("CTIBench adapter provenance does not match the gate config")
    if (
        output.get("sha256") != evidence.benchmark.cases.sha256
        or output.get("record_count") != evidence.benchmark.cases.record_count
        or snapshot.get("sha256") != evidence.benchmark.snapshot_manifest.sha256
    ):
        raise ValueError("CTIBench adapter output or snapshot hash is not frozen consistently")
    expected_dataset_sha256 = _canonical_json_sha256(
        {
            "snapshot_manifest_sha256": evidence.benchmark.snapshot_manifest.sha256,
            "output_sha256": evidence.benchmark.cases.sha256,
        }
    )
    if (
        manifest.get("dataset_sha256") != expected_dataset_sha256
        or evidence.benchmark.dataset_sha256 != expected_dataset_sha256
    ):
        raise ValueError("CTIBench dataset fingerprint is inconsistent")
    required_controls: dict[str, object] = {
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
    }
    if any(controls.get(name) is not value for name, value in required_controls.items()):
        raise ValueError("CTIBench adapter safety controls are incomplete or disabled")
    by_task = statistics.get("by_task")
    if not isinstance(by_task, dict):
        raise ValueError("CTIBench adapter manifest lacks per-task counts")
    expected_counts = {str(task): rule.sample_count for task, rule in expected.tasks.items()}
    if (
        by_task != expected_counts
        or statistics.get("emitted_count") != expected.expected_sample_count
    ):
        raise ValueError("CTIBench adapter task counts do not match the fixed benchmark")
    return manifest


def _verify_cases(
    cases: Sequence[CTIBenchEvalCase],
    evidence: FrozenReleaseEvidence,
    config: StrictReleaseGateConfig,
    snapshot_files: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(cases) != config.benchmark.expected_sample_count:
        raise ValueError("CTIBench case count does not match the fixed benchmark config")
    counts = Counter(case.task for case in cases)
    expected_counts = {task: rule.sample_count for task, rule in config.benchmark.tasks.items()}
    if counts != expected_counts:
        raise ValueError("CTIBench case task membership does not match the frozen config")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("CTIBench case IDs are not unique")
    if len({case.content_sha256 for case in cases}) != len(cases):
        raise ValueError("CTIBench case content hashes are not unique")
    for case in cases:
        provenance = case.provenance
        file_info = snapshot_files.get(provenance.source_file)
        if (
            provenance.upstream_revision != evidence.benchmark.upstream_revision
            or file_info is None
            or file_info.get("sha256") != provenance.source_file_sha256
        ):
            raise ValueError("CTIBench case provenance does not resolve to the frozen snapshot")
    for task, rule in config.benchmark.tasks.items():
        reference_classes = {
            _canonical_answer(task, case.answer) for case in cases if case.task == task
        }
        if len(reference_classes) < rule.minimum_reference_classes:
            raise ValueError(f"{task}: reference class count is below its configured minimum")


def _verify_training_manifest(
    content: bytes,
    corpus: TrainingCorpusReference,
) -> None:
    manifest = _load_json_object(content, "prepared training manifest")
    if manifest.get("dataset_sha256") != corpus.dataset_sha256:
        raise ValueError("training dataset hash does not match its prepared manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(corpus.split), dict):
        raise ValueError("prepared training manifest lacks the frozen train artifact")
    artifact = artifacts[corpus.split]
    if (
        artifact.get("sha256") != corpus.records.sha256
        or artifact.get("record_count") != corpus.records.record_count
    ):
        raise ValueError("training file hash or count does not match its prepared manifest")
    artifact_hashes: dict[str, str] = {}
    for split_name in ("evaluation", "test", "train", "validation"):
        split_artifact = artifacts.get(split_name)
        if not isinstance(split_artifact, dict):
            raise ValueError("prepared training manifest has an incomplete artifact inventory")
        digest = split_artifact.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(_SHA256_PATTERN, digest) is None:
            raise ValueError("prepared training manifest contains an invalid artifact hash")
        artifact_hashes[split_name] = digest
    if manifest.get("dataset_sha256") != _canonical_json_sha256(artifact_hashes):
        raise ValueError("prepared training dataset fingerprint is inconsistent")
    split_counts = manifest.get("split_counts")
    if (
        not isinstance(split_counts, dict)
        or split_counts.get(corpus.split) != corpus.records.record_count
    ):
        raise ValueError("training split count does not match its prepared manifest")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("prepared training manifest lacks source provenance")
    for source in sources:
        if (
            not isinstance(source, dict)
            or source.get("source_id") == "ctibench"
            or source.get("policy_class") != "allow_train"
            or "train" not in source.get("allowed_purposes", [])
        ):
            raise ValueError("training source is unproven, eval-only, or not approved for training")


def _scan_training_records(
    training_contents: Sequence[tuple[TrainingCorpusReference, bytes]],
    cases: Sequence[CTIBenchEvalCase],
) -> tuple[int, int]:
    benchmark_hashes = {
        digest
        for case in cases
        for digest in (
            case.provenance.source_prompt_normalized_sha256,
            case.provenance.input_normalized_sha256,
            case.provenance.rendered_input_normalized_sha256,
        )
    }
    benchmark_texts = {
        _normalize_text(text)
        for case in cases
        for text in (
            case.input_text,
            "\n".join(
                (
                    case.input_text,
                    *(f"{label}. {case.choices[label]}" for label in ("A", "B", "C", "D")),
                )
            )
            if case.choices is not None
            else case.input_text,
        )
    }
    seen_record_ids: set[str] = set()
    scanned = 0
    overlaps = 0
    for corpus, content in training_contents:
        records = _load_jsonl_models(
            content,
            reference=corpus.records,
            model=SecurityRecord,
            description="training corpus",
        )
        for typed in records:
            assert isinstance(typed, SecurityRecord)
            record = typed
            if record.record_id in seen_record_ids:
                raise ValueError("duplicate training record ID appears across release corpora")
            seen_record_ids.add(record.record_id)
            scanned += 1
            if record.provenance.content_sha256.lower() != record.canonical_hash():
                raise ValueError("training record provenance checksum does not match its content")
            if record.benchmark_holdout or record.provenance.source_id == "ctibench":
                overlaps += 1
                continue
            contaminated = False
            for message in record.messages:
                normalized = _normalize_text(message.content)
                if _normalized_sha256(normalized) in benchmark_hashes or any(
                    benchmark_text and benchmark_text in normalized
                    for benchmark_text in benchmark_texts
                ):
                    contaminated = True
                    break
            overlaps += contaminated
    return scanned, overlaps


def _canonical_answer(task: CTIBenchTask, value: str) -> str:
    candidate = " ".join(value.strip().split())
    if task == CTIBenchTask.MULTIPLE_CHOICE:
        normalized = candidate.upper()
        return normalized if re.fullmatch(r"[ABCD]", normalized) else _INVALID_OUTPUT
    if task in {CTIBenchTask.CWE_MAPPING, CTIBenchTask.CWE_MAPPING_2021}:
        parts = [part.strip().upper() for part in candidate.split(",")]
        if (
            not parts
            or len(parts) != len(set(parts))
            or any(re.fullmatch(r"CWE-[0-9]+", part) is None for part in parts)
        ):
            return _INVALID_OUTPUT
        return ", ".join(sorted(parts, key=lambda item: int(item.removeprefix("CWE-"))))
    if task == CTIBenchTask.ATTACK_TECHNIQUE_EXTRACTION:
        parts = [part.strip().upper() for part in candidate.split(",")]
        if (
            not parts
            or len(parts) != len(set(parts))
            or any(re.fullmatch(r"T[0-9]{4}", part) is None for part in parts)
        ):
            return _INVALID_OUTPUT
        return ", ".join(sorted(parts))
    if task == CTIBenchTask.CVSS_VECTOR:
        normalized = candidate.upper()
        pattern = (
            r"CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/"
            r"C:[NLH]/I:[NLH]/A:[NLH]"
        )
        return normalized if re.fullmatch(pattern, normalized) else _INVALID_OUTPUT
    raise ValueError(f"unsupported CTIBench task: {task}")


def _class_rows(y_true: Sequence[str], y_pred: Sequence[str]) -> list[dict[str, Any]]:
    labels = tuple(dict.fromkeys((*y_true, *y_pred)))
    rows: list[dict[str, Any]] = []
    for label in labels:
        support = sum(value == label for value in y_true)
        predicted_count = sum(value == label for value in y_pred)
        true_positives = sum(
            expected == label and observed == label
            for expected, observed in zip(y_true, y_pred, strict=True)
        )
        false_positives = predicted_count - true_positives
        false_negatives = support - true_positives
        precision = true_positives / (true_positives + false_positives) if predicted_count else 0.0
        recall = true_positives / support if support else 0.0
        denominator = 2 * true_positives + false_positives + false_negatives
        f1 = 0.0 if denominator == 0 else 2 * true_positives / denominator
        rows.append(
            {
                "label": label,
                "support": support,
                "predicted_count": predicted_count,
                "true_positives": true_positives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def _scope_report(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    metrics = compute_classification_metrics(y_true, y_pred)
    return {
        "sample_count": metrics.sample_count,
        "reference_class_count": len(set(y_true)),
        "observed_class_count": metrics.class_count,
        "metrics": {
            "accuracy": metrics.accuracy,
            "balanced_accuracy": metrics.balanced_accuracy,
            "macro_f1": metrics.macro_f1,
        },
        "classes": _class_rows(y_true, y_pred),
    }


def _evaluate_predictions(
    cases: Sequence[CTIBenchEvalCase],
    predictions: Sequence[FrozenPrediction],
    config: StrictReleaseGateConfig,
) -> tuple[dict[str, Any], list[str]]:
    expected_ids = [case.case_id for case in cases]
    observed_ids = [prediction.case_id for prediction in predictions]
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("prediction case IDs are duplicated")
    if observed_ids != expected_ids:
        raise ValueError("predictions are missing, reordered, or contain non-benchmark case IDs")

    truth_by_task: dict[CTIBenchTask, list[str]] = {task: [] for task in config.benchmark.tasks}
    prediction_by_task: dict[CTIBenchTask, list[str]] = {
        task: [] for task in config.benchmark.tasks
    }
    overall_truth: list[str] = []
    overall_predictions: list[str] = []
    for case, prediction in zip(cases, predictions, strict=True):
        expected = _canonical_answer(case.task, case.answer)
        if expected == _INVALID_OUTPUT:
            raise ValueError(f"benchmark answer has invalid syntax for {case.task}")
        observed = _canonical_answer(case.task, prediction.raw_output)
        truth_by_task[case.task].append(expected)
        prediction_by_task[case.task].append(observed)
        overall_truth.append(f"{case.task}\x1f{expected}")
        overall_predictions.append(f"{case.task}\x1f{observed}")

    quality_shortfalls: list[str] = []
    thresholds = config.metric_thresholds.model_dump()
    overall = _scope_report(overall_truth, overall_predictions)
    for metric_name, threshold in thresholds.items():
        value = overall["metrics"][metric_name]
        if value < threshold:
            quality_shortfalls.append(f"overall {metric_name} {value:.6f} < target {threshold:.6f}")

    tasks: dict[str, Any] = {}
    for task in config.benchmark.tasks:
        task_report = _scope_report(truth_by_task[task], prediction_by_task[task])
        tasks[str(task)] = task_report
        if config.require_per_task_metrics:
            for metric_name, threshold in thresholds.items():
                value = task_report["metrics"][metric_name]
                if value < threshold:
                    quality_shortfalls.append(
                        f"{task} {metric_name} {value:.6f} < target {threshold:.6f}"
                    )
    return {"overall": overall, "tasks": tasks}, quality_shortfalls


def _strict_evaluate(
    evidence_content: bytes,
    evidence_path: Path,
    config: StrictReleaseGateConfig,
) -> GateResult:
    evidence_raw = _load_json_object(evidence_content, "release evidence manifest")
    evidence = FrozenReleaseEvidence.model_validate(evidence_raw)
    if evidence.protocol != config.protocol:
        raise ValueError("release evidence protocol does not match gate config")
    _verify_candidate(evidence, config)
    benchmark_rule = config.benchmark
    if (
        evidence.benchmark.benchmark_id != benchmark_rule.benchmark_id
        or evidence.benchmark.repository_id != benchmark_rule.repository_id
        or evidence.benchmark.upstream_revision != benchmark_rule.upstream_revision
        or evidence.benchmark.license_id != benchmark_rule.license_id
        or evidence.benchmark.snapshot_manifest.sha256 != benchmark_rule.snapshot_manifest_sha256
        or evidence.benchmark.adapter_manifest.sha256 != benchmark_rule.adapter_manifest_sha256
        or evidence.benchmark.cases.sha256 != benchmark_rule.cases_sha256
        or evidence.benchmark.dataset_sha256 != benchmark_rule.dataset_sha256
        or evidence.benchmark.cases.record_count != benchmark_rule.expected_sample_count
    ):
        raise ValueError("benchmark evidence does not match the pinned gate config")
    if evidence.predictions.record_count != evidence.benchmark.cases.record_count:
        raise ValueError("prediction count does not match the frozen benchmark")
    if evidence.contamination.algorithm != config.contamination_algorithm:
        raise ValueError("contamination algorithm does not match gate config")
    if (
        evidence.publication.release_tier != config.release_tier
        or evidence.publication.visibility != config.required_visibility
        or evidence.publication.quality_target_is_publication_blocker
        != config.quality_target_is_publication_blocker
    ):
        raise ValueError("experimental publication scope does not match gate config")

    root = evidence_path.parent.resolve(strict=True)
    review_hashes = _verify_blocking_reviews(root, evidence)
    checkpoint_manifest_content = _read_verified_file(
        root, evidence.candidate.checkpoint_manifest, maximum_bytes=_MAX_MANIFEST_BYTES
    )
    if not _load_json_object(checkpoint_manifest_content, "checkpoint manifest"):
        raise ValueError("checkpoint manifest must not be empty")
    training_run_manifest_content = _read_verified_file(
        root, evidence.candidate.training_run_manifest, maximum_bytes=_MAX_MANIFEST_BYTES
    )
    if not _load_json_object(training_run_manifest_content, "training run manifest"):
        raise ValueError("training run manifest must not be empty")
    snapshot_content = _read_verified_file(
        root, evidence.benchmark.snapshot_manifest, maximum_bytes=_MAX_MANIFEST_BYTES
    )
    snapshot_files = _verify_snapshot_manifest(snapshot_content, evidence, config)
    adapter_content = _read_verified_file(
        root, evidence.benchmark.adapter_manifest, maximum_bytes=_MAX_MANIFEST_BYTES
    )
    _verify_adapter_manifest(adapter_content, evidence, config)
    cases_content = _read_verified_file(
        root, evidence.benchmark.cases, maximum_bytes=_MAX_EVALUATION_BYTES
    )
    case_models = _load_jsonl_models(
        cases_content,
        reference=evidence.benchmark.cases,
        model=CTIBenchEvalCase,
        description="CTIBench case",
    )
    cases = [model for model in case_models if isinstance(model, CTIBenchEvalCase)]
    if any(case.content_sha256 != case.canonical_hash() for case in cases):
        raise ValueError("CTIBench per-record content checksum mismatch")
    _verify_cases(cases, evidence, config, snapshot_files)

    predictions_content = _read_verified_file(
        root, evidence.predictions, maximum_bytes=_MAX_EVALUATION_BYTES
    )
    prediction_models = _load_jsonl_models(
        predictions_content,
        reference=evidence.predictions,
        model=FrozenPrediction,
        description="frozen prediction",
    )
    predictions = [model for model in prediction_models if isinstance(model, FrozenPrediction)]

    training_contents: list[tuple[TrainingCorpusReference, bytes]] = []
    for corpus in evidence.training_corpora:
        manifest_content = _read_verified_file(
            root, corpus.prepared_manifest, maximum_bytes=_MAX_MANIFEST_BYTES
        )
        _verify_training_manifest(manifest_content, corpus)
        records_content = _read_verified_file(
            root, corpus.records, maximum_bytes=_MAX_TRAINING_BYTES
        )
        training_contents.append((corpus, records_content))
    scanned_count, overlap_count = _scan_training_records(training_contents, cases)
    if scanned_count != evidence.contamination.scanned_training_record_count:
        raise ValueError("contamination scan count does not match frozen evidence")

    failures: list[str] = []
    if overlap_count != config.max_contamination_overlap_count:
        failures.append(
            f"training/evaluation contamination overlap_count {overlap_count} != required 0"
        )
    metric_report, quality_shortfalls = _evaluate_predictions(cases, predictions, config)
    quality_target_met = not quality_shortfalls
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": _RELEASE_PROTOCOL,
        "passed": not failures,
        "failures": failures,
        "quality_target_met": quality_target_met,
        "target_94_met": quality_target_met,
        "quality_shortfalls": quality_shortfalls,
        "evidence_manifest_sha256": _sha256_bytes(evidence_content),
        "evaluation_id": evidence.evaluation_id,
        "candidate": {
            "candidate_id": evidence.candidate.candidate_id,
            "model_family": evidence.candidate.model_family,
            "model_id": evidence.candidate.model_id,
            "checkpoint_sha256": evidence.candidate.checkpoint_sha256,
            "checkpoint_manifest_sha256": evidence.candidate.checkpoint_manifest.sha256,
            "training_run_manifest_sha256": evidence.candidate.training_run_manifest.sha256,
            "base_model_id": evidence.candidate.base_model_id,
            "base_model_revision": evidence.candidate.base_model_revision,
            "shadowcrafter_git_commit": evidence.candidate.shadowcrafter_git_commit,
        },
        "benchmark": {
            "benchmark_id": evidence.benchmark.benchmark_id,
            "repository_id": evidence.benchmark.repository_id,
            "upstream_revision": evidence.benchmark.upstream_revision,
            "dataset_sha256": evidence.benchmark.dataset_sha256,
            "sample_count": len(cases),
        },
        "predictions": {
            "sha256": evidence.predictions.sha256,
            "record_count": len(predictions),
            "frozen": True,
        },
        "contamination": {
            "algorithm": _CONTAMINATION_ALGORITHM,
            "scanned_training_record_count": scanned_count,
            "overlap_count": overlap_count,
        },
        "authorization": {
            "scope": config.authorization_scope,
            "commercial_use_permitted": config.commercial_use_permitted,
            "model_publication_authorized": config.model_publication_authorized and not failures,
            "release_tier": config.release_tier,
            "required_visibility": config.required_visibility,
            "public_publication_authorized": config.public_publication_authorized,
            "quality_target_is_publication_blocker": (config.quality_target_is_publication_blocker),
            "private_evidence_only": evidence.benchmark_license.private_evidence_only,
            "license_review_sha256": evidence.benchmark_license.license_review_sha256,
            "blocking_review_sha256": review_hashes,
        },
        **metric_report,
    }
    return GateResult(passed=not failures, failures=tuple(failures), report=report)


def load_and_evaluate(evidence_path: Path, config_path: Path) -> GateResult:
    """Validate a self-contained frozen-evidence bundle and recompute its scores.

    Malformed, missing, unhashed, mutable, legacy aggregate-only, or contaminated
    evidence always returns a failed result. No caller-supplied metric value is
    accepted by this release path.
    """

    try:
        if evidence_path.is_symlink() or config_path.is_symlink():
            raise ValueError("release evidence and gate config may not be symlinks")
        evidence_content = evidence_path.read_bytes()
        if len(evidence_content) > _MAX_MANIFEST_BYTES:
            raise ValueError("release evidence manifest exceeds its size bound")
        config_content = config_path.read_text(encoding="utf-8")
        config = _validate_config(yaml.safe_load(config_content))
        return _strict_evaluate(evidence_content, evidence_path, config)
    except (
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        KeyError,
        ValidationError,
        yaml.YAMLError,
    ) as exc:
        return GateResult(passed=False, failures=(f"invalid release evidence: {exc}",))


def write_gate_report(result: GateResult, output_path: Path) -> None:
    """Write a gate report once without overwriting prior release evidence."""

    if result.report is None:
        raise ValueError("gate result does not contain a strict release report")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result.report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
