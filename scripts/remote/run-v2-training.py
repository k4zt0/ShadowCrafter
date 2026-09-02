#!/usr/bin/env python3
"""Build the pinned 174K-record multitask V2 corpus and train one immutable 9B candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shadowcrafter.data.adapters import canonicalize_nist_juliet
from shadowcrafter.data.augmentation import (
    derive_attack_technique_id_jsonl,
    derive_juliet_cwe_mapping_jsonl,
)
from shadowcrafter.data.ctibench import (
    find_ctibench_training_contamination,
    load_ctibench_eval_cases,
)
from shadowcrafter.data.manifest import sha256_file, write_json_exclusive
from shadowcrafter.data.prepare import SplitMode, prepare_jsonl_many
from shadowcrafter.data.registry import load_registry
from shadowcrafter.schemas import SecurityRecord
from shadowcrafter.training.sft import DenseTrainingError, TrainingPins, train_sft

_PROJECT_ROOT = Path("/root/ShadowCrafter")
_SOURCE_ROOT = Path("/root/ShadowCrafter-source")
_V1_TRAIN = (
    _PROJECT_ROOT / "data/processed/security-expanded-20260901-v8-blackbox-train-only/train.jsonl"
)
_V1_TRAIN_SHA256 = "8b0be9434be7452bf8129650eec485a00d2ce3efabeb725dc2f81908e18b7c7f"
_V1_RECORD_COUNT = 28_140
_JULIET_ARCHIVE = (
    _PROJECT_ROOT / "data/raw/snapshots/nist-juliet-sard/juliet-cpp-1.3/juliet-cpp-1.3.zip"
)
_JULIET_ARCHIVE_SHA256 = "ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb"
_JULIET_RECORD_COUNT = 64_099
_JULIET_CWE_MAPPING_COUNT = 64_099
_ATTACK_TECHNIQUE_ID_COUNT = 17_639
_CTIBENCH_CASES = _PROJECT_ROOT / "artifacts/evaluations/ctibench-9237e163/cases.jsonl"
_CTIBENCH_CASES_SHA256 = "2455b46b4851ed998ce3094ba7d9f796365bd0d71ce51264ff665f1c5203b423"
_CTIBENCH_CASE_COUNT = 5_533
_BASE_MODEL = _PROJECT_ROOT / "artifacts/base_models/Ornith-1.5-9B"
_BASE_MANIFEST_RELATIVE = Path("artifacts/manifests/ornith-1.5-9b.json")
_BASE_MANIFEST_SHA256 = "9a8c8c0c909311654a8ced2181b838cfc6d1db08d82f81b841cefa9030178f94"
_MODEL_CONFIG_RELATIVE = Path("configs/models/shadowcrafter-9b.yaml")
_REGISTRY_RELATIVE = Path("configs/data/sources.yaml")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class V2TrainingError(RuntimeError):
    """The V2 data or training boundary could not be proven."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_sha(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
        raise V2TrainingError(f"{label} does not match its immutable SHA-256 pin")


def _source(revision: str) -> Path:
    if _REVISION.fullmatch(revision) is None:
        raise V2TrainingError("source revision must be an exact lowercase Git SHA-1")
    source = _SOURCE_ROOT / revision
    if source.parent != _SOURCE_ROOT or source.is_symlink() or not source.is_dir():
        raise V2TrainingError("immutable source snapshot is missing")
    git = shutil.which("git")
    if git is None:
        raise V2TrainingError("git is required to verify the source snapshot")
    head = subprocess.run(  # noqa: S603 - resolved git with a fixed argument shape.
        (git, "rev-parse", "HEAD"),
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    status = subprocess.run(  # noqa: S603 - resolved git with a fixed argument shape.
        (git, "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=source,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if head.returncode != 0 or head.stdout.strip() != revision or status.returncode != 0:
        raise V2TrainingError("source snapshot revision could not be verified")
    if status.stdout:
        raise V2TrainingError("source snapshot is not clean")
    return source


def _records(path: Path, *, source_id: str | None = None) -> Iterator[SecurityRecord]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = SecurityRecord.model_validate_json(line)
            except Exception as exc:
                raise V2TrainingError(f"invalid training record at line {line_number}") from exc
            if source_id is None or record.provenance.source_id == source_id:
                yield record


def _records_many(paths: Sequence[Path]) -> Iterator[SecurityRecord]:
    for path in paths:
        yield from _records(path)


def _prepare(source: Path, revision: str) -> tuple[Path, Path, dict[str, Any], Path]:
    short_revision = revision[:7]
    canonical_root = (
        _PROJECT_ROOT / "data/canonical/nist-juliet-sard" / f"juliet-cpp-1.3-{short_revision}"
    )
    juliet_output = canonical_root / "records.jsonl"
    juliet_cwe_output = canonical_root / "cwe-mapping-records.jsonl"
    attack_id_output = canonical_root / "attack-technique-id-records.jsonl"
    processed_root = (
        _PROJECT_ROOT
        / "data/processed"
        / f"security-v2-multitask-174k-20260902-{short_revision}-train-only"
    )
    report_path = (
        _PROJECT_ROOT
        / "artifacts/iterations/shadowcrafter-9b/v2.0"
        / f"training-inputs-{short_revision}.json"
    )
    registry_path = source / _REGISTRY_RELATIVE
    retrieved_at = datetime.now(UTC)
    juliet_manifest = canonicalize_nist_juliet(
        _JULIET_ARCHIVE,
        juliet_output,
        upstream_revision="nist-sard-suite-112-juliet-cpp-1.3",
        retrieved_at=retrieved_at,
        registry_path=registry_path,
    )
    juliet_cwe_manifest = derive_juliet_cwe_mapping_jsonl(
        juliet_output,
        juliet_cwe_output,
        expected_record_count=_JULIET_CWE_MAPPING_COUNT,
    )
    attack_id_manifest = derive_attack_technique_id_jsonl(
        _V1_TRAIN,
        attack_id_output,
        expected_input_count=_V1_RECORD_COUNT,
        expected_output_count=_ATTACK_TECHNIQUE_ID_COUNT,
    )
    cases = load_ctibench_eval_cases(_CTIBENCH_CASES)
    if len(cases) != _CTIBENCH_CASE_COUNT:
        raise V2TrainingError("CTIBench case count differs from the immutable evaluation pin")
    added_count = sum(
        int(manifest["output"]["record_count"])
        for manifest in (juliet_manifest, juliet_cwe_manifest, attack_id_manifest)
    )
    if added_count != (
        _JULIET_RECORD_COUNT + _JULIET_CWE_MAPPING_COUNT + _ATTACK_TECHNIQUE_ID_COUNT
    ):
        raise V2TrainingError("V2 adapters did not emit all three complete multitask views")
    added_paths = (juliet_output, juliet_cwe_output, attack_id_output)
    matches = find_ctibench_training_contamination(_records_many(added_paths), cases)
    if matches:
        raise V2TrainingError(
            f"V2 added corpus overlaps CTIBench evaluation content ({len(matches)} records)"
        )

    prepared = prepare_jsonl_many(
        [_V1_TRAIN, *added_paths],
        processed_root,
        registry_path=registry_path,
        split_mode=SplitMode.TRAIN_ONLY,
    )
    expected_count = (
        _V1_RECORD_COUNT
        + _JULIET_RECORD_COUNT
        + _JULIET_CWE_MAPPING_COUNT
        + _ATTACK_TECHNIQUE_ID_COUNT
    )
    if (
        prepared.get("record_count") != expected_count
        or prepared.get("split_counts", {}).get("train") != expected_count
        or prepared.get("split_counts", {}).get("validation") != 0
        or prepared.get("split_counts", {}).get("test") != 0
        or prepared.get("split_counts", {}).get("evaluation") != 0
    ):
        raise V2TrainingError("V2 prepared split does not preserve the expected 173,977 records")
    if (
        prepared.get("exact_duplicate_count") != 0
        or prepared.get("normalized_duplicate_count") != 0
    ):
        raise V2TrainingError("V2 prepared split contains duplicate training content")

    train_path = processed_root / "train.jsonl"
    manifest_path = processed_root / "manifest.json"
    final_juliet = _records(train_path, source_id="nist-juliet-sard")
    final_matches = find_ctibench_training_contamination(final_juliet, cases)
    if final_matches:
        raise V2TrainingError("prepared V2 corpus overlaps the pinned CTIBench evaluation")
    report = {
        "schema_version": 1,
        "version": "v2.0",
        "source_revision": revision,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "v1_training": {
                "path": str(_V1_TRAIN),
                "sha256": _V1_TRAIN_SHA256,
                "record_count": _V1_RECORD_COUNT,
            },
            "nist_juliet": juliet_manifest,
            "nist_juliet_cwe_mapping": juliet_cwe_manifest,
            "mitre_attack_technique_id": attack_id_manifest,
            "ctibench_evaluation": {
                "path": str(_CTIBENCH_CASES),
                "sha256": _CTIBENCH_CASES_SHA256,
                "record_count": _CTIBENCH_CASE_COUNT,
                "training_overlap": 0,
            },
        },
        "prepared": {
            "path": str(processed_root),
            "train_sha256": sha256_file(train_path),
            "manifest_sha256": sha256_file(manifest_path),
            "dataset_sha256": prepared["dataset_sha256"],
            "record_count": expected_count,
            "source_count": len(prepared["sources"]),
            "exact_duplicates": 0,
            "normalized_duplicates": 0,
            "ctibench_overlap": 0,
        },
        "controls": {
            "train_only": True,
            "source_lineage_grouped": True,
            "raw_binaries_excluded": True,
            "source_never_executed": True,
            "separate_external_evaluation_required": True,
        },
    }
    write_json_exclusive(report_path, report)
    return train_path, manifest_path, prepared, report_path


def main() -> int:
    args = _arguments()
    try:
        source = _source(args.source_revision)
        if shutil.disk_usage(_PROJECT_ROOT).free < 250 * 1024**3:
            raise V2TrainingError("remote free space is below the 250 GiB V2 budget")
        _check_sha(_V1_TRAIN, _V1_TRAIN_SHA256, "v1 training corpus")
        _check_sha(_JULIET_ARCHIVE, _JULIET_ARCHIVE_SHA256, "NIST Juliet archive")
        _check_sha(_CTIBENCH_CASES, _CTIBENCH_CASES_SHA256, "CTIBench cases")
        base_manifest = source / _BASE_MANIFEST_RELATIVE
        _check_sha(base_manifest, _BASE_MANIFEST_SHA256, "base-model manifest")
        train_path, manifest_path, _prepared, report_path = _prepare(source, args.source_revision)
        config_path = source / _MODEL_CONFIG_RELATIVE
        registry_path = source / _REGISTRY_RELATIVE
        registry_sha256 = load_registry(registry_path).canonical_sha256()
        output_dir = (
            _PROJECT_ROOT
            / "artifacts/checkpoints/shadowcrafter-9b"
            / f"v2-expanded-multitask-174k-20260902-{args.source_revision[:7]}-r1"
        )
        pins = TrainingPins(
            config_sha256=sha256_file(config_path),
            train_sha256=sha256_file(train_path),
            validation_sha256=None,
            dataset_manifest_sha256=sha256_file(manifest_path),
            registry_sha256=registry_sha256,
            git_revision=args.source_revision,
        )
        manifest = train_sft(
            config_path=config_path,
            train_path=train_path,
            validation_path=None,
            dataset_manifest_path=manifest_path,
            registry_path=registry_path,
            base_model_path=_BASE_MODEL,
            base_model_manifest_path=base_manifest,
            base_model_manifest_sha256=_BASE_MANIFEST_SHA256,
            output_dir=output_dir,
            pins=pins,
            max_steps=-1,
        )
    except (DenseTrainingError, OSError, ValueError, V2TrainingError) as error:
        print(f"ShadowCrafter V2 training refused: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "version": "v2.0",
                "checkpoint": str(output_dir),
                "run_manifest": str(output_dir / "run-manifest.json"),
                "training_inputs": str(report_path),
                "record_count": manifest["input"]["train"]["record_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
