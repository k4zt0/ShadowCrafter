import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shadowcrafter.data.hygiene import stable_split
from shadowcrafter.data.manifest import sha256_file
from shadowcrafter.data.prepare import (
    SplitMode,
    TemporalSplit,
    prepare_jsonl,
    prepare_jsonl_many,
)
from shadowcrafter.schemas import Message, Provenance, SecurityRecord, TaskType

REGISTRY = Path("configs/data/sources.yaml")


def _record(
    record_id: str,
    group: str,
    prompt: str,
    *,
    source_id: str = "odytssey-curated-security-instructions",
    license_id: str = "Apache-2.0",
    published_at: str = "2024-01-01T00:00:00Z",
    benchmark_holdout: bool = False,
) -> SecurityRecord:
    record = SecurityRecord(
        record_id=record_id,
        task=TaskType.CVE_TRIAGE,
        messages=[
            Message(role="user", content=prompt),
            Message(role="assistant", content=f"Defensive remediation for {record_id}."),
        ],
        labels={"published_at": published_at},
        provenance=Provenance(
            source_id=source_id,
            license=license_id,
            retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
            upstream_revision="revision-123",
            record_key=record_id,
            content_sha256="0" * 64,
        ),
        split_group=group,
        benchmark_holdout=benchmark_holdout,
    )
    record.provenance.content_sha256 = record.canonical_hash()
    return record


def _write_jsonl(path: Path, records: list[SecurityRecord]) -> None:
    path.write_text("".join(record.model_dump_json() + "\n" for record in records))


def _read_ids(path: Path) -> set[str]:
    return {json.loads(line)["record_id"] for line in path.read_text().splitlines() if line.strip()}


def test_prepare_applies_conservative_time_plus_group_and_eval_isolation(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"
    records = [
        _record("lineage-old", "same-lineage", "Old variant"),
        _record(
            "lineage-new",
            "same-lineage",
            "New variant",
            published_at="2026-02-01T00:00:00Z",
        ),
        _record("train-old", "train-lineage", "Training case"),
        _record(
            "validation-mid",
            "validation-lineage",
            "Validation case",
            published_at="2025-06-01T00:00:00Z",
        ),
        _record(
            "eval-only",
            "blind-benchmark",
            "Blind evaluation case",
            source_id="ctibench",
            license_id="CC-BY-NC-SA-4.0",
        ),
    ]
    _write_jsonl(input_path, records)

    manifest = prepare_jsonl(
        input_path,
        output_dir,
        registry_path=REGISTRY,
        temporal_split=TemporalSplit(
            validation_after=datetime(2025, 1, 1, tzinfo=UTC),
            test_after=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )

    assert _read_ids(output_dir / "train.jsonl") == {"train-old"}
    assert _read_ids(output_dir / "validation.jsonl") == {"validation-mid"}
    assert _read_ids(output_dir / "test.jsonl") == {"lineage-old", "lineage-new"}
    assert _read_ids(output_dir / "evaluation.jsonl") == {"eval-only"}
    assert manifest["split_policy"]["strategy"] == "time_plus_group"
    assert all(not any(counts.values()) for counts in manifest["isolation"].values())
    assert manifest["artifacts"]["test"]["sha256"] == sha256_file(output_dir / "test.jsonl")


def test_prepare_forces_entire_benchmark_lineage_to_test(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"
    _write_jsonl(
        input_path,
        [
            _record("ordinary", "one-lineage", "ordinary"),
            _record("holdout", "one-lineage", "holdout", benchmark_holdout=True),
        ],
    )
    prepare_jsonl(input_path, output_dir, registry_path=REGISTRY)
    assert _read_ids(output_dir / "test.jsonl") == {"ordinary", "holdout"}


def test_prepare_rejects_rag_only_source(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(
        input_path,
        [
            _record(
                "kev",
                "CVE-2026-0001",
                "KEV record",
                source_id="cisa-kev",
                license_id="US-Government-Work",
            )
        ],
    )
    with pytest.raises(ValueError, match="rag_only; supervised preparation denied"):
        prepare_jsonl(input_path, tmp_path / "prepared", registry_path=REGISTRY)


def test_prepare_rejects_train_eval_normalized_duplicate(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    train = _record("train", "train-group", "Same prompt")
    evaluation = _record(
        "evaluation",
        "eval-group",
        "Same prompt",
        source_id="ctibench",
        license_id="CC-BY-NC-SA-4.0",
    )
    evaluation.messages[-1].content = train.messages[-1].content
    evaluation.labels["published_at"] = "2025-01-01T00:00:00Z"
    evaluation.provenance.content_sha256 = evaluation.canonical_hash()
    _write_jsonl(input_path, [train, evaluation])
    with pytest.raises(ValueError, match="normalized duplicate"):
        prepare_jsonl(input_path, tmp_path / "prepared", registry_path=REGISTRY)


def test_prepare_rejects_embedded_executable_bytes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    payload = base64.b64encode(b"MZ" + (b"\x00" * 2048)).decode()
    _write_jsonl(input_path, [_record("binary", "binary", payload)])
    with pytest.raises(ValueError, match="prohibited raw executable payload"):
        prepare_jsonl(input_path, tmp_path / "prepared", registry_path=REGISTRY)


def test_prepare_refuses_to_overwrite_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"
    _write_jsonl(input_path, [_record("one", "one", "One")])
    prepare_jsonl(input_path, output_dir, registry_path=REGISTRY)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_jsonl(input_path, output_dir, registry_path=REGISTRY)


def test_prepare_many_is_order_independent_and_records_each_input(tmp_path: Path) -> None:
    alpha_path = tmp_path / "alpha.jsonl"
    zulu_path = tmp_path / "zulu.jsonl"
    _write_jsonl(alpha_path, [_record("alpha", "alpha-lineage", "Alpha")])
    _write_jsonl(zulu_path, [_record("zulu", "zulu-lineage", "Zulu")])

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = prepare_jsonl_many([zulu_path, alpha_path], first_dir, registry_path=REGISTRY)
    second = prepare_jsonl_many([alpha_path, zulu_path], second_dir, registry_path=REGISTRY)

    expected_paths = [str(alpha_path.resolve()), str(zulu_path.resolve())]
    assert [entry["path"] for entry in first["inputs"]] == expected_paths
    assert first["input"] is None
    assert first["input_set_sha256"] == second["input_set_sha256"]
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["input_record_count"] == 2
    for entry in first["inputs"]:
        path = Path(entry["path"])
        assert entry == {
            "path": str(path),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "record_count": 1,
        }
    for split_name in ("train", "validation", "test", "evaluation"):
        assert (first_dir / f"{split_name}.jsonl").read_bytes() == (
            second_dir / f"{split_name}.jsonl"
        ).read_bytes()


def test_prepare_many_rejects_duplicate_input_paths(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"
    _write_jsonl(input_path, [_record("one", "one", "One")])

    with pytest.raises(ValueError, match="duplicate canonical JSONL input path"):
        prepare_jsonl_many([input_path, input_path.resolve()], output_dir, registry_path=REGISTRY)
    assert not output_dir.exists()


def test_prepare_many_rejects_record_id_collision_across_files(tmp_path: Path) -> None:
    first_path = tmp_path / "a.jsonl"
    second_path = tmp_path / "b.jsonl"
    _write_jsonl(first_path, [_record("collision", "first-lineage", "First")])
    _write_jsonl(second_path, [_record("collision", "second-lineage", "Second")])

    with pytest.raises(ValueError, match="duplicate record_id 'collision'.*first seen"):
        prepare_jsonl_many([second_path, first_path], tmp_path / "prepared", registry_path=REGISTRY)


def test_prepare_many_deduplicates_globally_before_split(tmp_path: Path) -> None:
    first_path = tmp_path / "a.jsonl"
    second_path = tmp_path / "b.jsonl"
    first = _record("alpha", "alpha-lineage", "Same normalized prompt")
    second = _record("zulu", "zulu-lineage", "Same normalized prompt   ")
    second.messages[-1].content = first.messages[-1].content + "   "
    second.provenance.content_sha256 = second.canonical_hash()
    _write_jsonl(first_path, [first])
    _write_jsonl(second_path, [second])

    output_dir = tmp_path / "prepared"
    manifest = prepare_jsonl_many([second_path, first_path], output_dir, registry_path=REGISTRY)

    assert manifest["input_record_count"] == 2
    assert manifest["record_count"] == 1
    assert manifest["exact_duplicate_count"] == 0
    assert manifest["normalized_duplicate_count"] == 1
    prepared_ids = set().union(
        *(
            _read_ids(output_dir / f"{split_name}.jsonl")
            for split_name in ("train", "validation", "test", "evaluation")
        )
    )
    assert prepared_ids == {"alpha"}


def test_prepare_default_mode_remains_fail_closed_on_cross_split_content(
    tmp_path: Path,
) -> None:
    train_group = next(
        f"group-{index}" for index in range(1_000) if stable_split(f"group-{index}") == "train"
    )
    test_group = next(
        f"group-{index}" for index in range(1_000) if stable_split(f"group-{index}") == "test"
    )
    first = _record("first", train_group, "Repeated question")
    second = _record("second", test_group, "Repeated question")
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(input_path, [first, second])

    with pytest.raises(ValueError, match="cross-split contamination detected"):
        prepare_jsonl(input_path, tmp_path / "prepared", registry_path=REGISTRY)
    assert not (tmp_path / "prepared").exists()


def test_prepare_many_train_only_requires_explicit_mode_and_writes_no_holdouts(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "a.jsonl"
    second_path = tmp_path / "b.jsonl"
    _write_jsonl(
        first_path,
        [
            _record("alpha", "shared-lineage", "Alpha"),
            _record("beta", "shared-lineage", "Beta"),
        ],
    )
    _write_jsonl(second_path, [_record("zulu", "another-lineage", "Zulu")])

    output_dir = tmp_path / "prepared"
    manifest = prepare_jsonl_many(
        [second_path, first_path],
        output_dir,
        registry_path=REGISTRY,
        split_mode=SplitMode.TRAIN_ONLY,
    )

    assert _read_ids(output_dir / "train.jsonl") == {"alpha", "beta", "zulu"}
    for split_name in ("validation", "test", "evaluation"):
        assert (output_dir / f"{split_name}.jsonl").read_text() == ""
        assert manifest["split_counts"][split_name] == 0
    assert manifest["split_policy"] == {
        "mode": "train_only",
        "strategy": "train_only_no_internal_evaluation",
        "benchmark_holdout": "rejected",
        "eligible_for_release_evaluation": False,
        "separate_evaluation_required": True,
        "evaluation_requirement": (
            "Use a separately governed eval-only benchmark or a future temporal holdout; "
            "these train-only artifacts provide no internal validation or test evidence."
        ),
        "validation_after": None,
        "test_after": None,
    }


def test_prepare_train_only_rejects_temporal_split(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"
    _write_jsonl(input_path, [_record("one", "one", "One")])

    with pytest.raises(ValueError, match="cannot be combined with a temporal split"):
        prepare_jsonl(
            input_path,
            output_dir,
            registry_path=REGISTRY,
            temporal_split=TemporalSplit(
                validation_after=datetime(2025, 1, 1, tzinfo=UTC),
                test_after=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            split_mode=SplitMode.TRAIN_ONLY,
        )
    assert not output_dir.exists()


def test_prepare_train_only_rejects_evaluation_only_source(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"
    _write_jsonl(
        input_path,
        [
            _record(
                "evaluation",
                "evaluation",
                "External benchmark",
                source_id="ctibench",
                license_id="CC-BY-NC-SA-4.0",
            )
        ],
    )

    with pytest.raises(ValueError, match="rejects evaluation-only source ctibench"):
        prepare_jsonl(
            input_path,
            output_dir,
            registry_path=REGISTRY,
            split_mode=SplitMode.TRAIN_ONLY,
        )
    assert not output_dir.exists()


def test_prepare_train_only_rejects_benchmark_holdout(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"
    _write_jsonl(
        input_path,
        [_record("holdout", "holdout", "Holdout", benchmark_holdout=True)],
    )

    with pytest.raises(ValueError, match="rejects benchmark_holdout record 'holdout'"):
        prepare_jsonl(
            input_path,
            output_dir,
            registry_path=REGISTRY,
            split_mode=SplitMode.TRAIN_ONLY,
        )
    assert not output_dir.exists()
