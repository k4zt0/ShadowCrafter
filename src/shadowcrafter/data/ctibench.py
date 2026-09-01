"""Evaluation-only CTIBench adapter with prompt and answer-key isolation."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import re
import stat
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shadowcrafter.data.hygiene import (
    contains_executable_binary_payload,
    record_contains_raw_binary,
)
from shadowcrafter.data.manifest import canonical_json_sha256, sha256_file, write_json_exclusive
from shadowcrafter.data.registry import DataSource, PolicyClass, Purpose, load_registry
from shadowcrafter.schemas import RiskTier, SecurityRecord

CTIBENCH_REPO_ID: Literal["AI4Sec/cti-bench"] = "AI4Sec/cti-bench"
CTIBENCH_REVIEWED_REVISION = "9237e1636ee3e168fbe5ebdcc1c571de0525e568"
CTIBENCH_LICENSE_ID: Literal["CC-BY-NC-SA-4.0"] = "CC-BY-NC-SA-4.0"
CTIBENCH_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CWE_ANSWER = re.compile(r"^CWE-[0-9]+(?:\s*,\s*CWE-[0-9]+)*$")
_CVSS_ANSWER = re.compile(
    r"^CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/"
    r"C:[NLH]/I:[NLH]/A:[NLH]$"
)
_ATTACK_ANSWER = re.compile(r"^T[0-9]{4}(?:\s*,\s*T[0-9]{4})*$")
_SOURCE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _().,+&'-]{0,239}(?:\.[A-Za-z0-9]{1,12})?$")
_BLOCKED_MARKUP = re.compile(r"(?is)<(?:[A-Za-z0-9_-]+:)?(?:script|iframe|object|embed|code)\b|```")
_ACTIONABLE_PAYLOAD = (
    re.compile(r"(?im)^\s*(?:curl|wget|powershell|pwsh|cmd\.exe|bash|sh)\s+[-/]"),
    re.compile(r"(?i)(?:^|[.;:]\s+)(?:curl|wget|powershell|pwsh|cmd\.exe|bash|sh)\s+[-/]"),
    re.compile(r"(?i)\b(?:msfvenom|meterpreter|reverse\s+shell|bind\s+shell)\b"),
    re.compile(r"(?i)\b(?:shellcode|weaponized\s+payload|exploit\s+payload)\b"),
    re.compile(r"(?i)https?://\S+\.(?:exe|dll|ps1|bat|cmd|sh|bin)(?:\W|$)"),
)
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_ROWS_PER_FILE = 10_000
_MAX_FIELD_CHARS = 250_000
_MAX_EVAL_INPUT_CHARS = 40_000


class CTIBenchTask(StrEnum):
    ATTACK_TECHNIQUE_EXTRACTION = "cti-ate"
    MULTIPLE_CHOICE = "cti-mcq"
    CWE_MAPPING = "cti-rcm"
    CWE_MAPPING_2021 = "cti-rcm-2021"
    CVSS_VECTOR = "cti-vsp"


class CTIBenchProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: Literal["ctibench"] = "ctibench"
    repo_id: Literal["AI4Sec/cti-bench"] = CTIBENCH_REPO_ID
    license_id: Literal["CC-BY-NC-SA-4.0"] = CTIBENCH_LICENSE_ID
    upstream_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    retrieved_at: datetime
    source_file: str
    source_row: int = Field(ge=0)
    source_reference: str
    source_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_prompt_normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rendered_input_normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CTIBenchEvalCase(BaseModel):
    """Runner input kept structurally separate from every training record schema."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    case_id: str = Field(pattern=r"^ctibench:[a-z0-9-]+:[0-9]{6}$")
    task: CTIBenchTask
    benchmark_split: Literal["test"] = "test"
    risk_tier: Literal[RiskTier.DEFENSIVE] = RiskTier.DEFENSIVE
    input_text: str = Field(min_length=1, max_length=_MAX_EVAL_INPUT_CHARS)
    choices: dict[str, str] | None = None
    answer: str = Field(min_length=1, max_length=2_000)
    provenance: CTIBenchProvenance
    benchmark_holdout: Literal[True] = True
    eval_only: Literal[True] = True
    prompt_training_eligible: Literal[False] = False
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_task_contract(self) -> CTIBenchEvalCase:
        if self.task == CTIBenchTask.MULTIPLE_CHOICE:
            if self.choices is None or set(self.choices) != {"A", "B", "C", "D"}:
                raise ValueError("CTI-MCQ cases require exactly choices A, B, C, and D")
            if self.answer not in self.choices:
                raise ValueError("CTI-MCQ answer must identify one available choice")
        elif self.choices is not None:
            raise ValueError("only CTI-MCQ cases may contain choices")
        if self.provenance.retrieved_at.tzinfo is None:
            raise ValueError("CTIBench retrieved_at must be timezone-aware")
        return self

    def canonical_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        return canonical_json_sha256(payload)


@dataclass(frozen=True)
class _ConfigSpec:
    task: CTIBenchTask | None
    headers: tuple[str, ...]
    input_field: str | None
    expected_rows: int


_CONFIGS: dict[str, _ConfigSpec] = {
    "cti-ate.tsv": _ConfigSpec(
        CTIBenchTask.ATTACK_TECHNIQUE_EXTRACTION,
        ("URL", "Platform", "Description", "Prompt", "GT"),
        "Description",
        60,
    ),
    "cti-mcq.tsv": _ConfigSpec(
        CTIBenchTask.MULTIPLE_CHOICE,
        ("URL", "Question", "Option A", "Option B", "Option C", "Option D", "Prompt", "GT"),
        "Question",
        2_500,
    ),
    "cti-rcm-2021.tsv": _ConfigSpec(
        CTIBenchTask.CWE_MAPPING_2021,
        ("URL", "Description", "Prompt", "GT"),
        "Description",
        1_000,
    ),
    "cti-rcm.tsv": _ConfigSpec(
        CTIBenchTask.CWE_MAPPING,
        ("URL", "Description", "Prompt", "GT"),
        "Description",
        1_000,
    ),
    # CTI-TAA has no ground truth in the reviewed repository. It is verified and
    # counted, but no unscorable prompt or report text enters runner artifacts.
    "cti-taa.tsv": _ConfigSpec(
        None,
        ("URL", "Text", "Prompt"),
        None,
        50,
    ),
    "cti-vsp.tsv": _ConfigSpec(
        CTIBenchTask.CVSS_VECTOR,
        ("URL", "Description", "Prompt", "GT"),
        "Description",
        1_000,
    ),
}
_SNAPSHOT_FILES = frozenset({".gitattributes", "README.md", *_CONFIGS})
_SNAPSHOT_HASH_ALGORITHM = "sha256-canonical-json-v1(path,size,sha256)"
_REQUIRED_REGISTRY_FILTERS = {
    "answer-key-isolation",
    "contamination-scan",
    "drop-upstream-prompt",
    "evaluation-only",
    "never-in-prompts-or-training",
    "noncommercial-use-only",
}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _normalized_sha256(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode()).hexdigest()


def _safe_eval_text(value: str) -> str | None:
    raw = html.unescape(value).strip()
    if not raw or len(raw) > _MAX_EVAL_INPUT_CHARS:
        return None
    if contains_executable_binary_payload(raw) or _BLOCKED_MARKUP.search(raw):
        return None
    plain = " ".join(raw.split())
    if not plain or any(pattern.search(plain) for pattern in _ACTIONABLE_PAYLOAD):
        return None
    return plain


def _safe_source_reference(value: str) -> str | None:
    candidate = " ".join(value.split())
    if not candidate or len(candidate) > 512 or contains_executable_binary_payload(candidate):
        return None
    parsed = urlparse(candidate)
    try:
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme in {"http", "https"}:
        expected_ports = (None, 443) if parsed.scheme == "https" else (None, 80)
        if not parsed.hostname or parsed.username or parsed.password or port not in expected_ports:
            return None
        # Source references are provenance only and are never fetched or rendered.
        return candidate
    if parsed.scheme or "/" in candidate or "\\" in candidate or ".." in candidate:
        return None
    return candidate if _SOURCE_LABEL.fullmatch(candidate) else None


def _validated_answer(task: CTIBenchTask, value: str) -> str | None:
    answer = " ".join(value.split())
    if task == CTIBenchTask.MULTIPLE_CHOICE:
        answer = answer.upper()
        return answer if answer in {"A", "B", "C", "D"} else None
    if task in {CTIBenchTask.CWE_MAPPING, CTIBenchTask.CWE_MAPPING_2021}:
        return answer if _CWE_ANSWER.fullmatch(answer) else None
    if task == CTIBenchTask.CVSS_VECTOR:
        return answer if _CVSS_ANSWER.fullmatch(answer) else None
    if task == CTIBenchTask.ATTACK_TECHNIQUE_EXTRACTION:
        return answer if _ATTACK_ANSWER.fullmatch(answer) else None
    raise AssertionError(f"unsupported CTIBench task: {task}")


def render_ctibench_input(case: CTIBenchEvalCase) -> str:
    """Render only the case input and choices; runner instructions remain trusted code."""

    if case.content_sha256 != case.canonical_hash():
        raise ValueError(f"CTIBench case checksum mismatch: {case.case_id}")
    if case.choices is None:
        return case.input_text
    choice_lines = [f"{label}. {case.choices[label]}" for label in ("A", "B", "C", "D")]
    return "\n".join((case.input_text, *choice_lines))


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"invalid CTIBench snapshot directory: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"CTIBench snapshot directory must not be a symlink: {path}")
    return metadata.st_dev, metadata.st_ino


def _read_regular_file_once(path: Path, *, max_bytes: int, label: str) -> bytes:
    """Read one immutable regular file through a verified descriptor exactly once."""

    try:
        path_metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(path_metadata.st_mode) or path_metadata.st_nlink != 1:
        raise ValueError(f"{label} must be a non-linked regular file: {path}")
    if path_metadata.st_size > max_bytes:
        raise ValueError(f"{label} exceeds its size bound: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"unable to safely open {label}: {path}") from exc
    chunks: list[bytes] = []
    size = 0
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino) != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise ValueError(f"{label} changed before it was opened: {path}")
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"{label} exceeds its size bound: {path}")
            chunks.append(chunk)
        after = os.fstat(handle.fileno())

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if identity_after != identity_before or size != before.st_size:
        raise ValueError(f"{label} changed while it was being read: {path}")
    try:
        final_path_metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} path changed while it was being read: {path}") from exc
    if (
        final_path_metadata.st_dev,
        final_path_metadata.st_ino,
        final_path_metadata.st_size,
        final_path_metadata.st_mtime_ns,
        final_path_metadata.st_ctime_ns,
        final_path_metadata.st_nlink,
    ) != identity_after:
        raise ValueError(f"{label} path changed while it was being read: {path}")
    return b"".join(chunks)


def _snapshot_inventory_hash(files: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(
        [
            {"path": entry["path"], "sha256": entry["sha256"], "size": entry["size"]}
            for entry in files
        ]
    )


def _validated_registry_source(registry_path: Path) -> tuple[DataSource, str]:
    registry = load_registry(registry_path)
    source = registry.require_purpose("ctibench", Purpose.EVALUATE)
    if (
        source.policy_class != PolicyClass.EVAL_ONLY
        or source.allowed_purposes != {Purpose.EVALUATE}
        or source.repo_id != CTIBENCH_REPO_ID
        or source.license.id != CTIBENCH_LICENSE_ID
        or str(source.license.url) != CTIBENCH_LICENSE_URL
        or source.license.status != "verified"
        or source.license.redistribution != "terms_limited"
        or not source.license.attribution_required
        or not source.attribution
        or source.safety.raw_malware_binaries
        or source.safety.executable_content
        or not _REQUIRED_REGISTRY_FILTERS.issubset(source.safety.required_filters)
    ):
        raise ValueError("CTIBench registry policy or CC-BY-NC-SA contract changed")
    return source, registry.canonical_sha256()


def _load_snapshot_manifest(
    snapshot_dir: Path, registry_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], tuple[int, int], str]:
    root_identity = _directory_identity(snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    try:
        manifest_bytes = _read_regular_file_once(
            manifest_path,
            max_bytes=_MAX_MANIFEST_BYTES,
            label="CTIBench snapshot manifest",
        )
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid CTIBench snapshot manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("CTIBench snapshot manifest must be a JSON object")
    if manifest.get("schema_version") != 2:
        raise ValueError("CTIBench snapshot manifest must use reviewed schema version 2")

    source, registry_sha256 = _validated_registry_source(registry_path)
    source_manifest = manifest.get("source")
    if not isinstance(source_manifest, dict):
        raise ValueError("CTIBench snapshot manifest lacks source metadata")
    license_manifest = source_manifest.get("license")
    if not isinstance(license_manifest, dict):
        raise ValueError("CTIBench snapshot manifest lacks license metadata")
    if (
        source_manifest.get("id") != source.id
        or source_manifest.get("repo_id") != source.repo_id
        or source_manifest.get("policy_class") != PolicyClass.EVAL_ONLY
        or license_manifest.get("id") != source.license.id
        or source_manifest.get("allowed_purposes") != [Purpose.EVALUATE]
        or source_manifest.get("provider") != source.provider
        or source_manifest.get("type") != source.type
        or license_manifest.get("url") != str(source.license.url)
        or license_manifest.get("status") != source.license.status
        or license_manifest.get("redistribution") != source.license.redistribution
        or license_manifest.get("attribution") != source.attribution
    ):
        raise ValueError("CTIBench snapshot source, policy, repository, or license mismatch")
    revision = manifest.get("upstream_revision")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError("CTIBench snapshot must pin a full commit SHA")
    if revision != CTIBENCH_REVIEWED_REVISION:
        raise ValueError("CTIBench snapshot revision has not completed repository review")
    if manifest.get("snapshot_id") != f"ctibench-{revision}":
        raise ValueError("CTIBench snapshot identifier does not match its pinned revision")
    if manifest.get("registry_sha256") != registry_sha256:
        raise ValueError("CTIBench snapshot registry hash does not match current policy")

    file_values = manifest.get("files")
    if not isinstance(file_values, list):
        raise ValueError("CTIBench snapshot manifest lacks a file inventory")
    files: dict[str, dict[str, Any]] = {}
    for value in file_values:
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ValueError("CTIBench snapshot contains an invalid file inventory entry")
        path = value["path"]
        if path in files:
            raise ValueError(f"CTIBench snapshot contains a duplicate file path: {path}")
        size = value.get("size")
        digest = value.get("sha256")
        git_blob_id = value.get("git_blob_id")
        lfs = value.get("lfs")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"CTIBench snapshot contains an invalid file size: {path}")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"CTIBench snapshot contains an invalid file checksum: {path}")
        if git_blob_id is not None and (
            not isinstance(git_blob_id, str) or not re.fullmatch(r"[0-9a-f]{40}", git_blob_id)
        ):
            raise ValueError(f"CTIBench snapshot contains an invalid Git blob ID: {path}")
        if lfs is not None:
            if not isinstance(lfs, dict) or lfs.get("size") != size or lfs.get("sha256") != digest:
                raise ValueError(f"CTIBench snapshot contains invalid LFS identity: {path}")
        elif git_blob_id is None:
            raise ValueError(f"CTIBench snapshot contains no immutable content identity: {path}")
        files[path] = value
    if set(files) != _SNAPSHOT_FILES:
        raise ValueError("CTIBench snapshot file inventory differs from the reviewed repository")
    ordered_files = [files[path] for path in sorted(files)]
    if manifest.get("snapshot_sha256_algorithm") != _SNAPSHOT_HASH_ALGORITHM:
        raise ValueError("CTIBench snapshot uses an unreviewed inventory hash algorithm")
    if manifest.get("snapshot_sha256") != _snapshot_inventory_hash(ordered_files):
        raise ValueError("CTIBench snapshot inventory checksum mismatch")
    return manifest, files, root_identity, hashlib.sha256(manifest_bytes).hexdigest()


def _verified_snapshot_text(
    snapshot_dir: Path,
    path: str,
    metadata: Mapping[str, Any],
    *,
    root_identity: tuple[int, int],
) -> str:
    if _directory_identity(snapshot_dir) != root_identity:
        raise ValueError("CTIBench snapshot directory changed during verification")
    artifact = snapshot_dir / path
    if path not in _SNAPSHOT_FILES or artifact.parent != snapshot_dir:
        raise ValueError(f"unsafe CTIBench snapshot path: {path}")
    expected_size = metadata.get("size")
    expected_sha256 = metadata.get("sha256")
    if not isinstance(expected_size, int) or expected_size < 0 or expected_size > _MAX_FILE_BYTES:
        raise ValueError(f"invalid or excessive CTIBench file size for {path}")
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise ValueError(f"invalid CTIBench SHA-256 for {path}")
    content = _read_regular_file_once(
        artifact,
        max_bytes=expected_size,
        label=f"CTIBench snapshot file {path}",
    )
    if len(content) != expected_size:
        raise ValueError(f"CTIBench snapshot file size mismatch: {path}")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"CTIBench snapshot file checksum mismatch: {path}")
    git_blob_id = metadata.get("git_blob_id")
    if git_blob_id is not None:
        header = f"blob {len(content)}\0".encode()
        observed_blob_id = hashlib.sha1(  # noqa: S324 - Git content identity.
            header + content,
            usedforsecurity=False,
        ).hexdigest()
        if observed_blob_id != git_blob_id:
            raise ValueError(f"CTIBench snapshot Git blob mismatch: {path}")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"CTIBench snapshot file must be UTF-8: {path}") from exc
    if contains_executable_binary_payload(text):
        raise ValueError(f"raw executable or archive payload detected in CTIBench file: {path}")
    return text


def _read_tsv(text: str, path: str, spec: _ConfigSpec) -> list[dict[str, str]]:
    # The pinned files use one physical row per example and literal, unescaped
    # quote characters inside fields. Treat only tabs/newlines as structure;
    # RFC-style quote interpretation would corrupt or reject those source rows.
    reader = csv.DictReader(
        io.StringIO(text, newline=""),
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        strict=True,
    )
    if reader.fieldnames is None or tuple(reader.fieldnames) != spec.headers:
        raise ValueError(f"CTIBench TSV header mismatch: {path}")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError(f"CTIBench TSV contains duplicate headers: {path}")
    rows: list[dict[str, str]] = []
    try:
        for row in reader:
            if len(rows) >= _MAX_ROWS_PER_FILE:
                raise ValueError(f"CTIBench TSV exceeds row limit: {path}")
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CTIBench TSV row shape mismatch: {path}:{reader.line_num}")
            typed_row = {
                key: value for key, value in row.items() if key is not None and value is not None
            }
            if any(len(value) > _MAX_FIELD_CHARS for value in typed_row.values()):
                raise ValueError(f"CTIBench TSV field exceeds limit: {path}:{reader.line_num}")
            rows.append(typed_row)
    except csv.Error as exc:
        raise ValueError(f"invalid CTIBench TSV syntax: {path}:{reader.line_num}") from exc
    if len(rows) != spec.expected_rows:
        raise ValueError(
            f"CTIBench row count mismatch for reviewed revision {path}: "
            f"{len(rows)} != {spec.expected_rows}"
        )
    return rows


def _build_case(
    *,
    task: CTIBenchTask,
    source_file: str,
    source_file_sha256: str,
    row_index: int,
    row: Mapping[str, str],
    input_text: str,
    choices: dict[str, str] | None,
    answer: str,
    revision: str,
    retrieved_at: datetime,
) -> CTIBenchEvalCase:
    source_prompt = row["Prompt"]
    rendered = input_text
    if choices is not None:
        rendered = "\n".join(
            (input_text, *(f"{label}. {choices[label]}" for label in ("A", "B", "C", "D")))
        )
    case = CTIBenchEvalCase(
        case_id=f"ctibench:{task}:{row_index:06d}",
        task=task,
        input_text=input_text,
        choices=choices,
        answer=answer,
        provenance=CTIBenchProvenance(
            upstream_revision=revision,
            retrieved_at=retrieved_at.astimezone(UTC),
            source_file=source_file,
            source_row=row_index,
            source_reference=row["URL"],
            source_file_sha256=source_file_sha256,
            source_prompt_sha256=hashlib.sha256(source_prompt.encode()).hexdigest(),
            source_prompt_normalized_sha256=_normalized_sha256(source_prompt),
            input_normalized_sha256=_normalized_sha256(input_text),
            rendered_input_normalized_sha256=_normalized_sha256(rendered),
        ),
        content_sha256="0" * 64,
    )
    case.content_sha256 = case.canonical_hash()
    return case


def canonicalize_ctibench_evaluation(
    snapshot_dir: Path,
    output_path: Path,
    *,
    registry_path: Path = Path("configs/data/sources.yaml"),
) -> dict[str, Any]:
    """Create answer-key-bearing runner cases that cannot validate as training records."""

    snapshot_manifest, files, root_identity, snapshot_manifest_sha256 = _load_snapshot_manifest(
        snapshot_dir, registry_path
    )
    source, registry_sha256 = _validated_registry_source(registry_path)
    if snapshot_manifest.get("registry_sha256") != registry_sha256:
        raise ValueError("CTIBench registry changed during snapshot verification")
    revision = str(snapshot_manifest["upstream_revision"])
    retrieved_raw = snapshot_manifest.get("retrieved_at_utc")
    if not isinstance(retrieved_raw, str):
        raise ValueError("CTIBench snapshot manifest lacks retrieved_at_utc")
    try:
        retrieved_at = datetime.fromisoformat(retrieved_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("CTIBench snapshot retrieved_at_utc is invalid") from exc
    if retrieved_at.tzinfo is None:
        raise ValueError("CTIBench snapshot retrieved_at_utc must be timezone-aware")

    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite CTIBench evaluation artifacts: {existing}")

    cases: list[CTIBenchEvalCase] = []
    stats = Counter[str]()
    by_task = Counter[str]()
    verified_text = {
        path: _verified_snapshot_text(
            snapshot_dir,
            path,
            metadata,
            root_identity=root_identity,
        )
        for path, metadata in sorted(files.items())
    }
    for source_file, spec in sorted(_CONFIGS.items()):
        rows = _read_tsv(verified_text[source_file], source_file, spec)
        stats["candidate_count"] += len(rows)
        if spec.task is None:
            stats["missing_ground_truth_skipped"] += len(rows)
            continue

        for row_index, row in enumerate(rows):
            source_reference = _safe_source_reference(row["URL"])
            input_text = _safe_eval_text(row[str(spec.input_field)])
            answer = _validated_answer(spec.task, row["GT"])
            choices: dict[str, str] | None = None
            if spec.task == CTIBenchTask.MULTIPLE_CHOICE:
                choice_values = {
                    label: _safe_eval_text(row[f"Option {label}"]) for label in ("A", "B", "C", "D")
                }
                if all(value is not None for value in choice_values.values()):
                    choices = {
                        label: value for label, value in choice_values.items() if value is not None
                    }
            if source_reference is None or input_text is None or answer is None:
                stats["unsafe_or_invalid_skipped"] += 1
                continue
            if spec.task == CTIBenchTask.MULTIPLE_CHOICE and choices is None:
                stats["unsafe_or_invalid_skipped"] += 1
                continue
            row_with_url = dict(row)
            row_with_url["URL"] = source_reference
            case = _build_case(
                task=spec.task,
                source_file=source_file,
                source_file_sha256=str(files[source_file]["sha256"]),
                row_index=row_index,
                row=row_with_url,
                input_text=input_text,
                choices=choices,
                answer=answer,
                revision=revision,
                retrieved_at=retrieved_at,
            )
            cases.append(case)
            by_task[str(spec.task)] += 1

    cases.sort(key=lambda case: case.case_id)
    if not cases:
        raise ValueError("CTIBench adapter produced no safe, scorable evaluation cases")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("CTIBench adapter produced duplicate case IDs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as writer:
        for case in cases:
            writer.write(case.model_dump_json() + "\n")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "adapter": "ctibench_eval_only_v1",
        "source_id": "ctibench",
        "repo_id": CTIBENCH_REPO_ID,
        "source_policy_class": PolicyClass.EVAL_ONLY,
        "license_id": CTIBENCH_LICENSE_ID,
        "license": {
            "id": source.license.id,
            "url": str(source.license.url),
            "attribution": source.attribution,
            "noncommercial_only": True,
            "attribution_required_when_shared": True,
            "share_alike_required_when_adapted_material_is_shared": True,
            "no_additional_downstream_restrictions": True,
        },
        "upstream_revision": revision,
        "retrieved_at_utc": retrieved_at.astimezone(UTC).isoformat(),
        "registry_sha256": registry_sha256,
        "snapshot_manifest": {
            "path": str(snapshot_dir / "manifest.json"),
            "sha256": snapshot_manifest_sha256,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size": output_path.stat().st_size,
            "record_count": len(cases),
        },
        "statistics": {
            "candidate_count": stats["candidate_count"],
            "emitted_count": len(cases),
            "missing_ground_truth_skipped": stats["missing_ground_truth_skipped"],
            "unsafe_or_invalid_skipped": stats["unsafe_or_invalid_skipped"],
            "by_task": dict(sorted(by_task.items())),
        },
        "controls": {
            "evaluation_only": True,
            "benchmark_holdout": True,
            "prompt_training_eligible": False,
            "security_record_schema_used": False,
            "upstream_prompt_preserved": False,
            "upstream_prompt_hash_only": True,
            "trusted_runner_template_required": True,
            "source_references_are_not_fetched_or_rendered": True,
            "answer_key_isolated": True,
            "raw_binary_rejection": True,
            "actionable_payload_filter": True,
            "training_contamination_check_required": True,
            "commercial_use_permitted": False,
            "attribution_required": True,
            "share_alike_required_when_adapted_material_is_shared": True,
        },
    }
    manifest["dataset_sha256"] = canonical_json_sha256(
        {
            "snapshot_manifest_sha256": manifest["snapshot_manifest"]["sha256"],
            "output_sha256": manifest["output"]["sha256"],
        }
    )
    write_json_exclusive(manifest_path, manifest)
    return manifest


def load_ctibench_eval_cases(path: Path) -> list[CTIBenchEvalCase]:
    cases: list[CTIBenchEvalCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = CTIBenchEvalCase.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid CTIBench case at {path}:{line_number}: {exc}") from exc
            if case.content_sha256 != case.canonical_hash():
                raise ValueError(f"CTIBench case checksum mismatch at {path}:{line_number}")
            if case.case_id in seen:
                raise ValueError(f"duplicate CTIBench case ID at {path}:{line_number}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError("CTIBench evaluation file contains no cases")
    return cases


def assert_no_ctibench_training_contamination(
    training_records: Iterable[SecurityRecord],
    evaluation_cases: Sequence[CTIBenchEvalCase],
) -> None:
    """Reject CTIBench provenance and exact or embedded normalized benchmark text."""

    if not evaluation_cases:
        raise ValueError("CTIBench contamination scan requires non-empty evaluation cases")
    case_ids: set[str] = set()
    benchmark_texts: set[str] = set()
    for case in evaluation_cases:
        if case.content_sha256 != case.canonical_hash():
            raise ValueError(f"CTIBench case checksum mismatch: {case.case_id}")
        if case.provenance.upstream_revision != CTIBENCH_REVIEWED_REVISION:
            raise ValueError(f"unreviewed CTIBench case revision: {case.case_id}")
        if case.case_id in case_ids:
            raise ValueError(f"duplicate CTIBench case ID: {case.case_id}")
        case_ids.add(case.case_id)
        benchmark_texts.add(_normalize_text(case.input_text))
        benchmark_texts.add(_normalize_text(render_ctibench_input(case)))

    benchmark_hashes = {
        digest
        for case in evaluation_cases
        for digest in (
            case.provenance.source_prompt_normalized_sha256,
            case.provenance.input_normalized_sha256,
            case.provenance.rendered_input_normalized_sha256,
        )
    }
    contaminated: list[str] = []
    for record in training_records:
        if record.provenance.source_id == "ctibench":
            contaminated.append(f"{record.record_id}:ctibench-provenance")
            continue
        if record_contains_raw_binary(record):
            raise ValueError(f"training record contains raw binary payload: {record.record_id}")
        for message in record.messages:
            normalized = _normalize_text(message.content)
            if _normalized_sha256(normalized) in benchmark_hashes or any(
                (len(benchmark) >= 80 and benchmark in normalized)
                or (len(normalized) >= 80 and normalized in benchmark)
                for benchmark in benchmark_texts
            ):
                contaminated.append(f"{record.record_id}:benchmark-content")
                break
    if contaminated:
        raise ValueError(f"CTIBench training contamination detected: {sorted(contaminated)}")
