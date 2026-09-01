"""Immutable input and source-identity checks for the audited 9B trainer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shadowcrafter.data.manifest import canonical_json_sha256
from shadowcrafter.data.registry import Purpose, load_registry

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


class TrainingSafetyError(RuntimeError):
    """Raised when the audited 9B training boundary cannot remain fail-closed."""


@dataclass(frozen=True)
class TrainingPins:
    """Operator-supplied immutable identities for code, config, and inputs."""

    config_sha256: str
    train_sha256: str
    validation_sha256: str | None
    dataset_manifest_sha256: str
    registry_sha256: str
    git_revision: str

    def validate(self) -> None:
        """Reject missing, uppercase, shortened, or otherwise ambiguous pins."""
        for name, value in (
            ("config_sha256", self.config_sha256),
            ("train_sha256", self.train_sha256),
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("registry_sha256", self.registry_sha256),
        ):
            if _SHA256_PATTERN.fullmatch(value) is None:
                raise TrainingSafetyError(f"{name} must be an exact lowercase SHA-256")
        if (
            self.validation_sha256 is not None
            and _SHA256_PATTERN.fullmatch(self.validation_sha256) is None
        ):
            raise TrainingSafetyError("validation_sha256 must be an exact lowercase SHA-256")
        if _GIT_SHA_PATTERN.fullmatch(self.git_revision) is None:
            raise TrainingSafetyError("git_revision must be an exact lowercase 40-character SHA")


@dataclass(frozen=True)
class VerifiedTrainingInputs:
    """Verified local input observations retained for the run manifest."""

    train_path: Path
    validation_path: Path | None
    dataset_manifest_path: Path
    registry_path: Path
    train_sha256: str
    validation_sha256: str | None
    dataset_manifest_sha256: str
    dataset_sha256: str
    registry_sha256: str
    train_record_count: int
    validation_record_count: int | None
    source_licenses: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class GitObservation:
    """Pinned repository identity captured before and after training."""

    root: Path
    revision: str


@dataclass(frozen=True)
class FrozenTrainingFiles:
    """Private immutable byte copies consumed by the 9B trainer."""

    config_path: Path
    train_path: Path
    validation_path: Path | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_pinned_file(source: Path, destination: Path, expected_sha256: str, label: str) -> None:
    """Copy one exact pinned byte stream into a private, read-only snapshot."""
    digest = hashlib.sha256()
    try:
        with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
            for chunk in iter(lambda: source_handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                destination_handle.write(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    except OSError as error:
        raise TrainingSafetyError(f"could not materialize pinned {label}: {error}") from error
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != expected_sha256:
        destination.unlink(missing_ok=True)
        raise TrainingSafetyError(
            f"{label} changed while creating the private training snapshot: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    destination.chmod(0o400)


@contextmanager
def frozen_training_files(
    config_path: Path,
    inputs: VerifiedTrainingInputs,
    pins: TrainingPins,
) -> Iterator[FrozenTrainingFiles]:
    """Yield private copies so the trainer never reopens mutable inputs."""
    with tempfile.TemporaryDirectory(prefix="shadowcrafter-training-input-") as temporary:
        root = Path(temporary)
        root.chmod(0o700)
        config_snapshot = root / "config.yaml"
        train_snapshot = root / "train.jsonl"
        validation_snapshot = root / "validation.jsonl" if inputs.validation_path else None
        _copy_pinned_file(config_path, config_snapshot, pins.config_sha256, "configuration")
        _copy_pinned_file(inputs.train_path, train_snapshot, pins.train_sha256, "training split")
        if inputs.validation_path is not None and validation_snapshot is not None:
            if pins.validation_sha256 is None:  # Defensive redundancy after TrainingPins.validate.
                raise TrainingSafetyError("validation snapshot has no immutable SHA-256 pin")
            _copy_pinned_file(
                inputs.validation_path,
                validation_snapshot,
                pins.validation_sha256,
                "validation split",
            )
        yield FrozenTrainingFiles(config_snapshot, train_snapshot, validation_snapshot)


def _regular_file(path: Path, label: str, *, allow_empty: bool = False) -> Path:
    if path.is_symlink():
        raise TrainingSafetyError(f"{label} must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise TrainingSafetyError(f"{label} does not exist: {path}") from error
    if not resolved.is_file():
        raise TrainingSafetyError(f"{label} is not a regular file: {resolved}")
    if not allow_empty and resolved.stat().st_size == 0:
        raise TrainingSafetyError(f"{label} must not be empty")
    return resolved


def _json_mapping(path: Path, label: str, *, content: bytes | None = None) -> dict[str, Any]:
    try:
        encoded = path.read_bytes() if content is None else content
        payload: object = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingSafetyError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TrainingSafetyError(f"{label} must contain a JSON object")
    return payload


def _manifest_artifact(manifest: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TrainingSafetyError("dataset manifest has no artifacts mapping")
    artifact = artifacts.get(split)
    if not isinstance(artifact, Mapping):
        raise TrainingSafetyError(f"dataset manifest has no {split!r} artifact")
    return artifact


def _artifact_record_count(artifact: Mapping[str, Any], split: str) -> int:
    count = artifact.get("record_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise TrainingSafetyError(f"dataset manifest {split} record_count is invalid")
    return count


def verify_training_inputs(
    *,
    train_path: Path,
    validation_path: Path | None,
    dataset_manifest_path: Path,
    registry_path: Path,
    pins: TrainingPins,
) -> VerifiedTrainingInputs:
    """Verify every local byte input against explicit pins and registry policy."""
    pins.validate()
    train = _regular_file(train_path, "training split")
    manifest_file = _regular_file(dataset_manifest_path, "dataset manifest")
    registry_file = _regular_file(registry_path, "source registry")
    if (validation_path is None) != (pins.validation_sha256 is None):
        raise TrainingSafetyError(
            "validation path and validation SHA-256 must either both be supplied or both omitted"
        )
    validation = (
        _regular_file(validation_path, "validation split") if validation_path is not None else None
    )

    observed_train_sha = _sha256(train)
    if observed_train_sha != pins.train_sha256:
        raise TrainingSafetyError(
            f"training split SHA-256 mismatch: expected {pins.train_sha256}, "
            f"observed {observed_train_sha}"
        )
    observed_validation_sha = _sha256(validation) if validation is not None else None
    if observed_validation_sha != pins.validation_sha256:
        raise TrainingSafetyError(
            "validation split SHA-256 mismatch: "
            f"expected {pins.validation_sha256}, observed {observed_validation_sha}"
        )
    try:
        manifest_content = manifest_file.read_bytes()
    except OSError as error:
        raise TrainingSafetyError(f"could not read dataset manifest: {manifest_file}") from error
    observed_manifest_sha = hashlib.sha256(manifest_content).hexdigest()
    if observed_manifest_sha != pins.dataset_manifest_sha256:
        raise TrainingSafetyError(
            f"dataset manifest SHA-256 mismatch: expected {pins.dataset_manifest_sha256}, "
            f"observed {observed_manifest_sha}"
        )

    # Parse the exact bytes that produced the pinned digest. Reopening the path
    # here would otherwise leave a hash/JSON TOCTOU window.
    manifest = _json_mapping(manifest_file, "dataset manifest", content=manifest_content)
    if manifest.get("schema_version") != 2:
        raise TrainingSafetyError("dataset manifest schema_version must be exactly 2")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise TrainingSafetyError("dataset manifest artifacts mapping is empty")
    artifact_hashes: dict[str, str] = {}
    for split, raw_artifact in artifacts.items():
        if not isinstance(split, str) or not isinstance(raw_artifact, Mapping):
            raise TrainingSafetyError("dataset manifest contains an invalid artifact entry")
        artifact_sha = raw_artifact.get("sha256")
        if not isinstance(artifact_sha, str) or _SHA256_PATTERN.fullmatch(artifact_sha) is None:
            raise TrainingSafetyError(f"dataset manifest {split} SHA-256 is invalid")
        artifact_hashes[split] = artifact_sha
    observed_dataset_sha = canonical_json_sha256(artifact_hashes)
    if manifest.get("dataset_sha256") != observed_dataset_sha:
        raise TrainingSafetyError("dataset manifest aggregate dataset_sha256 is inconsistent")

    train_artifact = _manifest_artifact(manifest, "train")
    if train_artifact.get("sha256") != observed_train_sha:
        raise TrainingSafetyError("training split is not the train artifact pinned by the manifest")
    if train_artifact.get("size") != train.stat().st_size:
        raise TrainingSafetyError("training split size differs from the dataset manifest")
    train_count = _artifact_record_count(train_artifact, "train")
    if train_count < 1:
        raise TrainingSafetyError("training split manifest must contain at least one record")

    validation_count: int | None = None
    if validation is not None:
        validation_artifact = _manifest_artifact(manifest, "validation")
        if validation_artifact.get("sha256") != observed_validation_sha:
            raise TrainingSafetyError(
                "validation split is not the validation artifact pinned by the manifest"
            )
        if validation_artifact.get("size") != validation.stat().st_size:
            raise TrainingSafetyError("validation split size differs from the dataset manifest")
        validation_count = _artifact_record_count(validation_artifact, "validation")
        if validation_count < 1:
            raise TrainingSafetyError("supplied validation split must contain at least one record")

    registry = load_registry(registry_file)
    observed_registry_sha = registry.canonical_sha256()
    if observed_registry_sha != pins.registry_sha256:
        raise TrainingSafetyError(
            f"registry canonical SHA-256 mismatch: expected {pins.registry_sha256}, "
            f"observed {observed_registry_sha}"
        )
    manifest_registry = manifest.get("registry")
    if (
        not isinstance(manifest_registry, Mapping)
        or manifest_registry.get("sha256") != observed_registry_sha
    ):
        raise TrainingSafetyError("dataset manifest does not pin the verified source registry")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise TrainingSafetyError("dataset manifest contains no source provenance")
    source_licenses: dict[str, str] = {}
    for source_entry in sources:
        if not isinstance(source_entry, Mapping) or not isinstance(
            source_entry.get("source_id"), str
        ):
            raise TrainingSafetyError("dataset manifest contains invalid source provenance")
        source = registry.require_purpose(str(source_entry["source_id"]), Purpose.TRAIN)
        if source_entry.get("policy_class") != source.policy_class:
            raise TrainingSafetyError(
                f"source policy drift for {source.id}: manifest and registry disagree"
            )
        if source_entry.get("license_id") != source.license.id:
            raise TrainingSafetyError(
                f"source license drift for {source.id}: manifest and registry disagree"
            )
        previous_license = source_licenses.setdefault(source.id, source.license.id)
        if previous_license != source.license.id:
            raise TrainingSafetyError(f"source {source.id} has conflicting manifest licenses")

    return VerifiedTrainingInputs(
        train_path=train,
        validation_path=validation,
        dataset_manifest_path=manifest_file,
        registry_path=registry_file,
        train_sha256=observed_train_sha,
        validation_sha256=observed_validation_sha,
        dataset_manifest_sha256=observed_manifest_sha,
        dataset_sha256=observed_dataset_sha,
        registry_sha256=observed_registry_sha,
        train_record_count=train_count,
        validation_record_count=validation_count,
        source_licenses=tuple(sorted(source_licenses.items())),
    )


def _run_git(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    if git is None:
        raise TrainingSafetyError("git is required to prove the training source revision")
    result = subprocess.run(  # noqa: S603 - resolved executable and fixed argument shapes.
        [git, *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise TrainingSafetyError(
            f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def verify_git_revision(
    expected_revision: str,
    *,
    allowed_untracked: tuple[Path, ...] = (),
) -> GitObservation:
    """Require the executing source repository to be pinned and otherwise clean."""
    if _GIT_SHA_PATTERN.fullmatch(expected_revision) is None:
        raise TrainingSafetyError("git_revision must be an exact lowercase 40-character SHA")
    module_path = Path(__file__).resolve(strict=True)
    root_result = _run_git(["rev-parse", "--show-toplevel"], cwd=module_path.parent)
    root = Path(root_result.stdout.strip()).resolve(strict=True)
    if not module_path.is_relative_to(root):
        raise TrainingSafetyError("training runner source is outside the observed git repository")
    revision = _run_git(["rev-parse", "HEAD"], cwd=root).stdout.strip().lower()
    if revision != expected_revision:
        raise TrainingSafetyError(
            f"git revision mismatch: expected {expected_revision}, observed {revision}"
        )
    if _run_git(["diff", "--quiet", "--"], cwd=root, check=False).returncode != 0:
        raise TrainingSafetyError("git working tree has modified tracked files")
    if _run_git(["diff", "--cached", "--quiet", "--"], cwd=root, check=False).returncode != 0:
        raise TrainingSafetyError("git index contains uncommitted changes")
    untracked_result = _run_git(["ls-files", "--others", "--exclude-standard"], cwd=root)
    observed_untracked = {
        (root / line).resolve() for line in untracked_result.stdout.splitlines() if line
    }
    allowed = {path.resolve() for path in allowed_untracked}
    unexpected = sorted(str(path) for path in observed_untracked - allowed)
    if unexpected:
        raise TrainingSafetyError(f"git repository contains untracked files: {unexpected[:12]}")
    return GitObservation(root=root, revision=revision)


__all__ = [
    "GitObservation",
    "TrainingPins",
    "TrainingSafetyError",
    "VerifiedTrainingInputs",
    "frozen_training_files",
    "verify_git_revision",
    "verify_training_inputs",
]
