"""Fail-closed, immutable snapshotting for the reviewed CTIBench dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import yaml

from shadowcrafter.data.hygiene import contains_executable_binary_payload
from shadowcrafter.data.manifest import (
    canonical_json_sha256,
    sha256_bytes,
)
from shadowcrafter.data.registry import (
    ContentKind,
    DataSource,
    LicenseStatus,
    PolicyClass,
    Purpose,
    Redistribution,
    SourceRegistry,
    SourceType,
    load_registry,
)

CTIBENCH_SOURCE_ID = "ctibench"
CTIBENCH_REPO_ID = "AI4Sec/cti-bench"
CTIBENCH_REVIEWED_REVISION = "9237e1636ee3e168fbe5ebdcc1c571de0525e568"

_CARD_LICENSE = "cc-by-nc-sa-4.0"
_REGISTRY_LICENSE = "CC-BY-NC-SA-4.0"
_REGISTRY_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode"
_MAX_TOTAL_BYTES = 16 * 1024 * 1024
_MAX_FILE_BYTES = {
    ".gitattributes": 16 * 1024,
    "README.md": 128 * 1024,
    "cti-ate.tsv": 4 * 1024 * 1024,
    "cti-mcq.tsv": 4 * 1024 * 1024,
    "cti-rcm-2021.tsv": 4 * 1024 * 1024,
    "cti-rcm.tsv": 4 * 1024 * 1024,
    "cti-taa.tsv": 4 * 1024 * 1024,
    "cti-vsp.tsv": 4 * 1024 * 1024,
}
_EXPECTED_PATHS = frozenset(_MAX_FILE_BYTES)
_EXPECTED_CONFIG_FILES = {
    "cti-ate": "cti-ate.tsv",
    "cti-mcq": "cti-mcq.tsv",
    "cti-rcm": "cti-rcm.tsv",
    "cti-rcm-2021": "cti-rcm-2021.tsv",
    "cti-taa": "cti-taa.tsv",
    "cti-vsp": "cti-vsp.tsv",
}
_REQUIRED_SOURCE_FILTERS = {
    "contamination-scan",
    "evaluation-only",
    "never-in-prompts-or-training",
}
_GIT_OID = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_READ_CHUNK_BYTES = 1024 * 1024


class DatasetInfoApi(Protocol):
    """Subset of :class:`huggingface_hub.HfApi` used by this module."""

    def dataset_info(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        files_metadata: bool = False,
        token: bool | str | None = None,
    ) -> object: ...


class HubFileDownloader(Protocol):
    """Narrow, injectable equivalent of ``huggingface_hub.hf_hub_download``."""

    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str,
        cache_dir: Path,
        token: bool,
    ) -> str | Path: ...


@runtime_checkable
class _CardData(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _LfsIdentity:
    sha256: str
    size: int
    pointer_size: int


@dataclass(frozen=True)
class _RemoteFile:
    path: str
    size: int
    git_blob_id: str | None
    lfs: _LfsIdentity | None


def _new_hub_api() -> DatasetInfoApi:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - exercised without the data extra
        raise RuntimeError("huggingface-hub is required for CTIBench snapshots") from exc
    return cast(DatasetInfoApi, HfApi())


def _official_download(
    *,
    repo_id: str,
    filename: str,
    repo_type: str,
    revision: str,
    cache_dir: Path,
    token: bool,
) -> str | Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised without the data extra
        raise RuntimeError("huggingface-hub is required for CTIBench snapshots") from exc

    result = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    if not isinstance(result, str):
        raise TypeError("Hugging Face downloader returned an unexpected result")
    return result


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return cast(Mapping[object, object], value).get(name)
    return cast(object | None, getattr(value, name, None))


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Hugging Face metadata omitted {field_name}")
    return value


def _required_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Hugging Face metadata has an invalid {field_name}")
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        candidate = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in candidate):
            raise ValueError(f"{field_name} must use string keys")
        return cast(Mapping[str, object], candidate)
    raise ValueError(f"{field_name} must be a mapping")


def _card_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, _CardData):
        return _mapping(value.to_dict(), "Hugging Face card data")
    return _mapping(value, "Hugging Face card data")


def _validate_test_only_card(card: Mapping[str, object], *, origin: str) -> None:
    if card.get("license") != _CARD_LICENSE:
        raise ValueError(f"{origin} must declare license {_CARD_LICENSE}")

    raw_configs = card.get("configs")
    if isinstance(raw_configs, (str, bytes)) or not isinstance(raw_configs, Sequence):
        raise ValueError(f"{origin} must declare CTIBench configs")

    seen: dict[str, str] = {}
    for index, raw_config in enumerate(raw_configs):
        config = _mapping(raw_config, f"{origin} config {index}")
        name = _required_string(config.get("config_name"), f"{origin} config_name")
        if name in seen:
            raise ValueError(f"{origin} contains duplicate config {name}")
        expected_path = _EXPECTED_CONFIG_FILES.get(name)
        if expected_path is None:
            raise ValueError(f"{origin} contains an unreviewed config {name}")
        if config.get("sep") != "\t":
            raise ValueError(f"{origin} config {name} must use a tab separator")

        data_files = config.get("data_files")
        if isinstance(data_files, (str, bytes)) or not isinstance(data_files, Sequence):
            raise ValueError(f"{origin} config {name} must declare one test file")
        if len(data_files) != 1:
            raise ValueError(f"{origin} config {name} must contain only the test split")
        data_file = _mapping(data_files[0], f"{origin} config {name} data file")
        if data_file.get("split") != "test" or data_file.get("path") != expected_path:
            raise ValueError(f"{origin} config {name} is not the reviewed test-only mapping")
        seen[name] = expected_path

    if seen != _EXPECTED_CONFIG_FILES:
        missing = sorted(set(_EXPECTED_CONFIG_FILES) - set(seen))
        raise ValueError(f"{origin} has an incomplete CTIBench config set: missing {missing}")


def _validate_registry_source(registry: SourceRegistry) -> DataSource:
    source = registry.source(CTIBENCH_SOURCE_ID)
    if source.type != SourceType.HUGGINGFACE_DATASET or source.repo_id != CTIBENCH_REPO_ID:
        raise ValueError("ctibench registry locator is not the reviewed Hugging Face dataset")
    if source.policy_class != PolicyClass.EVAL_ONLY:
        raise ValueError("ctibench must remain eval_only")
    if source.allowed_purposes != {Purpose.EVALUATE}:
        raise ValueError("ctibench must permit evaluation only")
    if not source.mutable or not source.temporal_snapshot:
        raise ValueError("ctibench must require immutable snapshots of its mutable upstream")

    license_metadata = source.license
    if (
        license_metadata.id != _REGISTRY_LICENSE
        or str(license_metadata.url) != _REGISTRY_LICENSE_URL
        or license_metadata.status != LicenseStatus.VERIFIED
        or license_metadata.redistribution != Redistribution.TERMS_LIMITED
        or not license_metadata.attribution_required
        or not source.attribution
    ):
        raise ValueError("ctibench registry license contract is not the reviewed CC license")

    if (
        source.safety.content_kind != ContentKind.EVALUATION_BENCHMARK
        or source.safety.raw_malware_binaries
        or source.safety.executable_content
        or not _REQUIRED_SOURCE_FILTERS.issubset(source.safety.required_filters)
    ):
        raise ValueError("ctibench registry safety contract is not evaluation-only")
    return source


def _parse_lfs(value: object, path: str, api_size: int) -> _LfsIdentity | None:
    if value is None:
        return None
    sha256 = _required_string(_field(value, "sha256"), f"{path} LFS sha256").lower()
    if not _SHA256.fullmatch(sha256):
        raise ValueError(f"{path} has an invalid LFS sha256")
    size = _required_nonnegative_int(_field(value, "size"), f"{path} LFS size")
    pointer_size = _required_nonnegative_int(
        _field(value, "pointer_size"), f"{path} LFS pointer size"
    )
    if size != api_size or pointer_size == 0:
        raise ValueError(f"{path} has inconsistent LFS metadata")
    return _LfsIdentity(sha256=sha256, size=size, pointer_size=pointer_size)


def _validated_remote_files(info: object, max_total_bytes: int) -> dict[str, _RemoteFile]:
    siblings = _field(info, "siblings")
    if isinstance(siblings, (str, bytes)) or not isinstance(siblings, Sequence):
        raise ValueError("Hugging Face API omitted file metadata")

    files: dict[str, _RemoteFile] = {}
    total_size = 0
    for raw_sibling in siblings:
        path = _required_string(_field(raw_sibling, "rfilename"), "file path")
        if path in files:
            raise ValueError(f"Hugging Face API returned duplicate path: {path}")
        size = _required_nonnegative_int(_field(raw_sibling, "size"), f"{path} size")
        path_limit = _MAX_FILE_BYTES.get(path)
        if path_limit is None:
            raise ValueError(f"Hugging Face revision contains an unreviewed path: {path}")
        if size == 0 or size > path_limit:
            raise ValueError(f"{path} exceeds its reviewed size bound: {size} > {path_limit}")

        raw_blob_id = _field(raw_sibling, "blob_id")
        git_blob_id: str | None = None
        if raw_blob_id is not None:
            git_blob_id = _required_string(raw_blob_id, f"{path} Git blob id").lower()
            if not _GIT_OID.fullmatch(git_blob_id):
                raise ValueError(f"{path} has an invalid Git blob id")
        lfs = _parse_lfs(_field(raw_sibling, "lfs"), path, size)
        if git_blob_id is None and lfs is None:
            raise ValueError(f"{path} has no Git blob or LFS content identity")

        total_size += size
        if total_size > max_total_bytes:
            raise ValueError(
                f"CTIBench snapshot exceeds its total size bound: {total_size} > {max_total_bytes}"
            )
        files[path] = _RemoteFile(
            path=path,
            size=size,
            git_blob_id=git_blob_id,
            lfs=lfs,
        )

    if set(files) != _EXPECTED_PATHS:
        missing = sorted(_EXPECTED_PATHS - set(files))
        extra = sorted(set(files) - _EXPECTED_PATHS)
        raise ValueError(f"CTIBench revision path set changed: missing={missing}, extra={extra}")
    return files


def _validate_hub_info(info: object, revision: str, max_total_bytes: int) -> dict[str, _RemoteFile]:
    if _field(info, "id") != CTIBENCH_REPO_ID:
        raise ValueError("Hugging Face API returned the wrong dataset")
    if _field(info, "sha") != revision:
        raise ValueError("Hugging Face API did not resolve the reviewed revision exactly")
    if _field(info, "private") is not False:
        raise ValueError("CTIBench must remain public")
    if _field(info, "gated") is not False:
        raise ValueError("CTIBench must remain ungated")
    if _field(info, "disabled") is not False:
        raise ValueError("CTIBench dataset is disabled or its state is unknown")

    card = _card_mapping(_field(info, "card_data"))
    _validate_test_only_card(card, origin="Hugging Face card metadata")
    return _validated_remote_files(info, max_total_bytes)


def _read_bounded_file(
    path: Path,
    expected_size: int,
    max_bytes: int,
    *,
    cache_dir: Path,
) -> bytes:
    """Read a downloader result once from the private cache without following it outside."""

    try:
        cache_root = cache_dir.resolve(strict=True)
        downloaded = path.resolve(strict=True)
        downloaded.relative_to(cache_root)
        path_metadata = downloaded.lstat()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Hugging Face downloader returned a path outside its private cache: {path}"
        ) from exc
    if not stat.S_ISREG(path_metadata.st_mode):
        raise ValueError(f"Hugging Face downloader did not return a regular file: {path}")
    if path_metadata.st_size != expected_size:
        raise ValueError(
            f"downloaded size mismatch for {path.name}: {path_metadata.st_size} != {expected_size}"
        )

    chunks: list[bytes] = []
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(downloaded, flags)
    except OSError as exc:
        raise ValueError(f"unable to safely open downloaded file: {path}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
            path_metadata.st_dev,
            path_metadata.st_ino,
        ):
            raise ValueError(f"downloaded file changed before it was opened: {path.name}")
        while chunk := handle.read(_READ_CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes or size > expected_size:
                raise ValueError(f"downloaded file exceeds its bound: {path.name}")
            chunks.append(chunk)
        after = os.fstat(handle.fileno())

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_after != identity_before:
        raise ValueError(f"downloaded file changed while it was being read: {path.name}")
    if size != expected_size:
        raise ValueError(f"downloaded file was truncated: {path.name}")
    return b"".join(chunks)


def _git_blob_id(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def _validated_text(content: bytes, path: str) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8") from exc
    if contains_executable_binary_payload(text):
        raise ValueError(f"{path} contains a prohibited raw executable or archive payload")
    return text


def _read_card_front_matter(readme: str) -> Mapping[str, object]:
    lines = readme.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("README.md is missing Hugging Face card front matter")
    try:
        closing_index = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("README.md has unterminated Hugging Face card front matter") from exc
    try:
        payload = yaml.safe_load("\n".join(lines[1:closing_index]))
    except yaml.YAMLError as exc:
        raise ValueError("README.md has invalid Hugging Face card front matter") from exc
    return _mapping(payload, "README.md card front matter")


def _snapshot_hash(files: Sequence[Mapping[str, Any]]) -> str:
    content_index = [
        {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "size": entry["size"],
        }
        for entry in files
    ]
    return canonical_json_sha256(content_index)


def _retrieval_time(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware")
    return timestamp.astimezone(UTC)


def _created_directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise FileExistsError(f"snapshot target is not a directory: {path}")
    return metadata.st_dev, metadata.st_ino


def _write_exclusive_at(directory_fd: int, name: str, content: bytes) -> None:
    """Create one direct child without resolving a caller-controlled path component."""

    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"unsafe snapshot output name: {name!r}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        metadata = os.fstat(handle.fileno())
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"snapshot output is not a private regular file: {name}")


def _remove_created_directory(path: Path, identity: tuple[int, int]) -> None:
    """Remove only the directory inode created by this invocation."""

    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    current_identity = (metadata.st_dev, metadata.st_ino)
    if stat.S_ISDIR(metadata.st_mode) and current_identity == identity:
        shutil.rmtree(path)


def snapshot_ctibench(
    config_path: Path,
    output_dir: Path,
    *,
    revision: str = CTIBENCH_REVIEWED_REVISION,
    api: DatasetInfoApi | None = None,
    downloader: HubFileDownloader | None = None,
    retrieved_at: datetime | None = None,
) -> dict[str, Any]:
    """Snapshot the exact reviewed CTIBench revision for isolated evaluation only.

    The API client and single-file downloader are injectable so tests can run without
    network access. Production defaults always use ``token=False`` and request each of
    the eight reviewed files explicitly; no repository-wide download is performed.
    """

    if revision != CTIBENCH_REVIEWED_REVISION:
        raise ValueError("only the reviewed CTIBench revision may be snapshotted")

    registry = load_registry(config_path)
    source = _validate_registry_source(registry)
    timestamp = _retrieval_time(retrieved_at)
    snapshot_id = f"{source.id}-{revision}"
    target = output_dir / source.id / snapshot_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite CTIBench snapshot: {target}")

    active_api = api if api is not None else _new_hub_api()
    info = active_api.dataset_info(
        CTIBENCH_REPO_ID,
        revision=revision,
        files_metadata=True,
        token=False,
    )
    max_total_bytes = min(registry.policy.max_snapshot_bytes, _MAX_TOTAL_BYTES)
    remote_files = _validate_hub_info(info, revision, max_total_bytes)

    active_downloader = downloader if downloader is not None else _official_download
    completed = False
    target_identity: tuple[int, int] | None = None
    target_fd: int | None = None
    cache_dir: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=False)
        target_identity = _created_directory_identity(target)
        target_fd = os.open(
            target,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_target = os.fstat(target_fd)
        if (
            not stat.S_ISDIR(opened_target.st_mode)
            or (opened_target.st_dev, opened_target.st_ino) != target_identity
        ):
            raise ValueError("CTIBench snapshot target changed before it was opened")
        cache_dir = Path(tempfile.mkdtemp(prefix="shadowcrafter-ctibench-hub-"))

        file_entries: list[dict[str, Any]] = []
        readme_text: str | None = None
        total_downloaded = 0
        for path in sorted(remote_files):
            remote = remote_files[path]
            downloaded_path = Path(
                active_downloader(
                    repo_id=CTIBENCH_REPO_ID,
                    filename=path,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=cache_dir,
                    token=False,
                )
            )
            content = _read_bounded_file(
                downloaded_path,
                remote.size,
                _MAX_FILE_BYTES[path],
                cache_dir=cache_dir,
            )
            text = _validated_text(content, path)
            digest = sha256_bytes(content)
            if remote.lfs is not None:
                if digest != remote.lfs.sha256:
                    raise ValueError(f"{path} does not match its LFS sha256")
            elif remote.git_blob_id is None or _git_blob_id(content) != remote.git_blob_id:
                raise ValueError(f"{path} does not match its Git blob id")

            total_downloaded += len(content)
            if total_downloaded > max_total_bytes:
                raise ValueError("CTIBench download exceeded its total size bound")
            _write_exclusive_at(target_fd, path, content)

            entry: dict[str, Any] = {
                "git_blob_id": remote.git_blob_id,
                "path": path,
                "sha256": digest,
                "size": len(content),
            }
            if remote.lfs is not None:
                entry["lfs"] = {
                    "pointer_size": remote.lfs.pointer_size,
                    "sha256": remote.lfs.sha256,
                    "size": remote.lfs.size,
                }
            file_entries.append(entry)
            if path == "README.md":
                readme_text = text

        if readme_text is None:
            raise AssertionError("reviewed path set unexpectedly omitted README.md")
        readme_card = _read_card_front_matter(readme_text)
        _validate_test_only_card(readme_card, origin="README.md card")

        snapshot_sha256 = _snapshot_hash(file_entries)
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "snapshot_id": snapshot_id,
            "source": {
                "allowed_purposes": sorted(source.allowed_purposes),
                "id": source.id,
                "license": {
                    "attribution": source.attribution,
                    "id": source.license.id,
                    "redistribution": source.license.redistribution,
                    "status": source.license.status,
                    "url": str(source.license.url),
                },
                "policy_class": source.policy_class,
                "provider": source.provider,
                "repo_id": source.repo_id,
                "type": source.type,
            },
            "registry_sha256": registry.canonical_sha256(),
            "upstream_revision": revision,
            "retrieved_at_utc": timestamp.isoformat(),
            "files": file_entries,
            "snapshot_sha256": snapshot_sha256,
            "snapshot_sha256_algorithm": "sha256-canonical-json-v1(path,size,sha256)",
            "validation": {
                "api_file_sizes": "passed",
                "bounded_individual_downloads": "passed",
                "card_license": _CARD_LICENSE,
                "content_identity": "git-blob-or-lfs-sha256-verified",
                "eval_only_registry_policy": "passed",
                "exact_path_set": "passed",
                "public_ungated": "passed",
                "raw_executable_binary": "absent",
                "test_only_configs": sorted(_EXPECTED_CONFIG_FILES),
                "utf8": "passed",
            },
        }
        manifest_bytes = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        _write_exclusive_at(target_fd, "manifest.json", manifest_bytes)
        if _created_directory_identity(target) != target_identity:
            raise ValueError("CTIBench snapshot target changed during creation")
        completed = True
        return manifest
    finally:
        if target_fd is not None:
            os.close(target_fd)
        if cache_dir is not None:
            shutil.rmtree(cache_dir, ignore_errors=True)
        if not completed and target_identity is not None:
            _remove_created_directory(target, target_identity)
