"""Resumable, integrity-bound Colab QLoRA training for the pinned 9B model."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from shadowcrafter.data.manifest import sha256_file, write_json_exclusive
from shadowcrafter.training.sft import (
    EXPECTED_RUNTIME_VERSIONS,
    AdapterExpectation,
    DenseTrainingError,
    TrainingPins,
    _assert_runtime,
    _assert_single_process_environment,
    _assert_trl_029_api,
    _load_dense_config,
    _load_dense_runtime,
    _load_training_components,
    _lora_state_sha256,
    _map_prompt_completion,
    _offline_training_environment,
    _require_lora_surface,
    _save_verified_adapter,
    _sft_kwargs,
    _verify_completion_boundaries,
    _verify_dataset_counts,
    _verify_local_model_tree,
    _verify_record_provenance,
    verify_detached_git_revision,
)
from shadowcrafter.training.training_safety import verify_training_inputs

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CHECKPOINT = re.compile(r"^checkpoint-([1-9][0-9]*)$")
_MARKER_NAME = ".shadowcrafter-complete.json"


class ColabTrainingError(DenseTrainingError):
    """A resumable Colab training or checkpoint boundary failed closed."""


@dataclass(frozen=True)
class ColabCheckpointBinding:
    """Immutable identities that every resumable checkpoint must retain."""

    source_revision: str
    config_sha256: str
    train_sha256: str
    dataset_manifest_sha256: str
    dataset_sha256: str

    def __post_init__(self) -> None:
        if _GIT_SHA.fullmatch(self.source_revision) is None:
            raise ColabTrainingError("checkpoint source revision is invalid")
        for name, value in (
            ("config_sha256", self.config_sha256),
            ("train_sha256", self.train_sha256),
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("dataset_sha256", self.dataset_sha256),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ColabTrainingError(f"checkpoint {name} is invalid")

    def as_dict(self) -> dict[str, str]:
        return {
            "source_revision": self.source_revision,
            "config_sha256": self.config_sha256,
            "train_sha256": self.train_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_sha256": self.dataset_sha256,
        }


def _checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT.fullmatch(path.name)
    if match is None:
        raise ColabTrainingError(f"invalid checkpoint directory name: {path.name}")
    return int(match.group(1))


def _atomic_marker_write(path: Path, content: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ColabTrainingError("short write while sealing checkpoint marker")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            # Google Drive FUSE can reject directory fsync even after file fsync + rename.
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_inventory(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_dir():
        raise ColabTrainingError("checkpoint must be a non-linked directory")
    files: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise ColabTrainingError("checkpoint contains a symbolic link")
        if not candidate.is_file() or candidate.name == _MARKER_NAME:
            continue
        relative = candidate.relative_to(path).as_posix()
        metadata = candidate.stat()
        if metadata.st_nlink != 1 or ".cache" in Path(relative).parts:
            raise ColabTrainingError("checkpoint contains an unsafe file surface")
        files.append(
            {
                "path": relative,
                "size": metadata.st_size,
                "sha256": sha256_file(candidate),
            }
        )
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
    }
    observed = {entry["path"] for entry in files}
    if not required.issubset(observed):
        raise ColabTrainingError(f"checkpoint is incomplete; missing {sorted(required - observed)}")
    return files


def write_checkpoint_marker(path: Path, binding: ColabCheckpointBinding) -> dict[str, Any]:
    """Write a deterministic marker only after a trainer checkpoint is complete."""

    step = _checkpoint_step(path)
    state_path = path / "trainer_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ColabTrainingError("checkpoint trainer state is invalid") from error
    if not isinstance(state, Mapping) or state.get("global_step") != step:
        raise ColabTrainingError("checkpoint trainer state does not match its directory")
    marker = {
        "schema_version": 1,
        "global_step": step,
        "binding": binding.as_dict(),
        "files": _checkpoint_inventory(path),
    }
    encoded = (
        json.dumps(
            marker,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    marker_path = path / _MARKER_NAME
    if marker_path.exists():
        if marker_path.is_symlink() or marker_path.read_bytes() != encoded:
            raise ColabTrainingError("existing checkpoint marker differs from verified content")
    else:
        _atomic_marker_write(marker_path, encoded)
    return marker


def _verify_checkpoint_marker(path: Path, binding: ColabCheckpointBinding) -> dict[str, Any]:
    marker_path = path / _MARKER_NAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ColabTrainingError("checkpoint completion marker is missing or linked")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ColabTrainingError("checkpoint completion marker is invalid") from error
    expected = {
        "schema_version": 1,
        "global_step": _checkpoint_step(path),
        "binding": binding.as_dict(),
        "files": _checkpoint_inventory(path),
    }
    if marker != expected:
        raise ColabTrainingError("checkpoint marker, binding, or file hashes differ")
    return expected


def latest_complete_checkpoint(
    root: Path, binding: ColabCheckpointBinding
) -> tuple[Path | None, dict[str, Any] | None]:
    """Return the newest hash-valid checkpoint, ignoring only unmarked partial writes."""

    if root.is_symlink():
        raise ColabTrainingError("checkpoint root must not be a symbolic link")
    if not root.exists():
        return None, None
    if not root.is_dir():
        raise ColabTrainingError("checkpoint root must be a directory")
    complete: list[tuple[int, Path, dict[str, Any]]] = []
    for candidate in root.iterdir():
        if candidate.is_symlink() and _CHECKPOINT.fullmatch(candidate.name) is not None:
            raise ColabTrainingError("checkpoint root contains a symbolic checkpoint")
        if not candidate.is_dir() or _CHECKPOINT.fullmatch(candidate.name) is None:
            continue
        marker_path = candidate / _MARKER_NAME
        if not marker_path.exists():
            continue
        marker = _verify_checkpoint_marker(candidate, binding)
        complete.append((_checkpoint_step(candidate), candidate, marker))
    if not complete:
        return None, None
    _, path, marker = max(complete, key=lambda item: item[0])
    return path, marker


def _quarantine_partial_checkpoints(root: Path) -> tuple[str, ...]:
    quarantined: list[str] = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name):
        if (
            candidate.is_dir()
            and not candidate.is_symlink()
            and _CHECKPOINT.fullmatch(candidate.name) is not None
            and not (candidate / _MARKER_NAME).exists()
        ):
            quarantine = root / "incomplete"
            quarantine.mkdir(mode=0o700, exist_ok=True)
            destination = quarantine / f"{candidate.name}-{uuid.uuid4().hex}"
            candidate.rename(destination)
            quarantined.append(str(destination))
    return tuple(quarantined)


def _checkpoint_marker_callback(binding: ColabCheckpointBinding) -> Any:
    """Build a real Transformers callback without requiring it for non-training imports."""

    from transformers import TrainerCallback

    class CheckpointMarkerCallback(TrainerCallback):  # type: ignore[misc]
        def on_save(self, args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            step = getattr(state, "global_step", None)
            if not isinstance(step, int) or isinstance(step, bool) or step < 1:
                raise ColabTrainingError("trainer attempted to save an invalid global step")
            write_checkpoint_marker(Path(args.output_dir) / f"checkpoint-{step}", binding)
            return control

    return CheckpointMarkerCallback()


def _final_manifest(
    *,
    config: Mapping[str, Any],
    inputs: Any,
    base: Any,
    source_revision: str,
    final_dir: Path,
    adapter: Any,
    global_step: int,
    train_loss: float,
    before_sha256: str,
    after_sha256: str,
    resumed_checkpoint: Path | None,
    resumed_marker: Mapping[str, Any] | None,
    save_steps: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "project": config.get("project"),
        "version": "v2.0-colab-candidate",
        "base_model": {
            "id": "ornith-ai/Ornith-1.5-9B",
            "revision": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
            "local_path": str(base.root),
            "manifest_path": str(base.manifest_path),
            "manifest_sha256": base.manifest_sha256,
            "file_count": base.file_count,
            "total_size": base.total_size,
            "local_files_only": True,
        },
        "input": {
            "train": {
                "path": str(inputs.train_path),
                "sha256": inputs.train_sha256,
                "record_count": inputs.train_record_count,
            },
            "dataset_manifest": {
                "path": str(inputs.dataset_manifest_path),
                "sha256": inputs.dataset_manifest_sha256,
                "dataset_sha256": inputs.dataset_sha256,
            },
            "registry": {
                "path": str(inputs.registry_path),
                "canonical_sha256": inputs.registry_sha256,
            },
            "source_licenses": [
                {"source_id": source, "license_id": license_id}
                for source, license_id in inputs.source_licenses
            ],
        },
        "training": dict(config["training"]),
        "training_observation": {
            "global_step": global_step,
            "train_loss": train_loss,
            "eval_loss": None,
            "losses_finite": True,
            "lora_parameters_changed": before_sha256 != after_sha256,
            "lora_state_before_sha256": before_sha256,
            "lora_state_after_sha256": after_sha256,
        },
        "effective_training_invariants": {
            "completion_only_loss": True,
            "enable_thinking": False,
            "packing": False,
            "push_to_hub": False,
            "reporting": False,
            "resume_from_checkpoint": resumed_checkpoint is not None,
            "checkpoint_saving": True,
            "checkpoint_save_steps": save_steps,
            "checkpoint_integrity_markers": True,
            "trusted_private_drive_required": True,
            "clean_detached_source_snapshot": True,
        },
        "checkpoint_lineage": {
            "resumed_from": str(resumed_checkpoint) if resumed_checkpoint else None,
            "resumed_marker": dict(resumed_marker) if resumed_marker else None,
            "markers_are_integrity_manifests_not_signatures": True,
        },
        "adapter": {
            "path": str(final_dir / "adapter"),
            **adapter.to_dict(),
        },
        "environment": {
            "python": platform.python_version(),
            "git_revision": source_revision,
            **{name: package_version(name) for name in sorted(EXPECTED_RUNTIME_VERSIONS)},
        },
    }


def train_resumable_colab(
    *,
    config_path: Path,
    train_path: Path,
    dataset_manifest_path: Path,
    registry_path: Path,
    base_model_path: Path,
    base_model_manifest_path: Path,
    base_model_manifest_sha256: str,
    checkpoint_root: Path,
    final_dir: Path,
    pins: TrainingPins,
    save_steps: int = 100,
    save_total_limit: int = 3,
) -> dict[str, Any]:
    """Train with private-Drive resume checkpoints and publish one verified adapter."""

    if not 10 <= save_steps <= 10_000:
        raise ColabTrainingError("save_steps must be between 10 and 10,000")
    if not 2 <= save_total_limit <= 10:
        raise ColabTrainingError("save_total_limit must be between 2 and 10")
    if final_dir.exists() or final_dir.is_symlink():
        raise ColabTrainingError("final Colab candidate directory already exists")
    _assert_runtime()
    _assert_single_process_environment()
    config_sha256 = sha256_file(config_path)
    if config_sha256 != pins.config_sha256:
        raise ColabTrainingError("Colab model configuration differs from its pin")
    config = _load_dense_config(config_path, max_steps=-1)
    inputs = verify_training_inputs(
        train_path=train_path,
        validation_path=None,
        dataset_manifest_path=dataset_manifest_path,
        registry_path=registry_path,
        pins=pins,
    )
    base = _verify_local_model_tree(
        "ornith-ai/Ornith-1.5-9B",
        "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
        base_model_path,
        base_model_manifest_path,
        base_model_manifest_sha256,
    )
    source = verify_detached_git_revision(pins.git_revision)
    if (
        checkpoint_root.is_symlink()
        or checkpoint_root.is_relative_to(source.root)
        or final_dir.is_relative_to(source.root)
    ):
        raise ColabTrainingError("Colab outputs must be outside the source repository")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    binding = ColabCheckpointBinding(
        source_revision=pins.git_revision,
        config_sha256=pins.config_sha256,
        train_sha256=inputs.train_sha256,
        dataset_manifest_sha256=inputs.dataset_manifest_sha256,
        dataset_sha256=inputs.dataset_sha256,
    )
    resume_path, resume_marker = latest_complete_checkpoint(checkpoint_root, binding)
    quarantined_partials = _quarantine_partial_checkpoints(checkpoint_root)

    with (
        _offline_training_environment(),
        tempfile.TemporaryDirectory(prefix="shadowcrafter-colab-") as temporary,
    ):
        temporary_root = Path(temporary)
        runtime = _load_dense_runtime(config, base)
        components = _load_training_components()
        data_files = {"train": str(inputs.train_path)}
        dataset = components.load_dataset(
            "json", data_files=data_files, cache_dir=str(temporary_root / "datasets-cache")
        )
        _verify_dataset_counts(dataset, inputs)
        _verify_record_provenance(dataset, inputs)
        dataset = _map_prompt_completion(dataset)
        _verify_dataset_counts(dataset, inputs)
        _verify_completion_boundaries(
            dataset, runtime.tokenizer, int(config["training"]["max_sequence_length"])
        )
        expectation: AdapterExpectation = _require_lora_surface(runtime.model)
        kwargs = _sft_kwargs(
            config["training"],
            output_dir=checkpoint_root,
            max_steps=-1,
            has_validation=False,
        )
        kwargs.update(
            {
                "save_strategy": "steps",
                "save_steps": save_steps,
                "save_total_limit": save_total_limit,
                "save_only_model": False,
            }
        )
        _assert_trl_029_api(components.sft_config_type, components.sft_trainer_type)
        args = components.sft_config_type(**kwargs)
        components.set_seed(int(config["training"]["seed"]))
        trainer = components.sft_trainer_type(
            model=runtime.model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=None,
            processing_class=runtime.tokenizer,
            callbacks=[_checkpoint_marker_callback(binding)],
        )
        if getattr(trainer, "model", None) is not runtime.model:
            raise ColabTrainingError("TRL replaced the pinned in-memory model")
        before = _lora_state_sha256(runtime.model)
        result = trainer.train(
            resume_from_checkpoint=str(resume_path) if resume_path is not None else None
        )
        global_step = getattr(getattr(trainer, "state", None), "global_step", None)
        if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 1:
            raise ColabTrainingError("Colab trainer completed without an optimizer step")
        train_loss = getattr(result, "training_loss", None)
        if (
            not isinstance(train_loss, (int, float))
            or isinstance(train_loss, bool)
            or not math.isfinite(float(train_loss))
        ):
            raise ColabTrainingError("Colab trainer returned a non-finite training loss")
        after = _lora_state_sha256(runtime.model)
        if after == before or _require_lora_surface(runtime.model) != expectation:
            raise ColabTrainingError("Colab optimizer did not preserve and update the LoRA surface")
        local_adapter = temporary_root / "adapter"
        _save_verified_adapter(runtime.model, runtime.tokenizer, local_adapter, expectation)

        completed_inputs = verify_training_inputs(
            train_path=train_path,
            validation_path=None,
            dataset_manifest_path=dataset_manifest_path,
            registry_path=registry_path,
            pins=pins,
        )
        completed_base = _verify_local_model_tree(
            "ornith-ai/Ornith-1.5-9B",
            "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
            base_model_path,
            base_model_manifest_path,
            base_model_manifest_sha256,
        )
        completed_source = verify_detached_git_revision(pins.git_revision)
        if completed_inputs != inputs or completed_base != base or completed_source != source:
            raise ColabTrainingError("verified Colab training inputs changed during the run")

        staging = final_dir.parent / f".{final_dir.name}.staging-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            shutil.copytree(local_adapter, staging / "adapter")
            copied = _save_or_verify_copied_adapter(staging / "adapter", expectation)
            manifest = _final_manifest(
                config=config,
                inputs=inputs,
                base=base,
                source_revision=pins.git_revision,
                final_dir=final_dir,
                adapter=copied,
                global_step=global_step,
                train_loss=float(train_loss),
                before_sha256=before,
                after_sha256=after,
                resumed_checkpoint=resume_path,
                resumed_marker=resume_marker,
                save_steps=save_steps,
            )
            manifest["configuration"] = {
                "path": str(config_path),
                "sha256": config_sha256,
            }
            manifest["checkpoint_lineage"]["quarantined_partial_checkpoints"] = list(
                quarantined_partials
            )
            write_json_exclusive(staging / "run-manifest.json", manifest)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(final_dir)
        except Exception:
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging, ignore_errors=True)
            raise
    return manifest


def _save_or_verify_copied_adapter(path: Path, expectation: AdapterExpectation) -> Any:
    from shadowcrafter.training.sft import _verify_saved_adapter

    return _verify_saved_adapter(path, expectation)


__all__ = [
    "ColabCheckpointBinding",
    "ColabTrainingError",
    "latest_complete_checkpoint",
    "train_resumable_colab",
    "write_checkpoint_marker",
]
