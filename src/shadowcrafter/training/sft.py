"""Fail-closed dense QLoRA SFT runner for the pinned Ornith 1.5 9B model.

The operator-facing files are never consumed directly by the trainer. Every
input is bound to an explicit digest, copied into a private immutable snapshot,
and reverified after training. The final directory is published only after the
LoRA adapter has been reopened on CPU and its tensor surface has been proven.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from shadowcrafter.data.manifest import write_json_exclusive
from shadowcrafter.training.training_safety import (
    GitObservation,
    TrainingPins,
    TrainingSafetyError,
    VerifiedTrainingInputs,
    frozen_training_files,
    verify_git_revision,
    verify_training_inputs,
)


class DenseTrainingError(TrainingSafetyError, ValueError):
    """Raised whenever dense training cannot prove its audited boundary."""


EXPECTED_MODEL_ID = "ornith-ai/Ornith-1.5-9B"
EXPECTED_REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
EXPECTED_TEXT_MODEL_CLASS = "Qwen3_5ForCausalLM"
EXPECTED_ARCHITECTURE = "qwen3_5"
EXPECTED_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
EXPECTED_LORA_MODULE_COUNT = 248

EXPECTED_RUNTIME_VERSIONS = {
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.2",
    "datasets": "4.8.5",
    "huggingface-hub": "1.29.0",
    "peft": "0.19.1",
    "safetensors": "0.8.0",
    "torch": "2.10.0",
    "transformers": "5.12.1",
    "trl": "0.29.1",
}

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_LORA_TENSOR_PATTERN = re.compile(
    r"(?:^|\.)(?P<target>[A-Za-z0-9_]+)\.lora_(?P<side>[ab])"
    r"(?:\.[^.]+)?\.weight$",
    re.IGNORECASE,
)
_SAFE_ADAPTER_FILE_SUFFIXES = frozenset(
    {".jinja", ".json", ".md", ".model", ".safetensors", ".tiktoken", ".txt"}
)
_REQUIRED_SFT_CONFIG_PARAMETERS = frozenset(
    {
        "output_dir",
        "completion_only_loss",
        "assistant_only_loss",
        "packing",
        "eval_packing",
        "padding_free",
        "max_length",
        "push_to_hub",
        "hub_model_id",
        "hub_token",
        "hub_always_push",
        "report_to",
        "save_strategy",
        "eval_strategy",
        "resume_from_checkpoint",
    }
)
_REQUIRED_SFT_TRAINER_PARAMETERS = frozenset(
    {"model", "args", "train_dataset", "eval_dataset", "processing_class"}
)
_TRAINING_KEYS = frozenset(
    {
        "backend",
        "method",
        "load_in_4bit",
        "quant_type",
        "double_quant",
        "compute_dtype",
        "max_sequence_length",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "expected_target_module_count",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "warmup_ratio",
        "epochs",
        "gradient_checkpointing",
        "completion_only_loss",
        "assistant_only_loss",
        "packing",
        "eval_packing",
        "padding_free",
        "enable_thinking",
        "optimizer",
        "seed",
    }
)


@dataclass(frozen=True)
class BaseModelObservation:
    """One verified observation of the local immutable base-model tree."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    file_count: int
    total_size: int


@dataclass(frozen=True)
class AdapterExpectation:
    """Exact trainable LoRA surface observed before optimizer construction."""

    tensor_count: int
    parameter_count: int


@dataclass(frozen=True)
class AdapterVerification:
    """Evidence obtained by reopening the serialized adapter on CPU."""

    tensor_count: int
    parameter_count: int
    nonzero_parameter_values: int
    config_sha256: str
    weights_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
            "nonzero_parameter_values": self.nonzero_parameter_values,
            "adapter_config_sha256": self.config_sha256,
            "adapter_weights_sha256": self.weights_sha256,
            "finite": True,
            "lora_only": True,
            "safe_serialization": True,
        }


@dataclass(frozen=True)
class TrainingObservation:
    """Numerical evidence from one completed trainer invocation."""

    adapter: AdapterVerification
    global_step: int
    train_loss: float
    eval_loss: float | None
    lora_state_before_sha256: str
    lora_state_after_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "global_step": self.global_step,
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "losses_finite": True,
            "lora_parameters_changed": True,
            "lora_state_before_sha256": self.lora_state_before_sha256,
            "lora_state_after_sha256": self.lora_state_after_sha256,
        }


@dataclass(frozen=True)
class DenseRuntime:
    """Pinned dense model and tokenizer retained in the training process."""

    model: Any
    tokenizer: Any


@dataclass(frozen=True)
class TrainingComponents:
    """Late-bound datasets/TRL components used after all cheap checks pass."""

    sft_config_type: Any
    sft_trainer_type: Any
    load_dataset: Callable[..., Any]
    set_seed: Callable[[int], None]


@dataclass(frozen=True)
class OutputWorkspace:
    """Fresh private workspace whose root becomes the final run directory."""

    root: Path
    work: Path
    device: int
    inode: int


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def _assert_runtime() -> None:
    mismatches = {
        name: {"expected": expected, "actual": _package_version(name)}
        for name, expected in EXPECTED_RUNTIME_VERSIONS.items()
        if _package_version(name) != expected
    }
    if mismatches:
        raise DenseTrainingError(
            "dense training runtime does not match the audited lock: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise DenseTrainingError(f"{label} must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise DenseTrainingError(f"{label} does not exist: {path}") from error
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise DenseTrainingError(f"{label} must be a non-empty regular file")
    if resolved.stat().st_nlink != 1:
        raise DenseTrainingError(f"{label} must not be hard-linked")
    return resolved


def _verify_local_model_tree(
    model_id: str,
    revision: str,
    root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str | None = None,
) -> BaseModelObservation:
    """Verify an exact local tree and, when supplied, its operator pin."""
    if root.is_symlink():
        raise DenseTrainingError("local base-model path must not be a symbolic link")
    try:
        resolved_root = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise DenseTrainingError("local base-model path does not exist") from error
    if not resolved_root.is_dir():
        raise DenseTrainingError("local base-model path is not a directory")
    manifest_file = _regular_file(manifest_path, "base-model manifest")
    manifest_content = manifest_file.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
    if expected_manifest_sha256 is not None:
        if _SHA256_PATTERN.fullmatch(expected_manifest_sha256) is None:
            raise DenseTrainingError("base-model manifest pin must be an exact lowercase SHA-256")
        if manifest_sha256 != expected_manifest_sha256:
            raise DenseTrainingError(
                "base-model manifest SHA-256 mismatch: "
                f"expected {expected_manifest_sha256}, observed {manifest_sha256}"
            )
    try:
        manifest: object = json.loads(manifest_content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DenseTrainingError("base-model manifest is not valid UTF-8 JSON") from error
    if not isinstance(manifest, Mapping):
        raise DenseTrainingError("base-model manifest must contain a JSON object")
    if manifest.get("artifact_id") != model_id or manifest.get("revision") != revision:
        raise DenseTrainingError("base-model manifest does not match the configured model pin")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise DenseTrainingError("base-model manifest has no file inventory")

    expected: dict[str, tuple[int, str]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping) or not isinstance(raw_entry.get("path"), str):
            raise DenseTrainingError("base-model manifest contains an invalid file entry")
        relative = Path(str(raw_entry["path"]))
        size = raw_entry.get("size")
        sha256 = raw_entry.get("sha256")
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or ".cache" in relative.parts
            or relative.as_posix() in expected
        ):
            raise DenseTrainingError("base-model manifest contains an unsafe or duplicate path")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise DenseTrainingError("base-model manifest contains an invalid file size")
        if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
            raise DenseTrainingError("base-model manifest contains an invalid file SHA-256")
        expected[relative.as_posix()] = (size, sha256)

    tree_entries = tuple(resolved_root.rglob("*"))
    symlinks = [path for path in tree_entries if path.is_symlink()]
    if symlinks:
        raise DenseTrainingError(
            "base-model tree contains symbolic links: "
            + str([str(path.relative_to(resolved_root)) for path in symlinks[:12]])
        )
    actual_files = tuple(
        path
        for path in tree_entries
        if path.is_file() and ".cache" not in path.relative_to(resolved_root).parts
    )
    actual = {path.relative_to(resolved_root).as_posix() for path in actual_files}
    if actual != set(expected):
        raise DenseTrainingError("base-model tree differs from its immutable inventory")
    total_size = 0
    for relative_key, (expected_size, expected_sha) in sorted(expected.items()):
        path = resolved_root / relative_key
        before = path.stat()
        if not path.is_file() or before.st_nlink != 1 or before.st_size != expected_size:
            raise DenseTrainingError(f"base-model file metadata mismatch: {relative_key}")
        if _sha256(path) != expected_sha:
            raise DenseTrainingError(f"base-model file checksum mismatch: {relative_key}")
        after = path.stat()
        fingerprint_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        fingerprint_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        if fingerprint_before != fingerprint_after:
            raise DenseTrainingError(f"base-model file changed while hashing: {relative_key}")
        total_size += expected_size
    if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != manifest_sha256:
        raise DenseTrainingError("base-model manifest changed while verifying the model tree")
    return BaseModelObservation(
        root=resolved_root,
        manifest_path=manifest_file,
        manifest_sha256=manifest_sha256,
        file_count=len(expected),
        total_size=total_size,
    )


def _required_int(mapping: Mapping[str, Any], key: str, *, minimum: int, maximum: int) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise DenseTrainingError(f"training.{key} must be an integer in [{minimum}, {maximum}]")
    return value


def _required_float(
    mapping: Mapping[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    raw = mapping.get(key)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise DenseTrainingError(f"training.{key} must be a finite number")
    value = float(raw)
    lower_ok = value >= minimum if minimum_inclusive else value > minimum
    if not math.isfinite(value) or not lower_ok or value > maximum:
        bracket = "[" if minimum_inclusive else "("
        raise DenseTrainingError(
            f"training.{key} must be finite and in {bracket}{minimum}, {maximum}]"
        )
    return value


def _load_dense_config(path: Path, *, max_steps: int) -> dict[str, Any]:
    try:
        payload: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise DenseTrainingError("dense model configuration is not valid UTF-8 YAML") from error
    if not isinstance(payload, dict):
        raise DenseTrainingError("dense model configuration must contain a mapping")
    base = payload.get("base_model")
    training = payload.get("training")
    if not isinstance(base, Mapping) or not isinstance(training, Mapping):
        raise DenseTrainingError("dense model configuration is missing base_model or training")
    expected_base = {
        "id": EXPECTED_MODEL_ID,
        "revision": EXPECTED_REVISION,
        "license": "MIT",
        "architecture": EXPECTED_ARCHITECTURE,
        "text_model_class": EXPECTED_TEXT_MODEL_CLASS,
        "trust_remote_code": False,
    }
    if dict(base) != expected_base:
        raise DenseTrainingError("base_model configuration differs from the audited 9B pin")
    if set(training) != _TRAINING_KEYS:
        missing = sorted(_TRAINING_KEYS - set(training))
        extra = sorted(set(training) - _TRAINING_KEYS)
        raise DenseTrainingError(
            f"training configuration surface drifted: missing={missing}, extra={extra}"
        )
    exact = {
        "backend": "transformers_peft",
        "method": "qlora",
        "load_in_4bit": True,
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
        "gradient_checkpointing": True,
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "packing": False,
        "eval_packing": False,
        "padding_free": False,
        "enable_thinking": False,
        "optimizer": "paged_adamw_8bit",
    }
    drifted = {
        key: {"expected": expected, "actual": training.get(key)}
        for key, expected in exact.items()
        if training.get(key) != expected or type(training.get(key)) is not type(expected)
    }
    if drifted:
        raise DenseTrainingError(
            "dense training invariants drifted: " + json.dumps(drifted, sort_keys=True)
        )
    targets = training.get("target_modules")
    if not isinstance(targets, list) or tuple(targets) != EXPECTED_LORA_TARGETS:
        raise DenseTrainingError("training.target_modules differ from the audited dense surface")
    if training.get("expected_target_module_count") != EXPECTED_LORA_MODULE_COUNT:
        raise DenseTrainingError("expected dense LoRA module count differs from the architecture")
    _required_int(training, "max_sequence_length", minimum=64, maximum=32_768)
    _required_int(training, "lora_rank", minimum=1, maximum=512)
    _required_int(training, "lora_alpha", minimum=1, maximum=4096)
    _required_float(training, "lora_dropout", minimum=0.0, maximum=0.5)
    if _required_int(training, "per_device_train_batch_size", minimum=1, maximum=1) != 1:
        raise DenseTrainingError("audited dense path is single-example per device")
    _required_int(training, "gradient_accumulation_steps", minimum=1, maximum=4096)
    _required_float(training, "learning_rate", minimum=0.0, maximum=1.0, minimum_inclusive=False)
    _required_float(training, "warmup_ratio", minimum=0.0, maximum=1.0)
    _required_float(training, "epochs", minimum=0.0, maximum=100.0, minimum_inclusive=False)
    _required_int(training, "seed", minimum=0, maximum=2**32 - 1)
    if (
        not isinstance(max_steps, int)
        or isinstance(max_steps, bool)
        or max_steps == 0
        or max_steps < -1
    ):
        raise DenseTrainingError("max_steps must be -1 or a positive integer")
    return payload


def _assert_training_invariants(training: Mapping[str, Any]) -> None:
    """Compatibility helper retained for cheap unit-level invariant checks."""
    expected = {
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "packing": False,
        "eval_packing": False,
        "padding_free": False,
        "enable_thinking": False,
    }
    violations = {
        key: {"expected": value, "actual": training.get(key)}
        for key, value in expected.items()
        if training.get(key) is not value
    }
    if violations:
        raise DenseTrainingError(
            "training loss/packing configuration is outside the audited path: "
            + json.dumps(violations, sort_keys=True)
        )


def verify_detached_git_revision(
    expected_revision: str, *, allowed_untracked: tuple[Path, ...] = ()
) -> GitObservation:
    """Require a clean exact revision checked out in detached-HEAD state."""
    observation = verify_git_revision(
        expected_revision,
        allowed_untracked=allowed_untracked,
    )
    git = shutil.which("git")
    if git is None:
        raise DenseTrainingError("git is required to prove detached source identity")
    result = subprocess.run(  # noqa: S603 - executable resolved and arguments are fixed.
        [git, "symbolic-ref", "--quiet", "HEAD"],
        cwd=observation.root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        raise DenseTrainingError("dense training requires a detached clean source snapshot")
    if result.returncode != 1:
        raise DenseTrainingError(
            "could not prove detached HEAD state: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    tracked = subprocess.run(  # noqa: S603 - executable resolved and path is constant.
        [git, "ls-files", "--error-unmatch", "src/shadowcrafter/training/sft.py"],
        cwd=observation.root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise DenseTrainingError("dense runner source is not tracked by the pinned revision")
    return observation


def _workspace_identity(workspace: OutputWorkspace) -> None:
    if workspace.root.is_symlink():
        raise DenseTrainingError("training output root became a symbolic link")
    try:
        observed = workspace.root.stat()
    except FileNotFoundError as error:
        raise DenseTrainingError("training output root disappeared") from error
    if (observed.st_dev, observed.st_ino) != (workspace.device, workspace.inode):
        raise DenseTrainingError("training output root identity changed")


@contextmanager
def _fresh_output_workspace(output_dir: Path) -> Iterator[OutputWorkspace]:
    """Claim a fresh destination and remove all partial output after any failure."""
    if output_dir.is_symlink() or output_dir.exists():
        raise DenseTrainingError(f"refusing to reuse an existing output directory: {output_dir}")
    try:
        parent = output_dir.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise DenseTrainingError("training output parent does not exist") from error
    if not parent.is_dir() or output_dir.name in {"", ".", ".."}:
        raise DenseTrainingError("training output path is invalid")
    root = parent / output_dir.name
    root.mkdir(mode=0o700, exist_ok=False)
    observed = root.stat()
    work = root / ".training-staging"
    work.mkdir(mode=0o700, exist_ok=False)
    workspace = OutputWorkspace(root, work, observed.st_dev, observed.st_ino)
    completed = False
    try:
        yield workspace
        _workspace_identity(workspace)
        if work.exists():
            raise DenseTrainingError("training staging directory was not cleaned before publish")
        completed = True
    finally:
        if not completed and root.exists() and not root.is_symlink():
            current = root.stat()
            if (current.st_dev, current.st_ino) == (workspace.device, workspace.inode):
                shutil.rmtree(root, ignore_errors=True)


@contextmanager
def _offline_training_environment() -> Iterator[None]:
    """Disable Hub/network reporters for the full model-load and training boundary."""
    enforced = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_DISABLED": "true",
    }
    previous = {key: os.environ.get(key) for key in enforced}
    os.environ.update(enforced)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _assert_single_process_environment() -> None:
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        raw = os.environ.get(name)
        if raw is not None:
            try:
                value = int(raw)
            except ValueError as error:
                raise DenseTrainingError(f"{name} must be an integer") from error
            if value != 1:
                raise DenseTrainingError("dense audited runner supports exactly one process")
    forbidden = {
        "ACCELERATE_USE_DEEPSPEED": "true",
        "ACCELERATE_USE_FSDP": "true",
    }
    enabled = [
        key
        for key, expected in forbidden.items()
        if os.environ.get(key, "").strip().lower() == expected
    ]
    if enabled:
        raise DenseTrainingError(f"distributed training backends are not audited: {enabled}")


def _to_prompt_completion(example: Mapping[str, Any]) -> dict[str, Any]:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise DenseTrainingError("each training record must contain at least two messages")
    normalized: list[dict[str, str]] = []
    expected_role = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise DenseTrainingError(f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise DenseTrainingError(f"message {index} has a forbidden role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise DenseTrainingError(f"message {index} must contain non-empty text")
        if role == "system":
            if index != 0:
                raise DenseTrainingError("a system message is permitted only at position zero")
        else:
            if role != expected_role:
                raise DenseTrainingError(
                    f"message {index} must have role {expected_role!r}, observed {role!r}"
                )
            expected_role = "assistant" if role == "user" else "user"
        normalized.append({"role": str(role), "content": content})
    final = normalized[-1]
    if final["role"] != "assistant":
        raise DenseTrainingError("each training record must end with one assistant completion")
    return {
        "prompt": normalized[:-1],
        "completion": [final],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _verify_dataset_counts(dataset: Mapping[str, Any], inputs: VerifiedTrainingInputs) -> None:
    if "train" not in dataset or len(dataset["train"]) != inputs.train_record_count:
        raise DenseTrainingError("loaded training record count differs from the pinned manifest")
    if inputs.validation_path is not None and (
        "validation" not in dataset or len(dataset["validation"]) != inputs.validation_record_count
    ):
        raise DenseTrainingError("loaded validation record count differs from the pinned manifest")


def _verify_record_provenance(dataset: Mapping[str, Any], inputs: VerifiedTrainingInputs) -> None:
    allowed_sources = dict(inputs.source_licenses)
    if not allowed_sources:
        raise DenseTrainingError("verified inputs contain no train-approved source bindings")
    record_ids: set[str] = set()
    for split, split_dataset in dataset.items():
        for index, example in enumerate(split_dataset):
            if not isinstance(example, Mapping):
                raise DenseTrainingError(f"{split} record {index} is not a JSON object")
            record_id = example.get("record_id")
            if not isinstance(record_id, str) or not record_id.strip():
                raise DenseTrainingError(f"{split} record {index} has no record_id")
            if record_id in record_ids:
                raise DenseTrainingError(f"duplicate record_id across training splits: {record_id}")
            record_ids.add(record_id)
            if example.get("risk_tier") == "disallowed":
                raise DenseTrainingError(f"{split} record {record_id} is marked disallowed")
            if example.get("benchmark_holdout") is not False:
                raise DenseTrainingError(f"{split} record {record_id} is a benchmark holdout")
            provenance = example.get("provenance")
            if not isinstance(provenance, Mapping):
                raise DenseTrainingError(f"{split} record {record_id} has no provenance mapping")
            source_id = provenance.get("source_id")
            license_id = provenance.get("license")
            if not isinstance(source_id, str) or source_id not in allowed_sources:
                raise DenseTrainingError(
                    f"{split} record {record_id} cites a source absent from the pinned manifest"
                )
            if license_id != allowed_sources[source_id]:
                raise DenseTrainingError(
                    f"{split} record {record_id} license differs from the pinned registry"
                )


def _map_prompt_completion(dataset: Any) -> dict[str, Any]:
    return {
        split: dataset[split].map(
            _to_prompt_completion,
            remove_columns=list(dataset[split].column_names),
            load_from_cache_file=False,
            desc=f"building {split} prompt/completion records with thinking disabled",
        )
        for split in dataset
    }


def _token_ids(value: Any, label: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not value:
        raise DenseTrainingError(f"tokenizer returned no {label} token IDs")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise DenseTrainingError(f"tokenizer returned invalid {label} token IDs")
    return value


def _verify_completion_boundaries(
    dataset: Mapping[str, Any], tokenizer: Any, max_length: int
) -> None:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise DenseTrainingError("pinned tokenizer cannot apply its chat template")
    for split, split_dataset in dataset.items():
        for index, example in enumerate(split_dataset):
            if not isinstance(example, Mapping):
                raise DenseTrainingError(f"{split} record {index} lost its mapping shape")
            prompt = example.get("prompt")
            completion = example.get("completion")
            if (
                not isinstance(prompt, list)
                or not isinstance(completion, list)
                or example.get("chat_template_kwargs") != {"enable_thinking": False}
            ):
                raise DenseTrainingError(f"{split} record {index} lost completion metadata")
            prompt_ids = _token_ids(
                apply_chat_template(
                    prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                "prompt",
            )
            full_ids = _token_ids(
                apply_chat_template(
                    prompt + completion,
                    return_dict=True,
                    tokenize=True,
                    return_assistant_tokens_mask=False,
                    enable_thinking=False,
                ),
                "prompt/completion",
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise DenseTrainingError(
                    f"{split} record {index} has a non-prefix completion boundary"
                )
            if len(full_ids) <= len(prompt_ids) or len(prompt_ids) >= max_length:
                raise DenseTrainingError(
                    f"{split} record {index} has no completion token inside max_length"
                )
            if len(full_ids) > max_length:
                raise DenseTrainingError(f"{split} record {index} would truncate completion tokens")


def _assert_lora_targets(model: Any, training: Mapping[str, Any]) -> None:
    targets = set(str(value) for value in training["target_modules"])
    matched = [
        name
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in targets
        and getattr(getattr(module, "weight", None), "ndim", None) == 2
    ]
    expected = int(training["expected_target_module_count"])
    if len(matched) != expected:
        raise DenseTrainingError(
            f"LoRA target count mismatch: expected {expected}, found {len(matched)}; "
            f"sample={matched[:12]}"
        )


def _require_lora_surface(model: Any) -> AdapterExpectation:
    import torch

    trainable = sorted(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        key=lambda item: item[0],
    )
    if not trainable:
        raise DenseTrainingError("dense model has no trainable LoRA parameters")
    if len(trainable) != EXPECTED_LORA_MODULE_COUNT * 2:
        raise DenseTrainingError(
            "trainable LoRA tensor count differs from the audited module surface: "
            f"expected {EXPECTED_LORA_MODULE_COUNT * 2}, observed {len(trainable)}"
        )
    pairs: dict[str, set[str]] = {}
    parameter_count = 0
    for name, parameter in trainable:
        match = _LORA_TENSOR_PATTERN.search(name)
        if match is None or match.group("target") not in EXPECTED_LORA_TARGETS:
            raise DenseTrainingError(f"non-audited parameter became trainable: {name}")
        module = f"{name[: match.start()]}:{match.group('target')}"
        side = match.group("side").lower()
        if side in pairs.setdefault(module, set()):
            raise DenseTrainingError(f"duplicate LoRA {side.upper()} tensor: {name}")
        pairs[module].add(side)
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise DenseTrainingError(f"trainable LoRA tensor is non-finite: {name}")
        parameter_count += int(parameter.numel())
    incomplete = [module for module, sides in pairs.items() if sides != {"a", "b"}]
    if len(pairs) != EXPECTED_LORA_MODULE_COUNT or incomplete:
        raise DenseTrainingError("trainable LoRA A/B pairing differs from the audited surface")
    if parameter_count < 1:
        raise DenseTrainingError("trainable LoRA parameter count is zero")
    return AdapterExpectation(len(trainable), parameter_count)


def _lora_state_sha256(model: Any) -> str:
    import torch

    trainable = sorted(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        key=lambda item: item[0],
    )
    if not trainable:
        raise DenseTrainingError("cannot fingerprint an empty LoRA surface")
    digest = hashlib.sha256()
    for name, parameter in trainable:
        if _LORA_TENSOR_PATTERN.search(name) is None:
            raise DenseTrainingError(f"cannot fingerprint non-LoRA parameter: {name}")
        tensor = parameter.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(tensor).all().item()):
            raise DenseTrainingError(f"trainable LoRA state is non-finite: {name}")
        metadata = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(struct.pack(">Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _set_adapter_identity(model: Any) -> None:
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, Mapping) or not peft_config:
        raise DenseTrainingError("dense model exposes no PEFT adapter configuration")
    for adapter_config in peft_config.values():
        adapter_config.base_model_name_or_path = EXPECTED_MODEL_ID
        if not hasattr(adapter_config, "revision"):
            raise DenseTrainingError("PEFT adapter configuration cannot pin a revision")
        adapter_config.revision = EXPECTED_REVISION


def _load_dense_runtime(
    config: Mapping[str, Any], base_model: BaseModelObservation
) -> DenseRuntime:
    _assert_runtime()
    _assert_single_process_environment()
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise DenseTrainingError("CUDA with BF16 support is required")
    training = config["training"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model.root),
        local_files_only=True,
        trust_remote_code=False,
    )
    if getattr(tokenizer, "chat_template", None) is None:
        raise DenseTrainingError("pinned tokenizer exposes no chat template")
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise DenseTrainingError("pinned tokenizer exposes no EOS token")
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_class = getattr(transformers, EXPECTED_TEXT_MODEL_CLASS, None)
    if model_class is None:
        raise DenseTrainingError(
            f"required model class is unavailable: {EXPECTED_TEXT_MODEL_CLASS}"
        )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = model_class.from_pretrained(
        str(base_model.root),
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    if type(model).__name__ != EXPECTED_TEXT_MODEL_CLASS:
        raise DenseTrainingError("loaded model class differs from the audited text-only class")
    module_names = [name.lower() for name, _ in model.named_modules()]
    if any("vision" in name or "visual" in name or "experts" in name for name in module_names):
        raise DenseTrainingError("unexpected non-dense module appeared in the 9B model")
    _assert_lora_targets(model, training)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    lora = LoraConfig(
        r=int(training["lora_rank"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        target_modules=list(EXPECTED_LORA_TARGETS),
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None,
    )
    model = get_peft_model(model, lora)
    _set_adapter_identity(model)
    _require_lora_surface(model)
    model.config.use_cache = False
    return DenseRuntime(model=model, tokenizer=tokenizer)


def _assert_trl_029_api(sft_config_type: Any, sft_trainer_type: Any) -> None:
    config_parameters = set(inspect.signature(sft_config_type).parameters)
    trainer_parameters = set(inspect.signature(sft_trainer_type).parameters)
    missing_config = sorted(_REQUIRED_SFT_CONFIG_PARAMETERS - config_parameters)
    missing_trainer = sorted(_REQUIRED_SFT_TRAINER_PARAMETERS - trainer_parameters)
    if missing_config or missing_trainer:
        raise DenseTrainingError(
            f"TRL 0.29 API drifted: SFTConfig={missing_config}, SFTTrainer={missing_trainer}"
        )


def _load_training_components() -> TrainingComponents:
    from datasets import load_dataset
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    _assert_trl_029_api(SFTConfig, SFTTrainer)
    return TrainingComponents(SFTConfig, SFTTrainer, load_dataset, set_seed)


def _sft_kwargs(
    training: Mapping[str, Any], *, output_dir: Path, max_steps: int, has_validation: bool
) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training["epochs"]),
        "max_steps": max_steps,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "logging_steps": 5,
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": True,
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "max_length": int(training["max_sequence_length"]),
        "packing": False,
        "eval_packing": False,
        "padding_free": False,
        "seed": int(training["seed"]),
        "eval_strategy": "no",
        "push_to_hub": False,
        "hub_model_id": None,
        "hub_token": None,
        "hub_always_push": False,
        "load_best_model_at_end": False,
        "resume_from_checkpoint": None,
        "do_train": True,
        "do_eval": has_validation,
        "dataloader_num_workers": 0,
    }


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _verify_effective_sft_args(args: Any, output_dir: Path, has_validation: bool) -> None:
    expected_bools = {
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "packing": False,
        "eval_packing": False,
        "padding_free": False,
        "push_to_hub": False,
        "hub_always_push": False,
        "load_best_model_at_end": False,
        "do_train": True,
        "do_eval": has_validation,
    }
    drifted = {
        key: getattr(args, key, "missing")
        for key, value in expected_bools.items()
        if getattr(args, key, "missing") is not value
    }
    if drifted:
        raise DenseTrainingError(f"effective SFT boolean invariants drifted: {drifted}")
    for key in ("hub_model_id", "hub_token", "resume_from_checkpoint"):
        if getattr(args, key, "missing") is not None:
            raise DenseTrainingError(f"effective SFT args unexpectedly set {key}")
    if _enum_value(getattr(args, "save_strategy", None)) != "no":
        raise DenseTrainingError("effective SFT args enable checkpoint saving")
    if _enum_value(getattr(args, "eval_strategy", None)) != "no":
        raise DenseTrainingError("validation must run only at the explicit completion boundary")
    report_to = getattr(args, "report_to", "missing")
    if report_to not in ("none", None, []) and not (isinstance(report_to, tuple) and not report_to):
        raise DenseTrainingError(f"effective SFT args enable external reporters: {report_to!r}")
    if Path(str(getattr(args, "output_dir", ""))).resolve() != output_dir.resolve():
        raise DenseTrainingError("effective SFT output differs from the private staging path")


def _verify_saved_adapter(
    adapter_dir: Path,
    expectation: AdapterExpectation,
    *,
    safe_open_fn: Callable[..., Any] | None = None,
    torch_module: Any | None = None,
) -> AdapterVerification:
    if adapter_dir.is_symlink() or not adapter_dir.is_dir():
        raise DenseTrainingError("adapter path must be a regular directory")
    config_path = _regular_file(adapter_dir / "adapter_config.json", "adapter config")
    weights_path = _regular_file(adapter_dir / "adapter_model.safetensors", "adapter weights")
    entries = tuple(adapter_dir.rglob("*"))
    symlinks = [str(path.relative_to(adapter_dir)) for path in entries if path.is_symlink()]
    if symlinks:
        raise DenseTrainingError(f"adapter contains symbolic links: {symlinks[:12]}")
    files = tuple(path for path in entries if path.is_file())
    unsafe = [
        str(path.relative_to(adapter_dir))
        for path in files
        if path.suffix.lower() in {".bin", ".pt", ".pth", ".pkl", ".pickle"}
    ]
    unsupported = [
        str(path.relative_to(adapter_dir))
        for path in files
        if path.suffix.lower() not in _SAFE_ADAPTER_FILE_SUFFIXES
    ]
    tensor_files = [
        path for path in files if path.suffix.lower() == ".safetensors" and path != weights_path
    ]
    if unsafe or unsupported or tensor_files:
        raise DenseTrainingError(
            f"adapter serialization surface is unsafe: unsafe={unsafe}, "
            f"unsupported={unsupported}, extra_tensors={tensor_files}"
        )
    config_content = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_content).hexdigest()
    weights_sha256 = _sha256(weights_path)
    try:
        config: object = json.loads(config_content.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DenseTrainingError("adapter configuration is not valid UTF-8 JSON") from error
    if not isinstance(config, Mapping):
        raise DenseTrainingError("adapter configuration must be a mapping")
    invariants = {
        "base_model_name_or_path": EXPECTED_MODEL_ID,
        "revision": EXPECTED_REVISION,
        "bias": "none",
    }
    if any(config.get(key) != value for key, value in invariants.items()):
        raise DenseTrainingError("saved adapter identity differs from the pinned dense model")
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise DenseTrainingError("saved adapter is not LoRA")
    if config.get("modules_to_save") not in (None, []):
        raise DenseTrainingError("saved adapter declares forbidden modules_to_save")
    if config.get("target_parameters") not in (None, []):
        raise DenseTrainingError("saved adapter declares forbidden target_parameters")
    if config.get("use_dora") not in (None, False) or config.get("lora_bias") not in (None, False):
        raise DenseTrainingError("saved adapter enables an unaudited LoRA variant")
    targets = config.get("target_modules")
    if (
        not isinstance(targets, list)
        or len(targets) != len(EXPECTED_LORA_TARGETS)
        or set(targets) != set(EXPECTED_LORA_TARGETS)
    ):
        raise DenseTrainingError("saved adapter target_modules differ from the audited surface")
    if safe_open_fn is None:
        from safetensors import safe_open  # type: ignore[import-not-found]

        safe_open_fn = safe_open
    if torch_module is None:
        import torch

        torch_module = torch
    tensor_count = 0
    parameter_count = 0
    nonzero_count = 0
    pairs: dict[str, set[str]] = {}
    with safe_open_fn(weights_path, framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            match = _LORA_TENSOR_PATTERN.search(key)
            if match is None or match.group("target") not in EXPECTED_LORA_TARGETS:
                raise DenseTrainingError(f"adapter contains a non-audited tensor: {key}")
            module = f"{key[: match.start()]}:{match.group('target')}"
            side = match.group("side").lower()
            if side in pairs.setdefault(module, set()):
                raise DenseTrainingError(f"adapter contains duplicate LoRA tensor: {key}")
            pairs[module].add(side)
            tensor = handle.get_tensor(key)
            if not bool(torch_module.isfinite(tensor).all().item()):
                raise DenseTrainingError(f"adapter contains non-finite values: {key}")
            tensor_count += 1
            parameter_count += int(tensor.numel())
            nonzero_count += int(torch_module.count_nonzero(tensor).item())
    if any(sides != {"a", "b"} for sides in pairs.values()):
        raise DenseTrainingError("adapter contains incomplete LoRA A/B pairs")
    if tensor_count != expectation.tensor_count or parameter_count != expectation.parameter_count:
        raise DenseTrainingError("serialized adapter shape differs from the live LoRA surface")
    if nonzero_count < 1:
        raise DenseTrainingError("serialized adapter contains no nonzero values")
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != config_sha256:
        raise DenseTrainingError("adapter config changed during CPU verification")
    if _sha256(weights_path) != weights_sha256:
        raise DenseTrainingError("adapter weights changed during CPU verification")
    return AdapterVerification(
        tensor_count=tensor_count,
        parameter_count=parameter_count,
        nonzero_parameter_values=nonzero_count,
        config_sha256=config_sha256,
        weights_sha256=weights_sha256,
    )


def _save_verified_adapter(
    model: Any, tokenizer: Any, destination: Path, expectation: AdapterExpectation
) -> AdapterVerification:
    if destination.exists():
        raise DenseTrainingError("adapter staging destination already exists")
    model.save_pretrained(destination, safe_serialization=True, push_to_hub=False)
    tokenizer.save_pretrained(destination, push_to_hub=False)
    return _verify_saved_adapter(destination, expectation)


def _train_runtime(
    runtime: DenseRuntime,
    training: Mapping[str, Any],
    inputs: VerifiedTrainingInputs,
    workspace: OutputWorkspace,
    max_steps: int,
    components: TrainingComponents,
) -> TrainingObservation:
    model = runtime.model
    tokenizer = runtime.tokenizer
    expectation = _require_lora_surface(model)
    data_files = {"train": str(inputs.train_path)}
    if inputs.validation_path is not None:
        data_files["validation"] = str(inputs.validation_path)
    cache = workspace.work / "datasets-cache"
    dataset = components.load_dataset(
        "json",
        data_files=data_files,
        cache_dir=str(cache),
    )
    _verify_dataset_counts(dataset, inputs)
    _verify_record_provenance(dataset, inputs)
    dataset = _map_prompt_completion(dataset)
    _verify_dataset_counts(dataset, inputs)
    _verify_completion_boundaries(dataset, tokenizer, int(training["max_sequence_length"]))
    trainer_output = workspace.work / "trainer"
    kwargs = _sft_kwargs(
        training,
        output_dir=trainer_output,
        max_steps=max_steps,
        has_validation=inputs.validation_path is not None,
    )
    _assert_trl_029_api(components.sft_config_type, components.sft_trainer_type)
    args = components.sft_config_type(**kwargs)
    _verify_effective_sft_args(args, trainer_output, inputs.validation_path is not None)
    components.set_seed(int(training["seed"]))
    trainer = components.sft_trainer_type(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
    )
    if getattr(trainer, "model", None) is not model:
        raise DenseTrainingError("TRL replaced the pinned in-memory dense model")
    if getattr(trainer, "processing_class", tokenizer) is not tokenizer:
        raise DenseTrainingError("TRL replaced the pinned tokenizer")
    if _require_lora_surface(model) != expectation:
        raise DenseTrainingError("TRL changed the LoRA surface before training")
    before = _lora_state_sha256(model)
    result = trainer.train(resume_from_checkpoint=None)
    global_step = getattr(getattr(trainer, "state", None), "global_step", None)
    if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 1:
        raise DenseTrainingError("trainer completed without an optimizer step")
    if max_steps > 0 and global_step != max_steps:
        raise DenseTrainingError(
            f"trainer stopped at global_step={global_step}, expected {max_steps}"
        )
    raw_train_loss = getattr(result, "training_loss", None)
    if (
        not isinstance(raw_train_loss, (int, float))
        or isinstance(raw_train_loss, bool)
        or not math.isfinite(float(raw_train_loss))
    ):
        raise DenseTrainingError(f"trainer returned non-finite train loss: {raw_train_loss!r}")
    after = _lora_state_sha256(model)
    if after == before:
        raise DenseTrainingError("optimizer did not change the LoRA state")
    if _require_lora_surface(model) != expectation:
        raise DenseTrainingError("LoRA surface changed during training")
    eval_loss: float | None = None
    if inputs.validation_path is not None:
        evaluation = trainer.evaluate()
        raw_eval_loss = evaluation.get("eval_loss") if isinstance(evaluation, Mapping) else None
        if (
            not isinstance(raw_eval_loss, (int, float))
            or isinstance(raw_eval_loss, bool)
            or not math.isfinite(float(raw_eval_loss))
        ):
            raise DenseTrainingError(f"trainer returned non-finite eval loss: {raw_eval_loss!r}")
        eval_loss = float(raw_eval_loss)
    adapter_stage = workspace.work / "adapter"
    verification = _save_verified_adapter(model, tokenizer, adapter_stage, expectation)
    return TrainingObservation(
        adapter=verification,
        global_step=global_step,
        train_loss=float(raw_train_loss),
        eval_loss=eval_loss,
        lora_state_before_sha256=before,
        lora_state_after_sha256=after,
    )


def _output_files(root: Path) -> tuple[Path, ...]:
    entries = tuple(root.rglob("*"))
    symlinks = [str(path.relative_to(root)) for path in entries if path.is_symlink()]
    if symlinks:
        raise DenseTrainingError(f"training output contains symbolic links: {symlinks[:12]}")
    return tuple(path for path in entries if path.is_file())


def train_sft(
    *,
    config_path: Path,
    train_path: Path,
    validation_path: Path | None,
    dataset_manifest_path: Path,
    registry_path: Path,
    base_model_path: Path,
    base_model_manifest_path: Path,
    base_model_manifest_sha256: str,
    output_dir: Path,
    pins: TrainingPins,
    max_steps: int = -1,
) -> dict[str, Any]:
    """Train and publish one locally verified dense LoRA adapter.

    This entry point deliberately has no digest inference, environment-variable
    fallback, Hub download, output reuse, checkpoint resume, or automatic upload.
    """
    pins.validate()
    if _SHA256_PATTERN.fullmatch(base_model_manifest_sha256) is None:
        raise DenseTrainingError("base-model manifest pin must be an exact lowercase SHA-256")
    if output_dir.is_symlink() or output_dir.exists():
        raise DenseTrainingError(f"refusing to reuse an existing output directory: {output_dir}")
    config_file = _regular_file(config_path, "model configuration")
    observed_config_sha256 = _sha256(config_file)
    if observed_config_sha256 != pins.config_sha256:
        raise DenseTrainingError(
            f"configuration SHA-256 mismatch: expected {pins.config_sha256}, "
            f"observed {observed_config_sha256}"
        )
    initial_inputs = verify_training_inputs(
        train_path=train_path,
        validation_path=validation_path,
        dataset_manifest_path=dataset_manifest_path,
        registry_path=registry_path,
        pins=pins,
    )
    initial_base = _verify_local_model_tree(
        EXPECTED_MODEL_ID,
        EXPECTED_REVISION,
        base_model_path,
        base_model_manifest_path,
        base_model_manifest_sha256,
    )
    initial_git = verify_detached_git_revision(pins.git_revision)

    with frozen_training_files(config_file, initial_inputs, pins) as frozen:
        config = _load_dense_config(frozen.config_path, max_steps=max_steps)
        with _offline_training_environment():
            runtime = _load_dense_runtime(config, initial_base)
            components = _load_training_components()
            frozen_inputs = replace(
                initial_inputs,
                train_path=frozen.train_path,
                validation_path=frozen.validation_path,
            )
            with _fresh_output_workspace(output_dir) as workspace:
                observation = _train_runtime(
                    runtime,
                    config["training"],
                    frozen_inputs,
                    workspace,
                    max_steps,
                    components,
                )
                if _sha256(config_file) != observed_config_sha256:
                    raise DenseTrainingError("model configuration changed during training")
                completed_inputs = verify_training_inputs(
                    train_path=train_path,
                    validation_path=validation_path,
                    dataset_manifest_path=dataset_manifest_path,
                    registry_path=registry_path,
                    pins=pins,
                )
                if completed_inputs != initial_inputs:
                    raise DenseTrainingError("verified training inputs changed during training")
                completed_base = _verify_local_model_tree(
                    EXPECTED_MODEL_ID,
                    EXPECTED_REVISION,
                    base_model_path,
                    base_model_manifest_path,
                    base_model_manifest_sha256,
                )
                if completed_base != initial_base:
                    raise DenseTrainingError("base-model observation changed during training")
                staged_files = _output_files(workspace.root)
                completed_git = verify_detached_git_revision(
                    pins.git_revision,
                    allowed_untracked=staged_files,
                )
                if completed_git != initial_git:
                    raise DenseTrainingError("source snapshot identity changed during training")
                staged_adapter = workspace.work / "adapter"
                final_adapter = workspace.root / "adapter"
                if final_adapter.exists() or not staged_adapter.is_dir():
                    raise DenseTrainingError("verified adapter staging boundary is invalid")
                staged_adapter.rename(final_adapter)
                shutil.rmtree(workspace.work)
                manifest: dict[str, Any] = {
                    "schema_version": 1,
                    "created_at": datetime.now(UTC).isoformat(),
                    "project": config.get("project"),
                    "base_model": {
                        "id": EXPECTED_MODEL_ID,
                        "revision": EXPECTED_REVISION,
                        "local_path": str(initial_base.root),
                        "manifest_path": str(initial_base.manifest_path),
                        "manifest_sha256": initial_base.manifest_sha256,
                        "file_count": initial_base.file_count,
                        "total_size": initial_base.total_size,
                        "local_files_only": True,
                    },
                    "configuration": {
                        "path": str(config_file),
                        "sha256": observed_config_sha256,
                    },
                    "input": {
                        "train": {
                            "path": str(initial_inputs.train_path),
                            "sha256": initial_inputs.train_sha256,
                            "record_count": initial_inputs.train_record_count,
                        },
                        "validation": (
                            {
                                "path": str(initial_inputs.validation_path),
                                "sha256": initial_inputs.validation_sha256,
                                "record_count": initial_inputs.validation_record_count,
                            }
                            if initial_inputs.validation_path is not None
                            else None
                        ),
                        "dataset_manifest": {
                            "path": str(initial_inputs.dataset_manifest_path),
                            "sha256": initial_inputs.dataset_manifest_sha256,
                            "dataset_sha256": initial_inputs.dataset_sha256,
                        },
                        "registry": {
                            "path": str(initial_inputs.registry_path),
                            "canonical_sha256": initial_inputs.registry_sha256,
                        },
                        "source_licenses": [
                            {"source_id": source, "license_id": license_id}
                            for source, license_id in initial_inputs.source_licenses
                        ],
                    },
                    "training": dict(config["training"]),
                    "training_observation": observation.to_dict(),
                    "effective_training_invariants": {
                        "prompt_completion": True,
                        "completion_boundary_verified": True,
                        "enable_thinking": False,
                        "completion_only_loss": True,
                        "packing": False,
                        "push_to_hub": False,
                        "reporting": False,
                        "resume_from_checkpoint": False,
                        "checkpoint_saving": False,
                        "private_immutable_input_snapshots": True,
                        "private_datasets_cache": True,
                        "clean_detached_source_snapshot": True,
                        "max_steps": max_steps,
                    },
                    "adapter": {
                        "path": str(final_adapter),
                        **observation.adapter.to_dict(),
                    },
                    "environment": {
                        "python": platform.python_version(),
                        "git_revision": completed_git.revision,
                        **{
                            name: _package_version(name)
                            for name in sorted(EXPECTED_RUNTIME_VERSIONS)
                        },
                    },
                }
                json.dumps(manifest, allow_nan=False)
                write_json_exclusive(workspace.root / "run-manifest.json", manifest)
                final_files = _output_files(workspace.root)
                final_git = verify_detached_git_revision(
                    pins.git_revision,
                    allowed_untracked=final_files,
                )
                if final_git != initial_git:
                    raise DenseTrainingError("source snapshot changed at publication boundary")
                expected_output = {
                    "run-manifest.json",
                    *{
                        path.relative_to(workspace.root).as_posix()
                        for path in final_files
                        if path.is_relative_to(final_adapter)
                    },
                }
                observed_output = {
                    path.relative_to(workspace.root).as_posix() for path in final_files
                }
                if observed_output != expected_output:
                    raise DenseTrainingError("published output contains unexpected files")
                return manifest


__all__ = [
    "AdapterExpectation",
    "AdapterVerification",
    "BaseModelObservation",
    "DenseTrainingError",
    "TrainingObservation",
    "TrainingPins",
    "train_sft",
    "verify_detached_git_revision",
]
