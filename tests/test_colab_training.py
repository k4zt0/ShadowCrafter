import json
from pathlib import Path

import pytest

from shadowcrafter.training.colab import (
    ColabCheckpointBinding,
    ColabTrainingError,
    latest_complete_checkpoint,
    train_resumable_colab,
    write_checkpoint_marker,
)
from shadowcrafter.training.training_safety import TrainingPins


def _binding(suffix: str = "a") -> ColabCheckpointBinding:
    return ColabCheckpointBinding(
        source_revision=suffix * 40,
        config_sha256=suffix * 64,
        train_sha256=suffix * 64,
        dataset_manifest_sha256=suffix * 64,
        dataset_sha256=suffix * 64,
    )


def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    files: dict[str, bytes] = {
        "adapter_config.json": b"{}\n",
        "adapter_model.safetensors": b"safe-placeholder",
        "optimizer.pt": b"trusted-private-optimizer-state",
        "rng_state.pth": b"trusted-private-rng-state",
        "scheduler.pt": b"trusted-private-scheduler-state",
        "trainer_state.json": json.dumps({"global_step": step}).encode() + b"\n",
    }
    for name, content in files.items():
        (checkpoint / name).write_bytes(content)
    return checkpoint


def test_colab_checkpoint_marker_selects_latest_complete_and_ignores_partial(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    first = _checkpoint(root, 100)
    second = _checkpoint(root, 200)
    partial = _checkpoint(root, 300)
    binding = _binding()
    write_checkpoint_marker(first, binding)
    second_marker = write_checkpoint_marker(second, binding)

    selected, marker = latest_complete_checkpoint(root, binding)

    assert selected == second
    assert marker == second_marker
    assert marker["global_step"] == 200
    assert not (partial / ".shadowcrafter-complete.json").exists()


def test_colab_checkpoint_marker_rejects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = _checkpoint(root, 100)
    binding = _binding()
    write_checkpoint_marker(checkpoint, binding)
    (checkpoint / "optimizer.pt").write_bytes(b"changed")

    with pytest.raises(ColabTrainingError, match="hashes differ"):
        latest_complete_checkpoint(root, binding)


def test_colab_checkpoint_marker_rejects_other_training_lineage(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = _checkpoint(root, 100)
    write_checkpoint_marker(checkpoint, _binding("a"))

    with pytest.raises(ColabTrainingError, match="binding"):
        latest_complete_checkpoint(root, _binding("b"))


def test_colab_training_rejects_unknown_checkpoint_storage(tmp_path: Path) -> None:
    pins = TrainingPins(
        config_sha256="a" * 64,
        train_sha256="b" * 64,
        validation_sha256=None,
        dataset_manifest_sha256="c" * 64,
        registry_sha256="d" * 64,
        git_revision="e" * 40,
    )

    with pytest.raises(ColabTrainingError, match="checkpoint_storage"):
        train_resumable_colab(
            config_path=tmp_path / "config.yaml",
            train_path=tmp_path / "train.jsonl",
            dataset_manifest_path=tmp_path / "manifest.json",
            registry_path=tmp_path / "sources.yaml",
            base_model_path=tmp_path / "model",
            base_model_manifest_path=tmp_path / "model-manifest.json",
            base_model_manifest_sha256="f" * 64,
            checkpoint_root=tmp_path / "checkpoints",
            final_dir=tmp_path / "final",
            pins=pins,
            checkpoint_storage="external",  # type: ignore[arg-type]
        )
