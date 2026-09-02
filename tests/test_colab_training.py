import json
from pathlib import Path

import pytest

from shadowcrafter.training.colab import (
    ColabCheckpointBinding,
    ColabTrainingError,
    latest_complete_checkpoint,
    write_checkpoint_marker,
)


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
