"""Independently verify a saved PEFT adapter and emit a checksum report."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
import typer
from safetensors import safe_open


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(
    adapter_dir: Path,
    report: Path,
    expected_base_model: str,
    expected_revision: str,
    expected_tensor_count: int,
    expected_parameter_count: int,
) -> None:
    """Fail unless the adapter is portable, LoRA-only, finite, and complete."""
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError("adapter_config.json and adapter_model.safetensors are required")
    config = json.loads(config_path.read_text())
    if config.get("base_model_name_or_path") != expected_base_model:
        raise RuntimeError("adapter base_model_name_or_path does not match the upstream ID")
    if config.get("revision") != expected_revision:
        raise RuntimeError("adapter revision does not match the immutable upstream pin")

    unsafe_tokens = ("router", "experts", "vision", "visual", "mtp")
    tensor_count = 0
    parameter_count = 0
    nonzero_count = 0
    unsafe_keys: list[str] = []
    nonfinite_keys: list[str] = []
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        for key in keys:
            tensor = handle.get_tensor(key)
            tensor_count += 1
            parameter_count += tensor.numel()
            if "lora_A" not in key and "lora_B" not in key:
                unsafe_keys.append(key)
            if any(token in key.lower() for token in unsafe_tokens):
                unsafe_keys.append(key)
            if not bool(torch.isfinite(tensor).all().item()):
                nonfinite_keys.append(key)
            nonzero_count += int(torch.count_nonzero(tensor).item())
    if tensor_count != expected_tensor_count:
        raise RuntimeError(
            f"adapter tensor count mismatch: expected {expected_tensor_count}, got {tensor_count}"
        )
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            "adapter parameter count mismatch: "
            f"expected {expected_parameter_count}, got {parameter_count}"
        )
    if unsafe_keys:
        raise RuntimeError(f"adapter contains a non-LoRA or forbidden tensor: {unsafe_keys[:8]}")
    if nonfinite_keys:
        raise RuntimeError(f"adapter contains non-finite tensors: {nonfinite_keys[:8]}")
    if nonzero_count == 0:
        raise RuntimeError("adapter contains no nonzero values")

    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "adapter_dir": str(adapter_dir.resolve()),
        "base_model": expected_base_model,
        "revision": expected_revision,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "nonzero_parameter_values": nonzero_count,
        "adapter_config_sha256": _sha256(config_path),
        "adapter_weights_sha256": _sha256(weights_path),
        "finite": True,
        "lora_only": True,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    typer.run(main)
