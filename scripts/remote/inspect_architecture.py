"""Inspect empty Qwen3.5 text architectures before choosing LoRA targets."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import transformers
import typer
import yaml
from accelerate import init_empty_weights
from transformers import AutoConfig


def main(config_path: Path) -> None:
    project = yaml.safe_load(config_path.read_text())
    base = project["base_model"]
    config = AutoConfig.from_pretrained(
        base["id"],
        revision=base["revision"],
        trust_remote_code=False,
    )
    if getattr(config, "_commit_hash", None) != base["revision"]:
        raise RuntimeError("resolved model revision does not match the immutable pin")
    text_config = getattr(config, "text_config", config)
    model_class = getattr(transformers, base["text_model_class"])
    with init_empty_weights():
        model = model_class(text_config)

    linear_suffixes: set[str] = set()
    matching_modules: list[str] = []
    requested = set(project["training"]["target_modules"])
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.rsplit(".", maxsplit=1)[-1]
            linear_suffixes.add(suffix)
            if suffix in requested:
                matching_modules.append(name)
    routed_parameters = [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in model.named_parameters()
        if parameter.ndim == 3 or "router" in name.lower()
    ]
    expected_count = int(project["training"]["expected_target_module_count"])
    payload = {
        "model": base["id"],
        "revision": base["revision"],
        "model_class": base["text_model_class"],
        "linear_suffixes": sorted(linear_suffixes),
        "requested_targets": sorted(requested),
        "missing_requested_targets": sorted(requested - linear_suffixes),
        "matched_target_count": len(matching_modules),
        "expected_target_count": expected_count,
        "target_count_matches": len(matching_modules) == expected_count,
        "matched_target_sample": matching_modules[:20],
        "routed_or_3d_parameters": routed_parameters,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if requested - linear_suffixes or len(matching_modules) != expected_count:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    typer.run(main)
