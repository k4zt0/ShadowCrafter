#!/usr/bin/env python3
"""Run fail-closed dense 9B SFT from fully pinned local artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shadowcrafter.training.sft import DenseTrainingError, TrainingPins, train_sft


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the pinned local Ornith-1.5-9B tree from immutable data inputs. "
            "The source checkout must be clean and detached at --git-revision."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--validation-sha256")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--base-model-manifest", type=Path, required=True)
    parser.add_argument("--base-model-manifest-sha256", required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    pins = TrainingPins(
        config_sha256=args.config_sha256,
        train_sha256=args.train_sha256,
        validation_sha256=args.validation_sha256,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        registry_sha256=args.registry_sha256,
        git_revision=args.git_revision,
    )
    try:
        manifest = train_sft(
            config_path=args.config,
            train_path=args.train,
            validation_path=args.validation,
            dataset_manifest_path=args.dataset_manifest,
            registry_path=args.registry,
            base_model_path=args.base_model,
            base_model_manifest_path=args.base_model_manifest,
            base_model_manifest_sha256=args.base_model_manifest_sha256,
            output_dir=args.output_dir,
            pins=pins,
            max_steps=args.max_steps,
        )
    except (DenseTrainingError, OSError, ValueError) as error:
        print(f"9B dense SFT refused: {error}", file=sys.stderr)
        return 2
    print("Dense SFT completed; the local safe LoRA adapter was verified and not uploaded.")
    print(f"Adapter: {manifest['adapter']['path']}")
    print(f"Run manifest: {args.output_dir / 'run-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
