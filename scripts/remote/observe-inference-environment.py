#!/usr/bin/env python3
"""Write one exclusive runtime-identity manifest for a later pinned inference request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from shadowcrafter.evaluation.inference import InferenceError, observe_runtime_environment


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observe the exact Python/package/CUDA identity without loading a model."
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.output.exists() or args.output.is_symlink():
        print("Refusing to overwrite an environment manifest.", file=sys.stderr)
        return 2
    try:
        observation = observe_runtime_environment()
        content = (
            json.dumps(
                observation.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        args.output.chmod(0o400)
    except (InferenceError, OSError, ValueError):
        print("Environment observation refused.", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "completed": True,
                "sha256": hashlib.sha256(content).hexdigest(),
                "schema_version": 1,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
