#!/usr/bin/env python3
"""Produce one offline frozen CTIBench prediction bundle on the remote GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shadowcrafter.evaluation.inference import InferenceError, run_frozen_inference


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a hash-pinned local Ornith base plus verified PEFT adapter against the "
            "frozen CTIBench inputs. No Hub access, scoring, resume, or overwrite is allowed."
        )
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument(
        "--request-sha256",
        required=True,
        help="Exact lowercase SHA-256 of the immutable inference request",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        manifest = run_frozen_inference(args.request, args.request_sha256)
    except (InferenceError, OSError, ValueError):
        # Do not echo library exceptions: benchmark examples and raw generations
        # are deliberately excluded from process logs.
        print("Frozen inference refused; inspect hash-only supervisor evidence.", file=sys.stderr)
        return 2
    predictions = manifest["predictions"]
    print(
        json.dumps(
            {
                "completed": True,
                "evaluation_id": manifest["evaluation_id"],
                "predictions_sha256": predictions["sha256"],
                "record_count": predictions["record_count"],
                "scores_computed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
