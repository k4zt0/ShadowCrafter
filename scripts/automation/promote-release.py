#!/usr/bin/env python3
"""Atomically promote a pre-hashed adapter-only remote release tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shadowcrafter.automation.promotion import (  # noqa: E402
    PromotionError,
    load_promotion_request,
    promote_release,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = load_promotion_request(args.request, args.request_sha256)
        manifest = promote_release(request, args.output_manifest)
    except (OSError, ValueError, PromotionError) as error:
        print(f"release promotion refused: {error}", file=sys.stderr)
        return 2
    print(manifest.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
