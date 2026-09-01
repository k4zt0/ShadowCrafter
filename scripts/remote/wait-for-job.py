#!/usr/bin/env python3
"""Wait for one immutable remote-worker job before starting a dependent step."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from shadowcrafter.automation.remote_worker import status


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for an existing ShadowCrafter remote-worker job to succeed."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--job-root",
        type=Path,
        default=Path("/root/ShadowCrafter/artifacts/automation"),
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not 60 <= args.timeout_seconds <= 604800:
        print("wait timeout is outside the allowed range", file=sys.stderr)
        return 2
    if not 5 <= args.poll_seconds <= 60:
        print("wait poll interval is outside the allowed range", file=sys.stderr)
        return 2

    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        observed = status(args.job_id, args.job_root)
        if observed.status == "succeeded":
            print(f"dependency succeeded: {args.job_id}")
            return 0
        if observed.status == "failed":
            print(f"dependency failed: {args.job_id}", file=sys.stderr)
            return 2
        time.sleep(args.poll_seconds)
    print(f"dependency wait timed out: {args.job_id}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
