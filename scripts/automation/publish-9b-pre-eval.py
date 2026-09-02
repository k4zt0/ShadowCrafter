#!/usr/bin/env python3
"""Publish one staged public ShadowCrafter-9B release before accuracy evaluation."""

from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token

from shadowcrafter.data.manifest import sha256_file, write_json_exclusive
from shadowcrafter.release.remote_huggingface import publish_remote_official_release

_REPO_ID = "KaztoRay/ShadowCrafter-9B"
_RELEASE_ID = "v1.0-pre-eval"


class PreEvaluationPublishError(RuntimeError):
    """The staged pre-evaluation publication failed its fixed contract."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    return parser.parse_args()


def _regular(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    metadata = resolved.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise PreEvaluationPublishError(f"expected a regular non-symlink file: {path}")
    return resolved


def _load_ready(release_root: Path) -> tuple[Path, dict[str, Any]]:
    publication = release_root / "publication"
    if publication.is_symlink() or not publication.is_dir():
        raise PreEvaluationPublishError("publication directory is missing or linked")
    for path in publication.rglob("*"):
        if path.is_symlink():
            raise PreEvaluationPublishError("publication evidence contains a symbolic link")
    ready_path = _regular(publication / "ready.json")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    evaluation = ready.get("evaluation")
    if (
        ready.get("version") != _RELEASE_ID
        or ready.get("release_id") != _RELEASE_ID
        or ready.get("repo_id") != _REPO_ID
        or ready.get("visibility") != "public"
        or ready.get("commercial_release") is not False
        or not isinstance(evaluation, dict)
        or evaluation.get("status") != "not-yet-evaluated"
        or not isinstance(evaluation.get("reason"), str)
        or evaluation.get("quality_target_met") is not None
    ):
        raise PreEvaluationPublishError("ready evidence is not the fixed pre-evaluation release")
    return publication, ready


def main() -> int:
    args = _arguments()
    try:
        release_root = args.release_root.resolve(strict=True)
        publication, ready = _load_ready(release_root)
        key = _regular(args.ssh_key)
        if key.stat().st_mode & 0o077:
            raise PreEvaluationPublishError("SSH key permissions are broader than 0600")
        api = HfApi(token=get_token())
        before = api.model_info(_REPO_ID, files_metadata=False)
        if before.private is not False or not isinstance(before.sha, str):
            raise PreEvaluationPublishError("Hugging Face repository is not public")
        approvals: dict[str, dict[str, str]] = {}
        for name in ("artifact_integrity", "provenance", "license", "privacy", "safety"):
            remote_reference = ready["approvals"][name]
            local_path = _regular(publication / remote_reference["path"])
            if sha256_file(local_path) != remote_reference["sha256"]:
                raise PreEvaluationPublishError(f"{name} approval checksum differs")
            approvals[name] = {
                "path": local_path.relative_to(release_root).as_posix(),
                "sha256": remote_reference["sha256"],
            }
        manifest = {
            "schema_version": 1,
            "release_id": _RELEASE_ID,
            "repo_id": _REPO_ID,
            "release_tier": "Official Release",
            "visibility": "public",
            "commercial_release": False,
            "parent_commit": before.sha,
            "candidate_checkpoint_sha256": ready["candidate_checkpoint_sha256"],
            "remote_root": ready["remote_root"],
            "ssh": {"host": "capella.cloud.vessl.ai", "port": 31044, "user": "root"},
            "files": ready["files"],
            "total_bytes": ready["total_bytes"],
            "evaluation": {
                "status": "not-yet-evaluated",
                "evidence_manifest_sha256": None,
                "reason": ready["evaluation"]["reason"],
            },
            "approvals": approvals,
        }
        manifest_path = release_root / "remote-release-manifest.json"
        write_json_exclusive(manifest_path, manifest)
        result = publish_remote_official_release(
            manifest_path,
            manifest_sha256=sha256_file(manifest_path),
            ssh_key=key,
            evidence_path=None,
            gate_config=Path("configs/eval/release-gates.yaml").resolve(strict=True),
        )
        api.create_tag(
            _REPO_ID,
            tag=_RELEASE_ID,
            revision=result.commit_sha,
            repo_type="model",
            exist_ok=False,
        )
        receipt = result.as_dict()
        receipt["tag"] = _RELEASE_ID
        write_json_exclusive(release_root / "publication-receipt.json", receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(f"pre-evaluation publication stopped: {type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
