"""Hugging Face publication boundaries.

The legacy directory uploader is intentionally disabled so a stale local mirror cannot
become a release source. The supported publisher in ``remote_huggingface`` streams
manifest-verified remote artifacts while keeping Hugging Face credentials local.
"""

from __future__ import annotations

from pathlib import Path


def publish_private_model(
    repo_id: str,
    model_dir: Path,
    evidence_path: Path,
    gate_config: Path,
) -> str:
    del repo_id, model_dir, evidence_path, gate_config
    raise RuntimeError(
        "local directory publication is disabled by the immutable release policy; "
        "use the approved remote-stream publisher that verifies the remote manifest and "
        "keeps the Hugging Face token local"
    )
