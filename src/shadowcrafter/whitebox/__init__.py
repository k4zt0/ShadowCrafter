"""Authorized static-only white-box vulnerability assessment."""

from shadowcrafter.whitebox.assessor import (
    AuthorizedWhiteBoxAssessor,
    WhiteBoxAuthorizationError,
    WhiteBoxLimitError,
    compute_python_source_snapshot_sha256,
)
from shadowcrafter.whitebox.models import (
    StaticEvidenceRecord,
    WhiteBoxAssessmentResult,
    WhiteBoxAuthorizationArtifact,
)

__all__ = [
    "AuthorizedWhiteBoxAssessor",
    "StaticEvidenceRecord",
    "WhiteBoxAuthorizationArtifact",
    "WhiteBoxAuthorizationError",
    "WhiteBoxAssessmentResult",
    "WhiteBoxLimitError",
    "compute_python_source_snapshot_sha256",
]
