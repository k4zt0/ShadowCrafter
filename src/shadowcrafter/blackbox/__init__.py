"""Authorized, passive black-box HTTP/TLS assessment.

The public entry points in this package require an immutable authorization
artifact and never send payloads, credentials, or state-changing requests.
"""

from shadowcrafter.blackbox.assessor import (
    AuthorizedBlackBoxAssessor,
    assess_authorized_targets,
    run_authorized_assessment,
)
from shadowcrafter.blackbox.authorization import AuthorizationError
from shadowcrafter.blackbox.models import (
    AuthorizationArtifact,
    BlackBoxAssessmentResult,
    EvidenceRecord,
    SafetyLimits,
    TLSMetadata,
)
from shadowcrafter.blackbox.network import NetworkSafetyError

__all__ = [
    "AuthorizationArtifact",
    "AuthorizationError",
    "AuthorizedBlackBoxAssessor",
    "BlackBoxAssessmentResult",
    "EvidenceRecord",
    "NetworkSafetyError",
    "SafetyLimits",
    "TLSMetadata",
    "assess_authorized_targets",
    "run_authorized_assessment",
]
