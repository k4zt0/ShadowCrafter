"""Authorized, passive black-box HTTP/TLS assessment.

The public entry points in this package require an immutable authorization
artifact and never send payloads, credentials, or state-changing requests.
"""

from shadowcrafter.blackbox.assessor import (
    AuthorizedBlackBoxAssessor,
    assess_authorized_targets,
    run_authorized_assessment,
)
from shadowcrafter.blackbox.authorization import AuthorizationError, read_blackbox_scope
from shadowcrafter.blackbox.models import (
    AuthorizationArtifact,
    BlackBoxAssessmentResult,
    EvidenceRecord,
    PassiveBodySignal,
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
    "PassiveBodySignal",
    "SafetyLimits",
    "TLSMetadata",
    "assess_authorized_targets",
    "read_blackbox_scope",
    "run_authorized_assessment",
]
