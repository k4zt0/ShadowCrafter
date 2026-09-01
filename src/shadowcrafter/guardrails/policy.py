"""Deterministic minimum safety policy independent of model output."""

from __future__ import annotations

from shadowcrafter.schemas import ActionDecision, ActionRequest, RiskTier


def decide_action(request: ActionRequest) -> ActionDecision:
    """Fail closed for destructive, credential, evasion, or unauthorized actions."""
    if request.destructive:
        return ActionDecision(
            allowed=False,
            risk_tier=RiskTier.DISALLOWED,
            reason="Destructive actions are not available through ShadowCrafter.",
        )
    if request.requests_credentials:
        return ActionDecision(
            allowed=False,
            risk_tier=RiskTier.DISALLOWED,
            reason="Credential theft or collection is disallowed.",
        )
    if request.requests_evasion:
        return ActionDecision(
            allowed=False,
            risk_tier=RiskTier.DISALLOWED,
            reason="Malware evasion and stealth guidance is disallowed.",
        )
    if request.sandboxed and request.authorization_evidence:
        return ActionDecision(
            allowed=True,
            risk_tier=RiskTier.DUAL_USE_CONTROLLED,
            reason="Authorized, sandboxed validation is permitted with human approval.",
            requires_human_approval=True,
        )
    if request.intent in {
        "detect",
        "classify",
        "triage",
        "report",
        "remediate",
        "write_detection_rule",
        "incident_response",
    }:
        return ActionDecision(
            allowed=True,
            risk_tier=RiskTier.DEFENSIVE,
            reason="Defensive analysis is permitted.",
        )
    return ActionDecision(
        allowed=False,
        risk_tier=RiskTier.DUAL_USE_CONTROLLED,
        reason="Unscoped dual-use action requires explicit authorization and a sandbox.",
        requires_human_approval=True,
    )
