from shadowcrafter.guardrails.policy import decide_action
from shadowcrafter.schemas import ActionRequest, RiskTier


def test_defensive_action_is_allowed() -> None:
    decision = decide_action(ActionRequest(intent="triage"))
    assert decision.allowed
    assert decision.risk_tier == RiskTier.DEFENSIVE


def test_destructive_action_is_blocked() -> None:
    decision = decide_action(ActionRequest(intent="execute", destructive=True, sandboxed=True))
    assert not decision.allowed
    assert decision.risk_tier == RiskTier.DISALLOWED


def test_sandboxed_dual_use_requires_approval() -> None:
    decision = decide_action(
        ActionRequest(
            intent="validate",
            target="local-lab",
            sandboxed=True,
            authorization_evidence="engagement-42",
        )
    )
    assert decision.allowed
    assert decision.requires_human_approval
