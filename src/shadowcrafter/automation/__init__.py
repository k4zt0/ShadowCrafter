"""Durable, fail-closed background workflow automation."""

from shadowcrafter.automation.controller import AutomationController
from shadowcrafter.automation.models import AutomationConfig, WorkflowState

__all__ = ["AutomationConfig", "AutomationController", "WorkflowState"]
