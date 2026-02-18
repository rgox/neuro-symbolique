"""
Agents Module - Autonomous Systems.

Autonomous agents with perception-reasoning-planning-action loops.
All agents are registered as plugins via the Registry.
"""

from nesy.agents.autonomous import (
    AutonomousAgent,
    AgentConfig,
    AgentState,
    Observation,
    Goal,
)
from nesy.agents.multi_agent import (
    MultiAgentCoordinator,
    AgentRole,
    AgentMessage,
    MessageType,
    SharedKnowledge,
)
from nesy.agents.base import AgentBase

# Trigger plugin registration
import nesy.agents.plugins  # noqa: F401

__all__ = [
    "AgentBase",
    "AutonomousAgent",
    "AgentConfig",
    "AgentState",
    "Observation",
    "Goal",
    "MultiAgentCoordinator",
    "AgentRole",
    "AgentMessage",
    "MessageType",
    "SharedKnowledge",
]
