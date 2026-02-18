"""Agent plugin registrations."""

from nesy.core.registry import register_plugin
from nesy.agents.base import AgentBase
from nesy.agents.autonomous import AutonomousAgent, AgentConfig


@register_plugin(AgentBase, "autonomous")
class AutonomousAgentPlugin(AgentBase):
    """
    Plugin wrapper for AutonomousAgent.

    Adapts the existing AutonomousAgent to the AgentBase ABC interface.
    """

    def __init__(self, config: dict = None, **kwargs):
        agent_config = AgentConfig(**(config or {}))
        self._agent = AutonomousAgent(config=agent_config)

    def observe(self, sensor_data):
        import numpy as np
        if isinstance(sensor_data, np.ndarray):
            return self._agent.observe(sensor_data).__dict__
        return {"status": "no_observation"}

    def reason(self):
        return self._agent.reason()

    def plan(self, goal):
        from nesy.agents.autonomous import Goal
        if isinstance(goal, dict):
            g = Goal(**goal)
        elif isinstance(goal, Goal):
            g = goal
        else:
            g = Goal(description=str(goal))
        result = self._agent.plan(g)
        if result is None:
            return []
        return [{"action": str(a)} for a in result]

    def execute(self, plan):
        success = self._agent.execute(plan)
        return {"success": success}

    def get_state(self):
        return {
            "state": self._agent.state.value,
            "metrics": self._agent.metrics,
        }
