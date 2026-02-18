"""
Agent Base Class - Abstract Interface for Backend-Agnostic Agents.

Defines the ABC contract for autonomous agents.

Example:
    >>> from nesy.core.registry import Registry
    >>> from nesy.agents.base import AgentBase
    >>> agent = Registry.create(AgentBase, "autonomous")
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AgentBase(ABC):
    """
    Abstract base class for autonomous agents.

    All agent implementations must support the observe-reason-plan-execute cycle.
    """

    @abstractmethod
    def observe(self, sensor_data: Any) -> Dict[str, Any]:
        """
        Process sensor data (perception step).

        Args:
            sensor_data: Raw sensor input (image, point cloud, etc.)

        Returns:
            Observation result dict
        """
        ...

    @abstractmethod
    def reason(self) -> Dict[str, Any]:
        """
        Apply reasoning to current knowledge (inference step).

        Returns:
            Reasoning results dict
        """
        ...

    @abstractmethod
    def plan(self, goal: Any) -> List[Dict[str, Any]]:
        """
        Generate a plan to achieve a goal.

        Args:
            goal: Goal specification

        Returns:
            Ordered list of actions
        """
        ...

    @abstractmethod
    def execute(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a plan.

        Args:
            plan: List of actions to execute

        Returns:
            Execution result
        """
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        ...
