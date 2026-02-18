"""
Reasoning Base Classes - Abstract Interfaces for Backend-Agnostic Reasoning.

Defines the ABC contracts for logic engines and planners.
Concrete backends (Scallop, Minalog, PDDL, etc.) register via the plugin system.

Example:
    >>> from nesy.core.registry import Registry
    >>> from nesy.reasoning.base import LogicEngineBase, PlannerBase
    >>>
    >>> engine = Registry.create(LogicEngineBase, "minalog")
    >>> engine.add_fact("on", "cup1", "table1")
    >>> results = engine.query("on")
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional, Dict


class LogicEngineBase(ABC):
    """
    Abstract base class for logic/inference engines.

    All reasoning backends (Scallop, Minalog, forward-chainer, etc.)
    must implement this interface.
    """

    @abstractmethod
    def add_relation(self, name: str, types: List[str]) -> None:
        """Define a relation schema."""
        ...

    @abstractmethod
    def add_fact(self, relation: str, *args) -> None:
        """Add a ground fact."""
        ...

    @abstractmethod
    def add_rule(self, rule: str) -> None:
        """Add an inference rule (Datalog syntax)."""
        ...

    @abstractmethod
    def query(self, relation: str) -> List[Tuple]:
        """Query a relation and return all matching tuples."""
        ...

    @abstractmethod
    def clear_facts(self) -> None:
        """Clear all facts (keep rules and relations)."""
        ...

    @abstractmethod
    def get_num_facts(self) -> int:
        """Return the number of facts."""
        ...

    @abstractmethod
    def get_num_rules(self) -> int:
        """Return the number of rules."""
        ...

    def get_relations(self) -> List[str]:
        """Return all defined relation names (optional)."""
        return []


class PlannerBase(ABC):
    """
    Abstract base class for planners.

    All planning backends (PDDL, hierarchical, etc.) must implement this.
    """

    @abstractmethod
    def plan(
        self,
        initial_state: Dict[str, Any],
        goal: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate a plan from initial state to goal.

        Args:
            initial_state: Starting state description
            goal: Goal state description

        Returns:
            Ordered list of actions (each a dict with 'action', 'params', etc.)
        """
        ...

    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """List all available action types."""
        ...
