"""PDDL planner plugin."""

from typing import List, Dict, Any

from nesy.core.registry import register_plugin
from nesy.reasoning.base import PlannerBase


@register_plugin(PlannerBase, "pddl")
class PDDLPlannerPlugin(PlannerBase):
    """
    PDDL-based planner (wraps the existing PDDLPlanner/PDDLDomain).

    Creates a domain on-the-fly from the provided actions config.
    """

    def __init__(self, **kwargs):
        from nesy.reasoning.planning.pddl import PDDLDomain
        self.domain = PDDLDomain(kwargs.get("domain_name", "default"))
        self._actions: List[str] = []

    def add_action(
        self,
        name: str,
        parameters: List[str],
        preconditions: List[str],
        effects: List[str],
    ) -> None:
        """Add an action to the planning domain."""
        self.domain.add_action(name, parameters, preconditions, effects)
        self._actions.append(name)

    def plan(
        self,
        initial_state: Dict[str, Any],
        goal: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        from nesy.reasoning.planning.pddl import PDDLPlanner, PDDLProblem

        initial = initial_state.get("predicates", [])
        goal_preds = goal.get("predicates", [])

        problem = PDDLProblem(
            name="task",
            domain=self.domain,
            objects=initial_state.get("objects", []),
            initial_state=initial,
            goal=goal_preds,
        )

        planner = PDDLPlanner(self.domain)
        plan_result = planner.plan(problem)

        return [
            {"action": str(action)}
            for action in (plan_result or [])
        ]

    def get_available_actions(self) -> List[str]:
        return self._actions
