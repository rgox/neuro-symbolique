"""
PDDL Planning Module.

Implements Planning Domain Definition Language (PDDL) support for
goal-driven action planning.

Features:
- Domain/problem parsing
- Planning algorithms (A*, forward search)
- Action execution
- Integration with reasoning engine

Example:
    >>> from nesy.reasoning.planning import PDDLDomain, PDDLPlanner
    >>> 
    >>> # Define domain
    >>> domain = PDDLDomain("robot")
    >>> domain.add_action("move", 
    ...     parameters=["from", "to"],
    ...     preconditions=["at(from)", "connected(from, to)"],
    ...     effects=["not at(from)", "at(to)"]
    ... )
    >>> 
    >>> # Plan
    >>> planner = PDDLPlanner(domain)
    >>> plan = planner.plan(
    ...     initial=["at(room1)", "connected(room1, room2)"],
    ...     goal=["at(room2)"]
    ... )
    >>> # → [move(room1, room2)]
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PredicateType(Enum):
    """Type of PDDL predicate."""
    POSITIVE = "positive"  # at(X)
    NEGATIVE = "negative"  # not at(X)


@dataclass
class Predicate:
    """PDDL predicate (fact)."""
    name: str
    params: List[str] = field(default_factory=list)
    is_negative: bool = False
    
    def __str__(self) -> str:
        params_str = ", ".join(self.params) if self.params else ""
        pred = f"{self.name}({params_str})"
        return f"not {pred}" if self.is_negative else pred
    
    def __hash__(self) -> int:
        return hash((self.name, tuple(self.params), self.is_negative))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Predicate):
            return False
        return (self.name == other.name and 
                self.params == other.params and
                self.is_negative == other.is_negative)

    def __lt__(self, other) -> bool:
        """Comparison for sorting."""
        if not isinstance(other, Predicate):
            return NotImplemented
        return (self.name, self.params, self.is_negative) < (other.name, other.params, other.is_negative)


@dataclass
class Action:
    """PDDL action schema."""
    name: str
    parameters: List[str] = field(default_factory=list)
    preconditions: List[Predicate] = field(default_factory=list)
    effects: List[Predicate] = field(default_factory=list)
    
    def is_applicable(self, state: Set[Predicate]) -> bool:
        """Check if action is applicable in state."""
        for precond in self.preconditions:
            if precond.is_negative:
                # Negative precondition: fact should NOT be in state
                positive = Predicate(precond.name, precond.params, is_negative=False)
                if positive in state:
                    return False
            else:
                # Positive precondition: fact should be in state
                if precond not in state:
                    return False
        return True
    
    def apply(self, state: Set[Predicate]) -> Set[Predicate]:
        """Apply action to state, returning new state."""
        new_state = state.copy()
        
        for effect in self.effects:
            if effect.is_negative:
                # Negative effect: remove fact
                positive = Predicate(effect.name, effect.params, is_negative=False)
                new_state.discard(positive)
            else:
                # Positive effect: add fact
                new_state.add(effect)
        
        return new_state
    
    def __str__(self) -> str:
        params_str = ", ".join(self.parameters) if self.parameters else ""
        return f"{self.name}({params_str})"

    def __lt__(self, other) -> bool:
        """Comparison for sorting/heapq."""
        if not isinstance(other, Action):
            return NotImplemented
        return (self.name, tuple(self.parameters)) < (other.name, tuple(other.parameters))


@dataclass
class PDDLDomain:
    """PDDL planning domain."""
    name: str
    predicates: Dict[str, List[str]] = field(default_factory=dict)  # predicate_name → param_types
    actions: Dict[str, Action] = field(default_factory=dict)
    
    def add_predicate(self, name: str, param_types: List[str]):
        """Add predicate definition."""
        self.predicates[name] = param_types
    
    def add_action(
        self,
        name: str,
        parameters: List[str],
        preconditions: List[str],
        effects: List[str]
    ):
        """
        Add action to domain.
        
        Args:
            name: Action name
            parameters: List of parameter names
            preconditions: List of precondition strings
            effects: List of effect strings
        """
        # Parse preconditions
        preconds = [self._parse_predicate(p) for p in preconditions]
        
        # Parse effects
        effs = [self._parse_predicate(e) for e in effects]
        
        action = Action(
            name=name,
            parameters=parameters,
            preconditions=preconds,
            effects=effs
        )
        
        self.actions[name] = action
    
    def _parse_predicate(self, pred_str: str) -> Predicate:
        """Parse predicate string to Predicate object."""
        pred_str = pred_str.strip()
        
        # Check for negation
        is_negative = False
        if pred_str.startswith("not "):
            is_negative = True
            pred_str = pred_str[4:].strip()
        
        # Parse name and params
        if "(" in pred_str:
            name = pred_str[:pred_str.index("(")]
            params_str = pred_str[pred_str.index("(")+1:pred_str.index(")")]
            params = [p.strip() for p in params_str.split(",") if p.strip()]
        else:
            name = pred_str
            params = []
        
        return Predicate(name, params, is_negative)
    
    def get_action(self, name: str) -> Optional[Action]:
        """Get action by name."""
        return self.actions.get(name)


@dataclass
class PDDLProblem:
    """PDDL planning problem."""
    name: str
    domain: PDDLDomain
    initial_state: Set[Predicate] = field(default_factory=set)
    goal: Set[Predicate] = field(default_factory=set)
    
    def add_initial_fact(self, fact_str: str):
        """Add fact to initial state."""
        pred = self.domain._parse_predicate(fact_str)
        self.initial_state.add(pred)
    
    def add_goal(self, goal_str: str):
        """Add goal predicate."""
        pred = self.domain._parse_predicate(goal_str)
        self.goal.add(pred)
    
    def is_goal_satisfied(self, state: Set[Predicate]) -> bool:
        """Check if goal is satisfied in state."""
        for goal_pred in self.goal:
            if goal_pred.is_negative:
                # Negative goal: fact should NOT be in state
                positive = Predicate(goal_pred.name, goal_pred.params, is_negative=False)
                if positive in state:
                    return False
            else:
                # Positive goal: fact should be in state
                if goal_pred not in state:
                    return False
        return True


class PDDLPlanner:
    """
    PDDL planner with multiple search algorithms.
    
    Supports:
    - Forward search (breadth-first)
    - A* search with heuristics
    
    Example:
        >>> planner = PDDLPlanner(domain)
        >>> plan = planner.plan(problem, algorithm="forward")
        >>> # → [action1, action2, action3]
    """
    
    def __init__(self, domain: PDDLDomain):
        """Initialize planner with domain."""
        self.domain = domain
    
    def plan(
        self,
        problem: PDDLProblem,
        algorithm: str = "forward",
        max_steps: int = 100
    ) -> Optional[List[Action]]:
        """
        Generate plan to achieve goal.
        
        Args:
            problem: Planning problem
            algorithm: "forward" or "astar"
            max_steps: Maximum planning steps
        
        Returns:
            List of actions (plan) or None if no plan found
        """
        if algorithm == "forward":
            return self._forward_search(problem, max_steps)
        elif algorithm == "astar":
            return self._astar_search(problem, max_steps)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _ground_actions(self, state: Set[Predicate]) -> List[Action]:
        """
        Ground parameterized actions with concrete values from state.
        
        Generates all possible ground instances of each action template
        by substituting parameters with values found in the state.
        
        Args:
            state: Current state predicates
            
        Returns:
            List of grounded actions
        """
        from itertools import product
        
        # Collect all constants from state
        constants = set()
        for pred in state:
            for p in pred.params:
                constants.add(p)
        
        grounded = []
        for action in self.domain.actions.values():
            if not action.parameters:
                grounded.append(action)
                continue
            
            # Try all assignments of constants to parameters
            for assignment in product(constants, repeat=len(action.parameters)):
                binding = dict(zip(action.parameters, assignment))
                
                # Ground preconditions
                ground_preconds = []
                for p in action.preconditions:
                    params = [binding.get(x, x) for x in p.params]
                    ground_preconds.append(Predicate(p.name, params, p.is_negative))
                
                # Ground effects
                ground_effects = []
                for e in action.effects:
                    params = [binding.get(x, x) for x in e.params]
                    ground_effects.append(Predicate(e.name, params, e.is_negative))
                
                ground_action = Action(
                    name=action.name,
                    parameters=list(assignment),
                    preconditions=ground_preconds,
                    effects=ground_effects,
                )
                grounded.append(ground_action)
        
        return grounded
    
    def _forward_search(
        self,
        problem: PDDLProblem,
        max_steps: int
    ) -> Optional[List[Action]]:
        """
        Forward breadth-first search.
        
        Args:
            problem: Planning problem
            max_steps: Max search depth
        
        Returns:
            Plan or None
        """
        from collections import deque
        
        # Queue of (state, plan) tuples
        queue = deque([(problem.initial_state, [])])
        visited = set()
        
        steps = 0
        while queue and steps < max_steps:
            steps += 1
            state, plan = queue.popleft()
            
            # Check if goal reached
            if problem.is_goal_satisfied(state):
                logger.info(f"Plan found in {steps} steps: {len(plan)} actions")
                return plan
            
            # Skip if visited
            state_hash = frozenset(state)
            if state_hash in visited:
                continue
            visited.add(state_hash)
            
            # Try all grounded action instances
            for action in self._ground_actions(state):
                if action.is_applicable(state):
                    new_state = action.apply(state)
                    new_plan = plan + [action]
                    queue.append((new_state, new_plan))
        
        logger.warning(f"No plan found within {max_steps} steps")
        return None
    
    def _astar_search(
        self,
        problem: PDDLProblem,
        max_steps: int
    ) -> Optional[List[Action]]:
        """
        A* search with goal distance heuristic.
        
        Args:
            problem: Planning problem
            max_steps: Max search depth
        
        Returns:
            Plan or None
        """
        import heapq
        
        # Priority queue of (f_score, g_score, state, plan)
        initial_h = self._heuristic(problem.initial_state, problem.goal)
        pq = [(initial_h, 0, problem.initial_state, [])]
        visited = set()
        
        steps = 0
        while pq and steps < max_steps:
            steps += 1
            f_score, g_score, state, plan = heapq.heappop(pq)
            
            # Check if goal reached
            if problem.is_goal_satisfied(state):
                logger.info(f"Plan found with A* in {steps} steps: {len(plan)} actions")
                return plan
            
            # Skip if visited
            state_hash = frozenset(state)
            if state_hash in visited:
                continue
            visited.add(state_hash)
            
            # Try all grounded action instances
            for action in self._ground_actions(state):
                if action.is_applicable(state):
                    new_state = action.apply(state)
                    new_g = g_score + 1  # Uniform cost
                    new_h = self._heuristic(new_state, problem.goal)
                    new_f = new_g + new_h
                    new_plan = plan + [action]
                    heapq.heappush(pq, (new_f, new_g, new_state, new_plan))
        
        logger.warning(f"No plan found with A* within {max_steps} steps")
        return None
    
    def _heuristic(self, state: Set[Predicate], goal: Set[Predicate]) -> int:
        """
        Simple heuristic: count unsatisfied goal predicates.
        
        Args:
            state: Current state
            goal: Goal predicates
        
        Returns:
            Heuristic value (lower is better)
        """
        unsatisfied = 0
        for goal_pred in goal:
            if goal_pred.is_negative:
                positive = Predicate(goal_pred.name, goal_pred.params, is_negative=False)
                if positive in state:
                    unsatisfied += 1
            else:
                if goal_pred not in state:
                    unsatisfied += 1
        return unsatisfied


class ActionExecutor:
    """
    Execute PDDL plans in a world.
    
    Example:
        >>> executor = ActionExecutor(world_state)
        >>> executor.execute_plan(plan)
    """
    
    def __init__(self, initial_state: Set[Predicate]):
        """Initialize with initial world state."""
        self.state = initial_state.copy()
        self.history: List[Tuple[Action, Set[Predicate]]] = []
    
    def execute(self, action: Action) -> bool:
        """
        Execute single action.
        
        Args:
            action: Action to execute
        
        Returns:
            True if successful, False if preconditions not met
        """
        # Check preconditions
        if not action.is_applicable(self.state):
            logger.warning(f"Action {action} not applicable in current state")
            return False
        
        # Apply action
        new_state = action.apply(self.state)
        
        # Record
        self.history.append((action, self.state.copy()))
        self.state = new_state
        
        logger.info(f"Executed: {action}")
        return True
    
    def execute_plan(self, plan: List[Action]) -> bool:
        """
        Execute sequence of actions.
        
        Args:
            plan: List of actions
        
        Returns:
            True if all actions executed successfully
        """
        for action in plan:
            if not self.execute(action):
                return False
        return True
    
    def get_state(self) -> Set[Predicate]:
        """Get current world state."""
        return self.state.copy()
    
    def rollback(self, steps: int = 1):
        """Rollback execution by N steps."""
        for _ in range(min(steps, len(self.history))):
            action, prev_state = self.history.pop()
            self.state = prev_state
            logger.info(f"Rolled back: {action}")
