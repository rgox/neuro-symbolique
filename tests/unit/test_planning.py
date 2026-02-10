"""
Tests for PDDL Planning.

Tests domain/problem parsing, planning algorithms, and action execution.
"""

import pytest

from nesy.reasoning.planning import (
    PDDLDomain,
    PDDLProblem,
    PDDLPlanner,
    ActionExecutor,
    Predicate,
    Action,
)


class TestPredicate:
    """Test PDDL predicates."""
    
    def test_create_predicate(self):
        """Test creating predicates."""
        pred = Predicate("at", ["room1"])
        
        assert pred.name == "at"
        assert pred.params == ["room1"]
        assert not pred.is_negative
    
    def test_negative_predicate(self):
        """Test negative predicates."""
        pred = Predicate("at", ["room1"], is_negative=True)
        
        assert pred.is_negative
        assert str(pred).startswith("not")
    
    def test_predicate_equality(self):
        """Test predicate equality."""
        pred1 = Predicate("at", ["room1"])
        pred2 = Predicate("at", ["room1"])
        pred3 = Predicate("at", ["room2"])
        
        assert pred1 == pred2
        assert pred1 != pred3


class TestAction:
    """Test PDDL actions."""
    
    @pytest.fixture
    def move_action(self):
        """Create a simple move action."""
        return Action(
            name="move",
            parameters=["from", "to"],
            preconditions=[
                Predicate("at", ["from"]),
                Predicate("connected", ["from", "to"])
            ],
            effects=[
                Predicate("at", ["from"], is_negative=True),
                Predicate("at", ["to"])
            ]
        )
    
    def test_action_applicable(self, move_action):
        """Test checking action applicability."""
        state = {
            Predicate("at", ["room1"]),
            Predicate("connected", ["room1", "room2"])
        }
        
        assert move_action.is_applicable(state)
    
    def test_action_not_applicable(self, move_action):
        """Test action not applicable."""
        state = {Predicate("at", ["room3"])}  # Missing preconditions
        
        assert not move_action.is_applicable(state)
    
    def test_action_apply(self, move_action):
        """Test applying action to state."""
        state = {
            Predicate("at", ["room1"]),
            Predicate("connected", ["room1", "room2"])
        }
        
        new_state = move_action.apply(state)
        
        # Should remove at(room1) and add at(room2)
        assert Predicate("at", ["room1"]) not in new_state
        assert Predicate("at", ["room2"]) in new_state
        assert Predicate("connected", ["room1", "room2"]) in new_state


class TestPDDLDomain:
    """Test PDDL domain."""
    
    @pytest.fixture
    def robot_domain(self):
        """Create simple robot domain."""
        domain = PDDLDomain("robot")
        
        domain.add_predicate("at", ["location"])
        domain.add_predicate("connected", ["location", "location"])
        
        domain.add_action(
            "move",
            parameters=["from", "to"],
            preconditions=["at(from)", "connected(from, to)"],
            effects=["not at(from)", "at(to)"]
        )
        
        return domain
    
    def test_domain_creation(self, robot_domain):
        """Test creating domain."""
        assert robot_domain.name == "robot"
        assert "at" in robot_domain.predicates
        assert "move" in robot_domain.actions
    
    def test_parse_predicate(self, robot_domain):
        """Test predicate parsing."""
        pred = robot_domain._parse_predicate("at(room1)")
        
        assert pred.name == "at"
        assert pred.params == ["room1"]
        assert not pred.is_negative
    
    def test_parse_negative_predicate(self, robot_domain):
        """Test parsing negative predicate."""
        pred = robot_domain._parse_predicate("not at(room1)")
        
        assert pred.name == "at"
        assert pred.is_negative


class TestPDDLProblem:
    """Test PDDL problem."""
    
    @pytest.fixture
    def domain(self):
        """Create domain."""
        domain = PDDLDomain("robot")
        domain.add_action(
            "move",
            ["from", "to"],
            ["at(from)", "connected(from, to)"],
            ["not at(from)", "at(to)"]
        )
        return domain
    
    def test_problem_creation(self, domain):
        """Test creating problem."""
        problem = PDDLProblem("test", domain)
        
        problem.add_initial_fact("at(room1)")
        problem.add_initial_fact("connected(room1, room2)")
        problem.add_goal("at(room2)")
        
        assert len(problem.initial_state) == 2
        assert len(problem.goal) == 1
    
    def test_goal_satisfied(self, domain):
        """Test goal satisfaction checking."""
        problem = PDDLProblem("test", domain)
        problem.add_goal("at(room2)")
        
        state = {Predicate("at", ["room2"])}
        
        assert problem.is_goal_satisfied(state)
    
    def test_goal_not_satisfied(self, domain):
        """Test goal not satisfied."""
        problem = PDDLProblem("test", domain)
        problem.add_goal("at(room2)")
        
        state = {Predicate("at", ["room1"])}
        
        assert not problem.is_goal_satisfied(state)


class TestPDDLPlanner:
    """Test PDDL planner."""
    
    @pytest.fixture
    def simple_problem(self):
        """Create simple planning problem."""
        domain = PDDLDomain("robot")
        domain.add_action(
            "move",
            ["from", "to"],
            ["at(from)", "connected(from, to)"],
            ["not at(from)", "at(to)"]
        )
        
        problem = PDDLProblem("simple", domain)
        problem.add_initial_fact("at(room1)")
        problem.add_initial_fact("connected(room1, room2)")
        problem.add_goal("at(room2)")
        
        return problem
    
    def test_forward_search(self, simple_problem):
        """Test forward search planning."""
        planner = PDDLPlanner(simple_problem.domain)
        
        plan = planner.plan(simple_problem, algorithm="forward")
        
        # Should find plan with 1 action
        assert plan is not None
        assert len(plan) == 1
        assert plan[0].name == "move"
    
    def test_astar_search(self, simple_problem):
        """Test A* search planning."""
        planner = PDDLPlanner(simple_problem.domain)
        
        plan = planner.plan(simple_problem, algorithm="astar")
        
        # Should find plan
        assert plan is not None
        assert len(plan) >= 1
    
    def test_no_plan(self):
        """Test case with no solution."""
        domain = PDDLDomain("robot")
        domain.add_action(
            "move",
            ["from", "to"],
            ["at(from)", "connected(from, to)"],
            ["not at(from)", "at(to)"]
        )
        
        problem = PDDLProblem("impossible", domain)
        problem.add_initial_fact("at(room1)")
        # No connection to room2
        problem.add_goal("at(room2)")
        
        planner = PDDLPlanner(domain)
        plan = planner.plan(problem, max_steps=10)
        
        # Should not find plan
        assert plan is None


class TestActionExecutor:
    """Test action executor."""
    
    @pytest.fixture
    def executor(self):
        """Create executor with initial state."""
        initial = {
            Predicate("at", ["room1"]),
            Predicate("connected", ["room1", "room2"])
        }
        return ActionExecutor(initial)
    
    def test_execute_action(self, executor):
        """Test executing valid action."""
        action = Action(
            "move",
            ["from", "to"],
            [Predicate("at", ["room1"]), Predicate("connected", ["room1", "room2"])],
            [Predicate("at", ["room1"], is_negative=True), Predicate("at", ["room2"])]
        )
        
        success = executor.execute(action)
        
        assert success
        assert Predicate("at", ["room2"]) in executor.get_state()
        assert Predicate("at", ["room1"]) not in executor.get_state()
    
    def test_execute_invalid_action(self, executor):
        """Test executing action with unmet preconditions."""
        action = Action(
            "invalid",
            [],
            [Predicate("at", ["room3"])],  # Not in state
            []
        )
        
        success = executor.execute(action)
        
        assert not success
    
    def test_execute_plan(self, executor):
        """Test executing full plan."""
        action = Action(
            "move",
            ["room1", "room2"],
            [Predicate("at", ["room1"]), Predicate("connected", ["room1", "room2"])],
            [Predicate("at", ["room1"], is_negative=True), Predicate("at", ["room2"])]
        )
        
        plan = [action]
        success = executor.execute_plan(plan)
        
        assert success
        assert len(executor.history) == 1
    
    def test_rollback(self, executor):
        """Test rolling back execution."""
        initial_state = executor.get_state().copy()
        
        action = Action(
            "move",
            ["room1", "room2"],
            [Predicate("at", ["room1"]), Predicate("connected", ["room1", "room2"])],
            [Predicate("at", ["room1"], is_negative=True), Predicate("at", ["room2"])]
        )
        
        executor.execute(action)
        executor.rollback(1)
        
        assert executor.get_state() == initial_state
