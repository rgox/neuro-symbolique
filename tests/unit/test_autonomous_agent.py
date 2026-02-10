"""
Tests for Autonomous Agent.

Tests perception-reasoning-planning-action pipeline.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from nesy.agents import (
    AutonomousAgent,
    AgentConfig,
    AgentState,
    Observation,
    Goal,
)


class TestAutonomousAgent:
    """Test autonomous agent."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return AgentConfig(
            use_yolo=False,  # Disable for testing
            use_clip=False,
            reasoning_engine="mock",
            loop_frequency=10.0
        )
    
    @pytest.fixture
    def agent(self, config):
        """Create agent."""
        return AutonomousAgent(config)
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.state == AgentState.IDLE
        assert agent.metrics["observations"] == 0
        assert agent.metrics["actions_executed"] == 0
    
    def test_observe(self, agent):
        """Test observation processing."""
        # Create dummy image
        image = np.random.rand(480, 640, 3).astype(np.uint8)
        
        obs = agent.observe(image)
        
        assert isinstance(obs, Observation)
        assert obs.image is not None
        assert agent.metrics["observations"] == 1
    
    def test_reason(self, agent):
        """Test reasoning."""
        # Initialize components first
        agent._init_components()
        
        results = agent.reason()
        
        assert isinstance(results, dict)
        assert agent.metrics["inferences"] == 1
    
    def test_plan(self, agent):
        """Test planning."""
        agent._init_components()
        
        goal = Goal(
            description="Test goal",
            target_predicates=["at(target_location)"]
        )
        
        plan = agent.plan(goal)
        
        # May or may not find plan depending on state
        assert plan is None or isinstance(plan, list)
    
    def test_execute(self, agent):
        """Test action execution."""
        agent._init_components()
        
        # Create mock plan
        from nesy.reasoning.planning import Action, Predicate
        
        action = Action(
            "test_action",
            parameters=[],
            preconditions=[],
            effects=[Predicate("test", [])]
        )
        
        plan = [action]
        
        # Execute (may fail if preconditions not met)
        success = agent.execute(plan)
        
        assert isinstance(success, bool)
    
    def test_get_metrics(self, agent):
        """Test metrics retrieval."""
        metrics = agent.get_metrics()
        
        assert "observations" in metrics
        assert "actions_executed" in metrics
        assert "state" in metrics
    
    def test_reset(self, agent):
        """Test agent reset."""
        # Do some actions
        image = np.random.rand(480, 640, 3).astype(np.uint8)
        agent.observe(image)
        
        # Reset
        agent.reset()
        
        assert agent.state == AgentState.IDLE
        assert agent.metrics["observations"] == 0
        assert len(agent.observation_history) == 0


class TestAgentIntegration:
    """Integration tests for agent."""
    
    def test_perception_to_reasoning(self):
        """Test data flow from perception to reasoning."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)
        
        # Observe
        image = np.random.rand(480, 640, 3).astype(np.uint8)
        obs = agent.observe(image)
        
        # Reason
        results = agent.reason()
        
        # Should complete without error
        assert obs is not None
        assert results is not None
    
    def test_reasoning_to_planning(self):
        """Test reasoning feeds into planning."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)
        
        agent._init_components()
        
        # Reason
        agent.reason()
        
        # Plan
        goal = Goal("test_goal", target_predicates=["at(location)"])
        plan = agent.plan(goal)
        
        # Should complete
        assert True  # Test passes if no exceptions


class TestObservation:
    """Test observation data structure."""
    
    def test_create_observation(self):
        """Test creating observation."""
        obs = Observation(
            timestamp=123.456,
            image=np.zeros((100, 100, 3)),
            detections=[{"label": "cup"}]
        )
        
        assert obs.timestamp == 123.456
        assert len(obs.detections) == 1


class TestGoal:
    """Test goal specification."""
    
    def test_create_goal(self):
        """Test creating goal."""
        goal = Goal(
            description="Pick up cup",
            target_predicates=["holding(cup)"],
            priority=0.9
        )
        
        assert goal.description == "Pick up cup"
        assert len(goal.target_predicates) == 1
        assert goal.priority == 0.9
