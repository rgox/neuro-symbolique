"""
Tests for Multi-Agent System.

Tests coordinator, communication, task allocation, and collaborative planning.
"""

import pytest
from unittest.mock import Mock

from nesy.agents.multi_agent import (
    MultiAgentCoordinator,
    AgentRole,
    AgentMessage,
    MessageType,
    SharedKnowledge,
)


class TestMultiAgentCoordinator:
    """Test multi-agent coordinator."""
    
    @pytest.fixture
    def coordinator(self):
        """Create coordinator with agents."""
        coord = MultiAgentCoordinator()
        
        # Register mock agents
        agent1 = Mock()
        agent2 = Mock()
        agent3 = Mock()
        
        coord.register_agent("scout", agent1, AgentRole.OBSERVER, ["perceive", "survey"])
        coord.register_agent("worker1", agent2, AgentRole.WORKER, ["pick", "place"])
        coord.register_agent("worker2", agent3, AgentRole.WORKER, ["pick", "place", "navigate"])
        
        return coord
    
    def test_register_agents(self, coordinator):
        """Test agent registration."""
        assert len(coordinator.agents) == 3
        assert "scout" in coordinator.agents
        assert coordinator.agents["scout"].role == AgentRole.OBSERVER
    
    def test_unregister_agent(self, coordinator):
        """Test agent removal."""
        coordinator.unregister_agent("scout")
        assert "scout" not in coordinator.agents
        assert len(coordinator.agents) == 2
    
    def test_get_agents_by_role(self, coordinator):
        """Test filtering by role."""
        workers = coordinator.get_agents_by_role(AgentRole.WORKER)
        assert len(workers) == 2
        
        observers = coordinator.get_agents_by_role(AgentRole.OBSERVER)
        assert len(observers) == 1
    
    def test_get_available_agents(self, coordinator):
        """Test getting idle agents."""
        available = coordinator.get_available_agents()
        assert len(available) == 3  # All idle initially


class TestTaskManagement:
    """Test task allocation."""
    
    @pytest.fixture
    def coordinator(self):
        coord = MultiAgentCoordinator()
        coord.register_agent("a1", Mock(), AgentRole.WORKER, ["pick"])
        coord.register_agent("a2", Mock(), AgentRole.WORKER, ["pick", "navigate"])
        return coord
    
    def test_assign_task(self, coordinator):
        """Test direct task assignment."""
        success = coordinator.assign_task("a1", "pick_cup")
        
        assert success
        assert coordinator.agents["a1"].status == "busy"
        assert coordinator.metrics["tasks_assigned"] == 1
    
    def test_assign_unknown_agent(self, coordinator):
        """Test assigning to unknown agent."""
        success = coordinator.assign_task("unknown", "task")
        assert not success
    
    def test_allocate_task(self, coordinator):
        """Test automatic task allocation."""
        agent_id = coordinator.allocate_task("pick_cup", ["pick"])
        
        assert agent_id is not None
        assert coordinator.agents[agent_id].status == "busy"
    
    def test_allocate_no_capability(self, coordinator):
        """Test allocation with unmet capabilities."""
        agent_id = coordinator.allocate_task("fly", ["fly"])
        assert agent_id is None
    
    def test_complete_task(self, coordinator):
        """Test task completion."""
        coordinator.assign_task("a1", "pick_cup")
        task_id = coordinator.agents["a1"].current_task
        
        coordinator.complete_task(task_id, result="success")
        
        assert coordinator.agents["a1"].status == "idle"
        assert coordinator.metrics["tasks_completed"] == 1


class TestCommunication:
    """Test inter-agent communication."""
    
    @pytest.fixture
    def coordinator(self):
        coord = MultiAgentCoordinator()
        coord.register_agent("a1", Mock())
        coord.register_agent("a2", Mock())
        return coord
    
    def test_send_message(self, coordinator):
        """Test sending message."""
        msg = AgentMessage(
            sender="a1", receiver="a2",
            msg_type=MessageType.INFORM,
            content={"data": "hello"}
        )
        coordinator.send_message(msg)
        
        assert coordinator.metrics["messages_exchanged"] == 1
    
    def test_get_messages(self, coordinator):
        """Test retrieving messages."""
        msg = AgentMessage(sender="a1", receiver="a2", content={"data": "test"})
        coordinator.send_message(msg)
        
        messages = coordinator.get_messages("a2")
        assert len(messages) == 1
        assert messages[0].content["data"] == "test"
    
    def test_broadcast(self, coordinator):
        """Test broadcasting."""
        coordinator.broadcast("a1", {"alert": "obstacle detected"})
        
        assert coordinator.metrics["messages_exchanged"] == 1
        
        # Both agents should see broadcast
        msgs_a1 = coordinator.get_messages("a1")
        msgs_a2 = coordinator.get_messages("a2")
        
        # Broadcasts go to all (receiver="")
        assert len(msgs_a1) > 0 or len(msgs_a2) > 0


class TestSharedKnowledge:
    """Test shared knowledge base."""
    
    @pytest.fixture
    def kb(self):
        return SharedKnowledge()
    
    def test_add_fact(self, kb):
        """Test adding facts."""
        kb.add_fact("cup_location", "kitchen", source="robot1")
        
        assert kb.get_fact("cup_location") == "kitchen"
    
    def test_get_missing_fact(self, kb):
        """Test getting nonexistent fact."""
        assert kb.get_fact("nonexistent") is None
    
    def test_add_observation(self, kb):
        """Test adding observations."""
        kb.add_observation({"type": "detection", "objects": ["cup"]})
        
        assert len(kb.observations) == 1


class TestCollaborativePlanning:
    """Test collaborative planning."""
    
    @pytest.fixture
    def coordinator(self):
        coord = MultiAgentCoordinator()
        coord.register_agent("bot1", Mock(), AgentRole.WORKER)
        coord.register_agent("bot2", Mock(), AgentRole.WORKER)
        return coord
    
    def test_plan_collaborative(self, coordinator):
        """Test collaborative plan generation."""
        plan = coordinator.plan_collaborative("clean the room")
        
        assert len(plan) == 2  # One plan per agent
        assert "bot1" in plan
        assert "bot2" in plan
    
    def test_plan_patrol(self, coordinator):
        """Test patrol plan."""
        plan = coordinator.plan_collaborative("patrol the area")
        
        assert len(plan) == 2


class TestConflictResolution:
    """Test conflict detection and resolution."""
    
    @pytest.fixture
    def coordinator(self):
        coord = MultiAgentCoordinator()
        coord.register_agent("a1", Mock(), AgentRole.WORKER)
        coord.register_agent("a2", Mock(), AgentRole.LEADER)
        return coord
    
    def test_detect_location_conflict(self, coordinator):
        """Test location conflict detection."""
        coordinator.agents["a1"].location = "kitchen"
        coordinator.agents["a1"].status = "busy"
        coordinator.agents["a2"].location = "kitchen"
        coordinator.agents["a2"].status = "busy"
        
        conflicts = coordinator.detect_conflicts()
        
        assert len(conflicts) == 1
        assert conflicts[0]["type"] == "location"
    
    def test_resolve_conflicts(self, coordinator):
        """Test conflict resolution."""
        conflicts = [{"type": "location", "agents": ["a1", "a2"],
                      "location": "kitchen", "resolution": "priority"}]
        
        coordinator.resolve_conflicts(conflicts)
        
        assert coordinator.metrics["conflicts_resolved"] == 1
    
    def test_no_conflicts(self, coordinator):
        """Test no conflicts when agents in different locations."""
        coordinator.agents["a1"].location = "kitchen"
        coordinator.agents["a2"].location = "bedroom"
        
        conflicts = coordinator.detect_conflicts()
        assert len(conflicts) == 0


class TestCoordinatorStatus:
    """Test status reporting."""
    
    def test_get_status(self):
        """Test status report."""
        coord = MultiAgentCoordinator()
        coord.register_agent("bot1", Mock())
        
        status = coord.get_status()
        
        assert "agents" in status
        assert "active_tasks" in status
        assert "metrics" in status
        assert status["agents"]["bot1"]["status"] == "idle"
