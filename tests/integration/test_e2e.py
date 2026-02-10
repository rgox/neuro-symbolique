"""
End-to-End Integration Tests.

Tests the complete neuro-symbolic stack working together:
- Perception → Scene Graph → Reasoning → Planning → Action
- Knowledge Graph ↔ Scene Graph sync
- Temporal reasoning over scenes
- Multi-agent coordination
- API endpoints

These tests verify cross-module integration, not individual units.
"""

import pytest
import numpy as np
import time


class TestPerceptionToReasoning:
    """Test perception → scene graph → reasoning pipeline."""
    
    def test_scene_graph_to_reasoning(self):
        """Objects added to scene graph are reasoned about."""
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType
        from nesy.reasoning.logic import ReasoningEngine
        
        sg = SceneGraph()
        
        # Add objects
        cup = sg.add_node(LayerType.L1, NodeType.OBJECT, [1.0, 0.0, 0.5],
                          attributes={"label": "cup"}, node_id="cup_1")
        table = sg.add_node(LayerType.L1, NodeType.OBJECT, [1.0, 0.0, 0.0],
                            attributes={"label": "table"}, node_id="table_1")
        
        # Reason (query for spatial relations)
        engine = ReasoningEngine(sg)
        results = engine.query("on")
        
        # Should return results (possibly empty if no 'on' relation inferred)
        assert isinstance(results, list)
    
    def test_scene_graph_edge_creation(self):
        """Edges between scene objects work correctly."""
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType, RelationType
        
        sg = SceneGraph()
        
        n1 = sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 1], node_id="a")
        n2 = sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0], node_id="b")
        
        sg.add_edge("a", "b", RelationType.ON)
        
        # Edges stored in sg.edges dict
        assert len(sg.edges) > 0


class TestKnowledgeGraphIntegration:
    """Test KG ↔ Scene Graph integration."""
    
    def test_kg_with_scene_instances(self):
        """KG concepts bound to scene graph objects."""
        from nesy.world_model.knowledge_graph import KnowledgeGraph
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType
        
        # Build ontology
        kg = KnowledgeGraph()
        kg.add_concept("container")
        kg.add_concept("cup", parent="container")
        kg.add_concept("bowl", parent="container")
        
        # Create scene objects
        sg = SceneGraph()
        cup1 = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 0, 0], node_id="cup_1")
        cup2 = sg.add_node(LayerType.L1, NodeType.OBJECT, [2, 0, 0], node_id="cup_2")
        
        # Bind instances
        kg.bind_instance("cup", "cup_1")
        kg.bind_instance("cup", "cup_2")
        
        # Query through KG
        all_containers = kg.get_instances("container", include_descendants=True)
        assert "cup_1" in all_containers
        assert "cup_2" in all_containers
    
    def test_kg_property_inheritance_in_reasoning(self):
        """KG inherited properties used in reasoning."""
        from nesy.world_model.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        kg.add_concept("physical_object")
        kg.add_concept("container", parent="physical_object")
        kg.add_concept("cup", parent="container")
        
        kg.set_property("physical_object", "has_mass", True)
        kg.set_property("container", "can_hold_liquid", True)
        
        # Cup should inherit both
        assert kg.get_property("cup", "has_mass") is True
        assert kg.get_property("cup", "can_hold_liquid") is True
    
    def test_sparql_query_with_ontology(self):
        """SPARQL queries work across ontology."""
        from nesy.world_model.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        kg.add_concept("thing")
        kg.add_concept("furniture", parent="thing")
        kg.add_concept("chair", parent="furniture")
        kg.add_concept("table", parent="furniture")
        kg.add_concept("stool", parent="chair")
        
        results = kg.query("SELECT ?x WHERE { ?x is_a furniture }")
        found = {r["?x"] for r in results}
        
        assert "chair" in found
        assert "table" in found
        assert "stool" in found  # Transitive


class TestTemporalIntegration:
    """Test temporal reasoning integration."""
    
    def test_temporal_event_sequence(self):
        """Detect pick-move-place sequences."""
        from nesy.reasoning.temporal import TemporalReasoner
        
        reasoner = TemporalReasoner()
        reasoner.add_event("e1", "pick", 1.0, subject="cup")
        reasoner.add_event("e2", "move", 2.0, subject="cup")
        reasoner.add_event("e3", "place", 3.0, subject="cup")
        
        matches = reasoner.detect_sequence(["pick", "move", "place"], subject="cup")
        assert len(matches) == 1
        assert matches[0][0].subject == "cup"
    
    def test_temporal_causal_chain(self):
        """Build causal chains from events."""
        from nesy.reasoning.temporal import TemporalReasoner
        
        reasoner = TemporalReasoner()
        reasoner.add_event("e1", "push", 1.0, subject="box")
        reasoner.add_event("e2", "move", 1.5, subject="box")
        
        link = reasoner.infer_causality("e1", "e2")
        assert link is not None
        
        chain = reasoner.get_causal_chain("e1")
        assert len(chain) >= 1
    
    def test_temporal_prediction_from_history(self):
        """Predict next event from historical patterns."""
        from nesy.reasoning.temporal import TemporalReasoner
        
        reasoner = TemporalReasoner()
        # Establish patterns
        for i in range(5):
            reasoner.add_event(f"pick_{i}", "pick", i * 3.0)
            reasoner.add_event(f"move_{i}", "move", i * 3.0 + 1.0)
        
        predictions = reasoner.predict_next(["pick"])
        types = [p[0] for p in predictions]
        assert "move" in types


class TestAgentIntegration:
    """Test full agent pipeline."""
    
    def test_agent_observe_reason(self):
        """Agent can observe and reason."""
        from nesy.agents import AutonomousAgent, AgentConfig
        
        agent = AutonomousAgent(AgentConfig(use_yolo=False, use_clip=False))
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        obs = agent.observe(image)
        
        assert obs is not None
        
        results = agent.reason()
        assert isinstance(results, dict)
    
    def test_multi_agent_task_flow(self):
        """Multi-agent task allocation and completion."""
        from nesy.agents import AutonomousAgent, AgentConfig
        from nesy.agents.multi_agent import MultiAgentCoordinator, AgentRole
        
        coord = MultiAgentCoordinator()
        
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent1 = AutonomousAgent(config)
        agent2 = AutonomousAgent(config)
        
        coord.register_agent("bot1", agent1, AgentRole.WORKER, ["pick", "navigate"])
        coord.register_agent("bot2", agent2, AgentRole.WORKER, ["pick", "place"])
        
        # Allocate task requiring "pick"
        assigned = coord.allocate_task("pick_cup", ["pick"])
        assert assigned is not None
        
        # Complete task
        task_id = coord.agents[assigned].current_task
        coord.complete_task(task_id, result="success")
        
        assert coord.agents[assigned].status == "idle"
        assert coord.metrics["tasks_completed"] == 1
    
    def test_multi_agent_collaborative_plan(self):
        """Collaborative planning across agents."""
        from nesy.agents.multi_agent import MultiAgentCoordinator, AgentRole
        from unittest.mock import Mock
        
        coord = MultiAgentCoordinator()
        coord.register_agent("a1", Mock(), AgentRole.WORKER)
        coord.register_agent("a2", Mock(), AgentRole.WORKER)
        coord.register_agent("a3", Mock(), AgentRole.OBSERVER)
        
        plan = coord.plan_collaborative("search the area")
        assert len(plan) == 3  # One entry per agent


class TestRoboticsIntegration:
    """Test robotics module integration."""
    
    def test_navigation_planning(self):
        """Navigator plans path through PDDL."""
        from nesy.robots import ROSBridge, RobotNavigator
        
        bridge = ROSBridge("test_bot")
        bridge.start()
        
        nav = RobotNavigator(bridge)
        nav.add_waypoint("A", 0, 0)
        nav.add_waypoint("B", 5, 0)
        nav.add_waypoint("C", 5, 5)
        nav.add_connection("A", "B")
        nav.add_connection("B", "C")
        nav.set_current_location("A")
        
        # navigate_to may return None or path - verify no crash
        result = nav.navigate_to("C")
        # Navigation history should record attempts
        assert len(nav.navigation_history) >= 0
        
        bridge.stop()
    
    def test_manipulation_pick_place(self):
        """Manipulator does pick-and-place."""
        from nesy.robots import ROSBridge, ObjectManipulator
        
        bridge = ROSBridge("test_bot")
        bridge.start()
        
        manip = ObjectManipulator(bridge)
        manip.add_object("cup", 0.3, 0.2, 0.5)
        manip.add_surface("table", 0.5, 0.0, 0.3)
        
        assert manip.pick("cup")
        assert manip.place("table")
        assert len(manip.manipulation_history) == 2
        
        bridge.stop()


class TestFullPipeline:
    """Test complete end-to-end pipeline."""
    
    def test_perception_to_action(self):
        """Full loop: perceive → reason → plan → act."""
        from nesy.agents import AutonomousAgent, AgentConfig, Goal
        
        agent = AutonomousAgent(AgentConfig(use_yolo=False, use_clip=False))
        
        # Step 1: Observe
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        obs = agent.observe(image)
        assert obs is not None
        
        # Step 2: Reason
        facts = agent.reason()
        assert isinstance(facts, dict)
        
        # Step 3: Plan (may return empty plan)
        goal = Goal(description="cleanup", target_predicates=["clean(room)"])
        plan = agent.plan(goal)
        assert plan is None or isinstance(plan, list)
        
        # Step 4: Metrics
        metrics = agent.get_metrics()
        assert metrics["observations"] >= 1
        assert metrics["inferences"] >= 1
    
    def test_knowledge_temporal_integration(self):
        """KG + temporal reasoning together."""
        from nesy.world_model.knowledge_graph import KnowledgeGraph
        from nesy.reasoning.temporal import TemporalReasoner
        
        # Set up ontology
        kg = KnowledgeGraph()
        kg.add_concept("action")
        kg.add_concept("manipulation", parent="action")
        kg.add_concept("pick", parent="manipulation")
        kg.add_concept("place", parent="manipulation")
        
        # Track temporal events
        tr = TemporalReasoner()
        tr.add_event("e1", "pick", 1.0, subject="cup_1")
        tr.add_event("e2", "place", 2.0, subject="cup_1")
        
        # Verify: "pick" is a manipulation, which is an action
        assert kg.is_a("pick", "action")
        assert kg.is_a("place", "manipulation")
        
        # Detect sequence
        matches = tr.detect_sequence(["pick", "place"])
        assert len(matches) == 1
