"""
Coverage Boost Tests.

Target: engine.py, autonomous.py, grounding.py, navigation.py, device.py, 
hypervector.py, codebook.py, advanced.py, ontology.py, spatial_index.py.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import time
import tempfile
import os

# === REASONING ENGINE TESTS ===

from nesy.reasoning.logic.engine import ReasoningEngine
from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType, RelationType


class TestEngineSync:
    """Test reasoning engine sync and edge cases."""

    def test_sync_with_edges(self):
        """Test sync exports edges as spatial facts."""
        sg = SceneGraph()
        cup = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 0, 0],
                          attributes={"label": "cup", "class": "cup"}, node_id="cup_1")
        table = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 0, -0.5],
                            attributes={"label": "table", "class": "table"}, node_id="table_1")
        sg.add_edge(cup.id, table.id, RelationType.ON)

        engine = ReasoningEngine(sg)
        engine.sync_from_scene_graph()

        on_facts = engine.query("on")
        assert len(on_facts) > 0

    def test_sync_with_in_relation(self):
        """Test sync with 'in' relation."""
        sg = SceneGraph()
        obj = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 0, 0],
                          attributes={"class": "cup"}, node_id="cup1")
        container = sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0],
                                attributes={"class": "kitchen"}, node_id="kitchen1")
        sg.add_edge(obj.id, container.id, RelationType.IN)

        engine = ReasoningEngine(sg)
        engine.sync_from_scene_graph()

        in_facts = engine.query("in")
        assert len(in_facts) > 0

    def test_sync_with_near_relation(self):
        """Test sync with 'near' relation and symmetry."""
        sg = SceneGraph()
        a = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 0, 0],
                        attributes={"class": "cup"}, node_id="a1")
        b = sg.add_node(LayerType.L1, NodeType.OBJECT, [1.1, 0, 0],
                        attributes={"class": "book"}, node_id="b1")
        sg.add_edge(a.id, b.id, RelationType.NEAR)

        engine = ReasoningEngine(sg)
        engine.sync_from_scene_graph()

        near = engine.query("near")
        assert len(near) >= 1

    def test_load_rules_from_file(self):
        """Test loading rules from .scl file."""
        sg = SceneGraph()
        engine = ReasoningEngine(sg)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.scl', delete=False) as f:
            f.write("% Comment line\n")
            f.write("path(X, Y) :- near(X, Y)\n")
            f.name

        try:
            engine._load_rules_from_file(f.name)
        finally:
            os.unlink(f.name)

    def test_load_rules_file_not_found(self):
        """Test loading rules from non-existent file."""
        sg = SceneGraph()
        engine = ReasoningEngine(sg)

        # Should log warning but not raise
        engine._load_rules_from_file("/nonexistent/rules.scl")

    def test_engine_with_rules_file(self):
        """Test engine init with rules_file parameter."""
        sg = SceneGraph()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.scl', delete=False) as f:
            f.write("path(X, Y) :- near(X, Y)\n")
            fname = f.name

        try:
            engine = ReasoningEngine(sg, rules_file=fname)
            assert engine is not None
        finally:
            os.unlink(fname)

    def test_get_statistics(self):
        """Test statistics reporting."""
        sg = SceneGraph()
        engine = ReasoningEngine(sg)

        stats = engine.get_statistics()
        assert "num_facts" in stats
        assert "num_rules" in stats

    def test_find_all(self):
        """Test find_all method."""
        sg = SceneGraph()
        sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0],
                    attributes={"class": "cup"}, node_id="cup_1")

        engine = ReasoningEngine(sg)
        engine.sync_from_scene_graph()

        result = engine.find_all("cup")
        assert isinstance(result, list)

    def test_find_in_location(self):
        """Test find_in_location method."""
        sg = SceneGraph()
        kitchen = sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0],
                              attributes={"class": "kitchen"}, node_id="kitchen_1")

        engine = ReasoningEngine(sg)
        engine.sync_from_scene_graph()

        result = engine.find_in_location("kitchen_1")
        assert isinstance(result, list)

    def test_infer(self):
        """Test infer method."""
        sg = SceneGraph()
        engine = ReasoningEngine(sg)

        result = engine.infer("on('a', 'b')")
        assert isinstance(result, (bool, type(None)))

    def test_add_custom_rule(self):
        """Test adding custom rules."""
        sg = SceneGraph()
        engine = ReasoningEngine(sg)

        engine.add_custom_rule("test_rel(X, Y) :- near(X, Y)")
        # Should not raise

    def test_query_nonexistent_relation(self):
        """Test querying non-existent relation."""
        sg = SceneGraph()
        engine = ReasoningEngine(sg)

        result = engine.query("nonexistent_rel")
        assert isinstance(result, list)

    def test_find_similar_to(self):
        """Test VSA similarity search."""
        sg = SceneGraph()
        engine = ReasoningEngine(sg)

        result = engine.find_similar_to("cup_1")
        assert isinstance(result, list)


# === AUTONOMOUS AGENT TESTS ===

from nesy.agents.autonomous import (
    AutonomousAgent, AgentConfig, AgentState, Goal, Observation
)


class TestAgentAdvanced:
    """Test advanced agent features."""

    def test_init_perception(self):
        """Test perception initialization."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        agent._init_perception()
        assert agent.scene_graph is not None
        assert agent.model_zoo is not None

    def test_init_reasoning(self):
        """Test reasoning initialization."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        agent._init_perception()
        agent._init_reasoning()
        assert agent._reasoning is not None

    def test_init_planner(self):
        """Test planner initialization."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        agent._init_perception()
        agent._init_reasoning()
        agent._init_planner()
        assert agent._planner is not None
        assert agent.domain is not None

    def test_init_executor(self):
        """Test executor initialization."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        agent._init_perception()
        agent._init_reasoning()
        agent._init_planner()
        agent._init_executor()
        assert agent._executor is not None

    def test_observe_no_detector(self):
        """Test observation without detector initialized."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        obs = agent.observe(image)

        assert obs.timestamp > 0
        assert agent.metrics["observations"] == 1
        assert agent.state == AgentState.OBSERVING

    def test_reason(self):
        """Test reasoning step."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        agent.observe(image)

        results = agent.reason()
        assert "on" in results
        assert "in" in results
        assert agent.state == AgentState.REASONING

    def test_plan(self):
        """Test planning step."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        agent.observe(image)

        goal = Goal("test", target_predicates=["at(room2)"])
        plan = agent.plan(goal)

        # May be None if no valid plan
        assert agent.state == AgentState.PLANNING

    def test_execute_empty_plan(self):
        """Test executing empty plan."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        agent.observe(image)

        success = agent.execute([])
        assert success

    def test_execute_with_callback(self):
        """Test execute with step callback."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        agent.observe(image)

        callback_calls = []

        def cb(action, idx):
            callback_calls.append(idx)

        # Empty plan, callback not called
        agent.execute([], step_callback=cb)
        assert len(callback_calls) == 0

    def test_get_current_state(self):
        """Test getting current state as predicates."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        agent._init_perception()
        state = agent._get_current_state()
        assert isinstance(state, set)

    def test_get_current_facts(self):
        """Test getting current facts as strings."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        agent._init_perception()
        facts = agent._get_current_facts()
        assert isinstance(facts, list)

    def test_is_goal_achieved_empty(self):
        """Test goal check with no predicates."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        agent._init_perception()
        goal = Goal("test", target_predicates=[])

        assert agent._is_goal_achieved(goal)

    def test_is_goal_not_achieved(self):
        """Test goal check fails when predicates missing."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        agent._init_perception()
        goal = Goal("test", target_predicates=["at(room99)"])

        assert not agent._is_goal_achieved(goal)

    def test_get_metrics(self):
        """Test metrics reporting."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        metrics = agent.get_metrics()
        assert "state" in metrics
        assert "observations" in metrics

    def test_reset(self):
        """Test agent reset."""
        agent = AutonomousAgent(AgentConfig(use_yolo=False, use_clip=False))

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        agent.observe(image)
        assert agent.metrics["observations"] == 1

        agent.reset()
        assert agent.metrics["observations"] == 0
        assert agent.state == AgentState.IDLE

    def test_run_with_quick_goal(self):
        """Test run loop with goal achieved immediately."""
        config = AgentConfig(use_yolo=False, use_clip=False, loop_frequency=100.0)
        agent = AutonomousAgent(config)

        call_count = [0]

        def sensor():
            call_count[0] += 1
            return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Empty goal predicates = achieved immediately
        goal = Goal("test", target_predicates=[])

        result = agent.run(goal, sensor_callback=sensor, max_iterations=3)
        assert result is True

    def test_run_max_iterations(self):
        """Test run loop hitting max iterations."""
        config = AgentConfig(use_yolo=False, use_clip=False, loop_frequency=100.0)
        agent = AutonomousAgent(config)

        def sensor():
            return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        goal = Goal("impossible", target_predicates=["at(unreachable)"])

        result = agent.run(goal, sensor_callback=sensor, max_iterations=2)
        assert result is False

    def test_run_exception(self):
        """Test run loop with exception."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        def sensor():
            raise RuntimeError("sensor failed")

        goal = Goal("test", target_predicates=["at(room1)"])

        result = agent.run(goal, sensor_callback=sensor, max_iterations=1)
        assert result is False
        assert agent.state == AgentState.ERROR

    def test_create_planning_domain(self):
        """Test domain creation."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        agent._init_perception()
        domain = agent._create_planning_domain()

        assert domain.name == "robot_domain"
        assert "pick" in domain.actions
        assert "place" in domain.actions

    def test_add_domain_rules(self):
        """Test adding domain rules to reasoning."""
        config = AgentConfig(use_yolo=False, use_clip=False)
        agent = AutonomousAgent(config)

        agent._init_perception()
        agent._init_reasoning()
        # _add_domain_rules called in _init_reasoning
        # No error means success


# === VSA GROUNDING CACHE TESTS ===

from nesy.reasoning.vsa.grounding import NeuralGrounding, GroundingCache
from nesy.reasoning.vsa.codebook import VSACodebook
from nesy.reasoning.vsa.hypervector import HyperVector
from nesy.core.memory import UMA


class TestGroundingCache:
    """Test GroundingCache store/retrieve/clear."""

    def test_store_and_retrieve(self):
        """Test storing and retrieving HV."""
        uma = MagicMock()
        buffer_mock = MagicMock()
        # Mock buffer data as numpy array to match HV data
        buffer_mock.data = np.zeros(1000, dtype=np.float32)
        uma.get.return_value = buffer_mock
        uma.exists.return_value = False
        
        cache = GroundingCache(uma)

        hv = HyperVector.random(1000)
        key = cache.store("obj_1", hv)

        assert key == "vsa_grounded_obj_1"
        assert "obj_1" in cache.cache

        # For retrieve, we need to mock exist=True
        uma.exists.return_value = True
        retrieved = cache.retrieve("obj_1")
        assert retrieved is not None
        assert retrieved.dim == 1000

    def test_retrieve_not_found(self):
        """Test retrieving non-existent object."""
        uma = MagicMock()
        cache = GroundingCache(uma)

        result = cache.retrieve("nonexistent")
        assert result is None

    def test_clear(self):
        """Test clearing cache."""
        uma = MagicMock()
        buffer_mock = MagicMock()
        buffer_mock.data = np.zeros(1000, dtype=np.float32)
        uma.get.return_value = buffer_mock
        
        # dynamic side effect for exists: False (store 1), False (store 2), then True (clear checks)
        uma.exists.side_effect = lambda k: False 

        cache = GroundingCache(uma)

        hv = HyperVector.random(1000)
        cache.store("obj_1", hv)
        cache.store("obj_2", hv)
        
        # Change exists to True for clear() which checks existence before freeing
        uma.exists.side_effect = lambda k: True

        cache.clear()
        assert len(cache.cache) == 0

    def test_store_overwrite(self):
        """Test overwriting existing entry."""
        uma = MagicMock()
        buffer_mock = MagicMock()
        buffer_mock.data = np.zeros(1000, dtype=np.float32)
        uma.get.return_value = buffer_mock
        uma.exists.return_value = False

        cache = GroundingCache(uma)

        hv1 = HyperVector.random(1000)
        hv2 = HyperVector.random(1000)

        cache.store("obj_1", hv1)
        cache.store("obj_1", hv2)

        # Retrieve validation
        uma.exists.return_value = True
        retrieved = cache.retrieve("obj_1")
        assert retrieved is not None


class TestNeuralGroundingAdvanced:
    """Test NeuralGrounding edge cases."""

    def test_ground_with_position(self):
        """Test grounding with position."""
        codebook = VSACodebook(dim=1000)
        grounding = NeuralGrounding(neural_dim=64, vsa_dim=1000, codebook=codebook)

        embedding = np.random.randn(64).astype(np.float32)
        hv = grounding.ground_embedding(
            embedding=embedding,
            attributes={"color": "red"},
            object_class="cup",
            position=np.array([1.0, 2.0, 3.0])
        )

        assert hv.dim == 1000

    def test_ground_without_attributes(self):
        """Test grounding without attributes."""
        codebook = VSACodebook(dim=1000)
        grounding = NeuralGrounding(neural_dim=64, vsa_dim=1000, codebook=codebook)

        embedding = np.random.randn(64).astype(np.float32)
        hv = grounding.ground_embedding(
            embedding=embedding,
            attributes={},
            object_class="cup"
        )

        assert hv.dim == 1000

    def test_unbind_class(self):
        """Test class unbinding."""
        codebook = VSACodebook(dim=1000)
        grounding = NeuralGrounding(neural_dim=64, vsa_dim=1000, codebook=codebook)

        embedding = np.random.randn(64).astype(np.float32)
        hv = grounding.ground_embedding(
            embedding=embedding,
            attributes={},
            object_class="cup"
        )

        cls = grounding.unbind_class(hv)
        assert isinstance(cls, str)

    def test_unbind_attribute(self):
        """Test attribute unbinding."""
        codebook = VSACodebook(dim=1000)
        grounding = NeuralGrounding(neural_dim=64, vsa_dim=1000, codebook=codebook)

        embedding = np.random.randn(64).astype(np.float32)
        hv = grounding.ground_embedding(
            embedding=embedding,
            attributes={"color": "red"},
            object_class="cup"
        )

        result = grounding.unbind_attribute(hv, "color")
        # May or may not return correct attribute (VSA is fuzzy)
        assert result is None or isinstance(result, str)

    def test_similarity(self):
        """Test similarity computation."""
        codebook = VSACodebook(dim=1000)
        grounding = NeuralGrounding(neural_dim=64, vsa_dim=1000, codebook=codebook)

        hv1 = HyperVector.random(1000)
        hv2 = HyperVector.random(1000)

        sim = grounding.similarity(hv1, hv2)
        assert -1.0 <= sim <= 1.0

    def test_repr(self):
        """Test repr."""
        codebook = VSACodebook(dim=1000)
        grounding = NeuralGrounding(neural_dim=64, vsa_dim=1000, codebook=codebook)

        r = repr(grounding)
        assert "NeuralGrounding" in r

    def test_no_codebook(self):
        """Test grounding without providing codebook."""
        grounding = NeuralGrounding(neural_dim=64, vsa_dim=1000)

        assert grounding.codebook is not None


# === NAVIGATION TESTS ===

from nesy.robots.navigation import (
    RobotNavigator, TopologicalMap, Waypoint, NavigationPath,
    ObjectManipulator, VisualServoing
)


class TestTopologicalMap:
    """Test TopologicalMap."""

    def test_add_waypoint(self):
        """Test adding waypoints."""
        topo = TopologicalMap()
        topo.add_waypoint("room1", 0.0, 0.0)
        topo.add_waypoint("room2", 1.0, 0.0)

        assert "room1" in topo.waypoints
        assert "room2" in topo.waypoints

    def test_add_connection(self):
        """Test adding connections."""
        topo = TopologicalMap()
        topo.add_waypoint("room1", 0.0, 0.0)
        topo.add_waypoint("room2", 1.0, 0.0)
        topo.add_connection("room1", "room2")

        neighbors = topo.get_neighbors("room1")
        assert "room2" in neighbors

        # Bidirectional
        neighbors2 = topo.get_neighbors("room2")
        assert "room1" in neighbors2

    def test_distance(self):
        """Test distance computation."""
        topo = TopologicalMap()
        topo.add_waypoint("A", 0.0, 0.0)
        topo.add_waypoint("B", 3.0, 4.0)

        dist = topo.distance("A", "B")
        assert abs(dist - 5.0) < 0.01

    def test_distance_unknown(self):
        """Test distance with unknown waypoint."""
        topo = TopologicalMap()
        topo.add_waypoint("A", 0.0, 0.0)

        dist = topo.distance("A", "unknown")
        assert dist == float('inf')

    def test_to_pddl_facts(self):
        """Test PDDL facts generation."""
        topo = TopologicalMap()
        topo.add_waypoint("kitchen", 0.0, 0.0)
        topo.add_waypoint("hallway", 1.0, 0.0)
        topo.add_connection("kitchen", "hallway")

        facts = topo.to_pddl_facts()
        assert any("location" in f for f in facts)
        assert any("connected" in f for f in facts)


class TestRobotNavigator:
    """Test RobotNavigator."""

    def test_init(self):
        """Test initialization."""
        nav = RobotNavigator()
        assert nav.topo_map is not None
        assert nav.current_location is None
        assert not nav.is_navigating

    def test_add_waypoint(self):
        """Test waypoint addition via navigator."""
        nav = RobotNavigator()
        nav.add_waypoint("kitchen", 2.0, 3.0)
        assert "kitchen" in nav.topo_map.waypoints

    def test_add_connection(self):
        """Test connection addition via navigator."""
        nav = RobotNavigator()
        nav.add_waypoint("A", 0.0, 0.0)
        nav.add_waypoint("B", 1.0, 0.0)
        nav.add_connection("A", "B")
        assert "B" in nav.topo_map.get_neighbors("A")

    def test_set_current_location(self):
        """Test setting location."""
        nav = RobotNavigator()
        nav.set_current_location("kitchen")
        assert nav.current_location == "kitchen"

    def test_navigate_no_current(self):
        """Test navigating without current location set."""
        nav = RobotNavigator()
        nav.add_waypoint("kitchen", 2.0, 3.0)
        result = nav.navigate_to("kitchen")
        assert result is None

    def test_navigate_unknown_target(self):
        """Test navigating to unknown target."""
        nav = RobotNavigator()
        nav.set_current_location("hallway")
        result = nav.navigate_to("unknown_room")
        assert result is None

    def test_navigate_same_location(self):
        """Test navigating to current location."""
        nav = RobotNavigator()
        nav.add_waypoint("kitchen", 2.0, 3.0)
        nav.set_current_location("kitchen")
        result = nav.navigate_to("kitchen")
        assert result is not None
        assert len(result.waypoints) == 0

    def test_navigate_to_connected(self):
        """Test navigating to connected waypoint."""
        nav = RobotNavigator()
        nav.add_waypoint("hallway", 0.0, 0.0)
        nav.add_waypoint("kitchen", 3.0, 4.0)
        nav.add_connection("hallway", "kitchen")
        nav.set_current_location("hallway")

        result = nav.navigate_to("kitchen")
        # May be NavigationPath or None depending on PDDL planning
        assert isinstance(result, (NavigationPath, type(None)))


class TestObjectManipulator:
    """Test ObjectManipulator."""

    def test_init(self):
        """Test initialization."""
        manip = ObjectManipulator()
        assert manip.holding is None

    def test_add_object_and_surface(self):
        """Test adding objects and surfaces."""
        manip = ObjectManipulator()
        manip.add_object("cup", 0.3, 0.2, 0.1)
        manip.add_surface("table", 0.5, 0.0, 0.0)

        assert "cup" in manip.objects
        assert "table" in manip.surfaces

    def test_pick_success(self):
        """Test successful pick."""
        manip = ObjectManipulator()
        manip.add_object("cup", 0.3, 0.2, 0.1)

        result = manip.pick("cup")
        assert result is True
        assert manip.holding == "cup"

    def test_pick_already_holding(self):
        """Test pick while already holding."""
        manip = ObjectManipulator()
        manip.add_object("cup", 0.3, 0.2, 0.1)
        manip.add_object("plate", 0.4, 0.1, 0.0)
        manip.pick("cup")

        result = manip.pick("plate")
        assert result is False

    def test_pick_unknown_object(self):
        """Test picking unknown object."""
        manip = ObjectManipulator()
        result = manip.pick("nonexistent")
        assert result is False

    def test_pick_out_of_reach(self):
        """Test picking unreachable object."""
        manip = ObjectManipulator()
        manip.add_object("far_obj", 5.0, 5.0, 0.0)  # Way beyond workspace

        result = manip.pick("far_obj")
        assert result is False

    def test_place_success(self):
        """Test successful place."""
        manip = ObjectManipulator()
        manip.add_object("cup", 0.3, 0.2, 0.1)
        manip.add_surface("table", 0.5, 0.0, 0.0)
        manip.pick("cup")

        result = manip.place("table")
        assert result is True
        assert manip.holding is None

    def test_place_not_holding(self):
        """Test placing when not holding."""
        manip = ObjectManipulator()
        manip.add_surface("table", 0.5, 0.0, 0.0)

        result = manip.place("table")
        assert result is False

    def test_place_unknown_surface(self):
        """Test placing on unknown surface."""
        manip = ObjectManipulator()
        manip.add_object("cup", 0.3, 0.2, 0.1)
        manip.pick("cup")

        result = manip.place("unknown")
        assert result is False

    def test_plan_manipulation(self):
        """Test manipulation planning."""
        manip = ObjectManipulator()
        manip.add_object("cup", 0.3, 0.2, 0.1)
        manip.add_surface("table", 0.5, 0.0, 0.0)

        plan = manip.plan_manipulation([("cup", "table")])
        assert isinstance(plan, list)


class TestVisualServoing:
    """Test VisualServoing."""

    def test_init(self):
        """Test initialization."""
        servo = VisualServoing()
        assert not servo.is_servoing

    def test_set_target(self):
        """Test setting target."""
        servo = VisualServoing()
        servo.set_target(np.array([1.0, 2.0, 0.0]))
        np.testing.assert_array_equal(servo.target_position, [1.0, 2.0, 0.0])

    def test_compute_control_no_target(self):
        """Test control without target."""
        servo = VisualServoing()
        lin, ang = servo.compute_control(np.array([0.0, 0.0, 0.0]))
        assert lin == 0.0 and ang == 0.0

    def test_compute_control_at_target(self):
        """Test control when at target."""
        servo = VisualServoing()
        servo.set_target(np.array([1.0, 0.0, 0.0]))
        lin, ang = servo.compute_control(np.array([1.0, 0.0, 0.0]))
        assert lin == 0.0 and ang == 0.0

    def test_compute_control_move(self):
        """Test control when far from target."""
        servo = VisualServoing()
        servo.set_target(np.array([2.0, 0.0, 0.0]))
        lin, ang = servo.compute_control(np.array([0.0, 0.0, 0.0]))
        assert lin > 0.0

    def test_servo_step_reached(self):
        """Test servo step reaching target."""
        servo = VisualServoing()
        servo.set_target(np.array([1.0, 0.0, 0.0]))
        servo.is_servoing = True

        reached = servo.servo_step(np.array([1.0, 0.0, 0.0]))
        assert reached is True
        assert not servo.is_servoing

    def test_servo_step_not_reached(self):
        """Test servo step not yet at target."""
        servo = VisualServoing()
        servo.set_target(np.array([5.0, 0.0, 0.0]))
        servo.is_servoing = True

        reached = servo.servo_step(np.array([0.0, 0.0, 0.0]))
        assert reached is False


# === DEVICE TESTS ===

from nesy.hal.device import Device, DevicePool, Workload, WorkloadType, ExecutionResult
from nesy.core.memory import DeviceType


class TestDeviceAdvanced:
    """Test Device class edge cases."""

    def test_device_pool_find_no_device(self):
        """Test finding device when none registered."""
        uma = UMA()
        pool = DevicePool(uma=uma)

        workload = Workload(
            workload_type=WorkloadType.GENERAL_COMPUTE,
            operation=lambda: None,
            inputs={}
        )

        device = pool.find_device_for_workload(workload)
        assert device is None

    def test_device_pool_get_all_statistics(self):
        """Test getting statistics for all devices."""
        uma = UMA()
        pool = DevicePool(uma=uma)

        stats = pool.get_all_statistics()
        assert isinstance(stats, dict)

    def test_device_pool_reset_statistics(self):
        """Test resetting statistics."""
        uma = UMA()
        pool = DevicePool(uma=uma)

        pool.reset_all_statistics()
        # Should not raise


# === HYPERVECTOR ADDITIONAL TESTS ===


class TestHyperVectorAdvanced:
    """Test hypervector edge cases."""

    def test_from_vector_normalize(self):
        """Test creating HV from vector with normalization."""
        vec = np.random.randn(100).astype(np.float32) * 10
        hv = HyperVector.from_vector(vec, normalize=True)

        norm = np.linalg.norm(hv.data)
        assert abs(norm - 1.0) < 0.01

    def test_from_vector_no_normalize(self):
        """Test creating HV without normalization."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        hv = HyperVector.from_vector(vec, normalize=False)

        np.testing.assert_array_almost_equal(hv.data, vec)

    def test_random_hv(self):
        """Test random HV creation."""
        hv = HyperVector.random(500)
        assert hv.dim == 500

    def test_hv_bind(self):
        """Test binding operation."""
        hv1 = HyperVector.random(100)
        hv2 = HyperVector.random(100)

        bound = hv1.bind(hv2)
        assert bound.dim == 100

    def test_hv_bundle(self):
        """Test bundling operation."""
        hv1 = HyperVector.random(100)
        hv2 = HyperVector.random(100)

        bundled = hv1.bundle(hv2)
        assert bundled.dim == 100

    def test_hv_similarity_self(self):
        """Test self-similarity is 1."""
        hv = HyperVector.random(100)
        sim = hv.similarity(hv)
        assert abs(sim - 1.0) < 0.01

    def test_hv_similarity_orthogonal(self):
        """Test random HVs are approximately orthogonal."""
        hv1 = HyperVector.random(10000)
        hv2 = HyperVector.random(10000)

        sim = hv1.similarity(hv2)
        assert abs(sim) < 0.1  # Should be near 0


# === CODEBOOK ADDITIONAL TESTS ===


class TestCodebookAdvanced:
    """Test codebook edge cases."""

    def test_encode_position(self):
        """Test position encoding."""
        cb = VSACodebook(dim=1000)
        hv = cb.encode_position(np.array([1.0, 2.0, 3.0]))
        assert hv.dim == 1000

    def test_encode_attributes(self):
        """Test attribute encoding."""
        cb = VSACodebook(dim=1000)
        hv = cb.encode_attributes({"color": "red", "size": "large"})
        assert hv.dim == 1000

    def test_encode_same_symbol_cached(self):
        """Test encoding same symbol returns same HV."""
        cb = VSACodebook(dim=1000)
        hv1 = cb.encode("cup")
        hv2 = cb.encode("cup")

        sim = hv1.similarity(hv2)
        assert abs(sim - 1.0) < 0.01

    def test_size(self):
        """Test codebook size."""
        cb = VSACodebook(dim=1000)
        assert cb.size() == 0

        cb.encode("cup")
        assert cb.size() >= 1

    def test_decode(self):
        """Test decoding HV to symbols."""
        cb = VSACodebook(dim=1000)
        hv = cb.encode("cup")

        results = cb.decode(hv, top_k=3)
        assert len(results) >= 1
        assert results[0][0] == "cup"
