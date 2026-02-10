"""
Performance Benchmarks.

Measures execution time and throughput for key operations:
- Scene graph operations
- Reasoning inference
- Knowledge graph queries
- Temporal event processing
- Planning algorithms
- Agent pipeline

Run: python -m pytest tests/benchmarks/test_benchmarks.py -v -s
"""

import pytest
import numpy as np
import time
from contextlib import contextmanager
from typing import Dict, List


@contextmanager
def timer(label: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    result = {"elapsed": 0.0}
    yield result
    result["elapsed"] = (time.perf_counter() - start) * 1000
    print(f"  ⏱  {label}: {result['elapsed']:.2f} ms")


class TestSceneGraphBenchmarks:
    """Benchmark scene graph operations."""
    
    def test_add_1000_nodes(self):
        """Benchmark: add 1000 nodes."""
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType
        
        sg = SceneGraph()
        
        with timer("Add 1000 nodes") as t:
            for i in range(1000):
                sg.add_node(
                    LayerType.L1, NodeType.OBJECT,
                    [float(i), 0.0, 0.0],
                    attributes={"label": f"obj_{i}"},
                    node_id=f"node_{i}"
                )
        
        assert len(sg.nodes) == 1000
        assert t["elapsed"] < 5000  # Should be under 5s
        print(f"    Throughput: {1000 / (t['elapsed'] / 1000):.0f} nodes/sec")
    
    def test_query_nodes(self):
        """Benchmark: query nodes from large graph."""
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType
        
        sg = SceneGraph()
        for i in range(500):
            sg.add_node(LayerType.L1, NodeType.OBJECT, [float(i), 0, 0], node_id=f"n_{i}")
        
        with timer("Query 500 nodes") as t:
            for _ in range(100):
                nodes = list(sg.nodes.values())
        
        assert t["elapsed"] < 1000


class TestReasoningBenchmarks:
    """Benchmark reasoning performance."""
    
    def test_inference_speed(self):
        """Benchmark: reasoning inference."""
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType
        from nesy.reasoning.logic import ReasoningEngine
        
        sg = SceneGraph()
        for i in range(50):
            sg.add_node(LayerType.L1, NodeType.OBJECT, 
                       [float(i), 0.0, float(i % 5)],
                       attributes={"label": f"obj_{i}"}, node_id=f"n_{i}")
        
        engine = ReasoningEngine(sg)
        
        with timer("50-object query") as t:
            results = engine.query("on")
        
        assert isinstance(results, list)
        assert t["elapsed"] < 5000


class TestKnowledgeGraphBenchmarks:
    """Benchmark knowledge graph operations."""
    
    def test_build_deep_ontology(self):
        """Benchmark: build ontology with 100 concepts."""
        from nesy.world_model.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        
        with timer("Build 100-concept ontology") as t:
            kg.add_concept("root")
            for i in range(10):
                parent = f"level1_{i}"
                kg.add_concept(parent, parent="root")
                for j in range(9):
                    kg.add_concept(f"level2_{i}_{j}", parent=parent)
        
        assert len(kg.concepts) == 101  # root + 10 + 90
        assert t["elapsed"] < 1000
    
    def test_transitive_query_speed(self):
        """Benchmark: transitive IS_A queries."""
        from nesy.world_model.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        # Build chain: a → b → c → ... → z (26 deep)
        prev = "root"
        kg.add_concept(prev)
        for c in "abcdefghijklmnopqrstuvwxyz":
            kg.add_concept(c, parent=prev)
            prev = c
        
        with timer("1000 transitive IS_A queries") as t:
            for _ in range(1000):
                kg.is_a("z", "root")
        
        assert kg.is_a("z", "root")
        assert t["elapsed"] < 2000
    
    def test_sparql_query_speed(self):
        """Benchmark: SPARQL-like queries."""
        from nesy.world_model.knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        kg.add_concept("thing")
        for i in range(50):
            kg.add_concept(f"type_{i}", parent="thing")
        
        with timer("100 SPARQL queries") as t:
            for _ in range(100):
                kg.query("SELECT ?x WHERE { ?x is_a thing }")
        
        assert t["elapsed"] < 5000


class TestTemporalBenchmarks:
    """Benchmark temporal reasoning."""
    
    def test_add_1000_events(self):
        """Benchmark: add 1000 events."""
        from nesy.reasoning.temporal import TemporalReasoner
        
        r = TemporalReasoner()
        
        with timer("Add 1000 events") as t:
            for i in range(1000):
                r.add_event(f"e_{i}", "action", float(i))
        
        assert len(r.events) == 1000
        assert t["elapsed"] < 5000
    
    def test_sequence_detection_speed(self):
        """Benchmark: sequence pattern detection."""
        from nesy.reasoning.temporal import TemporalReasoner
        
        r = TemporalReasoner()
        types = ["pick", "move", "place"]
        for i in range(300):
            r.add_event(f"e_{i}", types[i % 3], float(i))
        
        with timer("Sequence detection in 300 events") as t:
            matches = r.detect_sequence(["pick", "move", "place"])
        
        assert len(matches) > 0
        assert t["elapsed"] < 5000
    
    def test_causal_inference_speed(self):
        """Benchmark: causal inference."""
        from nesy.reasoning.temporal import TemporalReasoner
        
        r = TemporalReasoner()
        for i in range(100):
            r.add_event(f"push_{i}", "push", float(i * 2), subject=f"obj_{i}")
            r.add_event(f"move_{i}", "move", float(i * 2 + 0.5), subject=f"obj_{i}")
        
        with timer("100 causal inferences") as t:
            for i in range(100):
                r.infer_causality(f"push_{i}", f"move_{i}")
        
        assert t["elapsed"] < 2000


class TestAgentBenchmarks:
    """Benchmark agent operations."""
    
    def test_agent_init_time(self):
        """Benchmark: agent initialization."""
        from nesy.agents import AutonomousAgent, AgentConfig
        
        with timer("Agent initialization") as t:
            agent = AutonomousAgent(AgentConfig(use_yolo=False, use_clip=False))
        
        assert agent is not None
        assert t["elapsed"] < 2000
    
    def test_observe_cycle_time(self):
        """Benchmark: observation cycle."""
        from nesy.agents import AutonomousAgent, AgentConfig
        
        agent = AutonomousAgent(AgentConfig(use_yolo=False, use_clip=False))
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with timer("10 observation cycles") as t:
            for _ in range(10):
                agent.observe(image)
        
        print(f"    Avg: {t['elapsed'] / 10:.2f} ms/cycle")
        assert t["elapsed"] < 10000


class TestMultiAgentBenchmarks:
    """Benchmark multi-agent operations."""
    
    def test_register_100_agents(self):
        """Benchmark: register 100 agents."""
        from nesy.agents.multi_agent import MultiAgentCoordinator, AgentRole
        from unittest.mock import Mock
        
        coord = MultiAgentCoordinator()
        
        with timer("Register 100 agents") as t:
            for i in range(100):
                coord.register_agent(f"bot_{i}", Mock(), AgentRole.WORKER)
        
        assert len(coord.agents) == 100
        assert t["elapsed"] < 1000
    
    def test_allocate_100_tasks(self):
        """Benchmark: allocate 100 tasks."""
        from nesy.agents.multi_agent import MultiAgentCoordinator, AgentRole
        from unittest.mock import Mock
        
        coord = MultiAgentCoordinator()
        for i in range(100):
            coord.register_agent(f"bot_{i}", Mock(), AgentRole.WORKER, ["pick"])
        
        with timer("Allocate 100 tasks") as t:
            for i in range(100):
                coord.allocate_task(f"task_{i}", ["pick"])
        
        assert coord.metrics["tasks_assigned"] == 100
        assert t["elapsed"] < 2000


class TestSummaryBenchmark:
    """Summary benchmark showing system capacity."""
    
    def test_system_capacity_summary(self):
        """Print system capacity summary."""
        print("\n" + "="*60)
        print("  📊 SYSTEM CAPACITY SUMMARY")
        print("="*60)
        
        # Scene Graph
        from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType
        sg = SceneGraph()
        start = time.perf_counter()
        for i in range(1000):
            sg.add_node(LayerType.L1, NodeType.OBJECT, [float(i), 0, 0], node_id=f"n_{i}")
        sg_time = (time.perf_counter() - start) * 1000
        
        # KG
        from nesy.world_model.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        start = time.perf_counter()
        kg.add_concept("root")
        for i in range(99):
            kg.add_concept(f"c_{i}", parent="root")
        kg_time = (time.perf_counter() - start) * 1000
        
        # Temporal
        from nesy.reasoning.temporal import TemporalReasoner
        tr = TemporalReasoner()
        start = time.perf_counter()
        for i in range(1000):
            tr.add_event(f"e_{i}", "action", float(i))
        tr_time = (time.perf_counter() - start) * 1000
        
        print(f"\n  Scene Graph:   1000 nodes     in {sg_time:.1f} ms")
        print(f"  Knowledge Graph: 100 concepts in {kg_time:.1f} ms")
        print(f"  Temporal:      1000 events    in {tr_time:.1f} ms")
        print(f"  SG throughput: {1000 / (sg_time / 1000):.0f} nodes/sec")
        print(f"  KG throughput: {100 / (kg_time / 1000):.0f} concepts/sec")
        print(f"  TR throughput: {1000 / (tr_time / 1000):.0f} events/sec")
        print("="*60 + "\n")
        
        assert sg_time < 10000
        assert kg_time < 5000
        assert tr_time < 10000
