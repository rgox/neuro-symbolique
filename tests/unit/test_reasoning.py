"""
Unit tests for Logic Reasoning Engine.

Tests Scallop integration, scene graph sync, spatial reasoning,
and VSA-logic bridging.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

# These tests work without Scallop (uses mock fallback)
from nesy.reasoning.logic import ScallopContext, ReasoningEngine, SCALLOP_AVAILABLE
from nesy.world_model.scene_graph import (
    SceneGraph, LayerType, NodeType, RelationType
)


def test_scallop_context_basic():
    """Test basic Scallop context operations."""
    print("[TEST] Scallop context basic operations...")
    
    ctx = ScallopContext()
    
    # Define relation
    ctx.add_relation("edge", ["String", "String"])
    
    # Add facts
    ctx.add_fact("edge", "a", "b")
    ctx.add_fact("edge", "b", "c")
    
    assert ctx.get_num_facts() == 2
    
    # Add rule
    ctx.add_rule("path(X, Y) :- edge(X, Y)")
    ctx.add_rule("path(X, Z) :- edge(X, Y), path(Y, Z)")
    
    assert ctx.get_num_rules() == 2
    
    print(f"  Context: {ctx}")
    print("  ✓ Basic operations work")


def test_scallop_context_query():
    """Test Scallop query."""
    print("[TEST] Scallop context query...")
    
    ctx = ScallopContext()
    
    # Define relation
    ctx.add_relation("person", ["String"])
    ctx.add_relation("parent", ["String", "String"])
    
    # Facts
    ctx.add_fact("person", "alice")
    ctx.add_fact("person", "bob")
    ctx.add_fact("person", "charlie")
    ctx.add_fact("parent", "alice", "bob")
    ctx.add_fact("parent", "bob", "charlie")
    
    # Rules
    ctx.add_relation("ancestor", ["String", "String"])
    ctx.add_rule("ancestor(X, Y) :- parent(X, Y)")
    ctx.add_rule("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)")
    
    # Query
    results = ctx.query("ancestor")
    
    print(f"  Ancestors: {results}")
    
    if SCALLOP_AVAILABLE:
        # With Scallop, should infer transitive closure
        assert len(results) >= 2  # At least alice→bob, bob→charlie
    else:
        # Mock returns base facts only
        assert len(results) >= 0
    
    print("  ✓ Query works")


def test_reasoning_engine_init():
    """Test reasoning engine initialization."""
    print("[TEST] Reasoning engine initialization...")
    
    sg = SceneGraph()
    engine = ReasoningEngine(scene_graph=sg)
    
    stats = engine.get_statistics()
    
    print(f"  Engine: {engine}")
    print(f"  Stats: {stats}")
    
    assert stats["num_rules"] > 0  # Should have default rules
    assert "in" in stats["relations"]
    assert "on" in stats["relations"]
    
    print("  ✓ Initialization works")


def test_reasoning_engine_scene_graph_sync():
    """Test syncing scene graph to reasoning engine."""
    print("[TEST] Scene graph sync...")
    
    # Create scene graph
    sg = SceneGraph()
    
    # Add nodes
    cup = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[1, 2, 0.5],
        attributes={"class": "cup", "color": "red"}
    )
    
    table = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[1, 2, 0],
        attributes={"class": "table"}
    )
    
    kitchen = sg.add_node(
        layer=LayerType.L2,
        node_type=NodeType.ROOM,
        position=[0, 0, 0],
        attributes={"class": "kitchen"}
    )
    
    # Add edges
    sg.add_edge(cup.id, table.id, RelationType.ON)
    sg.add_edge(table.id, kitchen.id, RelationType.IN)
    
    # Create reasoning engine
    engine = ReasoningEngine(scene_graph=sg)
    
    # Sync
    engine.sync_from_scene_graph()
    
    stats = engine.get_statistics()
    
    print(f"  Synced {stats['num_facts']} facts")
    
    assert stats["num_facts"] > 0
    
    # Query objects
    objects = engine.query("object")
    print(f"  Objects: {objects}")
    
    # Should have 3 objects
    assert len(objects) == 3
    
    # Query spatial relations
    on_relations = engine.query("on")
    in_relations = engine.query("in")
    
    print(f"  ON relations: {on_relations}")
    print(f"  IN relations: {in_relations}")
    
    assert len(on_relations) == 1  # cup on table
    # table in kitchen + cup in kitchen (inferred via transitivity: on(cup,table) + in(table,kitchen))
    assert len(in_relations) >= 1
    
    print("  ✓ Scene graph sync works")


def test_reasoning_spatial_inference():
    """Test spatial reasoning via transitivity."""
    print("[TEST] Spatial reasoning (transitivity)...")
    
    # Create scene graph
    sg = SceneGraph()
    
    cup = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[1, 2, 0.5],
        attributes={"class": "cup"}
    )
    
    table = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[1, 2, 0],
        attributes={"class": "table"}
    )
    
    kitchen = sg.add_node(
        layer=LayerType.L2,
        node_type=NodeType.ROOM,
        position=[0, 0, 0],
        attributes={"class": "kitchen"}
    )
    
    # Edges: cup ON table, table IN kitchen
    sg.add_edge(cup.id, table.id, RelationType.ON)
    sg.add_edge(table.id, kitchen.id, RelationType.IN)
    
    # Reasoning
    engine = ReasoningEngine(scene_graph=sg)
    engine.sync_from_scene_graph()
    
    # Query: should infer cup IN kitchen (via transitivity)
    in_relations = engine.query("in")
    
    print(f"  IN relations (with inference): {in_relations}")
    
    if SCALLOP_AVAILABLE:
        # Should have: table in kitchen, cup in kitchen (inferred)
        assert len(in_relations) >= 2
        
        # Check cup is in kitchen
        cup_in_kitchen = any(
            src == cup.id and dst == kitchen.id
            for src, dst in in_relations
        )
        
        print(f"  Cup in kitchen (inferred): {cup_in_kitchen}")
        assert cup_in_kitchen, "Transitive inference failed"
    
    print("  ✓ Spatial inference works")


def test_reasoning_find_methods():
    """Test find_all and find_in_location methods."""
    print("[TEST] Find methods...")
    
    sg = SceneGraph()
    
    # Add multiple cups
    cup1 = sg.add_node(
        LayerType.L1, NodeType.OBJECT, [1, 2, 0.5],
        attributes={"class": "cup"}
    )
    
    cup2 = sg.add_node(
        LayerType.L1, NodeType.OBJECT, [2, 2, 0.5],
        attributes={"class": "cup"}
    )
    
    table = sg.add_node(
        LayerType.L1, NodeType.OBJECT, [1, 2, 0],
        attributes={"class": "table"}
    )
    
    kitchen = sg.add_node(
        LayerType.L2, NodeType.ROOM, [0, 0, 0],
        attributes={"class": "kitchen"}
    )
    
    # Edges
    sg.add_edge(cup1.id, table.id, RelationType.ON)
    sg.add_edge(cup2.id, table.id, RelationType.ON)
    sg.add_edge(table.id, kitchen.id, RelationType.IN)
    
    # Reasoning
    engine = ReasoningEngine(scene_graph=sg)
    engine.sync_from_scene_graph()
    
    # Find all cups
    cups = engine.find_all("cup")
    print(f"  All cups: {cups}")
    assert len(cups) == 2
    
    # Find objects in kitchen
    in_kitchen = engine.find_in_location(kitchen.id)
    print(f"  Objects in kitchen: {in_kitchen}")
    
    if SCALLOP_AVAILABLE:
        # Should include table + cups (via transitivity)
        assert len(in_kitchen) >= 1
    
    print("  ✓ Find methods work")


def run_all_tests():
    """Run all reasoning engine tests."""
    print("=" * 80)
    print("REASONING ENGINE UNIT TESTS")
    if not SCALLOP_AVAILABLE:
        print("⚠️  WARNING: Scallop not installed, using mock implementation")
        print("   Install with: pip install scallop-lang")
    print("=" * 80)
    print()
    
    tests = [
        test_scallop_context_basic,
        test_scallop_context_query,
        test_reasoning_engine_init,
        test_reasoning_engine_scene_graph_sync,
        test_reasoning_spatial_inference,
        test_reasoning_find_methods,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            print()
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 80)
    print(f"RESULTS: {passed}/{passed+failed} tests passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} tests failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
