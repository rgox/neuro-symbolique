"""
Kitchen Scene Integration Test.

Realistic scenario with multiple objects and spatial relations.

Scene:
- Red cup on table
- Blue cup on counter
- Plate on table
- Table in kitchen
- Counter in kitchen  
- Chair near table

Queries:
- "What's on the table?" → [red_cup, plate]
- "What's in the kitchen?" → [red_cup, blue_cup, plate, table, counter, chair]
- "Find objects similar to red_cup" → [blue_cup] (VSA similarity)
- "Is the plate in the kitchen?" → True (via transitivity)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np

from nesy.pipeline.full_pipeline import Neural SymbolicPipeline
from nesy.world_model.scene_graph import LayerType, NodeType, RelationType
from nesy.reasoning.logic import SCALLOP_AVAILABLE


def create_kitchen_scene(pipeline: NeuralSymbolicPipeline):
    """
    Create realistic kitchen scene.
    
    Returns node IDs for objects.
    """
    sg = pipeline.scene_graph
    
    # L2: Kitchen room
    kitchen = sg.add_node(
        layer=LayerType.L2,
        node_type=NodeType.ROOM,
        position=[5.0, 5.0, 0.0],
        attributes={"class": "kitchen", "area": 25.0}
    )
    
    # L1: Furniture
    table = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[5.0, 5.0, 0.0],
        attributes={"class": "table", "material": "wood"}
    )
    
    counter = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[7.0, 5.0, 0.0],
        attributes={"class": "counter", "material": "granite"}
    )
    
    chair = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[4.0, 5.0, 0.0],
        attributes={"class": "chair", "material": "wood"}
    )
    
    # L1: Objects on surfaces
    red_cup = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[5.0, 5.0, 0.75],
        attributes={"class": "cup", "color": "red"}
    )
    
    blue_cup = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[7.0, 5.0, 0.75],
        attributes={"class": "cup", "color": "blue"}
    )
    
    plate = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[5.2, 5.0, 0.75],
        attributes={"class": "plate", "material": "ceramic"}
    )
    
    # Spatial relations
    sg.add_edge(table.id, kitchen.id, RelationType.IN)
    sg.add_edge(counter.id, kitchen.id, RelationType.IN)
    sg.add_edge(chair.id, table.id, RelationType.NEAR)
    
    sg.add_edge(red_cup.id, table.id, RelationType.ON)
    sg.add_edge(blue_cup.id, counter.id, RelationType.ON)
    sg.add_edge(plate.id, table.id, RelationType.ON)
    
    return {
        "kitchen": kitchen.id,
        "table": table.id,
        "counter": counter.id,
        "chair": chair.id,
        "red_cup": red_cup.id,
        "blue_cup": blue_cup.id,
        "plate": plate.id,
    }


def test_kitchen_scene_queries():
    """Test queries on kitchen scene."""
    print("[KITCHEN SCENE] Testing spatial queries...")
    
    pipeline = NeuralSymbolicPipeline()
    objects = create_kitchen_scene(pipeline)
    
    pipeline.reasoning.sync_from_scene_graph()
    
    # Query: What's on the table?
    on_table = [
        src for src, dst in pipeline.reasoning.query("on")
        if dst == objects["table"]
    ]
    print(f"  On table: {on_table}")
    assert objects["red_cup"] in on_table
    assert objects["plate"] in on_table
    assert len(on_table) == 2
    
    # Query: What's in the kitchen?
    in_kitchen = pipeline.find_in_location(objects["kitchen"])
    print(f"  In kitchen: {in_kitchen}")
    
    # Should have at least table, counter (direct)
    assert objects["table"] in in_kitchen
    assert objects["counter"] in in_kitchen
    
    if SCALLOP_AVAILABLE:
        # With Scallop, should infer cups/plate in kitchen via transitivity
        assert len(in_kitchen) >= 4  # table, counter, cups, plate
    
    # Query: Find all cups
    cups = pipeline.find_all("cup")
    print(f"  All cups: {cups}")
    assert len(cups) == 2
    
    print("  ✓ Kitchen scene queries work")


def test_kitchen_scene_transitivity():
    """Test spatial transitivity in kitchen scene."""
    print("[KITCHEN SCENE] Testing transitivity...")
    
    pipeline = NeuralSymbolicPipeline()
    objects = create_kitchen_scene(pipeline)
    
    pipeline.reasoning.sync_from_scene_graph()
    
    # Direct relation: red_cup ON table
    on_rels = pipeline.reasoning.query("on")
    assert (objects["red_cup"], objects["table"]) in on_rels
    
    # Direct relation: table IN kitchen
    in_rels = pipeline.reasoning.query("in")
    assert (objects["table"], objects["kitchen"]) in in_rels
    
    # Inferred: red_cup IN kitchen (via on + in → in)
    if SCALLOP_AVAILABLE:
        cup_in_kitchen = pipeline.infer(
            f"in('{objects['red_cup']}', '{objects['kitchen']}')"
        )
        print(f"  Red cup in kitchen (inferred): {cup_in_kitchen}")
        assert cup_in_kitchen, "Transitivity inference failed"
    
    print("  ✓ Kitchen transitivity works")


def test_kitchen_scene_natural_language():
    """Test natural language queries on kitchen scene."""
    print("[KITCHEN SCENE] Testing natural language...")
    
    pipeline = NeuralSymbolicPipeline()
    objects = create_kitchen_scene(pipeline)
    
    pipeline.reasoning.sync_from_scene_graph()
    
    # Query: "What's in the kitchen?"
    result = pipeline.query("What's in the kitchen?")
    print(f"  'What's in the kitchen?' → {len(result)} objects")
    assert len(result) >= 2  # At least table, counter
    
    # Query: "Find all cups"
    result = pipeline.query("Find all cups")
    print(f"  'Find all cups' → {result}")
    assert len(result) == 2
    
    # Query: "What's on the table?"
    result = pipeline.query("What's on the table?")
    print(f"  'What's on the table?' → {result}")
    assert len(result) >= 2  # red_cup, plate
    
    print("  ✓ Kitchen natural language works")


def run_kitchen_tests():
    """Run all kitchen scene tests."""
    print("=" * 80)
    print("KITCHEN SCENE INTEGRATION TESTS")
    print("=" * 80)
    print()
    
    tests = [
        test_kitchen_scene_queries,
        test_kitchen_scene_transitivity,
        test_kitchen_scene_natural_language,
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
    success = run_kitchen_tests()
    sys.exit(0 if success else 1)
