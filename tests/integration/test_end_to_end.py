"""
End-to-end integration tests for complete neural-symbolic pipeline.

Tests the full pipeline from image input to answer output:
Image → Perception → VSA → Scene Graph → Reasoning → Answer
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from nesy.pipeline.full_pipeline import NeuralSymbolicPipeline
from nesy.world_model.scene_graph import LayerType, NodeType, RelationType


def test_e2e_pipeline_initialization():
    """Test: Pipeline initializes all components."""
    print("[E2E] Testing pipeline initialization...")
    
    pipeline = NeuralSymbolicPipeline(
        detector_backend="mock",
        feature_backend="mock"
    )
    
    # Verify components exist
    assert pipeline.detector is not None
    assert pipeline.feature_extractor is not None
    assert pipeline.scene_graph is not None
    assert pipeline.codebook is not None
    assert pipeline.grounding is not None
    assert pipeline.reasoning is not None
    
    stats = pipeline.get_statistics()
    print(f"  Pipeline: {pipeline}")
    print(f"  Stats: {stats}")
    
    print("  ✓ Pipeline initialization works")


def test_e2e_single_object_detection():
    """Test: Single object detection → scene graph → reasoning."""
    print("[E2E] Testing single object detection...")
    
    pipeline = NeuralSymbolicPipeline(
        detector_backend="mock",
        feature_backend="mock"
    )
    
    # Create mock image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Process
    result = pipeline.process_image(
        image=image,
        ground_vsa=False,  # Skip VSA for now
        sync_reasoning=True
    )
    
    print(f"  Detections: {len(result['detections'])}")
    print(f"  Nodes: {result['num_nodes']}")
    print(f"  Facts: {result['num_facts']}")
    print(f"  Processing time: {result['processing_time']:.2f}ms")
    
    # Should have detected at least one object (mock detector)
    assert len(result['detections']) > 0
    assert result['num_nodes'] > 0
    assert result['num_facts'] > 0
    
    print("  ✓ Single object detection works")


def test_e2e_spatial_reasoning():
    """Test: Spatial reasoning via scene graph + logic."""
    print("[E2E] Testing spatial reasoning...")
    
    pipeline = NeuralSymbolicPipeline()
    
    # Manually create scene for testing
    sg = pipeline.scene_graph
    
    # Objects
    cup = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[1.0, 2.0, 0.5],
        attributes={"class": "cup", "color": "red"}
    )
    
    table = sg.add_node(
        layer=LayerType.L1,
        node_type=NodeType.OBJECT,
        position=[1.0, 2.0, 0.0],
        attributes={"class": "table"}
    )
    
    kitchen = sg.add_node(
        layer=LayerType.L2,
        node_type=NodeType.ROOM,
        position=[0.0, 0.0, 0.0],
        attributes={"class": "kitchen"}
    )
    
    # Spatial relations
    sg.add_edge(cup.id, table.id, RelationType.ON)
    sg.add_edge(table.id, kitchen.id, RelationType.IN)
    
    # Sync to reasoning
    pipeline.reasoning.sync_from_scene_graph()
    
    # Query: What's on the table?
    on_relations = pipeline.reasoning.query("on")
    print(f"  ON relations: {on_relations}")
    assert len(on_relations) == 1
    assert on_relations[0] == (cup.id, table.id)
    
    # Query: What's in the kitchen? (should include cup via transitivity)
    in_relations = pipeline.reasoning.query("in")
    print(f"  IN relations: {in_relations}")
    
    # Should have: table in kitchen (direct), cup in kitchen (inferred)
    # Note: Scallop availability affects this
    assert len(in_relations) >= 1  # At least table in kitchen
    
    # Infer: Is cup in kitchen?
    is_in_kitchen = pipeline.infer(f"in('{cup.id}', '{kitchen.id}')")
    print(f"  Cup in kitchen (inferred): {is_in_kitchen}")
    
    print("  ✓ Spatial reasoning works")


def test_e2e_find_operations():
    """Test: Find operations (find_all, find_in_location)."""
    print("[E2E] Testing find operations...")
    
    pipeline = NeuralSymbolicPipeline()
    sg = pipeline.scene_graph
    
    # Create multiple objects
    cup1 = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 2, 0.5],
                       attributes={"class": "cup"})
    cup2 = sg.add_node(LayerType.L1, NodeType.OBJECT, [2, 2, 0.5],
                       attributes={"class": "cup"})
    table = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 2, 0],
                        attributes={"class": "table"})
    kitchen = sg.add_node(LayerType.L2, NodeType.ROOM, [0, 0, 0],
                          attributes={"class": "kitchen"})
    
    # Relations
    sg.add_edge(cup1.id, table.id, RelationType.ON)
    sg.add_edge(cup2.id, table.id, RelationType.ON)
    sg.add_edge(table.id, kitchen.id, RelationType.IN)
    
    pipeline.reasoning.sync_from_scene_graph()
    
    # Find all cups
    cups = pipeline.find_all("cup")
    print(f"  All cups: {cups}")
    assert len(cups) == 2
    
    # Find objects in kitchen
    in_kitchen = pipeline.find_in_location(kitchen.id)
    print(f"  Objects in kitchen: {in_kitchen}")
    assert len(in_kitchen) >= 1  # At least table
    
    print("  ✓ Find operations work")


def test_e2e_natural_language_queries():
    """Test: Basic natural language query patterns."""
    print("[E2E] Testing natural language queries...")
    
    pipeline = NeuralSymbolicPipeline()
    sg = pipeline.scene_graph
    
    # Create scene
    cup = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 2, 0.5],
                      attributes={"class": "cup"})
    table = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 2, 0],
                        attributes={"class": "table"})
    kitchen = sg.add_node(LayerType.L2, NodeType.ROOM, [0, 0, 0],
                          attributes={"class": "kitchen"})
    
    sg.add_edge(cup.id, table.id, RelationType.ON)
    sg.add_edge(table.id, kitchen.id, RelationType.IN)
    
    pipeline.reasoning.sync_from_scene_graph()
    
    # Query: "Find all cups"
    result = pipeline.query("Find all cups")
    print(f"  'Find all cups' → {result}")
    assert len(result) >= 1
    
    # Query: "What's in the kitchen?"
    result = pipeline.query("What's in the kitchen?")
    print(f"  'What's in the kitchen?' → {result}")
    assert len(result) >= 1
    
    # Query: "What's on the table?"
    result = pipeline.query("What's on the table?")
    print(f"  'What's on the table?' → {result}")
    assert len(result) >= 1
    
    print("  ✓ Natural language queries work")


def test_e2e_pipeline_reset():
    """Test: Pipeline reset clears state."""
    print("[E2E] Testing pipeline reset...")
    
    pipeline = NeuralSymbolicPipeline()
    sg = pipeline.scene_graph
    
    # Add some nodes
    sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0],
                attributes={"class": "cup"})
    sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 0],
                attributes={"class": "table"})
    
    stats_before = pipeline.get_statistics()
    print(f"  Before reset: {stats_before['scene_graph']['total_nodes']} nodes")
    
    assert stats_before['scene_graph']['total_nodes'] > 0
    
    # Reset
    pipeline.reset()
    
    stats_after = pipeline.get_statistics()
    print(f"  After reset: {stats_after['scene_graph']['total_nodes']} nodes")
    
    assert stats_after['scene_graph']['total_nodes'] == 0
    assert stats_after['reasoning']['num_facts'] == 0
    
    print("  ✓ Pipeline reset works")


def run_all_tests():
    """Run all end-to-end integration tests."""
    print("=" * 80)
    print("END-TO-END INTEGRATION TESTS")
    print("=" * 80)
    print()
    
    tests = [
        test_e2e_pipeline_initialization,
        test_e2e_single_object_detection,
        test_e2e_spatial_reasoning,
        test_e2e_find_operations,
        test_e2e_natural_language_queries,
        test_e2e_pipeline_reset,
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
