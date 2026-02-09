"""
Complete Demo - Neural-Symbolic AI Platform.

Demonstrates the complete pipeline from image to answer.

Example:
    python examples/demo_complete.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from nesy.pipeline import NeuralSymbolicPipeline
from nesy.world_model.scene_graph import LayerType, NodeType, RelationType


def create_demo_scene(pipeline):
    """Create a demo scene for testing."""
    sg = pipeline.scene_graph
    
    # Create kitchen with objects
    kitchen = sg.add_node(
        LayerType.L2, NodeType.ROOM, [0, 0, 0],
        attributes={"class": "kitchen"}
    )
    
    table = sg.add_node(
        LayerType.L1, NodeType.OBJECT, [1, 1, 0],
        attributes={"class": "table"}
    )
    
    cup = sg.add_node(
        LayerType.L1, NodeType.OBJECT, [1, 1, 0.5],
        attributes={"class": "cup", "color": "red"}
    )
    
    plate = sg.add_node(
        LayerType.L1, NodeType.OBJECT, [1.2, 1, 0.5],
        attributes={"class": "plate"}
    )
    
    # Spatial relations
    sg.add_edge(cup.id, table.id, RelationType.ON)
    sg.add_edge(plate.id, table.id, RelationType.ON)
    sg.add_edge(table.id, kitchen.id, RelationType.IN)
    
    return {
        "kitchen": kitchen.id,
        "table": table.id,
        "cup": cup.id,
        "plate": plate.id,
    }


def main():
    """Run complete demo."""
    print("=" * 80)
    print("NEURAL-SYMBOLIC AI PLATFORM - COMPLETE DEMO")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    print("1. Initializing pipeline...")
    pipeline = NeuralSymbolicPipeline(
        detector_backend="mock",
        feature_backend="mock"
    )
    print(f"   {pipeline}")
    print()
    
    # Create scene
    print("2. Creating demo scene...")
    objects = create_demo_scene(pipeline)
    print(f"   Created {len(objects)} objects")
    print()
    
    # Sync to reasoning
    print("3. Syncing to reasoning engine...")
    pipeline.reasoning.sync_from_scene_graph()
    stats = pipeline.get_statistics()
    print(f"   Scene graph stats: {stats['scene_graph']}")
    print(f"   Reasoning stats: {stats['reasoning']}")
    print()
    
    # Query demonstrations
    print("4. Running queries...")
    print()
    
    # Query 1: Find all objects
    print("   Q: Find all cups")
    cups = pipeline.find_all("cup")
    print(f"   A: {cups}")
    print()
    
    # Query 2: What's in the kitchen?
    print("   Q: What's in the kitchen?")
    result = pipeline.query("What's in the kitchen?")
    print(f"   A: {result}")
    print()
    
    # Query 3: What's on the table?
    print("   Q: What's on the table?")
    result = pipeline.query("What's on the table?")
    print(f"   A: {result}")
    print()
    
    # Query 4: Is cup in kitchen? (inference)
    print("   Q: Is cup in kitchen? (via transitivity)")
    is_true = pipeline.infer(f"in('{objects['cup']}', '{objects['kitchen']}')")
    print(f"   A: {is_true}")
    print()
    
    print("=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
