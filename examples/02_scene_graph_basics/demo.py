"""
Scene Graph Basics Demo.

This demo showcases the 3D scene graph capabilities:
- Building a multi-layer scene (objects + rooms)
- Adding spatial relations (on, in, near)
- Semantic queries (find all cups, find objects in kitchen)
- Spatial queries (radius search, bounding box search)
- Octree spatial indexing for performance

The demo creates a simple kitchen scene with:
- Kitchen room (L2)
- Table, cup, robot objects (L1)
- Spatial relations between them
"""

from pathlib import Path
import numpy as np
from nesy import NeSyPlatform
from nesy.world_model import (
    SceneGraph,
    ObjectLayer,
    RoomLayer,
    LayerType,
    NodeType,
    RelationType,
    Octree,
    compute_spatial_relations,
    create_spatial_index_for_scene_graph,
)


def main():
    print("=" * 70)
    print("3D SCENE GRAPH - BASIC DEMONSTRATION")
    print("=" * 70)

    # Initialize platform
    print("\n1. Initializing NeSy Platform...")
    # Resolve config relative to repository root, regardless of cwd
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "minimal.yaml"
    platform = NeSyPlatform.from_config(config_path)
    print(f"   ✓ Platform initialized")
    print(f"   ✓ UMA available for embeddings")

    # Create scene graph
    print("\n2. Creating Scene Graph...")
    scene_graph = SceneGraph(uma=platform.uma, logger=platform.logger)
    print(f"   ✓ SceneGraph created")

    # Initialize layers
    obj_layer = ObjectLayer(scene_graph)
    room_layer = RoomLayer(scene_graph)
    print(f"   ✓ ObjectLayer (L1) and RoomLayer (L2) initialized")

    # Demo 1: Build a kitchen scene
    print("\n" + "=" * 70)
    print("DEMO 1: Building a Kitchen Scene (L1 Objects + L2 Room)")
    print("=" * 70)

    # Create kitchen room (L2)
    print("\n   Creating kitchen room...")
    kitchen = room_layer.create_room(
        name="kitchen",
        center=[5.0, 3.0, 0.0],
        bounds_min=[0.0, 0.0, 0.0],
        bounds_max=[10.0, 6.0, 3.0],
        room_type="indoor",
        description="Main kitchen area",
    )
    print(f"   ✓ Kitchen room created: {kitchen.id[:8]}...")
    print(f"     - Center: {kitchen.position}")
    print(f"     - Bounds: {kitchen.bbox[0]} to {kitchen.bbox[1]}")

    # Create objects (L1)
    print("\n   Creating objects in the kitchen...")

    table = obj_layer.create_object(
        class_name="table",
        position=[3.0, 2.0, 0.0],
        bbox_min=[2.0, 1.5, 0.0],
        bbox_max=[4.0, 2.5, 0.8],
        confidence=0.95,
        color="brown",
        material="wood",
    )
    print(f"   ✓ Table: {table.id[:8]}... at {table.position}")

    cup1 = obj_layer.create_object(
        class_name="cup",
        position=[3.0, 2.0, 0.85],
        bbox_min=[2.9, 1.9, 0.8],
        bbox_max=[3.1, 2.1, 1.0],
        confidence=0.92,
        color="red",
        material="ceramic",
    )
    print(f"   ✓ Cup 1 (red): {cup1.id[:8]}... at {cup1.position}")

    cup2 = obj_layer.create_object(
        class_name="cup",
        position=[3.5, 2.0, 0.85],
        bbox_min=[3.4, 1.9, 0.8],
        bbox_max=[3.6, 2.1, 1.0],
        confidence=0.88,
        color="blue",
    )
    print(f"   ✓ Cup 2 (blue): {cup2.id[:8]}... at {cup2.position}")

    chair = obj_layer.create_object(
        class_name="chair",
        position=[2.5, 3.5, 0.0],
        bbox_min=[2.0, 3.0, 0.0],
        bbox_max=[3.0, 4.0, 1.0],
        confidence=0.90,
        color="black",
    )
    print(f"   ✓ Chair: {chair.id[:8]}... at {chair.position}")

    robot = obj_layer.create_object(
        class_name="robot",
        position=[7.0, 4.0, 0.0],
        bbox_min=[6.5, 3.5, 0.0],
        bbox_max=[7.5, 4.5, 1.5],
        confidence=1.0,
        status="idle",
    )
    print(f"   ✓ Robot: {robot.id[:8]}... at {robot.position}")

    # Add spatial relations
    print("\n   Adding spatial relations...")
    obj_layer.add_spatial_relation(cup1.id, table.id, "on", confidence=0.95)
    obj_layer.add_spatial_relation(cup2.id, table.id, "on", confidence=0.95)
    print(f"   ✓ cup1 --[on]--> table")
    print(f"   ✓ cup2 --[on]--> table")

    # Add containment relations (objects in room)
    for obj in [table, cup1, cup2, chair, robot]:
        room_layer.add_object_to_room(obj.id, kitchen.id, confidence=1.0)
    print(f"   ✓ All objects added to kitchen room")

    # Compute additional spatial relations
    print("\n   Computing additional spatial relations...")
    num_relations = compute_spatial_relations(
        scene_graph,
        threshold_on=0.1,
        threshold_near=2.0,
    )
    print(f"   ✓ Computed {num_relations} spatial relations")

    # Demo 2: Semantic Queries
    print("\n" + "=" * 70)
    print("DEMO 2: Semantic Queries")
    print("=" * 70)

    print("\n   Query 1: Find all cups")
    cups = obj_layer.get_objects_by_class("cup")
    print(f"   ✓ Found {len(cups)} cups:")
    for c in cups:
        print(f"     - Cup ({c.attributes.get('color', 'unknown')}) at {c.position}")

    print("\n   Query 2: Find all objects in kitchen")
    kitchen_objects = room_layer.get_objects_in_room(kitchen.id)
    print(f"   ✓ Found {len(kitchen_objects)} objects in kitchen:")
    for obj in kitchen_objects:
        print(f"     - {obj.attributes.get('class', 'unknown')} at {obj.position}")

    print("\n   Query 3: Find red objects")
    red_objects = scene_graph.query_by_attributes(layer=LayerType.L1, color="red")
    print(f"   ✓ Found {len(red_objects)} red objects:")
    for obj in red_objects:
        print(f"     - {obj.attributes.get('class', 'unknown')} at {obj.position}")

    print("\n   Query 4: What is on the table?")
    on_table = scene_graph.get_neighbors(table.id, RelationType.ON, direction="in")
    print(f"   ✓ Found {len(on_table)} objects on table:")
    for obj in on_table:
        print(f"     - {obj.attributes.get('class', 'unknown')} ({obj.attributes.get('color', 'unknown')})")

    # Demo 3: Spatial Queries
    print("\n" + "=" * 70)
    print("DEMO 3: Spatial Queries (Naive O(n))")
    print("=" * 70)

    print("\n   Query 1: Find all objects within 1.5m of table center")
    near_table = scene_graph.query_spatial(
        center=table.position,
        radius=1.5,
        layer=LayerType.L1,
    )
    print(f"   ✓ Found {len(near_table)} objects near table:")
    for obj in near_table:
        distance = np.linalg.norm(obj.position - table.position)
        print(f"     - {obj.attributes.get('class', 'unknown')} (distance: {distance:.2f}m)")

    print("\n   Query 2: Find all objects in bounding box [6.0, 3.0, 0.0] to [8.0, 5.0, 2.0]")
    bbox_objects = scene_graph.query_bbox(
        min_xyz=[6.0, 3.0, 0.0],
        max_xyz=[8.0, 5.0, 2.0],
        layer=LayerType.L1,
    )
    print(f"   ✓ Found {len(bbox_objects)} objects in bbox:")
    for obj in bbox_objects:
        print(f"     - {obj.attributes.get('class', 'unknown')} at {obj.position}")

    # Demo 4: Octree Spatial Index
    print("\n" + "=" * 70)
    print("DEMO 4: Octree Spatial Index (O(log n))")
    print("=" * 70)

    print("\n   Creating octree spatial index...")
    octree = create_spatial_index_for_scene_graph(
        scene_graph,
        layer=LayerType.L1,
        size=20.0,
    )
    stats = octree.get_statistics()
    print(f"   ✓ Octree created:")
    print(f"     - Total objects indexed: {stats['total_objects']}")
    print(f"     - Tree depth: {stats['max_depth']}")
    print(f"     - Total nodes: {stats['total_nodes']}")
    print(f"     - Leaf nodes: {stats['leaf_nodes']}")

    print("\n   Query: Find objects within 1.5m of [3.0, 2.0, 0.5] using octree")
    nearby_ids = octree.query_radius(center=[3.0, 2.0, 0.5], radius=1.5)
    print(f"   ✓ Found {len(nearby_ids)} objects (fast O(log n) query):")
    for obj_id in nearby_ids:
        obj = scene_graph.get_node(obj_id)
        if obj:
            distance = np.linalg.norm(obj.position - [3.0, 2.0, 0.5])
            print(f"     - {obj.attributes.get('class', 'unknown')} (distance: {distance:.2f}m)")

    # Demo 5: Statistics and Analysis
    print("\n" + "=" * 70)
    print("DEMO 5: Scene Graph Statistics")
    print("=" * 70)

    stats = scene_graph.get_statistics()
    print(f"\n   Scene Graph Statistics:")
    print(f"   - Total nodes: {stats['total_nodes']}")
    print(f"   - Total edges: {stats['total_edges']}")
    print(f"   - Nodes by layer: {stats['nodes_by_layer']}")
    print(f"   - Edges by relation: {stats['edges_by_relation']}")
    print(f"   - Active layers: {stats['layers']}")

    print(f"\n   Object Layer (L1) Statistics:")
    objects = obj_layer.get_nodes()
    print(f"   - Total objects: {len(objects)}")
    classes = {}
    for obj in objects:
        cls = obj.attributes.get('class', 'unknown')
        classes[cls] = classes.get(cls, 0) + 1
    print(f"   - Object classes: {classes}")

    print(f"\n   Room Layer (L2) Statistics:")
    rooms = room_layer.get_nodes()
    print(f"   - Total rooms: {len(rooms)}")
    for room in rooms:
        obj_count = len(room_layer.get_objects_in_room(room.id))
        print(f"     - {room.attributes['name']}: {obj_count} objects")

    # Demo 6: Export
    print("\n" + "=" * 70)
    print("DEMO 6: Export Scene Graph")
    print("=" * 70)

    export = scene_graph.to_dict()
    print(f"\n   ✓ Scene graph exported to dictionary")
    print(f"   - {len(export['nodes'])} nodes")
    print(f"   - {len(export['edges'])} edges")
    print(f"   - Export can be saved to JSON for persistence")

    print("\n" + "=" * 70)
    print("SCENE GRAPH DEMO COMPLETE!")
    print("=" * 70)

    print("\n   Key Takeaways:")
    print("   ✓ Multi-layer scene graph (L1 objects + L2 rooms)")
    print("   ✓ Spatial relations (on, in, near) automatically computed")
    print("   ✓ Semantic queries (by class, color, room)")
    print("   ✓ Spatial queries (radius, bbox) with O(n) naive")
    print("   ✓ Octree spatial index for O(log n) queries")
    print("   ✓ Statistics and export capabilities")
    print("   ✓ Ready for integration with perception and reasoning!")

    print("\n   Next Steps:")
    print("   - Week 5: Perception → Scene Graph pipeline")
    print("   - Week 6: VSA for neural-symbolic grounding")
    print("   - Week 7: Reasoning over scene graph with Scallop")


if __name__ == "__main__":
    main()
