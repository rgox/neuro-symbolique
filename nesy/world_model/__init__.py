"""
World Model Module.

This module provides world representation components for the neuro-symbolic platform:
- Scene Graph: 3D hierarchical scene representation (L1-L5)
- Knowledge Graph: Semantic knowledge (ontologies, facts, rules)
- VSA: Vector-Symbolic Architectures for grounding
- Temporal: Temporal reasoning and change detection

For MVP, we implement Scene Graph (L1-L2).

Example:
    >>> from nesy.world_model import SceneGraph, ObjectLayer, RoomLayer
    >>> sg = SceneGraph(uma=uma, logger=logger)
    >>> obj_layer = ObjectLayer(sg)
    >>> room_layer = RoomLayer(sg)
"""

# Scene Graph (implemented in Week 4)
from nesy.world_model.scene_graph import (
    SceneGraph,
    Node,
    Edge,
    LayerType,
    NodeType,
    RelationType,
    ObjectLayer,
    RoomLayer,
    Octree,
    infer_room_from_position,
    compute_spatial_relations,
    create_spatial_index_for_scene_graph,
)

__all__ = [
    # Scene Graph
    "SceneGraph",
    "Node",
    "Edge",
    "LayerType",
    "NodeType",
    "RelationType",
    "ObjectLayer",
    "RoomLayer",
    "Octree",
    "infer_room_from_position",
    "compute_spatial_relations",
    "create_spatial_index_for_scene_graph",
]

# Knowledge Graph, VSA, Temporal will be added in later weeks
