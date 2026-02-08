"""
3D Scene Graph Module.

This module provides the multi-layer 3D scene graph implementation for
neuro-symbolic world representation.

Main Components:
    - SceneGraph: Core graph structure
    - Node, Edge: Graph elements with semantic attributes
    - LayerType, NodeType, RelationType: Type enumerations
    - ObjectLayer, RoomLayer: Layer-specific interfaces (L1, L2)
    - Octree: Spatial indexing for fast queries

Example:
    >>> from nesy.world_model.scene_graph import (
    ...     SceneGraph,
    ...     ObjectLayer,
    ...     RoomLayer,
    ...     LayerType,
    ...     NodeType,
    ...     RelationType,
    ... )
    >>> sg = SceneGraph(uma=uma, logger=logger)
    >>> obj_layer = ObjectLayer(sg)
    >>> cup = obj_layer.create_object(
    ...     class_name="cup",
    ...     position=[1.0, 2.0, 0.5],
    ...     bbox_min=[0.9, 1.9, 0.4],
    ...     bbox_max=[1.1, 2.1, 0.6],
    ... )
"""

from nesy.world_model.scene_graph.graph import (
    SceneGraph,
    Node,
    Edge,
    LayerType,
    NodeType,
    RelationType,
)
from nesy.world_model.scene_graph.layers import (
    Layer,
    ObjectLayer,
    RoomLayer,
    BuildingLayer,
    infer_room_from_position,
    compute_spatial_relations,
)
from nesy.world_model.scene_graph.spatial_index import (
    Octree,
    OctreeNode,
    create_spatial_index_for_scene_graph,
)

__all__ = [
    # Core graph
    "SceneGraph",
    "Node",
    "Edge",
    # Enums
    "LayerType",
    "NodeType",
    "RelationType",
    # Layers
    "Layer",
    "ObjectLayer",
    "RoomLayer",
    "BuildingLayer",
    # Spatial indexing
    "Octree",
    "OctreeNode",
    # Helper functions
    "infer_room_from_position",
    "compute_spatial_relations",
    "create_spatial_index_for_scene_graph",
]
