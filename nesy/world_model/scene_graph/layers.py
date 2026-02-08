"""
Scene Graph Layer Implementations.

This module implements the different layers of the 3D scene graph:
- L1 (Objects): Individual object instances (chairs, cups, tables, etc.)
- L2 (Rooms): Semantic places/regions (kitchen, bedroom, hallway, etc.)
- L3 (Buildings): Large structures (building level, optional for MVP)

Each layer provides convenient methods for creating and managing nodes at that
hierarchical level, along with layer-specific operations.

Architecture:
    - Layer interface defines common operations
    - ObjectLayer (L1): Manages detected objects with bboxes, classes, embeddings
    - RoomLayer (L2): Manages semantic regions with containment relations
    - BuildingLayer (L3): Manages building-level structures (post-MVP)
"""

from typing import Any, Dict, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

from nesy.world_model.scene_graph.graph import (
    SceneGraph,
    Node,
    LayerType,
    NodeType,
    RelationType,
)


class Layer(ABC):
    """
    Abstract base class for scene graph layers.

    Each layer represents a level in the hierarchical scene graph and provides
    methods for creating and querying nodes at that level.
    """

    def __init__(
        self,
        scene_graph: SceneGraph,
        layer_type: LayerType,
    ):
        """
        Initialize layer.

        Args:
            scene_graph: Parent scene graph
            layer_type: Layer type identifier
        """
        self.scene_graph = scene_graph
        self.layer_type = layer_type

    @abstractmethod
    def create_node(self, *args, **kwargs) -> Node:
        """Create a node in this layer."""
        pass

    def get_nodes(self) -> List[Node]:
        """Get all nodes in this layer."""
        return self.scene_graph.get_nodes(layer=self.layer_type)

    def query(self, **attributes) -> List[Node]:
        """Query nodes in this layer by attributes."""
        return self.scene_graph.query_by_attributes(
            layer=self.layer_type,
            **attributes
        )


class ObjectLayer(Layer):
    """
    L1: Object Layer - Individual object instances.

    This layer represents detected objects in the environment:
    - Physical objects (chairs, tables, cups, etc.)
    - Each object has: position, bounding box, class, confidence
    - Objects can have visual/semantic embeddings
    - Objects are related via spatial relations (on, near, supports)

    Example:
        >>> obj_layer = ObjectLayer(scene_graph)
        >>> cup = obj_layer.create_object(
        ...     class_name="cup",
        ...     position=[1.0, 2.0, 0.5],
        ...     bbox_min=[0.9, 1.9, 0.4],
        ...     bbox_max=[1.1, 2.1, 0.6],
        ...     color="red",
        ...     confidence=0.95
        ... )
        >>> table = obj_layer.create_object(...)
        >>> obj_layer.add_spatial_relation(cup.id, table.id, "on")
    """

    def __init__(self, scene_graph: SceneGraph):
        super().__init__(scene_graph, LayerType.L1)

    def create_object(
        self,
        class_name: str,
        position: Union[np.ndarray, List[float]],
        bbox_min: Optional[Union[np.ndarray, List[float]]] = None,
        bbox_max: Optional[Union[np.ndarray, List[float]]] = None,
        confidence: float = 1.0,
        embedding_key: Optional[str] = None,
        **attributes,
    ) -> Node:
        """
        Create an object node.

        Args:
            class_name: Object class (e.g., "cup", "chair", "table")
            position: Center position [x, y, z]
            bbox_min: Bounding box minimum corner [x, y, z]
            bbox_max: Bounding box maximum corner [x, y, z]
            confidence: Detection confidence [0, 1]
            embedding_key: Optional key to visual embedding in UMA
            **attributes: Additional attributes (color, material, etc.)

        Returns:
            Created object node
        """
        # Prepare bbox
        bbox = None
        if bbox_min is not None and bbox_max is not None:
            bbox = (
                np.array(bbox_min, dtype=np.float32),
                np.array(bbox_max, dtype=np.float32),
            )

        # Add class to attributes
        attributes["class"] = class_name

        # Create node
        node = self.scene_graph.add_node(
            layer=LayerType.L1,
            node_type=NodeType.OBJECT,
            position=position,
            attributes=attributes,
            bbox=bbox,
            confidence=confidence,
            embedding_key=embedding_key,
        )

        return node

    def add_spatial_relation(
        self,
        obj1_id: str,
        obj2_id: str,
        relation: str,
        confidence: float = 1.0,
        **properties,
    ) -> None:
        """
        Add a spatial relation between two objects.

        Args:
            obj1_id: First object ID
            obj2_id: Second object ID
            relation: Relation type ("on", "near", "supports", etc.)
            confidence: Relation confidence
            **properties: Additional properties (distance, etc.)

        Raises:
            ValueError: If relation type is invalid
        """
        # Map string to RelationType
        relation_map = {
            "on": RelationType.ON,
            "near": RelationType.NEAR,
            "supports": RelationType.SUPPORTS,
            "attached_to": RelationType.ATTACHED_TO,
            "part_of": RelationType.PART_OF,
        }

        if relation not in relation_map:
            raise ValueError(f"Invalid relation: {relation}")

        rel_type = relation_map[relation]

        # Add edge
        self.scene_graph.add_edge(
            src_id=obj1_id,
            dst_id=obj2_id,
            relation=rel_type,
            properties=properties,
            confidence=confidence,
        )

    def get_objects_by_class(self, class_name: str) -> List[Node]:
        """Get all objects of a specific class."""
        return self.query(**{"class": class_name})

    def get_objects_near(
        self,
        position: Union[np.ndarray, List[float]],
        radius: float,
    ) -> List[Node]:
        """Get all objects near a position."""
        return self.scene_graph.query_spatial(
            center=position,
            radius=radius,
            layer=LayerType.L1,
        )

    def compute_bbox_overlap(
        self,
        obj1: Node,
        obj2: Node,
    ) -> float:
        """
        Compute bounding box overlap (IoU) between two objects.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            IoU value [0, 1], or 0 if either bbox is None
        """
        if obj1.bbox is None or obj2.bbox is None:
            return 0.0

        # Unpack bboxes
        min1, max1 = obj1.bbox
        min2, max2 = obj2.bbox

        # Compute intersection
        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)

        # Check if intersection exists
        if np.any(inter_min >= inter_max):
            return 0.0

        # Compute volumes
        inter_volume = np.prod(inter_max - inter_min)
        vol1 = np.prod(max1 - min1)
        vol2 = np.prod(max2 - min2)

        # IoU
        union_volume = vol1 + vol2 - inter_volume
        iou = inter_volume / union_volume if union_volume > 0 else 0.0

        return float(iou)


class RoomLayer(Layer):
    """
    L2: Room Layer - Semantic places and regions.

    This layer represents semantic regions in the environment:
    - Rooms (kitchen, bedroom, bathroom, etc.)
    - Outdoor regions (garden, parking, etc.)
    - Each room has: centroid position, bounding region, semantic label
    - Rooms contain objects (L1) via CONTAINS relation
    - Rooms are connected via CONNECTED_TO relation (doors, hallways)

    Example:
        >>> room_layer = RoomLayer(scene_graph)
        >>> kitchen = room_layer.create_room(
        ...     name="kitchen",
        ...     center=[5.0, 3.0, 0.0],
        ...     bounds_min=[0.0, 0.0, 0.0],
        ...     bounds_max=[10.0, 6.0, 3.0]
        ... )
        >>> living_room = room_layer.create_room(...)
        >>> room_layer.connect_rooms(kitchen.id, living_room.id)
    """

    def __init__(self, scene_graph: SceneGraph):
        super().__init__(scene_graph, LayerType.L2)

    def create_room(
        self,
        name: str,
        center: Union[np.ndarray, List[float]],
        bounds_min: Optional[Union[np.ndarray, List[float]]] = None,
        bounds_max: Optional[Union[np.ndarray, List[float]]] = None,
        room_type: str = "indoor",
        **attributes,
    ) -> Node:
        """
        Create a room/place node.

        Args:
            name: Room name (e.g., "kitchen", "bedroom")
            center: Room center position [x, y, z]
            bounds_min: Bounding region minimum [x, y, z]
            bounds_max: Bounding region maximum [x, y, z]
            room_type: Type of room ("indoor", "outdoor", etc.)
            **attributes: Additional attributes

        Returns:
            Created room node
        """
        # Prepare bbox
        bbox = None
        if bounds_min is not None and bounds_max is not None:
            bbox = (
                np.array(bounds_min, dtype=np.float32),
                np.array(bounds_max, dtype=np.float32),
            )

        # Add name and type to attributes
        attributes["name"] = name
        attributes["room_type"] = room_type

        # Create node
        node = self.scene_graph.add_node(
            layer=LayerType.L2,
            node_type=NodeType.ROOM,
            position=center,
            attributes=attributes,
            bbox=bbox,
        )

        return node

    def add_object_to_room(
        self,
        object_id: str,
        room_id: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Add containment relation: object is IN room.

        Args:
            object_id: Object node ID (from L1)
            room_id: Room node ID (from L2)
            confidence: Relation confidence
        """
        self.scene_graph.add_edge(
            src_id=object_id,
            dst_id=room_id,
            relation=RelationType.IN,
            confidence=confidence,
        )

    def connect_rooms(
        self,
        room1_id: str,
        room2_id: str,
        connection_type: str = "door",
        confidence: float = 1.0,
    ) -> None:
        """
        Connect two rooms (e.g., via door or hallway).

        Args:
            room1_id: First room ID
            room2_id: Second room ID
            connection_type: Type of connection ("door", "hallway", etc.)
            confidence: Connection confidence
        """
        self.scene_graph.add_edge(
            src_id=room1_id,
            dst_id=room2_id,
            relation=RelationType.CONNECTED_TO,
            properties={"connection_type": connection_type},
            confidence=confidence,
        )

    def get_room_by_name(self, name: str) -> Optional[Node]:
        """Get room by name."""
        results = self.query(name=name)
        return results[0] if results else None

    def get_objects_in_room(self, room_id: str) -> List[Node]:
        """
        Get all objects contained in a room.

        Args:
            room_id: Room node ID

        Returns:
            List of object nodes in this room
        """
        # Get all nodes with incoming IN edge from this room
        neighbors = self.scene_graph.get_neighbors(
            room_id,
            relation=RelationType.IN,
            direction="in",
        )
        return [n for n in neighbors if n.node_type == NodeType.OBJECT]

    def get_connected_rooms(self, room_id: str) -> List[Node]:
        """
        Get all rooms connected to this room.

        Args:
            room_id: Room node ID

        Returns:
            List of connected room nodes
        """
        return self.scene_graph.get_neighbors(
            room_id,
            relation=RelationType.CONNECTED_TO,
            direction="both",
        )


class BuildingLayer(Layer):
    """
    L3: Building Layer - Large-scale structures (post-MVP).

    This layer represents building-level structures:
    - Entire buildings
    - Multi-story structures
    - Floor levels

    Note:
        This is a placeholder for post-MVP implementation.
        For MVP, we focus on L1 (objects) and L2 (rooms).
    """

    def __init__(self, scene_graph: SceneGraph):
        super().__init__(scene_graph, LayerType.L3)

    def create_node(self, *args, **kwargs) -> Node:
        """Create building node (placeholder)."""
        raise NotImplementedError("BuildingLayer is post-MVP")


# Helper functions for common operations

def infer_room_from_position(
    scene_graph: SceneGraph,
    position: Union[np.ndarray, List[float]],
) -> Optional[Node]:
    """
    Infer which room contains a given position.

    Args:
        scene_graph: Scene graph
        position: 3D position [x, y, z]

    Returns:
        Room node if found, None otherwise
    """
    position = np.array(position, dtype=np.float32)

    # Get all rooms
    rooms = scene_graph.get_nodes(layer=LayerType.L2)

    for room in rooms:
        if room.bbox is None:
            continue

        min_xyz, max_xyz = room.bbox
        if np.all(position >= min_xyz) and np.all(position <= max_xyz):
            return room

    return None


def compute_spatial_relations(
    scene_graph: SceneGraph,
    threshold_on: float = 0.05,  # Vertical distance threshold for "on"
    threshold_near: float = 1.0,  # Distance threshold for "near"
) -> int:
    """
    Automatically compute and add spatial relations between objects.

    This function analyzes object positions and bounding boxes to infer
    spatial relations like "on", "near", etc.

    Args:
        scene_graph: Scene graph
        threshold_on: Maximum vertical distance for "on" relation
        threshold_near: Maximum distance for "near" relation

    Returns:
        Number of relations added
    """
    objects = scene_graph.get_nodes(layer=LayerType.L1)
    relations_added = 0

    for i, obj1 in enumerate(objects):
        for obj2 in objects[i + 1:]:
            # Skip if either has no bbox
            if obj1.bbox is None or obj2.bbox is None:
                continue

            pos1 = obj1.position
            pos2 = obj2.position
            distance = np.linalg.norm(pos1 - pos2)

            # Check "near" relation
            if distance <= threshold_near:
                scene_graph.add_edge(
                    obj1.id,
                    obj2.id,
                    RelationType.NEAR,
                    properties={"distance": float(distance)},
                )
                relations_added += 1

            # Check "on" relation (obj1 on top of obj2)
            min1, max1 = obj1.bbox
            min2, max2 = obj2.bbox

            # obj1 bottom close to obj2 top
            if abs(min1[2] - max2[2]) < threshold_on:
                # Check horizontal overlap
                x_overlap = min(max1[0], max2[0]) - max(min1[0], min2[0])
                y_overlap = min(max1[1], max2[1]) - max(min1[1], min2[1])

                if x_overlap > 0 and y_overlap > 0:
                    scene_graph.add_edge(
                        obj1.id,
                        obj2.id,
                        RelationType.ON,
                        properties={"vertical_distance": float(abs(min1[2] - max2[2]))},
                    )
                    relations_added += 1

    return relations_added
