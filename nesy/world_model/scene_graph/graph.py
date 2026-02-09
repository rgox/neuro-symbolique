"""
Core Scene Graph Implementation.

This module implements the multi-layer 3D scene graph that serves as the central
world representation for the neuro-symbolic platform. The scene graph bridges
geometric/spatial information (neural) with semantic/relational information (symbolic).

Architecture:
    - Multi-layer design (L1-L5, MVP implements L1-L3)
    - Nodes represent entities (voxels, objects, places)
    - Edges represent spatial/semantic relations
    - Integrates with UMA for neural embeddings
    - Supports both geometric and symbolic queries

References:
    - 3D Scene Graph paper (Armeni et al.)
    - Hydra (Rosinol et al.)
    - SceneGraphFusion (Rosinol et al.)
"""

from typing import Any, Dict, Optional, Set, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import time
import uuid

from nesy.core.memory import UMA, DeviceType, DataType, MemoryBuffer
from nesy.core.telemetry import TelemetryLogger, EventType


class LayerType(Enum):
    """Scene graph layer types (hierarchical representation)."""
    L0 = "mesh"          # Dense 3D mesh/voxel grid (optional, dense geometry)
    L1 = "objects"       # Object instances (detected entities)
    L2 = "rooms"         # Rooms/places (semantic regions)
    L3 = "buildings"     # Buildings/structures (large-scale)
    L4 = "global"        # Global map (outdoor environments)


class NodeType(Enum):
    """Types of nodes in the scene graph."""
    VOXEL = "voxel"           # L0: Dense voxel
    OBJECT = "object"         # L1: Object instance (chair, cup, etc.)
    ROOM = "room"             # L2: Room/place (kitchen, bedroom, etc.)
    BUILDING = "building"     # L3: Building
    OUTDOOR = "outdoor"       # L4: Outdoor region


class RelationType(Enum):
    """Types of relations between nodes."""
    # Spatial relations
    ON = "on"                 # Object A is on object B
    IN = "in"                 # Object A is inside place B
    NEAR = "near"             # Object A is near object B
    CONTAINS = "contains"     # Place A contains object B
    CONNECTED_TO = "connected_to"  # Room A connected to room B
    PART_OF = "part_of"       # Object A is part of object B

    # Semantic relations
    SAME_AS = "same_as"       # Same entity across time
    SUPPORTS = "supports"     # Object A supports object B (physical)
    ATTACHED_TO = "attached_to"  # Object A attached to B


@dataclass
class Node:
    """
    Scene graph node representing an entity in the world.

    Attributes:
        id: Unique identifier (UUID)
        layer: Layer this node belongs to
        node_type: Type of node (OBJECT, ROOM, etc.)
        position: 3D position (x, y, z) in world coordinates
        attributes: Semantic attributes (color, class, etc.)
        embedding_key: Optional key to neural embedding in UMA
        vsa_embedding_key: Optional key to VSA hypervector in UMA
        bbox: Optional bounding box (min_xyz, max_xyz)
        confidence: Detection/existence confidence [0, 1]
        timestamp: Creation/update timestamp
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    layer: LayerType = LayerType.L1
    node_type: NodeType = NodeType.OBJECT
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding_key: Optional[str] = None
    vsa_embedding_key: Optional[str] = None  # NEW: VSA hypervector UMA key
    bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "id": self.id,
            "layer": self.layer.value,
            "node_type": self.node_type.value,
            "position": self.position.tolist(),
            "attributes": self.attributes,
            "embedding_key": self.embedding_key,
            "vsa_embedding_key": self.vsa_embedding_key,
            "bbox": [self.bbox[0].tolist(), self.bbox[1].tolist()] if self.bbox else None,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class Edge:
    """
    Scene graph edge representing a relation between nodes.

    Attributes:
        src: Source node ID
        dst: Destination node ID
        relation: Type of relation
        properties: Additional properties (distance, confidence, etc.)
        confidence: Relation confidence [0, 1]
        timestamp: Creation/update timestamp
    """
    src: str
    dst: str
    relation: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def __hash__(self):
        return hash((self.src, self.dst, self.relation.value))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.src == other.src and
                self.dst == other.dst and
                self.relation == other.relation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation.value,
            "properties": self.properties,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


class SceneGraph:
    """
    Multi-layer 3D scene graph for world representation.

    The scene graph maintains a hierarchical representation of the world:
    - L0 (optional): Dense mesh/voxel grid
    - L1: Object instances (detected entities)
    - L2: Rooms/places (semantic regions)
    - L3: Buildings (large structures)
    - L4: Global map (outdoor)

    For MVP, we implement L1-L2 (objects and rooms).

    Features:
    - Multi-layer graph structure
    - Spatial and semantic queries
    - Integration with UMA for embeddings
    - Incremental updates
    - Temporal tracking

    Example:
        >>> sg = SceneGraph(uma=uma, logger=logger)
        >>> node = sg.add_node(
        ...     layer=LayerType.L1,
        ...     node_type=NodeType.OBJECT,
        ...     position=[1.0, 2.0, 0.5],
        ...     attributes={"class": "cup", "color": "red"}
        ... )
        >>> sg.add_edge(node.id, room_id, RelationType.IN)
        >>> results = sg.query_spatial(center=[1.0, 2.0, 0.5], radius=1.0)
    """

    def __init__(
        self,
        uma: Optional[UMA] = None,
        logger: Optional[TelemetryLogger] = None,
    ):
        """
        Initialize scene graph.

        Args:
            uma: Unified Memory Architecture for embeddings
            logger: Optional telemetry logger
        """
        self.uma = uma
        self.logger = logger

        # Multi-layer storage
        self.nodes: Dict[str, Node] = {}  # node_id -> Node
        self.edges: Dict[str, List[Edge]] = defaultdict(list)  # src_id -> [Edge]
        self.reverse_edges: Dict[str, List[Edge]] = defaultdict(list)  # dst_id -> [Edge]

        # Layer indices for efficient layer-specific queries
        self.layer_nodes: Dict[LayerType, Set[str]] = defaultdict(set)

        # Spatial index (will be implemented in spatial_index.py)
        self.spatial_index = None  # Placeholder for octree/R-tree

        # Statistics
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "nodes_by_layer": defaultdict(int),
            "edges_by_relation": defaultdict(int),
        }

        if logger:
            logger.info("SceneGraph initialized", event_type=EventType.GRAPH)

    def add_node(
        self,
        layer: LayerType,
        node_type: NodeType,
        position: Union[np.ndarray, List[float]],
        attributes: Optional[Dict[str, Any]] = None,
        embedding_key: Optional[str] = None,
        bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        confidence: float = 1.0,
        node_id: Optional[str] = None,
    ) -> Node:
        """
        Add a node to the scene graph.

        Args:
            layer: Layer this node belongs to
            node_type: Type of node
            position: 3D position [x, y, z]
            attributes: Semantic attributes
            embedding_key: Optional key to embedding in UMA
            bbox: Optional bounding box (min, max)
            confidence: Node confidence
            node_id: Optional custom ID (generates UUID if None)

        Returns:
            Created Node object
        """
        # Create node
        node = Node(
            id=node_id or str(uuid.uuid4()),
            layer=layer,
            node_type=node_type,
            position=np.array(position, dtype=np.float32),
            attributes=attributes or {},
            embedding_key=embedding_key,
            bbox=bbox,
            confidence=confidence,
        )

        # Store node
        self.nodes[node.id] = node
        self.layer_nodes[layer].add(node.id)

        # Update statistics
        self.stats["total_nodes"] += 1
        self.stats["nodes_by_layer"][layer.value] += 1

        # Log
        if self.logger:
            self.logger.debug(
                f"Added node {node.id[:8]} to layer {layer.value}",
                event_type=EventType.GRAPH,
                data={
                    "node_id": node.id,
                    "layer": layer.value,
                    "type": node_type.value,
                    "position": position,
                },
            )

        return node

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        relation: RelationType,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> Edge:
        """
        Add an edge between two nodes.

        Args:
            src_id: Source node ID
            dst_id: Destination node ID
            relation: Type of relation
            properties: Additional properties
            confidence: Edge confidence

        Returns:
            Created Edge object

        Raises:
            ValueError: If source or destination node doesn't exist
        """
        # Validate nodes exist
        if src_id not in self.nodes:
            raise ValueError(f"Source node {src_id} not found")
        if dst_id not in self.nodes:
            raise ValueError(f"Destination node {dst_id} not found")

        # Create edge
        edge = Edge(
            src=src_id,
            dst=dst_id,
            relation=relation,
            properties=properties or {},
            confidence=confidence,
        )

        # Store edge (bidirectional indices)
        self.edges[src_id].append(edge)
        self.reverse_edges[dst_id].append(edge)

        # Update statistics
        self.stats["total_edges"] += 1
        self.stats["edges_by_relation"][relation.value] += 1

        # Log
        if self.logger:
            self.logger.debug(
                f"Added edge {src_id[:8]} --[{relation.value}]--> {dst_id[:8]}",
                event_type=EventType.GRAPH,
            )

        return edge

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_nodes(self, layer: Optional[LayerType] = None) -> List[Node]:
        """
        Get all nodes, optionally filtered by layer.

        Args:
            layer: Optional layer filter

        Returns:
            List of nodes
        """
        if layer is None:
            return list(self.nodes.values())

        node_ids = self.layer_nodes.get(layer, set())
        return [self.nodes[nid] for nid in node_ids]

    def get_neighbors(
        self,
        node_id: str,
        relation: Optional[RelationType] = None,
        direction: str = "out",
    ) -> List[Node]:
        """
        Get neighbors of a node.

        Args:
            node_id: Node ID
            relation: Optional relation type filter
            direction: "out" (outgoing), "in" (incoming), "both"

        Returns:
            List of neighbor nodes
        """
        neighbors = []

        # Outgoing edges
        if direction in ["out", "both"]:
            for edge in self.edges.get(node_id, []):
                if relation is None or edge.relation == relation:
                    neighbor = self.nodes.get(edge.dst)
                    if neighbor:
                        neighbors.append(neighbor)

        # Incoming edges
        if direction in ["in", "both"]:
            for edge in self.reverse_edges.get(node_id, []):
                if relation is None or edge.relation == relation:
                    neighbor = self.nodes.get(edge.src)
                    if neighbor:
                        neighbors.append(neighbor)

        return neighbors

    def query_by_attributes(
        self,
        layer: Optional[LayerType] = None,
        **attributes
    ) -> List[Node]:
        """
        Query nodes by attributes.

        Args:
            layer: Optional layer filter
            **attributes: Attribute filters (e.g., class="cup", color="red")

        Returns:
            List of matching nodes

        Example:
            >>> nodes = sg.query_by_attributes(layer=LayerType.L1, class_="cup")
        """
        candidates = self.get_nodes(layer)

        results = []
        for node in candidates:
            match = True
            for key, value in attributes.items():
                if node.attributes.get(key) != value:
                    match = False
                    break
            if match:
                results.append(node)

        return results

    def query_spatial(
        self,
        center: Union[np.ndarray, List[float]],
        radius: float,
        layer: Optional[LayerType] = None,
    ) -> List[Node]:
        """
        Query nodes within a spatial radius.

        Args:
            center: Center position [x, y, z]
            radius: Search radius
            layer: Optional layer filter

        Returns:
            List of nodes within radius

        Note:
            This is a naive O(n) implementation. Will be optimized with
            spatial index (octree) in spatial_index.py.
        """
        center = np.array(center, dtype=np.float32)
        candidates = self.get_nodes(layer)

        results = []
        for node in candidates:
            distance = np.linalg.norm(node.position - center)
            if distance <= radius:
                results.append(node)

        return results

    def query_bbox(
        self,
        min_xyz: Union[np.ndarray, List[float]],
        max_xyz: Union[np.ndarray, List[float]],
        layer: Optional[LayerType] = None,
    ) -> List[Node]:
        """
        Query nodes within a bounding box.

        Args:
            min_xyz: Minimum corner [x, y, z]
            max_xyz: Maximum corner [x, y, z]
            layer: Optional layer filter

        Returns:
            List of nodes within bounding box
        """
        min_xyz = np.array(min_xyz, dtype=np.float32)
        max_xyz = np.array(max_xyz, dtype=np.float32)
        candidates = self.get_nodes(layer)

        results = []
        for node in candidates:
            pos = node.position
            if np.all(pos >= min_xyz) and np.all(pos <= max_xyz):
                results.append(node)

        return results

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its edges.

        Args:
            node_id: Node ID to remove

        Returns:
            True if removed, False if not found
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        # Remove edges (check if exists first)
        if node_id in self.edges:
            del self.edges[node_id]
        if node_id in self.reverse_edges:
            del self.reverse_edges[node_id]

        # Remove node from layer index
        self.layer_nodes[node.layer].discard(node_id)

        # Remove node
        del self.nodes[node_id]

        # Update stats
        self.stats["total_nodes"] -= 1
        self.stats["nodes_by_layer"][node.layer.value] -= 1

        return True

    def clear(self, layer: Optional[LayerType] = None) -> None:
        """
        Clear all nodes and edges, optionally for a specific layer.

        Args:
            layer: Optional layer to clear (clears all if None)
        """
        if layer is None:
            # Clear everything
            self.nodes.clear()
            self.edges.clear()
            self.reverse_edges.clear()
            self.layer_nodes.clear()
            self.stats = {
                "total_nodes": 0,
                "total_edges": 0,
                "nodes_by_layer": defaultdict(int),
                "edges_by_relation": defaultdict(int),
            }
        else:
            # Clear specific layer
            node_ids = list(self.layer_nodes[layer])
            for node_id in node_ids:
                self.remove_node(node_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get scene graph statistics."""
        return {
            "total_nodes": self.stats["total_nodes"],
            "total_edges": self.stats["total_edges"],
            "nodes_by_layer": dict(self.stats["nodes_by_layer"]),
            "edges_by_relation": dict(self.stats["edges_by_relation"]),
            "layers": [layer.value for layer in self.layer_nodes.keys()],
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Export scene graph to dictionary.

        Returns:
            Dictionary representation with nodes and edges
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [
                edge.to_dict()
                for edges in self.edges.values()
                for edge in edges
            ],
            "statistics": self.get_statistics(),
        }

    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)

    def __repr__(self) -> str:
        return (
            f"SceneGraph(nodes={len(self.nodes)}, "
            f"edges={self.stats['total_edges']}, "
            f"layers={list(self.layer_nodes.keys())})"
        )
