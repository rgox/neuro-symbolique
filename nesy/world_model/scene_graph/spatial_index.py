"""
Spatial Indexing for Scene Graph.

This module provides spatial data structures for efficient spatial queries:
- Octree: 3D spatial partitioning for point/bbox queries
- R-tree: Bounding box indexing (post-MVP)

Spatial indexing accelerates queries from O(n) to O(log n):
- Find all nodes within radius
- Find all nodes in bounding box
- Nearest neighbor search
- Collision detection

For MVP, we implement a simple octree. Post-MVP will add R-tree and other
advanced structures.
"""

from typing import List, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass

from nesy.world_model.scene_graph.graph import Node, LayerType


@dataclass
class OctreeNode:
    """
    Node in the octree spatial index.

    An octree recursively subdivides 3D space into 8 octants.
    Each node either:
    - Is a leaf containing objects
    - Has 8 children (one per octant)

    Attributes:
        center: Center point of this octree node
        half_size: Half the side length of this node's bounding box
        depth: Depth in the tree (0 = root)
        is_leaf: True if this is a leaf node
        objects: Object IDs stored in this node (if leaf)
        children: 8 child octree nodes (if not leaf)
    """
    center: np.ndarray
    half_size: float
    depth: int
    is_leaf: bool = True
    objects: Set[str] = None  # Object IDs
    children: List[Optional['OctreeNode']] = None  # 8 children

    def __post_init__(self):
        if self.objects is None:
            self.objects = set()
        if self.children is None:
            self.children = [None] * 8


class Octree:
    """
    Octree spatial index for 3D scene graph.

    The octree recursively subdivides 3D space to enable fast spatial queries.
    Instead of checking all N nodes (O(n)), we only check nodes in relevant
    octants (O(log n) average case).

    Features:
    - Insert nodes by position
    - Query within radius (sphere)
    - Query within bounding box
    - Automatic subdivision when capacity exceeded
    - Balanced tree for uniform distributions

    Example:
        >>> octree = Octree(
        ...     center=[0, 0, 0],
        ...     size=10.0,
        ...     max_depth=5,
        ...     capacity=8
        ... )
        >>> octree.insert(node_id, position)
        >>> nearby = octree.query_radius(center=[1, 1, 1], radius=2.0)

    Performance:
        - Insert: O(log n)
        - Query radius: O(log n) + O(k) where k = results
        - Query bbox: O(log n) + O(k)
    """

    def __init__(
        self,
        center: np.ndarray,
        size: float,
        max_depth: int = 8,
        capacity: int = 8,
    ):
        """
        Initialize octree.

        Args:
            center: Center of the root bounding box [x, y, z]
            size: Full size of the root bounding box
            max_depth: Maximum depth of tree (prevents infinite subdivision)
            capacity: Maximum objects per leaf before subdivision
        """
        self.center = np.array(center, dtype=np.float32)
        self.size = size
        self.max_depth = max_depth
        self.capacity = capacity

        # Create root node
        self.root = OctreeNode(
            center=self.center,
            half_size=size / 2.0,
            depth=0,
        )

        # Store node positions for queries
        self.node_positions: dict[str, np.ndarray] = {}

    def insert(self, node_id: str, position: np.ndarray) -> bool:
        """
        Insert a node into the octree.

        Args:
            node_id: Unique node identifier
            position: Node position [x, y, z]

        Returns:
            True if inserted, False if outside bounds
        """
        position = np.array(position, dtype=np.float32)

        # Check if within bounds
        if not self._in_bounds(self.root, position):
            return False

        # Store position
        self.node_positions[node_id] = position

        # Insert into tree
        self._insert_recursive(self.root, node_id, position)
        return True

    def _insert_recursive(
        self,
        node: OctreeNode,
        object_id: str,
        position: np.ndarray,
    ) -> None:
        """Recursively insert object into octree."""
        if node.is_leaf:
            # Add to this leaf
            node.objects.add(object_id)

            # Subdivide if over capacity and not at max depth
            if len(node.objects) > self.capacity and node.depth < self.max_depth:
                self._subdivide(node)
        else:
            # Find which octant and recurse
            octant = self._get_octant(node, position)
            if node.children[octant] is None:
                # Should not happen if properly subdivided
                return

            self._insert_recursive(node.children[octant], object_id, position)

    def _subdivide(self, node: OctreeNode) -> None:
        """
        Subdivide a leaf node into 8 children.

        Octant numbering:
            0: (-x, -y, -z)
            1: (+x, -y, -z)
            2: (-x, +y, -z)
            3: (+x, +y, -z)
            4: (-x, -y, +z)
            5: (+x, -y, +z)
            6: (-x, +y, +z)
            7: (+x, +y, +z)
        """
        node.is_leaf = False
        new_half_size = node.half_size / 2.0
        new_depth = node.depth + 1

        # Create 8 children
        offsets = [
            [-1, -1, -1],  # 0
            [+1, -1, -1],  # 1
            [-1, +1, -1],  # 2
            [+1, +1, -1],  # 3
            [-1, -1, +1],  # 4
            [+1, -1, +1],  # 5
            [-1, +1, +1],  # 6
            [+1, +1, +1],  # 7
        ]

        for i, offset in enumerate(offsets):
            child_center = node.center + np.array(offset) * new_half_size
            node.children[i] = OctreeNode(
                center=child_center,
                half_size=new_half_size,
                depth=new_depth,
            )

        # Redistribute objects to children
        objects = list(node.objects)
        node.objects.clear()

        for obj_id in objects:
            position = self.node_positions[obj_id]
            octant = self._get_octant(node, position)
            node.children[octant].objects.add(obj_id)

    def _get_octant(self, node: OctreeNode, position: np.ndarray) -> int:
        """
        Get which octant a position falls into.

        Returns:
            Octant index (0-7)
        """
        octant = 0
        if position[0] >= node.center[0]:
            octant |= 1
        if position[1] >= node.center[1]:
            octant |= 2
        if position[2] >= node.center[2]:
            octant |= 4
        return octant

    def _in_bounds(self, node: OctreeNode, position: np.ndarray) -> bool:
        """Check if position is within node's bounding box."""
        return np.all(np.abs(position - node.center) <= node.half_size)

    def query_radius(
        self,
        center: np.ndarray,
        radius: float,
    ) -> List[str]:
        """
        Query all objects within a radius.

        Args:
            center: Query center [x, y, z]
            radius: Search radius

        Returns:
            List of object IDs within radius
        """
        center = np.array(center, dtype=np.float32)
        results = set()
        self._query_radius_recursive(self.root, center, radius, results)
        return list(results)

    def _query_radius_recursive(
        self,
        node: OctreeNode,
        center: np.ndarray,
        radius: float,
        results: Set[str],
    ) -> None:
        """Recursively query radius."""
        # Check if sphere intersects this node's bbox
        if not self._sphere_intersects_bbox(center, radius, node.center, node.half_size):
            return

        if node.is_leaf:
            # Check each object in this leaf
            for obj_id in node.objects:
                obj_pos = self.node_positions[obj_id]
                distance = np.linalg.norm(obj_pos - center)
                if distance <= radius:
                    results.add(obj_id)
        else:
            # Recurse into children
            for child in node.children:
                if child is not None:
                    self._query_radius_recursive(child, center, radius, results)

    def _sphere_intersects_bbox(
        self,
        sphere_center: np.ndarray,
        sphere_radius: float,
        bbox_center: np.ndarray,
        bbox_half_size: float,
    ) -> bool:
        """Check if sphere intersects axis-aligned bounding box."""
        # Find closest point on bbox to sphere center
        closest = np.clip(
            sphere_center,
            bbox_center - bbox_half_size,
            bbox_center + bbox_half_size,
        )

        # Check if closest point is within radius
        distance = np.linalg.norm(closest - sphere_center)
        return distance <= sphere_radius

    def query_bbox(
        self,
        min_xyz: np.ndarray,
        max_xyz: np.ndarray,
    ) -> List[str]:
        """
        Query all objects within a bounding box.

        Args:
            min_xyz: Minimum corner [x, y, z]
            max_xyz: Maximum corner [x, y, z]

        Returns:
            List of object IDs within bounding box
        """
        min_xyz = np.array(min_xyz, dtype=np.float32)
        max_xyz = np.array(max_xyz, dtype=np.float32)
        results = set()
        self._query_bbox_recursive(self.root, min_xyz, max_xyz, results)
        return list(results)

    def _query_bbox_recursive(
        self,
        node: OctreeNode,
        min_xyz: np.ndarray,
        max_xyz: np.ndarray,
        results: Set[str],
    ) -> None:
        """Recursively query bbox."""
        # Check if query bbox intersects this node's bbox
        node_min = node.center - node.half_size
        node_max = node.center + node.half_size

        if not self._bbox_intersects_bbox(min_xyz, max_xyz, node_min, node_max):
            return

        if node.is_leaf:
            # Check each object in this leaf
            for obj_id in node.objects:
                obj_pos = self.node_positions[obj_id]
                if np.all(obj_pos >= min_xyz) and np.all(obj_pos <= max_xyz):
                    results.add(obj_id)
        else:
            # Recurse into children
            for child in node.children:
                if child is not None:
                    self._query_bbox_recursive(child, min_xyz, max_xyz, results)

    def _bbox_intersects_bbox(
        self,
        min1: np.ndarray,
        max1: np.ndarray,
        min2: np.ndarray,
        max2: np.ndarray,
    ) -> bool:
        """Check if two axis-aligned bounding boxes intersect."""
        return (
            np.all(min1 <= max2) and
            np.all(max1 >= min2)
        )

    def remove(self, node_id: str) -> bool:
        """
        Remove a node from the octree.

        Args:
            node_id: Node ID to remove

        Returns:
            True if removed, False if not found

        Note:
            This does not rebalance the tree. For MVP, we accept
            this limitation. Post-MVP can add tree rebalancing.
        """
        if node_id not in self.node_positions:
            return False

        position = self.node_positions[node_id]
        removed = self._remove_recursive(self.root, node_id, position)

        if removed:
            del self.node_positions[node_id]

        return removed

    def _remove_recursive(
        self,
        node: OctreeNode,
        object_id: str,
        position: np.ndarray,
    ) -> bool:
        """Recursively remove object from octree."""
        if not self._in_bounds(node, position):
            return False

        if node.is_leaf:
            if object_id in node.objects:
                node.objects.remove(object_id)
                return True
            return False
        else:
            octant = self._get_octant(node, position)
            if node.children[octant] is None:
                return False
            return self._remove_recursive(node.children[octant], object_id, position)

    def clear(self) -> None:
        """Clear all objects from the octree."""
        self.root = OctreeNode(
            center=self.center,
            half_size=self.size / 2.0,
            depth=0,
        )
        self.node_positions.clear()

    def get_statistics(self) -> dict:
        """Get octree statistics."""
        stats = {
            "total_objects": len(self.node_positions),
            "max_depth": 0,
            "total_nodes": 0,
            "leaf_nodes": 0,
        }

        self._collect_statistics(self.root, stats)
        return stats

    def _collect_statistics(self, node: OctreeNode, stats: dict) -> None:
        """Recursively collect statistics."""
        stats["total_nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], node.depth)

        if node.is_leaf:
            stats["leaf_nodes"] += 1
        else:
            for child in node.children:
                if child is not None:
                    self._collect_statistics(child, stats)


def create_spatial_index_for_scene_graph(
    scene_graph,
    layer: Optional[LayerType] = None,
    size: float = 100.0,
    center: Optional[np.ndarray] = None,
) -> Octree:
    """
    Create a spatial index (octree) for a scene graph layer.

    Args:
        scene_graph: SceneGraph instance
        layer: Optional layer to index (indexes all if None)
        size: Size of the octree root bounding box
        center: Center of the octree (auto-computed if None)

    Returns:
        Octree spatial index with all nodes inserted
    """
    nodes = scene_graph.get_nodes(layer=layer)

    # Compute center if not provided
    if center is None:
        if nodes:
            positions = np.array([n.position for n in nodes])
            center = np.mean(positions, axis=0)
        else:
            center = np.zeros(3)

    # Create octree
    octree = Octree(center=center, size=size)

    # Insert all nodes
    for node in nodes:
        octree.insert(node.id, node.position)

    return octree
