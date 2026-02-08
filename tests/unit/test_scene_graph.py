"""
Unit tests for Scene Graph module.

Tests cover:
- SceneGraph core functionality
- Node and Edge creation
- Layer operations (ObjectLayer, RoomLayer)
- Spatial queries (radius, bbox)
- Octree spatial indexing
- Relations and queries
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nesy.world_model import (
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


class TestSceneGraph:
    """Test core SceneGraph functionality."""

    def test_create_scene_graph(self):
        """Test creating an empty scene graph."""
        sg = SceneGraph()
        assert len(sg) == 0
        assert sg.stats["total_nodes"] == 0
        assert sg.stats["total_edges"] == 0

    def test_add_node(self):
        """Test adding nodes to scene graph."""
        sg = SceneGraph()

        node = sg.add_node(
            layer=LayerType.L1,
            node_type=NodeType.OBJECT,
            position=[1.0, 2.0, 3.0],
            attributes={"class": "cup"},
        )

        assert node.id in sg.nodes
        assert len(sg) == 1
        assert np.array_equal(node.position, [1.0, 2.0, 3.0])
        assert node.attributes["class"] == "cup"

    def test_add_edge(self):
        """Test adding edges between nodes."""
        sg = SceneGraph()

        node1 = sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0])
        node2 = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 1])

        edge = sg.add_edge(node1.id, node2.id, RelationType.NEAR)

        assert len(sg.edges[node1.id]) == 1
        assert edge.src == node1.id
        assert edge.dst == node2.id
        assert edge.relation == RelationType.NEAR

    def test_get_nodes_by_layer(self):
        """Test filtering nodes by layer."""
        sg = SceneGraph()

        sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0])
        sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 1])
        sg.add_node(LayerType.L2, NodeType.ROOM, [5, 5, 0])

        l1_nodes = sg.get_nodes(layer=LayerType.L1)
        l2_nodes = sg.get_nodes(layer=LayerType.L2)

        assert len(l1_nodes) == 2
        assert len(l2_nodes) == 1

    def test_query_by_attributes(self):
        """Test querying nodes by attributes."""
        sg = SceneGraph()

        sg.add_node(
            LayerType.L1, NodeType.OBJECT, [0, 0, 0],
            attributes={"class": "cup", "color": "red"}
        )
        sg.add_node(
            LayerType.L1, NodeType.OBJECT, [1, 1, 1],
            attributes={"class": "cup", "color": "blue"}
        )
        sg.add_node(
            LayerType.L1, NodeType.OBJECT, [2, 2, 2],
            attributes={"class": "table"}
        )

        cups = sg.query_by_attributes(layer=LayerType.L1, **{"class": "cup"})
        red_cups = sg.query_by_attributes(layer=LayerType.L1, color="red")

        assert len(cups) == 2
        assert len(red_cups) == 1

    def test_query_spatial_radius(self):
        """Test spatial queries within radius."""
        sg = SceneGraph()

        sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0])
        sg.add_node(LayerType.L1, NodeType.OBJECT, [0.5, 0.5, 0.5])
        sg.add_node(LayerType.L1, NodeType.OBJECT, [5, 5, 5])

        results = sg.query_spatial(center=[0, 0, 0], radius=1.0)

        assert len(results) == 2  # First two nodes within 1.0

    def test_query_spatial_bbox(self):
        """Test spatial queries within bounding box."""
        sg = SceneGraph()

        sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 1])
        sg.add_node(LayerType.L1, NodeType.OBJECT, [2, 2, 2])
        sg.add_node(LayerType.L1, NodeType.OBJECT, [5, 5, 5])

        results = sg.query_bbox(
            min_xyz=[0, 0, 0],
            max_xyz=[3, 3, 3],
        )

        assert len(results) == 2  # First two nodes in bbox

    def test_get_neighbors(self):
        """Test getting neighbors of a node."""
        sg = SceneGraph()

        node1 = sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0])
        node2 = sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 1])
        node3 = sg.add_node(LayerType.L1, NodeType.OBJECT, [2, 2, 2])

        sg.add_edge(node1.id, node2.id, RelationType.NEAR)
        sg.add_edge(node1.id, node3.id, RelationType.ON)

        neighbors = sg.get_neighbors(node1.id)
        near_neighbors = sg.get_neighbors(node1.id, relation=RelationType.NEAR)

        assert len(neighbors) == 2
        assert len(near_neighbors) == 1

    def test_remove_node(self):
        """Test removing a node."""
        sg = SceneGraph()

        node = sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0])
        assert len(sg) == 1

        sg.remove_node(node.id)
        assert len(sg) == 0

    def test_clear(self):
        """Test clearing scene graph."""
        sg = SceneGraph()

        sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0])
        sg.add_node(LayerType.L2, NodeType.ROOM, [5, 5, 0])

        sg.clear()
        assert len(sg) == 0

    def test_statistics(self):
        """Test getting scene graph statistics."""
        sg = SceneGraph()

        sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0])
        sg.add_node(LayerType.L1, NodeType.OBJECT, [1, 1, 1])
        node1_id = sg.get_nodes()[0].id
        node2_id = sg.get_nodes()[1].id
        sg.add_edge(node1_id, node2_id, RelationType.NEAR)

        stats = sg.get_statistics()

        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1

    def test_to_dict(self):
        """Test exporting scene graph to dictionary."""
        sg = SceneGraph()

        sg.add_node(LayerType.L1, NodeType.OBJECT, [0, 0, 0])
        export = sg.to_dict()

        assert "nodes" in export
        assert "edges" in export
        assert len(export["nodes"]) == 1


class TestObjectLayer:
    """Test ObjectLayer (L1) functionality."""

    def test_create_object(self):
        """Test creating an object."""
        sg = SceneGraph()
        obj_layer = ObjectLayer(sg)

        cup = obj_layer.create_object(
            class_name="cup",
            position=[1, 2, 3],
            bbox_min=[0.9, 1.9, 2.9],
            bbox_max=[1.1, 2.1, 3.1],
            color="red",
        )

        assert cup.layer == LayerType.L1
        assert cup.node_type == NodeType.OBJECT
        assert cup.attributes["class"] == "cup"
        assert cup.attributes["color"] == "red"
        assert cup.bbox is not None

    def test_get_objects_by_class(self):
        """Test getting objects by class."""
        sg = SceneGraph()
        obj_layer = ObjectLayer(sg)

        obj_layer.create_object("cup", [0, 0, 0])
        obj_layer.create_object("cup", [1, 1, 1])
        obj_layer.create_object("table", [2, 2, 2])

        cups = obj_layer.get_objects_by_class("cup")

        assert len(cups) == 2

    def test_add_spatial_relation(self):
        """Test adding spatial relation between objects."""
        sg = SceneGraph()
        obj_layer = ObjectLayer(sg)

        cup = obj_layer.create_object("cup", [0, 0, 1])
        table = obj_layer.create_object("table", [0, 0, 0])

        obj_layer.add_spatial_relation(cup.id, table.id, "on")

        neighbors = sg.get_neighbors(cup.id, RelationType.ON)
        assert len(neighbors) == 1
        assert neighbors[0].id == table.id

    def test_get_objects_near(self):
        """Test getting objects near a position."""
        sg = SceneGraph()
        obj_layer = ObjectLayer(sg)

        obj_layer.create_object("cup", [0, 0, 0])
        obj_layer.create_object("table", [0.5, 0.5, 0.5])
        obj_layer.create_object("chair", [5, 5, 5])

        nearby = obj_layer.get_objects_near([0, 0, 0], radius=1.0)

        assert len(nearby) == 2

    def test_compute_bbox_overlap(self):
        """Test computing bounding box overlap."""
        sg = SceneGraph()
        obj_layer = ObjectLayer(sg)

        obj1 = obj_layer.create_object(
            "cup",
            [0, 0, 0],
            bbox_min=[0, 0, 0],
            bbox_max=[1, 1, 1],
        )
        obj2 = obj_layer.create_object(
            "table",
            [0.5, 0.5, 0.5],
            bbox_min=[0, 0, 0],
            bbox_max=[1, 1, 1],
        )

        iou = obj_layer.compute_bbox_overlap(obj1, obj2)

        assert iou == 1.0  # Perfect overlap


class TestRoomLayer:
    """Test RoomLayer (L2) functionality."""

    def test_create_room(self):
        """Test creating a room."""
        sg = SceneGraph()
        room_layer = RoomLayer(sg)

        kitchen = room_layer.create_room(
            name="kitchen",
            center=[5, 5, 0],
            bounds_min=[0, 0, 0],
            bounds_max=[10, 10, 3],
        )

        assert kitchen.layer == LayerType.L2
        assert kitchen.node_type == NodeType.ROOM
        assert kitchen.attributes["name"] == "kitchen"

    def test_add_object_to_room(self):
        """Test adding object to room."""
        sg = SceneGraph()
        obj_layer = ObjectLayer(sg)
        room_layer = RoomLayer(sg)

        cup = obj_layer.create_object("cup", [1, 1, 1])
        kitchen = room_layer.create_room("kitchen", [5, 5, 0])

        room_layer.add_object_to_room(cup.id, kitchen.id)

        objects_in_room = room_layer.get_objects_in_room(kitchen.id)
        assert len(objects_in_room) == 1
        assert objects_in_room[0].id == cup.id

    def test_connect_rooms(self):
        """Test connecting two rooms."""
        sg = SceneGraph()
        room_layer = RoomLayer(sg)

        kitchen = room_layer.create_room("kitchen", [0, 0, 0])
        living = room_layer.create_room("living_room", [10, 0, 0])

        room_layer.connect_rooms(kitchen.id, living.id, connection_type="door")

        connected = room_layer.get_connected_rooms(kitchen.id)
        assert len(connected) == 1

    def test_get_room_by_name(self):
        """Test getting room by name."""
        sg = SceneGraph()
        room_layer = RoomLayer(sg)

        room_layer.create_room("kitchen", [0, 0, 0])
        room_layer.create_room("bedroom", [10, 0, 0])

        kitchen = room_layer.get_room_by_name("kitchen")
        assert kitchen is not None
        assert kitchen.attributes["name"] == "kitchen"


class TestOctree:
    """Test Octree spatial indexing."""

    def test_create_octree(self):
        """Test creating an octree."""
        octree = Octree(center=[0, 0, 0], size=10.0)
        assert octree is not None

    def test_insert_and_query(self):
        """Test inserting and querying octree."""
        octree = Octree(center=[0, 0, 0], size=10.0)

        octree.insert("obj1", [0, 0, 0])
        octree.insert("obj2", [0.5, 0.5, 0.5])
        octree.insert("obj3", [5, 5, 5])

        results = octree.query_radius(center=[0, 0, 0], radius=1.0)

        assert len(results) == 2
        assert "obj1" in results
        assert "obj2" in results

    def test_query_bbox(self):
        """Test bounding box query on octree."""
        octree = Octree(center=[0, 0, 0], size=10.0)

        octree.insert("obj1", [1, 1, 1])
        octree.insert("obj2", [2, 2, 2])
        octree.insert("obj3", [5, 5, 5])

        results = octree.query_bbox(
            min_xyz=[0, 0, 0],
            max_xyz=[3, 3, 3],
        )

        assert len(results) == 2

    def test_remove(self):
        """Test removing from octree."""
        octree = Octree(center=[0, 0, 0], size=10.0)

        octree.insert("obj1", [0, 0, 0])
        octree.insert("obj2", [1, 1, 1])

        octree.remove("obj1")
        results = octree.query_radius(center=[0, 0, 0], radius=1.0)

        assert len(results) == 1
        assert "obj1" not in results

    def test_statistics(self):
        """Test getting octree statistics."""
        octree = Octree(center=[0, 0, 0], size=10.0, capacity=2)

        for i in range(5):
            octree.insert(f"obj{i}", [i, i, i])

        stats = octree.get_statistics()

        assert stats["total_objects"] == 5
        assert stats["total_nodes"] > 1  # Should have subdivided


class TestHelperFunctions:
    """Test helper functions."""

    def test_infer_room_from_position(self):
        """Test inferring room from position."""
        sg = SceneGraph()
        room_layer = RoomLayer(sg)

        kitchen = room_layer.create_room(
            "kitchen",
            [5, 5, 0],
            bounds_min=[0, 0, 0],
            bounds_max=[10, 10, 3],
        )

        room = infer_room_from_position(sg, [5, 5, 1])

        assert room is not None
        assert room.id == kitchen.id

    def test_compute_spatial_relations(self):
        """Test computing spatial relations automatically."""
        sg = SceneGraph()
        obj_layer = ObjectLayer(sg)

        cup = obj_layer.create_object(
            "cup",
            [0, 0, 1],
            bbox_min=[0, 0, 0.9],
            bbox_max=[0.1, 0.1, 1.1],
        )
        table = obj_layer.create_object(
            "table",
            [0, 0, 0],
            bbox_min=[0, 0, 0],
            bbox_max=[1, 1, 0.9],
        )

        num_relations = compute_spatial_relations(sg, threshold_on=0.2)

        assert num_relations > 0

    def test_create_spatial_index(self):
        """Test creating spatial index for scene graph."""
        sg = SceneGraph()
        obj_layer = ObjectLayer(sg)

        obj_layer.create_object("cup", [0, 0, 0])
        obj_layer.create_object("table", [1, 1, 1])

        octree = create_spatial_index_for_scene_graph(sg, layer=LayerType.L1)

        assert octree is not None
        assert octree.get_statistics()["total_objects"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
