"""
Temporal Scene Graph Implementation.

Extends the base scene graph with temporal reasoning:
- Timestamped nodes and edges
- Event history tracking
- Temporal queries (at time T, between T1-T2, etc.)
- Temporal relations (before, after, during, overlaps)

This allows answering questions like:
- "What was on the table at 10:00?"
- "How long was the cup on the table?"
- "What changed between 9:00 and 10:00?"
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import time
import numpy as np

from nesy.world_model.scene_graph import Node, Edge, SceneGraph, LayerType, NodeType, RelationType


class EventType(Enum):
    """Types of temporal events."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


class TemporalRelation(Enum):
    """Temporal relations between intervals (Allen's interval algebra)."""
    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    DURING = "during"
    CONTAINS = "contains"
    STARTS = "starts"
    STARTED_BY = "started_by"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"


@dataclass
class TemporalNode(Node):
    """
    Scene graph node with temporal information.
    
    Extends Node with:
    - created_at: When object first appeared
    - deleted_at: When object was removed (None if still exists)
    - history: List of modification events
    
    Attributes:
        created_at: Timestamp when node was created
        deleted_at: Timestamp when node was deleted (None if active)
        history: List of (timestamp, event_type, data) tuples
    """
    created_at: float = field(default_factory=time.time)
    deleted_at: Optional[float] = None
    history: List[Tuple[float, EventType, Dict[str, Any]]] = field(default_factory=list)
    
    def is_active_at(self, timestamp: float) -> bool:
        """Check if node exists at given timestamp."""
        if timestamp < self.created_at:
            return False
        if self.deleted_at is not None and timestamp >= self.deleted_at:
            return False
        return True
    
    def get_state_at(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Get node state at specific timestamp."""
        if not self.is_active_at(timestamp):
            return None
        
        # Start with initial state
        state = {
            "position": self.position.copy() if self.position is not None else None,
            "attributes": self.attributes.copy(),
        }
        
        # Apply history up to timestamp
        for t, event_type, data in self.history:
            if t > timestamp:
                break
            if event_type == EventType.MODIFIED:
                if "position" in data:
                    state["position"] = data["position"]
                if "attributes" in data:
                    state["attributes"].update(data["attributes"])
        
        return state
    
    def record_event(self, event_type: EventType, data: Optional[Dict[str, Any]] = None, timestamp: Optional[float] = None):
        """Record a temporal event in history."""
        if timestamp is None:
            timestamp = time.time()
        self.history.append((timestamp, event_type, data or {}))


@dataclass
class TemporalEdge(Edge):
    """
    Scene graph edge with temporal information.
    
    Extends Edge with:
    - created_at: When relation first established
    - deleted_at: When relation ended (None if still active)
    - temporal_relation: Type of temporal relation (before, during, etc.)
    
    Attributes:
        created_at: Timestamp when edge was created
        deleted_at: Timestamp when edge was deleted (None if active)
        temporal_relation: Optional temporal relation type
    """
    created_at: float = field(default_factory=time.time)
    deleted_at: Optional[float] = None
    temporal_relation: Optional[TemporalRelation] = None
    
    def is_active_at(self, timestamp: float) -> bool:
        """Check if edge exists at given timestamp."""
        if timestamp < self.created_at:
            return False
        if self.deleted_at is not None and timestamp >= self.deleted_at:
            return False
        return True


class TemporalSceneGraph(SceneGraph):
    """
    Scene graph with temporal reasoning capabilities.
    
    Extends SceneGraph to track object lifecycles and temporal relations.
    Automatically records creation, modification, and deletion events.
    
    Features:
        - Temporal nodes with timestamps
        - Event history tracking
        - Queries at specific times
        - Temporal relation reasoning
    
    Example:
        >>> tsg = TemporalSceneGraph()
        >>> 
        >>> # Add cup at t=0
        >>> cup = tsg.add_temporal_node(
        ...     class_name="cup",
        ...     position=[1, 2, 0],
        ...     timestamp=0.0
        ... )
        >>> 
        >>> # Move cup at t=5
        >>> tsg.update_node_position(cup.id, [2, 2, 0], timestamp=5.0)
        >>> 
        >>> # Remove cup at t=10
        >>> tsg.remove_temporal_node(cup.id, timestamp=10.0)
        >>> 
        >>> # Query what existed at t=7
        >>> nodes_at_7 = tsg.query_at_time(7.0)
        >>> print(len(nodes_at_7))  # 1 (cup exists)
        >>> 
        >>> # Query history
        >>> history = tsg.get_node_history(cup.id)
        >>> print(len(history))  # 3 events (created, moved, deleted)
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize temporal scene graph."""
        super().__init__(*args, **kwargs)
        
        # Track temporal nodes and edges
        self.temporal_nodes: Dict[str, TemporalNode] = {}
        self.temporal_edges: Dict[str, TemporalEdge] = {}
        
        # Event timeline
        self.events: List[Tuple[float, EventType, str, Dict[str, Any]]] = []
    
    def add_temporal_node(
        self,
        class_name: str,
        position: np.ndarray,
        layer: LayerType = LayerType.L1,
        node_type: NodeType = NodeType.OBJECT,
        timestamp: Optional[float] = None,
        **kwargs
    ) -> TemporalNode:
        """
        Add a temporal node to the scene graph.
        
        Args:
            class_name: Object class name
            position: 3D position [x, y, z]
            layer: Layer type
            node_type: Node type
            timestamp: Event timestamp (current time if None)
            **kwargs: Additional node attributes
        
        Returns:
            Created TemporalNode
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Create temporal node
        node = TemporalNode(
            id=kwargs.pop("node_id", str(uuid.uuid4())),
            layer=layer,
            node_type=node_type,
            position=np.array(position),
            attributes={"class": class_name, **kwargs.get("attributes", {})},
            created_at=timestamp,
            **kwargs
        )
        
        # Record creation event
        node.record_event(EventType.CREATED, {"class": class_name}, timestamp)
        
        # Add to base scene graph
        self.nodes[node.id] = node
        self.temporal_nodes[node.id] = node
        self.layer_nodes[layer].add(node.id)
        
        # Record global event
        self.events.append((timestamp, EventType.CREATED, node.id, {"class": class_name}))
        
        return node
    
    def remove_temporal_node(self, node_id: str, timestamp: Optional[float] = None):
        """
        Remove (soft delete) a temporal node.
        
        Args:
            node_id: Node ID to remove
            timestamp: Deletion timestamp (current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        if node_id in self.temporal_nodes:
            node = self.temporal_nodes[node_id]
            node.deleted_at = timestamp
            node.record_event(EventType.DELETED, {}, timestamp)
            
            # Record global event
            self.events.append((timestamp, EventType.DELETED, node_id, {}))
            
            # Note: Don't remove from scene graph to maintain history
            # Just mark as deleted
    
    def update_node_position(
        self,
        node_id: str,
        new_position: np.ndarray,
        timestamp: Optional[float] = None
    ):
        """
        Update node position and record movement event.
        
        Args:
            node_id: Node ID
            new_position: New 3D position
            timestamp: Event timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        if node_id in self.temporal_nodes:
            node = self.temporal_nodes[node_id]
            old_position = node.position.copy() if node.position is not None else None
            
            # Update position
            node.position = np.array(new_position)
            
            # Record move event
            node.record_event(
                EventType.MOVED,
                {"from": old_position.tolist() if old_position is not None else None, "to": new_position.tolist()},
                timestamp
            )
            
            # Record global event
            self.events.append((timestamp, EventType.MOVED, node_id, {
                "from": old_position.tolist() if old_position is not None else None,
                "to": new_position.tolist()
            }))
    
    def query_at_time(self, timestamp: float, layer: Optional[LayerType] = None) -> List[TemporalNode]:
        """
        Query all nodes that existed at a specific time.
        
        Args:
            timestamp: Query timestamp
            layer: Optional layer filter
        
        Returns:
            List of nodes active at timestamp
        """
        results = []
        for node in self.temporal_nodes.values():
            if node.is_active_at(timestamp):
                if layer is None or node.layer == layer:
                    results.append(node)
        return results
    
    def query_between_times(
        self,
        start_time: float,
        end_time: float,
        layer: Optional[LayerType] = None
    ) -> List[TemporalNode]:
        """
        Query nodes that existed at any point during time interval.
        
        Args:
            start_time: Start of interval
            end_time: End of interval
            layer: Optional layer filter
        
        Returns:
            List of nodes active during interval
        """
        results = []
        for node in self.temporal_nodes.values():
            # Node existed if it was created before end and deleted after start (or not deleted)
            if node.created_at <= end_time:
                if node.deleted_at is None or node.deleted_at >= start_time:
                    if layer is None or node.layer == layer:
                        results.append(node)
        return results
    
    def get_node_history(self, node_id: str) -> List[Tuple[float, EventType, Dict[str, Any]]]:
        """
        Get complete event history for a node.
        
        Args:
            node_id: Node ID
        
        Returns:
            List of (timestamp, event_type, data) tuples
        """
        if node_id in self.temporal_nodes:
            return self.temporal_nodes[node_id].history.copy()
        return []
    
    def get_events_between(
        self,
        start_time: float,
        end_time: float,
        event_type: Optional[EventType] = None
    ) -> List[Tuple[float, EventType, str, Dict[str, Any]]]:
        """
        Get all events in a time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            event_type: Optional event type filter
        
        Returns:
            List of (timestamp, event_type, node_id, data) tuples
        """
        results = []
        for t, e_type, node_id, data in self.events:
            if start_time <= t <= end_time:
                if event_type is None or e_type == event_type:
                    results.append((t, e_type, node_id, data))
        return sorted(results, key=lambda x: x[0])
    
    def get_object_duration(self, node_id: str) -> Optional[float]:
        """
        Get how long object existed (in seconds).
        
        Args:
            node_id: Node ID
        
        Returns:
            Duration in seconds, or None if not found
        """
        if node_id in self.temporal_nodes:
            node = self.temporal_nodes[node_id]
            end_time = node.deleted_at if node.deleted_at is not None else time.time()
            return end_time - node.created_at
        return None


# Import uuid at top
import uuid
