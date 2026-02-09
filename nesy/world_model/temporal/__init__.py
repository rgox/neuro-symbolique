"""
Temporal World Model.

This module extends the scene graph with temporal reasoning capabilities:
- TemporalNode: Nodes with timestamps and history
- TemporalEdge: Temporal relations between events
- Event tracking: Automatic tracking of object lifecycle
- Temporal queries: "What was here before?", "How long has X existed?"

Example:
    >>> from nesy.world_model.temporal import TemporalSceneGraph
    >>> tsg = TemporalSceneGraph()
    >>> 
    >>> # Add object at t=0
    >>> cup = tsg.add_temporal_node("cup", position=[1, 2, 0], timestamp=0.0)
    >>> 
    >>> # Query what exists at t=5
    >>> objects_at_t5 = tsg.query_at_time(5.0)
    >>> 
    >>> # Query history
    >>> history = tsg.get_node_history(cup.id)

Key Concepts:
    - **Temporal Node**: Scene graph node with timestamps
    - **Event**: Creation, modification, or deletion of nodes
    - **Temporal Relation**: before, after, during, overlaps
    - **Persistence**: Objects persist until explicitly removed
"""

from nesy.world_model.temporal.temporal_graph import (
    TemporalNode,
    TemporalEdge,
    TemporalSceneGraph,
    EventType,
    TemporalRelation,
)

__all__ = [
    "TemporalNode",
    "TemporalEdge",
    "TemporalSceneGraph",
    "EventType",
    "TemporalRelation",
]
