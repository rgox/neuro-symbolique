"""
Multi-Modal Perception Module.

Integrates vision and language for semantic understanding:
- Text → Object search
- Visual grounding
- Cross-modal similarity
- Semantic scene understanding

Example:
    >>> from nesy.perception.multimodal import MultiModalQuery
    >>> from nesy.world_model.scene_graph import SceneGraph
    >>> 
    >>> sg = SceneGraph()
    >>> query = MultiModalQuery(scene_graph=sg)
    >>> 
    >>> # Text search
    >>> results = query.find_objects("red cup on the table")
    >>> 
    >>> # Visual grounding
    >>> obj = query.ground("the leftmost chair")
"""

from nesy.perception.multimodal.query import MultiModalQuery, QueryResult

__all__ = [
    "MultiModalQuery",
    "QueryResult",
]
