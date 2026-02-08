"""
Logic Module - Symbolic Reasoning with Differentiable Logic Programming.

This module provides symbolic reasoning capabilities using Scallop (differentiable Datalog).

Components:
- ScallopContext: Wrapper for Scallop context (facts, rules, queries)
- ReasoningEngine: Main reasoning engine with scene graph integration
- VSA-logic bridging for hybrid neural-symbolic reasoning

Example:
    >>> from nesy.reasoning.logic import ReasoningEngine
    >>> from nesy.world_model.scene_graph import SceneGraph
    >>> 
    >>> # Create reasoning engine
    >>> engine = ReasoningEngine(scene_graph=sg, vsa_codebook=codebook)
    >>> 
    >>> # Sync from scene graph
    >>> engine.sync_from_scene_graph()
    >>> 
    >>> # Query spatial relations
    >>> results = engine.query("in")
    >>> # → [("cup1", "kitchen"), ...]
    >>> 
    >>> # Infer via transitivity
    >>> engine.infer("in('cup1', 'kitchen')")
    >>> # → True (cup1 on table1, table1 in kitchen)
    >>> 
    >>> # Find similar objects (via VSA)
    >>> similar = engine.find_similar_to("cup1", min_similarity=0.8)
    >>> # → [("cup2", 0.87), ...]
"""

from nesy.reasoning.logic.scallop_context import ScallopContext, SCALLOP_AVAILABLE
from nesy.reasoning.logic.engine import ReasoningEngine

__all__ = [
    'ScallopContext',
    'ReasoningEngine',
    'SCALLOP_AVAILABLE',
]
