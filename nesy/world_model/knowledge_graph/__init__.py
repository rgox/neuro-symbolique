"""
Knowledge Graph Layer L3.

This module extends the scene graph with abstract knowledge representation:
- Concepts: Abstract classes (furniture, kitchen, food)
- Ontology: is-a and part-of hierarchies
- Inheritance: Reason about class membership
- Abstraction: Map objects to concepts

The knowledge graph sits above the scene graph (L1-L2) and provides
semantic abstraction for reasoning.

Example:
    >>> from nesy.world_model.knowledge_graph import KnowledgeGraph, ConceptNode
    >>> kg = KnowledgeGraph()
    >>> 
    >>> # Build ontology
    >>> furniture = kg.add_concept("furniture")
    >>> chair = kg.add_concept("chair", parent=furniture)
    >>> table = kg.add_concept("table", parent=furniture)
    >>> 
    >>> # Query
    >>> assert kg.is_a("chair", "furniture")  # True - inheritance
    >>> descendants = kg.get_descendants("furniture")  # [chair, table]

Features:
    - **ConceptNode**: Abstract semantic concepts
    - **Ontology**: Hierarchical is-a relations
    - **Part-of**: Compositional relations
    - **Inference**: Transitive closure & inheritance
"""

from nesy.world_model.knowledge_graph.ontology import (
    ConceptNode,
    ConceptRelation,
    KnowledgeGraph,
)

__all__ = [
    "ConceptNode",
    "ConceptRelation",
    "KnowledgeGraph",
]
