"""
Knowledge Graph Ontology.

Hierarchical concept representation with is-a and part-of relations,
SPARQL-like queries, and semantic reasoning.

Features:
- ConceptNode: Abstract semantic concepts
- Ontology hierarchy (is-a, part-of, has-property)
- Transitive closure for inheritance
- SPARQL-like query language
- Scene graph integration

Example:
    >>> kg = KnowledgeGraph()
    >>> kg.add_concept("furniture")
    >>> kg.add_concept("chair", parent="furniture")
    >>> kg.is_a("chair", "furniture")  # True
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConceptRelation(Enum):
    """Types of concept relations."""
    IS_A = "is_a"           # Taxonomy: chair IS_A furniture
    PART_OF = "part_of"     # Mereology: leg PART_OF table
    HAS_PROPERTY = "has_property"  # Attribution: cup HAS_PROPERTY round
    RELATED_TO = "related_to"     # General association
    OPPOSITE_OF = "opposite_of"   # Antonymy: hot OPPOSITE_OF cold
    SIMILAR_TO = "similar_to"     # Similarity


@dataclass
class ConceptNode:
    """
    Abstract semantic concept.
    
    Represents a concept in the ontology, e.g. "furniture", "kitchen".
    """
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None  # IS_A parent
    instances: List[str] = field(default_factory=list)  # Scene graph object IDs


@dataclass
class ConceptEdge:
    """Relation between concepts."""
    source: str
    target: str
    relation: ConceptRelation
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    Knowledge graph with ontology and semantic reasoning.
    
    Provides hierarchical concept management, inheritance-based
    reasoning, and SPARQL-like queries.
    
    Example:
        >>> kg = KnowledgeGraph()
        >>> 
        >>> # Build ontology
        >>> kg.add_concept("thing")
        >>> kg.add_concept("furniture", parent="thing")
        >>> kg.add_concept("chair", parent="furniture")
        >>> kg.add_concept("table", parent="furniture")
        >>> 
        >>> # Properties
        >>> kg.set_property("chair", "has_legs", True)
        >>> kg.set_property("chair", "typical_count", 4)
        >>> 
        >>> # Queries
        >>> kg.is_a("chair", "furniture")  # True
        >>> kg.is_a("chair", "thing")      # True (transitive)
        >>> kg.get_descendants("furniture") # ["chair", "table"]
        >>> 
        >>> # SPARQL-like
        >>> results = kg.query("SELECT ?x WHERE { ?x is_a furniture }")
    """
    
    def __init__(self):
        """Initialize knowledge graph."""
        self.concepts: Dict[str, ConceptNode] = {}
        self.edges: List[ConceptEdge] = []
        self._children: Dict[str, Set[str]] = {}  # Parent → children cache
        
        logger.info("Knowledge graph initialized")
    
    # --- Concept Management ---
    
    def add_concept(
        self,
        name: str,
        parent: Optional[str] = None,
        description: str = "",
        properties: Optional[Dict] = None
    ) -> ConceptNode:
        """
        Add concept to ontology.
        
        Args:
            name: Concept name
            parent: Parent concept (IS_A relation)
            description: Concept description
            properties: Initial properties
        """
        node = ConceptNode(
            name=name,
            description=description,
            properties=properties or {},
            parent=parent
        )
        self.concepts[name] = node
        
        # Add IS_A relation
        if parent:
            if parent not in self.concepts:
                # Auto-create parent
                self.add_concept(parent)
            
            self.add_relation(name, parent, ConceptRelation.IS_A)
            
            # Update children cache
            if parent not in self._children:
                self._children[parent] = set()
            self._children[parent].add(name)
        
        if name not in self._children:
            self._children[name] = set()
        
        logger.debug(f"Added concept: {name}" + (f" (is_a {parent})" if parent else ""))
        return node
    
    def remove_concept(self, name: str):
        """Remove concept and its relations."""
        if name in self.concepts:
            # Remove from parent's children
            parent = self.concepts[name].parent
            if parent and parent in self._children:
                self._children[parent].discard(name)
            
            # Remove edges
            self.edges = [e for e in self.edges if e.source != name and e.target != name]
            
            # Remove children cache
            del self._children[name]
            del self.concepts[name]
    
    def get_concept(self, name: str) -> Optional[ConceptNode]:
        """Get concept by name."""
        return self.concepts.get(name)
    
    # --- Relations ---
    
    def add_relation(
        self,
        source: str,
        target: str,
        relation: ConceptRelation,
        weight: float = 1.0
    ):
        """Add relation between concepts."""
        edge = ConceptEdge(
            source=source, target=target,
            relation=relation, weight=weight
        )
        self.edges.append(edge)
    
    def get_relations(
        self,
        concept: str,
        relation_type: Optional[ConceptRelation] = None
    ) -> List[ConceptEdge]:
        """Get all relations for a concept."""
        results = [e for e in self.edges if e.source == concept or e.target == concept]
        if relation_type:
            results = [e for e in results if e.relation == relation_type]
        return results
    
    # --- Taxonomy / Inference ---
    
    def is_a(self, concept: str, ancestor: str) -> bool:
        """
        Check IS_A relation (with transitive closure).
        
        Args:
            concept: Child concept
            ancestor: Potential ancestor
        
        Returns:
            True if concept IS_A ancestor
        """
        if concept == ancestor:
            return True
        
        node = self.concepts.get(concept)
        if not node or not node.parent:
            return False
        
        # Direct parent
        if node.parent == ancestor:
            return True
        
        # Transitive: check parent's chain
        return self.is_a(node.parent, ancestor)
    
    def get_ancestors(self, concept: str) -> List[str]:
        """Get all ancestors (transitive IS_A closure)."""
        ancestors = []
        node = self.concepts.get(concept)
        
        while node and node.parent:
            ancestors.append(node.parent)
            node = self.concepts.get(node.parent)
        
        return ancestors
    
    def get_descendants(self, concept: str) -> List[str]:
        """Get all descendants (transitive)."""
        descendants = []
        direct = self._children.get(concept, set())
        
        for child in direct:
            descendants.append(child)
            descendants.extend(self.get_descendants(child))
        
        return descendants
    
    def get_children(self, concept: str) -> List[str]:
        """Get direct children."""
        return list(self._children.get(concept, set()))
    
    def get_siblings(self, concept: str) -> List[str]:
        """Get sibling concepts (same parent)."""
        node = self.concepts.get(concept)
        if not node or not node.parent:
            return []
        
        siblings = list(self._children.get(node.parent, set()))
        siblings = [s for s in siblings if s != concept]
        return siblings
    
    # --- Properties ---
    
    def set_property(self, concept: str, key: str, value: Any):
        """Set concept property."""
        if concept in self.concepts:
            self.concepts[concept].properties[key] = value
    
    def get_property(self, concept: str, key: str, inherit: bool = True) -> Optional[Any]:
        """
        Get concept property with optional inheritance.
        
        Args:
            concept: Concept name
            key: Property key
            inherit: Whether to check parent concepts
        """
        node = self.concepts.get(concept)
        if not node:
            return None
        
        # Direct property
        if key in node.properties:
            return node.properties[key]
        
        # Inherit from parent
        if inherit and node.parent:
            return self.get_property(node.parent, key, inherit=True)
        
        return None
    
    # --- Instance Binding ---
    
    def bind_instance(self, concept: str, instance_id: str):
        """Bind scene graph object to concept."""
        if concept in self.concepts:
            self.concepts[concept].instances.append(instance_id)
    
    def get_instances(self, concept: str, include_descendants: bool = True) -> List[str]:
        """Get all instances of a concept."""
        instances = []
        
        node = self.concepts.get(concept)
        if node:
            instances.extend(node.instances)
        
        if include_descendants:
            for desc in self.get_descendants(concept):
                desc_node = self.concepts.get(desc)
                if desc_node:
                    instances.extend(desc_node.instances)
        
        return instances
    
    # --- SPARQL-like Queries ---
    
    def query(self, sparql: str) -> List[Dict[str, str]]:
        """
        Execute SPARQL-like query.
        
        Supports simple patterns:
            SELECT ?x WHERE { ?x is_a concept }
            SELECT ?x WHERE { ?x has_property key }
            SELECT ?x ?y WHERE { ?x related_to ?y }
        
        Args:
            sparql: SPARQL-like query string
        
        Returns:
            List of variable bindings
        """
        results = []
        
        # Parse simple patterns
        sparql = sparql.strip()
        
        # Extract SELECT variables
        if "SELECT" in sparql.upper():
            select_part = sparql.split("WHERE")[0].replace("SELECT", "").strip()
            variables = [v.strip() for v in select_part.split() if v.startswith("?")]
        else:
            variables = []
        
        # Extract WHERE pattern
        if "WHERE" in sparql.upper() and "{" in sparql:
            where_part = sparql.split("{")[1].split("}")[0].strip()
            patterns = [p.strip() for p in where_part.split(".") if p.strip()]
        else:
            return results
        
        for pattern in patterns:
            parts = pattern.split()
            if len(parts) != 3:
                continue
            
            subject, predicate, obj = parts
            
            # Match pattern
            if predicate == "is_a" and subject.startswith("?"):
                # Find all concepts that IS_A object
                var_name = subject
                for name in self.concepts:
                    if self.is_a(name, obj) and name != obj:
                        results.append({var_name: name})
            
            elif predicate == "has_property" and subject.startswith("?"):
                var_name = subject
                for name, node in self.concepts.items():
                    if obj in node.properties:
                        results.append({var_name: name})
            
            elif predicate == "part_of" and subject.startswith("?"):
                var_name = subject
                for edge in self.edges:
                    if edge.relation == ConceptRelation.PART_OF and edge.target == obj:
                        results.append({var_name: edge.source})
            
            elif subject.startswith("?") and obj.startswith("?"):
                # Both variables
                rel_type = ConceptRelation(predicate) if predicate in [r.value for r in ConceptRelation] else None
                if rel_type:
                    for edge in self.edges:
                        if edge.relation == rel_type:
                            results.append({subject: edge.source, obj: edge.target})
        
        return results
    
    # --- Stats ---
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return {
            "concepts": len(self.concepts),
            "relations": len(self.edges),
            "max_depth": self._max_depth(),
            "root_concepts": len([c for c in self.concepts.values() if not c.parent]),
            "leaf_concepts": len([c for c in self.concepts.values()
                                  if not self._children.get(c.name)])
        }
    
    def _max_depth(self) -> int:
        """Calculate max ontology depth."""
        max_d = 0
        for name in self.concepts:
            depth = len(self.get_ancestors(name))
            max_d = max(max_d, depth)
        return max_d
    
    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            "concepts": {
                name: {
                    "description": c.description,
                    "parent": c.parent,
                    "properties": c.properties,
                    "instances": c.instances
                }
                for name, c in self.concepts.items()
            },
            "relations": [
                {"source": e.source, "target": e.target, "relation": e.relation.value}
                for e in self.edges
            ]
        }
