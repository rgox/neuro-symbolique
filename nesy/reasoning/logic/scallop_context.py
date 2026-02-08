"""
Scallop Context Wrapper - Differentiable Logic Programming.

This module provides a clean wrapper around Scallop's context for managing
facts, rules, and queries in a neurosymbolic reasoning system.

Scallop is a neurosymbolic programming language that combines:
- Datalog: Declarative logic programming
- Differentiability: Gradients flow through logical operations
- Provenance: Track reasoning paths and confidence
- Probabilistic: Handle uncertain facts

Example:
    >>> ctx = ScallopContext()
    >>> ctx.add_relation("edge", ["String", "String"])
    >>> ctx.add_fact("edge", "cup", "table")
    >>> ctx.add_fact("edge", "table", "kitchen")
    >>> 
    >>> ctx.add_rule("path(X, Z) :- edge(X, Z)")
    >>> ctx.add_rule("path(X, Z) :- edge(X, Y), path(Y, Z)")
    >>> 
    >>> results = ctx.query("path")
    >>> # → [("cup", "table"), ("cup", "kitchen"), ("table", "kitchen")]
"""

from typing import List, Tuple, Any, Optional, Dict
import logging

# Try to import scallopy, provide fallback if not available
try:
    import scallopy
    SCALLOP_AVAILABLE = True
except ImportError:
    SCALLOP_AVAILABLE = False
    logging.warning(
        "Scallop not available. Install with: pip install scallop-lang\n"
        "Falling back to mock implementation."
    )


class ScallopContext:
    """
    Wrapper for Scallop context with easy fact/rule management.
    
    Provides a clean API for adding relations, facts, rules, and querying
    the logical knowledge base.
    
    Attributes:
        provenance: Provenance semantics (difftopkproofs, minmaxprob, etc.)
        k: Number of proofs for top-k provenance
        ctx: Underlying Scallop context (if available)
        facts: List of all facts added
        rules: List of all rules added
    
    Example:
        >>> ctx = ScallopContext(provenance="difftopkproofs", k=3)
        >>> 
        >>> # Define schema
        >>> ctx.add_relation("object", ["String", "String"])
        >>> 
        >>> # Add facts
        >>> ctx.add_fact("object", "cup1", "cup")
        >>> ctx.add_fact("object", "table1", "table")
        >>> 
        >>> # Add rules
        >>> ctx.add_rule('same_room(X, Y) :- in(X, R), in(Y, R)')
        >>> 
        >>> # Query
        >>> results = ctx.query("same_room")
    """
    
    def __init__(
        self,
        provenance: str = "difftopkproofs",
        k: int = 3,
        train_k: Optional[int] = None
    ):
        """
        Initialize Scallop context.
        
        Args:
            provenance: Provenance semantics
                - "unit": No provenance tracking
                - "minmaxprob": Min-max probabilistic
                - "difftopkproofs": Differentiable top-k proofs (default)
            k: Number of proofs for top-k
            train_k: Number of proofs during training (if different)
        """
        self.provenance = provenance
        self.k = k
        self.train_k = train_k or k
        
        # Storage
        self.facts: List[Tuple[str, Tuple]] = []
        self.rules: List[str] = []
        self.relations: Dict[str, List[str]] = {}
        
        # Create Scallop context if available
        if SCALLOP_AVAILABLE:
            self.ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        else:
            self.ctx = None
            logging.warning("Using mock Scallop context (no inference)")
    
    def add_relation(self, name: str, types: List[str]) -> None:
        """
        Define a relation schema.
        
        Args:
            name: Relation name
            types: List of type names (e.g., ["String", "String", "f32"])
        
        Example:
            >>> ctx.add_relation("edge", ["String", "String", "String"])
            >>> # edge(src: String, relation: String, dst: String)
            >>> 
            >>> ctx.add_relation("similar", ["String", "String", "f32"])
            >>> # similar(obj1: String, obj2: String, score: f32)
        """
        self.relations[name] = types
        
        if self.ctx:
            self.ctx.add_relation(name, tuple(types))
    
    def add_fact(self, relation: str, *args) -> None:
        """
        Add a fact to the knowledge base.
        
        Args:
            relation: Relation name
            *args: Fact arguments matching relation schema
        
        Example:
            >>> ctx.add_fact("object", "cup1", "cup")
            >>> ctx.add_fact("on", "cup1", "table1")
            >>> ctx.add_fact("similar", "cup1", "cup2", 0.87)
        """
        self.facts.append((relation, args))
        
        if self.ctx:
            self.ctx.add_facts(relation, [args])
    
    def add_facts_batch(self, relation: str, facts: List[Tuple]) -> None:
        """
        Add multiple facts at once (more efficient).
        
        Args:
            relation: Relation name
            facts: List of fact tuples
        
        Example:
            >>> ctx.add_facts_batch("on", [
            ...     ("cup1", "table1"),
            ...     ("cup2", "table1"),
            ...     ("book1", "shelf1"),
            ... ])
        """
        for fact in facts:
            self.facts.append((relation, fact))
        
        if self.ctx:
            self.ctx.add_facts(relation, facts)
    
    def add_rule(self, rule: str) -> None:
        """
        Add a Datalog rule.
        
        Args:
            rule: Rule in Scallop/Datalog syntax
        
        Example:
            >>> # Transitive closure
            >>> ctx.add_rule('path(X, Z) :- edge(X, Z)')
            >>> ctx.add_rule('path(X, Z) :- edge(X, Y), path(Y, Z)')
            >>> 
            >>> # Spatial transitivity
            >>> ctx.add_rule('in(X, Z) :- on(X, Y), in(Y, Z)')
            >>> 
            >>> # Similarity threshold
            >>> ctx.add_rule('same_type(X, Y) :- similar(X, Y, S), S >= 0.8')
        """
        self.rules.append(rule)
        
        if self.ctx:
            self.ctx.add_rule(rule)
    
    def add_rules_batch(self, rules: List[str]) -> None:
        """
        Add multiple rules at once.
        
        Args:
            rules: List of rule strings
        """
        for rule in rules:
            self.add_rule(rule)
    
    def query(self, relation: str) -> List[Tuple]:
        """
        Query a relation and return all results.
        
        Args:
            relation: Relation name to query
        
        Returns:
            List of tuples matching the relation
        
        Example:
            >>> ctx.query("path")
            >>> # → [("cup1", "kitchen"), ("table1", "kitchen"), ...]
        """
        if not self.ctx:
            # Mock: return facts for this relation
            return [fact for rel, fact in self.facts if rel == relation]
        
        # Run inference
        self.ctx.run()
        
        # Get results
        results = list(self.ctx.relation(relation))
        
        return results
    
    def query_with_provenance(self, relation: str) -> List[Tuple[Tuple, Any]]:
        """
        Query with provenance information.
        
        Returns:
            List of (tuple, provenance) pairs
        """
        if not self.ctx:
            # Mock: no provenance
            return [(fact, 1.0) for rel, fact in self.facts if rel == relation]
        
        self.ctx.run()
        
        # Get results with tags (provenance)
        results = []
        for elem in self.ctx.relation(relation):
            results.append((elem, 1.0))  # Simplified
        
        return results
    
    def clear_facts(self) -> None:
        """
        Clear all facts but keep rules and schema.
        
        Useful for incremental updates.
        """
        self.facts = []
        
        if self.ctx:
            # Recreate context with same settings
            old_rules = self.rules.copy()
            old_relations = self.relations.copy()
            
            self.ctx = scallopy.ScallopContext(
                provenance=self.provenance,
                k=self.k
            ) if SCALLOP_AVAILABLE else None
            
            # Re-add relations
            for name, types in old_relations.items():
                self.add_relation(name, types)
            
            # Re-add rules
            for rule in old_rules:
                if self.ctx:
                    self.ctx.add_rule(rule)
    
    def clear_all(self) -> None:
        """Clear everything (facts, rules, relations)."""
        self.facts = []
        self.rules = []
        self.relations = {}
        
        if SCALLOP_AVAILABLE:
            self.ctx = scallopy.ScallopContext(
                provenance=self.provenance,
                k=self.k
            )
        else:
            self.ctx = None
    
    def get_num_facts(self) -> int:
        """Get total number of facts."""
        return len(self.facts)
    
    def get_num_rules(self) -> int:
        """Get total number of rules."""
        return len(self.rules)
    
    def get_relations(self) -> List[str]:
        """Get list of defined relations."""
        return list(self.relations.keys())
    
    def __repr__(self) -> str:
        return (
            f"ScallopContext("
            f"provenance={self.provenance}, "
            f"facts={len(self.facts)}, "
            f"rules={len(self.rules)}, "
            f"relations={len(self.relations)})"
        )
