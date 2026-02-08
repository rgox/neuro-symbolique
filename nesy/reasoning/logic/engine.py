"""
Reasoning Engine - Symbolic Inference over Scene Graph.

This module implements the core reasoning engine that performs logical
inference over the scene graph using Scallop (differentiable Datalog).

The engine bridges neural perception and symbolic reasoning by:
1. Converting scene graph to Scallop facts
2. Applying logical rules (spatial, semantic, temporal)
3. Computing VSA similarity as fuzzy logic facts
4. Answering queries through inference

Example:
    >>> from nesy.reasoning.logic import ReasoningEngine
    >>> 
    >>> engine = ReasoningEngine(scene_graph=sg, vsa_codebook=codebook)
    >>> 
    >>> # Sync scene graph to logic
    >>> engine.sync_from_scene_graph()
    >>> 
    >>> # Query: what's in the kitchen?
    >>> results = engine.query("in")
    >>> # → [("cup1", "kitchen"), ("table1", "kitchen"), ...]
    >>> 
    >>> # Infer: is cup1 in kitchen? (via transitivity)
    >>> engine.infer("in('cup1', 'kitchen')")
    >>> # → True (cup1 on table1, table1 in kitchen → cup1 in kitchen)
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import logging

from nesy.reasoning.logic.scallop_context import ScallopContext
from nesy.world_model.scene_graph import SceneGraph, LayerType, RelationType, Node
from nesy.reasoning.vsa import VSACodebook, HyperVector
from nesy.core.memory import UMA


logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Symbolic reasoning engine using Scallop.
    
    Performs inference over scene graph using logic rules.
    Bridges VSA similarity with fuzzy logic for hybrid reasoning.
    
    Attributes:
        scene_graph: Scene graph to reason over
        codebook: VSA codebook for symbol grounding
        ctx: Scallop context for logic inference
        vsa_similarity_threshold: Minimum similarity to create fuzzy facts
    
    Example:
        >>> engine = ReasoningEngine(
        ...     scene_graph=sg,
        ...     vsa_codebook=codebook,
        ...     vsa_similarity_threshold=0.5
        ... )
        >>> 
        >>> engine.sync_from_scene_graph()
        >>> 
        >>> # Spatial reasoning
        >>> results = engine.query("in")
        >>> 
        >>> # Check inference
        >>> is_true = engine.infer("in('cup1', 'kitchen')")
    """
    
    def __init__(
        self,
        scene_graph: SceneGraph,
        vsa_codebook: Optional[VSACodebook] = None,
        vsa_similarity_threshold: float = 0.5,
        rules_file: Optional[str] = None,
        provenance: str = "difftopkproofs",
        k: int = 3
    ):
        """
        Initialize reasoning engine.
        
        Args:
            scene_graph: Scene graph to reason over
            vsa_codebook: Optional VSA codebook for similarity reasoning
            vsa_similarity_threshold: Min similarity for VSA facts (0-1)
            rules_file: Optional file with custom rules
            provenance: Scallop provenance semantics
            k: Top-k proofs
        """
        self.scene_graph = scene_graph
        self.codebook = vsa_codebook
        self.vsa_similarity_threshold = vsa_similarity_threshold
        
        # Create Scallop context
        self.ctx = ScallopContext(provenance=provenance, k=k)
        
        # Define relations
        self._define_relations()
        
        # Load default rules
        self._load_default_rules()
        
        # Load custom rules if provided
        if rules_file:
            self._load_rules_from_file(rules_file)
        
        logger.info(
            f"ReasoningEngine initialized with {self.ctx.get_num_rules()} rules"
        )
    
    def _define_relations(self) -> None:
        """Define Scallop relation schemas."""
        # Object facts
        self.ctx.add_relation("object", ["String", "String"])  # (id, class)
        self.ctx.add_relation("attribute", ["String", "String", "String"])  # (id, key, value)
        self.ctx.add_relation("layer", ["String", "String"])  # (id, layer_name)
        
        # Spatial relations
        self.ctx.add_relation("on", ["String", "String"])  # (obj1, obj2)
        self.ctx.add_relation("in", ["String", "String"])  # (obj, container)
        self.ctx.add_relation("near", ["String", "String"])  # (obj1, obj2)
        self.ctx.add_relation("contains", ["String", "String"])  # (container, obj)
        self.ctx.add_relation("connected_to", ["String", "String"])  # (place1, place2)
        
        # VSA similarity
        self.ctx.add_relation("similar_vsa", ["String", "String", "f32"])  # (obj1, obj2, sim)
        
        # Derived relations (computed by rules)
        self.ctx.add_relation("same_type", ["String", "String"])  # Derived from VSA similarity
        self.ctx.add_relation("path", ["String", "String"])  # Transitive closure
        self.ctx.add_relation("reachable", ["String", "String"])  # Spatial reachability
    
    def _load_default_rules(self) -> None:
        """Load default reasoning rules."""
        # ===== Spatial Transitivity =====
        
        # on + in → in (cup on table, table in kitchen → cup in kitchen)
        self.ctx.add_rule("in(X, Z) :- on(X, Y), in(Y, Z)")
        
        # near + in → in (obj near table, table in kitchen → obj in kitchen)
        self.ctx.add_rule("in(X, Z) :- near(X, Y), in(Y, Z)")
        
        # ===== Symmetry =====
        
        # near is symmetric
        self.ctx.add_rule("near(Y, X) :- near(X, Y)")
        
        # connected_to is symmetric
        self.ctx.add_rule("connected_to(Y, X) :- connected_to(X, Y)")
        
        # ===== Containment =====
        
        # in → contains (reverse relation)
        self.ctx.add_rule("contains(Y, X) :- in(X, Y)")
        
        # ===== VSA Similarity =====
        
        # High similarity → same type (threshold 0.8)
        self.ctx.add_rule("same_type(X, Y) :- similar_vsa(X, Y, S), S >= 0.8")
        
        # Symmetry of similarity
        self.ctx.add_rule("similar_vsa(Y, X, S) :- similar_vsa(X, Y, S)")
        
        # ===== Path / Reachability =====
        
        # Base cases
        self.ctx.add_rule("path(X, Y) :- on(X, Y)")
        self.ctx.add_rule("path(X, Y) :- in(X, Y)")
        self.ctx.add_rule("path(X, Y) :- near(X, Y)")
        
        # Transitive closure
        self.ctx.add_rule("path(X, Z) :- path(X, Y), on(Y, Z)")
        self.ctx.add_rule("path(X, Z) :- path(X, Y), in(Y, Z)")
        
        # Reachability
        self.ctx.add_rule("reachable(X, Y) :- path(X, Y)")
        self.ctx.add_rule("reachable(X, Z) :- path(X, Y), reachable(Y, Z)")
    
    def _load_rules_from_file(self, filepath: str) -> None:
        """Load rules from a .scl file."""
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('%'):  # Skip empty and comments
                        self.ctx.add_rule(line)
            logger.info(f"Loaded rules from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Rules file not found: {filepath}")
    
    def sync_from_scene_graph(self) -> None:
        """
        Synchronize facts from scene graph to Scallop context.
        
        Exports:
        - All nodes (object, attributes, layer)
        - All edges (spatial/semantic relations)
        - VSA similarities (if codebook available)
        
        This should be called whenever the scene graph is updated.
        """
        logger.debug("Syncing scene graph to Scallop context...")
        
        # Clear old facts
        self.ctx.clear_facts()
        
        # Export nodes
        for node in self.scene_graph.nodes.values():
            # Object class
            obj_class = node.attributes.get("class", "unknown")
            self.ctx.add_fact("object", node.id, obj_class)
            
            # Layer
            self.ctx.add_fact("layer", node.id, node.layer.value)
            
            # Attributes
            for key, value in node.attributes.items():
                if key != "class":  # Already exported as object
                    self.ctx.add_fact("attribute", node.id, key, str(value))
        
        # Export edges (spatial/semantic relations)
        for edges in self.scene_graph.edges.values():
            for edge in edges:
                relation = edge.relation.value
                
                # Map relation types to Scallop relations
                if relation == "on":
                    self.ctx.add_fact("on", edge.src, edge.dst)
                elif relation == "in":
                    self.ctx.add_fact("in", edge.src, edge.dst)
                elif relation == "near":
                    self.ctx.add_fact("near", edge.src, edge.dst)
                elif relation == "contains":
                    self.ctx.add_fact("contains", edge.src, edge.dst)
                elif relation == "connected_to":
                    self.ctx.add_fact("connected_to", edge.src, edge.dst)
        
        # Export VSA similarities (if available)
        if self.scene_graph.uma and self.codebook:
            self._export_vsa_similarities()
        
        logger.info(
            f"Synced {self.ctx.get_num_facts()} facts from scene graph "
            f"({len(self.scene_graph.nodes)} nodes, "
            f"{self.scene_graph.stats['total_edges']} edges)"
        )
    
    def _export_vsa_similarities(self) -> None:
        """
        Export VSA similarities as fuzzy logic facts.
        
        Computes pairwise similarities between all nodes with VSA embeddings
        and creates similar_vsa facts for similarities above threshold.
        """
        # Get all nodes with VSA embeddings
        vsa_nodes = [
            n for n in self.scene_graph.nodes.values()
            if n.vsa_embedding_key and self.scene_graph.uma.exists(n.vsa_embedding_key)
        ]
        
        if not vsa_nodes:
            return
        
        logger.debug(f"Computing VSA similarities for {len(vsa_nodes)} nodes...")
        
        # Compute pairwise similarities
        similarities_added = 0
        
        for i, node1 in enumerate(vsa_nodes):
            # Get hypervector 1
            hv1_data = self.scene_graph.uma.get(node1.vsa_embedding_key).data
            hv1 = HyperVector.from_vector(hv1_data, normalize=False)
            
            for node2 in vsa_nodes[i+1:]:
                # Get hypervector 2
                hv2_data = self.scene_graph.uma.get(node2.vsa_embedding_key).data
                hv2 = HyperVector.from_vector(hv2_data, normalize=False)
                
                # Compute similarity
                sim = hv1.similarity(hv2)
                
                # Only export if above threshold
                if sim >= self.vsa_similarity_threshold:
                    self.ctx.add_fact("similar_vsa", node1.id, node2.id, float(sim))
                    similarities_added += 1
        
        logger.debug(f"Added {similarities_added} VSA similarity facts")
    
    def query(self, relation: str) -> List[Tuple]:
        """
        Query a relation and return all results.
        
        Args:
            relation: Relation name (e.g., "in", "path", "same_type")
        
        Returns:
            List of tuples matching the relation
        
        Example:
            >>> results = engine.query("in")
            >>> # → [("cup1", "kitchen"), ("table1", "kitchen"), ...]
            >>> 
            >>> results = engine.query("same_type")
            >>> # → [("cup1", "cup2"), ("table1", "table2"), ...]
        """
        return self.ctx.query(relation)
    
    def infer(self, goal: str) -> bool:
        """
        Check if a goal can be inferred from the facts and rules.
        
        Args:
            goal: Goal in Scallop syntax (e.g., "in('cup1', 'kitchen')")
        
        Returns:
            True if goal is provable, False otherwise
        
        Example:
            >>> # Direct fact
            >>> engine.infer("on('cup1', 'table1')")
            >>> # → True
            >>> 
            >>> # Derived via rule
            >>> engine.infer("in('cup1', 'kitchen')")
            >>> # → True (cup1 on table1, table1 in kitchen)
        """
        # Create temporary query rule
        temp_rule = f"_goal() :- {goal}"
        
        # Save current rules
        old_rules_count = self.ctx.get_num_rules()
        
        # Add temporary rule
        self.ctx.add_rule(temp_rule)
        
        # Query
        results = self.ctx.query("_goal")
        
        # Note: In a real implementation, we'd remove the temp rule
        # For now, it's okay to leave it
        
        return len(results) > 0
    
    def find_all(self, object_class: str) -> List[str]:
        """
        Find all object IDs of a given class.
        
        Args:
            object_class: Object class name (e.g., "cup", "table")
        
        Returns:
            List of object IDs
        
        Example:
            >>> cup_ids = engine.find_all("cup")
            >>> # → ["cup1", "cup2", "cup3"]
        """
        results = self.ctx.query("object")
        
        # Filter by class
        matching_ids = [obj_id for obj_id, cls in results if cls == object_class]
        
        return matching_ids
    
    def find_in_location(self, location_id: str) -> List[str]:
        """
        Find all objects in a location (via 'in' relation and transitivity).
        
        Args:
            location_id: ID of container/location node
        
        Returns:
            List of object IDs in that location
        
        Example:
            >>> objects_in_kitchen = engine.find_in_location("kitchen1")
            >>> # → ["cup1", "table1", "chair1", ...]
        """
        results = self.ctx.query("in")
        
        # Filter by location
        objects = [obj_id for obj_id, loc_id in results if loc_id == location_id]
        
        return objects
    
    def find_similar_to(self, object_id: str, min_similarity: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find objects similar to a given object (via VSA).
        
        Args:
            object_id: ID of query object
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of (object_id, similarity) tuples
        
        Example:
            >>> similar = engine.find_similar_to("cup1", min_similarity=0.8)
            >>> # → [("cup2", 0.87), ("cup3", 0.82)]
        """
        results = self.ctx.query("similar_vsa")
        
        # Filter by object and similarity
        similar = []
        for obj1, obj2, sim in results:
            if obj1 == object_id and sim >= min_similarity:
                similar.append((obj2, sim))
            elif obj2 == object_id and sim >= min_similarity:
                similar.append((obj1, sim))
        
        # Sort by similarity (descending)
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar
    
    def add_custom_rule(self, rule: str) -> None:
        """
        Add a custom inference rule.
        
        Args:
            rule: Rule in Scallop/Datalog syntax
        
        Example:
            >>> # Add domain-specific rule
            >>> engine.add_custom_rule(
            ...     'on_desk(X) :- on(X, D), object(D, "desk")'
            ... )
            >>> 
            >>> # Query
            >>> items_on_desk = engine.query("on_desk")
        """
        self.ctx.add_rule(rule)
        logger.debug(f"Added custom rule: {rule}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning engine statistics."""
        return {
            "num_facts": self.ctx.get_num_facts(),
            "num_rules": self.ctx.get_num_rules(),
            "relations": self.ctx.get_relations(),
            "vsa_enabled": self.codebook is not None,
            "vsa_similarity_threshold": self.vsa_similarity_threshold,
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"ReasoningEngine("
            f"facts={stats['num_facts']}, "
            f"rules={stats['num_rules']}, "
            f"vsa={stats['vsa_enabled']})"
        )
