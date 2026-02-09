"""
Multi-Modal Query API.

Enables natural language queries over visual scenes using CLIP.

Features:
    - Text → Object matching
    - Semantic similarity ranking
    - Visual grounding
    - Compositional queries
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

from nesy.world_model.scene_graph import SceneGraph, Node
from nesy.perception.models import CLIPEmbedder
from nesy.core.memory import UMA


@dataclass
class QueryResult:
    """
    Result of a multi-modal query.
    
    Attributes:
        node: Matched scene graph node
        similarity: Similarity score [0, 1]
        explanation: Human-readable explanation
        metadata: Additional query metadata
    """
    node: Node
    similarity: float
    explanation: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MultiModalQuery:
    """
    Multi-modal query interface for scene graphs.
    
    Enables natural language queries over visual scenes:
    - "Find the red cup"
    - "Show me objects on the table"
    - "Which chair is closest to the door?"
    
    Args:
        scene_graph: Scene graph to query
        clip_model: CLIP model name (default: "clip-vit-b32")
        uma: Optional UMA for cached embeddings
        top_k: Default number of results to return
    
    Example:
        >>> query = MultiModalQuery(scene_graph=sg)
        >>> results = query.find_objects("red cup")
        >>> for r in results:
        ...     print(f"{r.node.id}: {r.similarity:.2f}")
    """
    
    def __init__(
        self,
        scene_graph: SceneGraph,
        clip_model: str = "clip-vit-b32",
        uma: Optional[UMA] = None,
        top_k: int = 5,
    ):
        """Initialize multi-modal query."""
        self.scene_graph = scene_graph
        self.clip_model_name = clip_model
        self.uma = uma
        self.top_k = top_k
        
        # Lazy load CLIP
        self._clip = None
    
    def _get_clip(self) -> CLIPEmbedder:
        """Lazy load CLIP embedder."""
        if self._clip is None:
            self._clip = CLIPEmbedder(model_name=self.clip_model_name)
        return self._clip
    
    def find_objects(
        self,
        text_query: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.0,
        layer: Optional[Any] = None,
    ) -> List[QueryResult]:
        """
        Find objects matching text description.
        
        Args:
            text_query: Natural language query
            top_k: Number of results (default: self.top_k)
            min_similarity: Minimum similarity threshold
            layer: Optional layer filter
        
        Returns:
            List of QueryResult sorted by similarity
        
        Example:
            >>> results = query.find_objects("red cup on table")
            >>> best_match = results[0]
        """
        if top_k is None:
            top_k = self.top_k
        
        # Get CLIP text embedding
        clip = self._get_clip()
        text_emb = clip.encode_text(text_query)
        
        # Get all nodes
        nodes = list(self.scene_graph.nodes.values())
        
        # Filter by layer if specified
        if layer is not None:
            nodes = [n for n in nodes if n.layer == layer]
        
        # Compute similarities
        results = []
        for node in nodes:
            # Get node embedding (from UMA or attributes)
            node_emb = self._get_node_embedding(node)
            
            if node_emb is None:
                continue
            
            # Compute similarity
            similarity = clip.compute_similarity(text_emb, node_emb)
            
            if similarity >= min_similarity:
                results.append(QueryResult(
                    node=node,
                    similarity=similarity,
                    explanation=f"Matched '{text_query}' to {node.attributes.get('class', 'object')}",
                    metadata={"query": text_query}
                ))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]
    
    def ground(
        self,
        text_description: str,
        candidates: Optional[List[Node]] = None,
    ) -> Optional[QueryResult]:
        """
        Visual grounding: find single best object matching description.
        
        Args:
            text_description: Description (e.g., "the leftmost chair")
            candidates: Optional list of candidate nodes
        
        Returns:
            Best matching QueryResult or None
        
        Example:
            >>> result = query.ground("the red cup on the left")
            >>> if result:
            ...     print(f"Found: {result.node.id}")
        """
        # If no candidates, use all nodes
        if candidates is None:
            candidates = list(self.scene_graph.nodes.values())
        
        if not candidates:
            return None
        
        # Find best match
        clip = self._get_clip()
        text_emb = clip.encode_text(text_description)
        
        best_node = None
        best_similarity = -1.0
        
        for node in candidates:
            node_emb = self._get_node_embedding(node)
            if node_emb is None:
                continue
            
            similarity = clip.compute_similarity(text_emb, node_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_node = node
        
        if best_node is None:
            return None
        
        return QueryResult(
            node=best_node,
            similarity=best_similarity,
            explanation=f"Grounded '{text_description}' to {best_node.id}",
            metadata={"description": text_description}
        )
    
    def find_similar(
        self,
        reference_node: Node,
        top_k: Optional[int] = None,
        exclude_self: bool = True,
    ) -> List[QueryResult]:
        """
        Find objects visually similar to reference.
        
        Args:
            reference_node: Reference node
            top_k: Number of results
            exclude_self: Exclude reference from results
        
        Returns:
            List of similar objects
        
        Example:
            >>> cup1 = sg.get_node("cup_1")
            >>> similar = query.find_similar(cup1)
            >>> # Returns other cup instances
        """
        if top_k is None:
            top_k = self.top_k
        
        # Get reference embedding
        ref_emb = self._get_node_embedding(reference_node)
        if ref_emb is None:
            return []
        
        # Compare with all other nodes
        clip = self._get_clip()
        results = []
        
        for node in self.scene_graph.nodes.values():
            if exclude_self and node.id == reference_node.id:
                continue
            
            node_emb = self._get_node_embedding(node)
            if node_emb is None:
                continue
            
            similarity = clip.compute_similarity(ref_emb, node_emb)
            results.append(QueryResult(
                node=node,
                similarity=similarity,
                explanation=f"Similar to {reference_node.id}",
                metadata={"reference": reference_node.id}
            ))
        
        # Sort and return top_k
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]
    
    def semantic_search(
        self,
        class_names: List[str],
        min_confidence: float = 0.3,
    ) -> Dict[str, List[QueryResult]]:
        """
        Zero-shot semantic search for multiple classes.
        
        Args:
            class_names: List of class names to search for
            min_confidence: Minimum confidence threshold
        
        Returns:
            Dict mapping class names to matching nodes
        
        Example:
            >>> results = query.semantic_search(["cup", "plate", "fork"])
            >>> for cls, matches in results.items():
            ...     print(f"{cls}: {len(matches)} found")
        """
        clip = self._get_clip()
        
        # Encode all class names
        class_embs = clip.encode_text(class_names)
        
        # Results dict
        results = {cls: [] for cls in class_names}
        
        # Check each node
        for node in self.scene_graph.nodes.values():
            node_emb = self._get_node_embedding(node)
            if node_emb is None:
                continue
            
            # Compute similarities to all classes
            similarities = [
                clip.compute_similarity(node_emb, cls_emb)
                for cls_emb in class_embs
            ]
            
            # Find best matching class
            best_idx = int(np.argmax(similarities))
            best_sim = similarities[best_idx]
            
            if best_sim >= min_confidence:
                results[class_names[best_idx]].append(QueryResult(
                    node=node,
                    similarity=best_sim,
                    explanation=f"Classified as {class_names[best_idx]}",
                    metadata={"all_scores": dict(zip(class_names, similarities))}
                ))
        
        # Sort each class results
        for cls in class_names:
            results[cls].sort(key=lambda r: r.similarity, reverse=True)
        
        return results
    
    def _get_node_embedding(self, node: Node) -> Optional[np.ndarray]:
        """
        Get CLIP embedding for node.
        
        Tries:
        1. UMA cached embedding
        2. Node attributes embedding
        3. Generate from class name
        
        Args:
            node: Scene graph node
        
        Returns:
            Embedding array or None
        """
        # Try UMA cache
        if self.uma is not None:
            embedding_key = f"obj_{node.id}_emb"
            try:
                data = self.uma.read(embedding_key)
                if data is not None:
                    return data
            except:
                pass
        
        # Try node attributes
        if "embedding" in node.attributes:
            emb = node.attributes["embedding"]
            if hasattr(emb, "features"):
                return emb.features
            elif isinstance(emb, np.ndarray):
                return emb
        
        # Generate from class name
        if "class" in node.attributes:
            clip = self._get_clip()
            class_name = node.attributes["class"]
            text_emb = clip.encode_text(f"a photo of a {class_name}")
            return text_emb.features
        
        return None
