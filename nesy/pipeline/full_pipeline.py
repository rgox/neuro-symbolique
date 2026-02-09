"""
Full Neural-Symbolic Pipeline.

Unified interface combining all components:
- Perception (detection + feature extraction)
- VSA grounding (neural → hypervector)
- Scene graph (world model)
- Reasoning engine (symbolic inference)

Example:
    >>> from nesy.pipeline import NeuralSymbolicPipeline
    >>> 
    >>> pipeline = NeuralSymbolicPipeline()
    >>> result = pipeline.process_image(image)
    >>> answer = pipeline.query("What's in the kitchen?")
"""

from typing import Dict, Any, List, Optional
import numpy as np
import logging

from nesy.core.memory import UMA
from nesy.perception import ObjectDetector, FeatureExtractor, PerceptionPipeline
from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType, RelationType
from nesy.reasoning.vsa import VSACodebook, NeuralGrounding, GroundingCache
from nesy.reasoning.logic import ReasoningEngine


logger = logging.getLogger(__name__)


class NeuralSymbolicPipeline:
    """
    Complete neural-symbolic AI pipeline.
    
    Integrates perception, VSA grounding, scene graph, and reasoning
    into a single unified interface.
    
    Pipeline:
        Image → Perception → VSA → Scene Graph → Reasoning → Answer
    
    Attributes:
        uma: Unified Memory Architecture
        detector: Object detector
        feature_extractor: Feature extractor
        scene_graph: 3D scene graph
        codebook: VSA codebook
        grounding: Neural grounding layer
        reasoning: Reasoning engine
        perception: Perception pipeline
    
    Example:
        >>> pipeline = NeuralSymbolicPipeline(
        ...     detector_backend="mock",
        ...     feature_backend="mock"
        ... )
        >>> 
        >>> # Process image
        >>> result = pipeline.process_image(image)
        >>> 
        >>> # Query
        >>> cups = pipeline.find_all("cup")
        >>> in_kitchen = pipeline.find_in_location("kitchen")
        >>> 
        >>> # Natural language (basic)
        >>> answer = pipeline.query("What's on the table?")
    """
    
    def __init__(
        self,
        detector_backend: str = "mock",
        feature_backend: str = "mock",
        vsa_dim: int = 10000,
        vsa_similarity_threshold: float = 0.5,
        use_uma: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize complete pipeline.
        
        Args:
            detector_backend: Object detector backend (mock, yolo, faster_rcnn)
            feature_backend: Feature extractor backend (mock, clip, resnet, dinov2)
            vsa_dim: VSA hypervector dimensionality
            vsa_similarity_threshold: Threshold for VSA similarity facts
            use_uma: Whether to use Unified Memory Architecture
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Core infrastructure
        self.uma = UMA() if use_uma else None
        
        # World model
        self.scene_graph = SceneGraph(uma=self.uma, logger=self.logger)
        
        # VSA grounding
        self.codebook = VSACodebook(dim=vsa_dim)
        self.grounding = NeuralGrounding(
            neural_dim=512,  # Standard feature dim
            vsa_dim=vsa_dim,
            codebook=self.codebook,
            uma=self.uma
        )
        self.grounding_cache = GroundingCache(self.uma) if self.uma else None
        
        # Reasoning
        self.reasoning = ReasoningEngine(
            scene_graph=self.scene_graph,
            vsa_codebook=self.codebook,
            vsa_similarity_threshold=vsa_similarity_threshold
        )
        
        # Integrated perception pipeline
        self.perception = PerceptionPipeline(
            scene_graph=self.scene_graph,
            uma=self.uma,
            logger=self.logger,
            detector_backend=detector_backend,
            feature_backend=feature_backend
        )
        
        self.logger.info("NeuralSymbolicPipeline initialized successfully")
    
    def process_image(
        self,
        image: np.ndarray,
        camera_pose: Optional[np.ndarray] = None,
        ground_vsa: bool = True,
        sync_reasoning: bool = True
    ) -> Dict[str, Any]:
        """
        Process image through complete pipeline.
        
        Args:
            image: Input image (H, W, 3)
            camera_pose: Optional camera pose [x, y, z]
            ground_vsa: Whether to ground features to VSA
            sync_reasoning: Whether to sync to reasoning engine
        
        Returns:
            Dictionary containing:
            - detections: List of detected objects
            - num_nodes: Number of scene graph nodes
            - num_edges: Number of scene graph edges
            - num_facts: Number of reasoning facts (if synced)
            - processing_time: Time taken (ms)
        
        Example:
            >>> result = pipeline.process_image(image)
            >>> print(f"Detected {len(result['detections'])} objects")
        """
        import time
        start_time = time.time()
        
        # 1. Perception (detection + features + scene graph update)
        perception_result = self.perception.process_image(
            image=image,
            camera_pose=camera_pose
        )
        
        # 2. VSA grounding (optional)
        if ground_vsa:
            self._ground_detections_to_vsa(perception_result.detections)
        
        # 3. Sync to reasoning (optional)
        num_facts = 0
        if sync_reasoning:
            self.reasoning.sync_from_scene_graph()
            stats = self.reasoning.get_statistics()
            num_facts = stats["num_facts"]
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        sg_stats = self.scene_graph.get_statistics()
        
        return {
            "detections": perception_result.detections,
            "num_nodes": sg_stats["total_nodes"],
            "num_edges": sg_stats["total_edges"],
            "num_facts": num_facts,
            "processing_time": processing_time,
        }
    
    def _ground_detections_to_vsa(self, detections: List[Any]) -> None:
        """Ground detections to VSA hypervectors."""
        for detection in detections:
            if detection.embedding is None:
                continue
            
            # Ground embedding
            hv_grounded = self.grounding.ground_embedding(
                embedding=detection.embedding,
                attributes={"class": detection.class_name},
                object_class=detection.class_name,
                position=detection.position_3d if hasattr(detection, 'position_3d') else None
            )
            
            # Store in cache (UMA)
            if self.grounding_cache:
                # Find corresponding scene graph node (simplified)
                # In practice, would use object_id from tracking
                uma_key = self.grounding_cache.store(
                    object_id=f"det_{hash(detection.class_name)}",
                    hv=hv_grounded
                )
                
                # Link to scene graph node
                # (Would need proper node lookup in production)
    
    def query(self, query_str: str) -> List[str]:
        """
        Natural language query (simple pattern matching).
        
        Supports patterns:
        - "What's in the [location]?" → find_in_location
        - "Find all [class]" → find_all
        - "What's on the [object]?" → query spatial relations
        
        Args:
            query_str: Natural language query
        
        Returns:
            List of object IDs or class names
        
        Example:
            >>> pipeline.query("What's in the kitchen?")
            >>> # → ["cup1", "table1", ...]
            >>> 
            >>> pipeline.query("Find all cups")
            >>> # → ["cup1", "cup2"]
        """
        query_lower = query_str.lower()
        
        # Pattern: "What's in the [location]?"
        if "in the" in query_lower:
            location = query_lower.split("in the")[-1].strip(" ?")
            location_nodes = self.scene_graph.query_by_attributes(class_=location)
            if location_nodes:
                return self.reasoning.find_in_location(location_nodes[0].id)
        
        # Pattern: "Find all [class]"
        if "find all" in query_lower:
            obj_class = query_lower.split("find all")[-1].strip(" ?")
            return self.reasoning.find_all(obj_class)
        
        # Pattern: "What's on the [object]?"
        if "on the" in query_lower:
            target = query_lower.split("on the")[-1].strip(" ?")
            target_nodes = self.scene_graph.query_by_attributes(class_=target)
            if target_nodes:
                on_relations = self.reasoning.query("on")
                return [src for src, dst in on_relations if dst == target_nodes[0].id]
        
        self.logger.warning(f"Could not parse query: {query_str}")
        return []
    
    def find_all(self, object_class: str) -> List[str]:
        """Find all objects of a given class."""
        return self.reasoning.find_all(object_class)
    
    def find_in_location(self, location_id: str) -> List[str]:
        """Find all objects in a location (with transitivity)."""
        return self.reasoning.find_in_location(location_id)
    
    def infer(self, goal: str) -> bool:
        """Check if a goal can be inferred."""
        return self.reasoning.infer(goal)
    
    def find_similar_to(
        self,
        object_id: str,
        min_similarity: float = 0.7
    ) -> List[tuple]:
        """Find objects similar to given object (via VSA)."""
        return self.reasoning.find_similar_to(object_id, min_similarity)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "scene_graph": self.scene_graph.get_statistics(),
            "reasoning": self.reasoning.get_statistics(),
            "codebook_size": self.codebook.size(),
        }
    
    def reset(self) -> None:
        """Reset pipeline (clear scene graph and reasoning)."""
        self.scene_graph.clear()
        self.reasoning.sync_from_scene_graph()
        self.logger.info("Pipeline reset")
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"NeuralSymbolicPipeline("
            f"nodes={stats['scene_graph']['total_nodes']}, "
            f"edges={stats['scene_graph']['total_edges']}, "
            f"facts={stats['reasoning']['num_facts']})"
        )
