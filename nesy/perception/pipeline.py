"""
Perception Pipeline Module.

This module integrates object detection and feature extraction into a unified
perception pipeline that automatically updates the scene graph.

The pipeline:
1. Detect objects in image (ObjectDetector)
2. Extract visual features (FeatureExtractor)
3. Store embeddings in UMA (zero-copy)
4. Update scene graph (Objects layer L1)
5. Track objects across frames
6. Compute spatial relations

Example:
    >>> pipeline = PerceptionPipeline(
    ...     scene_graph=sg,
    ...     uma=uma,
    ...     detector_backend="mock",
    ...     feature_backend="mock"
    ... )
    >>> image = np.array(Image.open("scene.jpg"))
    >>> camera_pose = [0, 0, 1.5]  # Camera position in world coordinates
    >>> updates = pipeline.process_image(image, camera_pose=camera_pose)
    >>> print(f"Added {len(updates['objects_added'])} objects to scene graph")
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import torch
from dataclasses import dataclass
import time

from nesy.perception.object_detection import ObjectDetector, Detection
from nesy.perception.feature_extraction import FeatureExtractor, FeatureVector
from nesy.world_model.scene_graph import (
    SceneGraph,
    ObjectLayer,
    RoomLayer,
    Node,
    LayerType,
    NodeType,
    RelationType,
)
from nesy.core.memory import UMA, DeviceType, DataType


@dataclass
class PerceptionUpdate:
    """
    Result of processing an image through the perception pipeline.
    
    Attributes:
        objects_added: List of newly added object node IDs
        objects_updated: List of updated object node IDs
        embeddings_stored: List of embedding keys stored in UMA
        spatial_relations: Number of spatial relations computed
        processing_time: Time taken to process (seconds)
        detections: List of raw detections
    """
    objects_added: List[str]
    objects_updated: List[str]
    embeddings_stored: List[str]
    spatial_relations: int
    processing_time: float
    detections: List[Detection]


class PerceptionPipeline:
    """
    Unified perception pipeline for scene understanding.
    
    This pipeline integrates:
    - Object detection (YOLO/Faster R-CNN/etc.)
    - Feature extraction (CLIP/ResNet/etc.)
    - Scene graph updates
    - UMA embedding storage
    - Spatial relation inference
    
    The pipeline can process:
    - Single images
    - Video streams (with tracking)
    - Multi-camera setups
    
    Example:
        >>> pipeline = PerceptionPipeline(
        ...     scene_graph=scene_graph,
        ...     uma=uma,
        ...     logger=logger
        ... )
        >>> image = load_image("scene.jpg")
        >>> result = pipeline.process_image(image)
    """
    
    def __init__(
        self,
        scene_graph: SceneGraph,
        uma: Optional[UMA] = None,
        logger: Optional[Any] = None,
        detector_backend: str = "mock",
        detector_model: str = "yolov8n",
        detector_confidence: float = 0.5,
        feature_backend: str = "mock",
        feature_model: str = "ViT-B/32",
        device: str = "cpu",
        track_objects: bool = True,
        compute_relations: bool = True,
    ):
        """
        Initialize perception pipeline.
        
        Args:
            scene_graph: Scene graph to update
            uma: Unified Memory Architecture for embeddings
            logger: Optional telemetry logger
            detector_backend: Object detection backend
            detector_model: Detection model variant
            detector_confidence: Minimum detection confidence
            feature_backend: Feature extraction backend
            feature_model: Feature model variant
            device: Device to run on ("cpu", "cuda", etc.)
            track_objects: Whether to track objects across frames
            compute_relations: Whether to compute spatial relations
        """
        self.scene_graph = scene_graph
        self.uma = uma
        self.logger = logger
        
        # Initialize detector
        self.detector = ObjectDetector(
            backend=detector_backend,
            model=detector_model,
            confidence_threshold=detector_confidence,
            device=device,
        )
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            backend=feature_backend,
            model=feature_model,
            device=device,
            normalize=True,
        )
        
        # Layer interfaces
        self.object_layer = ObjectLayer(scene_graph)
        
        # Configuration
        self.track_objects = track_objects
        self.compute_relations = compute_relations
        
        # Tracking state
        self.object_tracking: Dict[str, Node] = {}  # class_name -> last_node
        
        if logger:
            logger.info(
                "PerceptionPipeline initialized",
                event_type="PERCEPTION",
                data={
                    "detector": f"{detector_backend}:{detector_model}",
                    "feature_extractor": f"{feature_backend}:{feature_model}",
                    "device": device,
                }
            )
    
    def process_image(
        self,
        image: np.ndarray,
        camera_pose: Optional[np.ndarray] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
    ) -> PerceptionUpdate:
        """
        Process an image through the perception pipeline.
        
        This method:
        1. Detects objects in the image
        2. Extracts visual features for each object
        3. Stores embeddings in UMA
        4. Updates scene graph with detected objects
        5. Optionally computes spatial relations
        
        Args:
            image: Input image as numpy array (H, W, 3) RGB
            camera_pose: Optional camera position [x, y, z] in world coordinates
            image_metadata: Optional metadata (timestamp, camera_id, etc.)
        
        Returns:
            PerceptionUpdate with processing results
        """
        start_time = time.time()
        
        objects_added = []
        objects_updated = []
        embeddings_stored = []
        
        # Step 1: Detect objects
        detections = self.detector.detect(image, return_embeddings=False)
        
        if self.logger:
            self.logger.debug(
                f"Detected {len(detections)} objects",
                event_type="PERCEPTION",
                data={"num_detections": len(detections)}
            )
        
        # Step 2: Process each detection
        for det in detections:
            # Extract visual features
            feature_vec = self.feature_extractor.extract(
                image,
                bbox=det.bbox
            )
            
            # Store embedding in UMA if available
            embedding_key = None
            if self.uma:
                embedding_key = f"obj_{det.class_name}_{len(objects_added)}_emb"
                try:
                    self.uma.allocate(
                        key=embedding_key,
                        shape=(feature_vec.dim,),
                        dtype=DataType.FLOAT32,
                        device=self.uma.device,  # Use UMA's actual device (CPU/NPU)
                        metadata={
                            "class": det.class_name,
                            "model": feature_vec.model,
                            "bbox": list(det.bbox),
                        }
                    )
                    # Copy features to UMA buffer
                    buffer = self.uma.get(embedding_key)
                    # Convert numpy array to torch tensor if needed
                    if isinstance(feature_vec.features, np.ndarray):
                        buffer.data[:] = torch.from_numpy(feature_vec.features)
                    else:
                        buffer.data[:] = feature_vec.features

                    embeddings_stored.append(embedding_key)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to store embedding: {e}")
            
            # Estimate 3D position from 2D detection
            if camera_pose is None:
                camera_pose = np.array([0, 0, 1.5])
            position_3d = self._estimate_3d_position(
                det,
                camera_pose=camera_pose,
                image_size=det.image_size,
            )
            
            # Add/update object in scene graph
            object_node = self.object_layer.create_object(
                class_name=det.class_name,
                position=position_3d,
                bbox_min=position_3d - np.array([0.1, 0.1, 0.1]),  # Rough bbox
                bbox_max=position_3d + np.array([0.1, 0.1, 0.1]),
                confidence=det.confidence,
                embedding_key=embedding_key,
                detected_bbox_2d=list(det.bbox),
                detection_confidence=det.confidence,
            )
            
            objects_added.append(object_node.id)
            
            # Track object
            if self.track_objects:
                self.object_tracking[det.class_name] = object_node
        
        # Step 3: Compute spatial relations if requested
        num_relations = 0
        if self.compute_relations and len(objects_added) > 1:
            from nesy.world_model.scene_graph.layers import compute_spatial_relations
            num_relations = compute_spatial_relations(
                self.scene_graph,
                threshold_on=0.1,
                threshold_near=1.0,
            )
        
        # Processing time
        processing_time = time.time() - start_time
        
        if self.logger:
            self.logger.info(
                f"Processed image: {len(objects_added)} objects, {num_relations} relations",
                event_type="PERCEPTION",
                data={
                    "objects_added": len(objects_added),
                    "embeddings_stored": len(embeddings_stored),
                    "spatial_relations": num_relations,
                    "processing_time_ms": processing_time * 1000,
                }
            )
        
        return PerceptionUpdate(
            objects_added=objects_added,
            objects_updated=objects_updated,
            embeddings_stored=embeddings_stored,
            spatial_relations=num_relations,
            processing_time=processing_time,
            detections=detections,
        )
    
    def _estimate_3d_position(
        self,
        detection: Detection,
        camera_pose: np.ndarray,
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Estimate 3D position from 2D detection.
        
        This is a simple heuristic  for MVP. In full version, this would use:
        - Depth estimation (monocular/stereo/RGB-D)
        - Camera intrinsics/extrinsics
        - Object size priors
        
        Args:
            detection: Detection object
            camera_pose: Camera position [x, y, z]
            image_size: Image size (width, height)
        
        Returns:
            Estimated 3D position [x, y, z]
        """
        # Get 2D center in normalized coordinates [-1, 1]
        center_x, center_y = detection.get_bbox_center()
        img_w, img_h = image_size
        
        norm_x = (center_x / img_w) * 2 - 1  # [-1, 1]
        norm_y = (center_y / img_h) * 2 - 1  # [-1, 1]
        
        # Simple heuristic: assume object is ~2m in front of camera
        # and map normalized x,y to world coordinates
        depth = 2.0  # meters
        
        # Rough camera projection (assumes FOV ~ 60 degrees)
        world_x = camera_pose[0] + norm_x * depth * 0.5
        world_y = camera_pose[1] + depth
        world_z = camera_pose[2] - norm_y * depth * 0.5
        
        return np.array([world_x, world_y,world_z], dtype=np.float32)
    
    def process_video_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        camera_pose: Optional[np.ndarray] = None,
    ) -> PerceptionUpdate:
        """
        Process a video frame (alias for process_image with frame metadata).
        
        Args:
            frame: Video frame as numpy array
            frame_id: Frame number
            camera_pose: Camera position
        
        Returns:
            PerceptionUpdate
        """
        metadata = {"frame_id": frame_id, "timestamp": time.time()}
        return self.process_image(frame, camera_pose, metadata)
    
    def get_tracked_objects(self) -> Dict[str, Node]:
        """Get currently tracked objects."""
        return self.object_tracking.copy()
    
    def clear_tracking(self):
        """Clear object tracking state."""
        self.object_tracking.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "detector": {
                "backend": self.detector.backend.value,
                "model": self.detector.model_name,
                "confidence_threshold": self.detector.confidence_threshold,
            },
            "feature_extractor": {
                "backend": self.feature_extractor.backend.value,
                "model": self.feature_extractor.model_name,
                "feature_dim": self.feature_extractor.feature_dim,
            },
            "tracked_objects": len(self.object_tracking),
        }
