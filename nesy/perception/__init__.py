"""
Perception Module.

This module provides perception capabilities for the neuro-symbolic platform:
- Object detection (YOLO, Faster R-CNN, etc.)
- Feature extraction (CLIP, ResNet, DINOv2, etc.)
- Perception pipeline (detection → features → scene graph)

The perception module bridges raw sensory data (images, video) with the
symbolic scene graph representation.

Example:
    >>> from nesy.perception import PerceptionPipeline
    >>> pipeline = PerceptionPipeline(
    ...     scene_graph=sg,
    ...     uma=uma,
    ...     detector_backend="mock",
    ...     feature_backend="mock"
    ... )
    >>> image = np.array(Image.open("scene.jpg"))
    >>> result = pipeline.process_image(image)
"""

from nesy.perception.object_detection import (
    ObjectDetector,
    Detection,
    DetectionBackend,
)
from nesy.perception.feature_extraction import (
    FeatureExtractor,
    FeatureVector,
    FeatureBackend,
)
from nesy.perception.pipeline import (
    PerceptionPipeline,
    PerceptionUpdate,
)

__all__ = [
    # Object Detection
    "ObjectDetector",
    "Detection",
    "DetectionBackend",
    # Feature Extraction
    "FeatureExtractor",
    "FeatureVector",
    "FeatureBackend",
    # Pipeline
    "PerceptionPipeline",
    "PerceptionUpdate",
]
