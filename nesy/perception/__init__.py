"""
Perception Module.

This module provides perception capabilities for the neuro-symbolic platform:
- Object detection (YOLO, Faster R-CNN, etc.)
- Feature extraction (CLIP, ResNet, DINOv2, etc.)
- Perception pipeline (detection -> features -> scene graph)

All backends are registered as plugins via the Registry. Use the ABCs
(ObjectDetectorBase, FeatureExtractorBase) as the primary interfaces.

Example:
    >>> from nesy.core.registry import Registry
    >>> from nesy.perception import ObjectDetectorBase, FeatureExtractorBase
    >>>
    >>> detector = Registry.create(ObjectDetectorBase, "mock")
    >>> extractor = Registry.create(FeatureExtractorBase, "mock")
"""

from nesy.perception.object_detection import (
    ObjectDetector,
    Detection,
    DetectionBackend,
    COCO_NAMES,
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
from nesy.perception.base import (
    ObjectDetectorBase,
    FeatureExtractorBase,
)

# Import plugin modules to trigger @register_plugin decorators
import nesy.perception.detectors  # noqa: F401
import nesy.perception.extractors  # noqa: F401

__all__ = [
    # ABCs (preferred interfaces)
    "ObjectDetectorBase",
    "FeatureExtractorBase",
    # Legacy classes (backward compatible)
    "ObjectDetector",
    "Detection",
    "DetectionBackend",
    "COCO_NAMES",
    # Feature Extraction
    "FeatureExtractor",
    "FeatureVector",
    "FeatureBackend",
    # Pipeline
    "PerceptionPipeline",
    "PerceptionUpdate",
]
