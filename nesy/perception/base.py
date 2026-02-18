"""
Perception Base Classes - Abstract Interfaces for Backend-Agnostic Perception.

Defines the ABC contracts for object detection and feature extraction.
Concrete backends (YOLO, CLIP, Mock, etc.) register via the plugin system.

Example:
    >>> from nesy.core.registry import Registry
    >>> from nesy.perception.base import ObjectDetectorBase, FeatureExtractorBase
    >>>
    >>> # List available backends
    >>> Registry.list_plugins(ObjectDetectorBase)
    ['mock', 'yolo', 'faster_rcnn']
    >>>
    >>> # Create via registry
    >>> detector = Registry.create(ObjectDetectorBase, "mock")
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass
from nesy.perception.object_detection import Detection
from nesy.perception.feature_extraction import FeatureVector


class ObjectDetectorBase(ABC):
    """
    Abstract base class for object detectors.

    All detection backends must implement this interface.
    """

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        return_embeddings: bool = False,
    ) -> List[Detection]:
        """
        Detect objects in an image.

        Args:
            image: Input image (H, W, 3) RGB
            return_embeddings: Whether to return visual embeddings

        Returns:
            List of Detection results
        """
        ...

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Return the list of class names this detector can identify."""
        ...


class FeatureExtractorBase(ABC):
    """
    Abstract base class for feature extractors.

    All feature extraction backends must implement this interface.
    """

    @abstractmethod
    def extract(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> FeatureVector:
        """
        Extract features from an image or crop.

        Args:
            image: Input image (H, W, 3) RGB
            bbox: Optional bounding box for crop

        Returns:
            FeatureVector with extracted features
        """
        ...

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        ...

    def extract_batch(self, images: List[np.ndarray]) -> List[FeatureVector]:
        """Extract features from multiple images (default: sequential)."""
        return [self.extract(img) for img in images]
