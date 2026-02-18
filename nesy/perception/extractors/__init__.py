"""Feature extraction backends."""

from nesy.perception.extractors.mock import MockFeatureExtractor
from nesy.perception.extractors.clip import CLIPExtractor
from nesy.perception.extractors.resnet import ResNetExtractor
from nesy.perception.extractors.dinov2 import DINOv2Extractor

__all__ = ["MockFeatureExtractor", "CLIPExtractor", "ResNetExtractor", "DINOv2Extractor"]
