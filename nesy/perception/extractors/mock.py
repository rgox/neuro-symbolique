"""Mock feature extractor for testing and development."""

from typing import List, Optional, Tuple
import numpy as np

from nesy.core.registry import register_plugin
from nesy.perception.base import FeatureExtractorBase
from nesy.perception.feature_extraction import FeatureVector


@register_plugin(FeatureExtractorBase, "mock")
class MockFeatureExtractor(FeatureExtractorBase):
    """
    Mock extractor returning random feature vectors.

    Useful for testing pipelines without loading real models.
    """

    def __init__(self, feature_dim: int = 512, normalize: bool = True, **kwargs):
        self.feature_dim = feature_dim
        self.normalize = normalize

    def extract(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> FeatureVector:
        features = np.random.randn(self.feature_dim).astype(np.float32)
        fv = FeatureVector(
            features=features,
            dim=self.feature_dim,
            model="mock",
            normalized=False,
        )
        if self.normalize:
            fv = fv.normalize()
        return fv

    def get_feature_dim(self) -> int:
        return self.feature_dim
