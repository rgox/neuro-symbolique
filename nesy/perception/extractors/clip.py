"""CLIP feature extractor backend (requires openai-clip)."""

from typing import Optional, Tuple
import numpy as np

from nesy.core.registry import register_plugin
from nesy.perception.base import FeatureExtractorBase
from nesy.perception.feature_extraction import FeatureVector


@register_plugin(FeatureExtractorBase, "clip")
class CLIPExtractor(FeatureExtractorBase):
    """
    CLIP feature extractor.

    Requires: pip install git+https://github.com/openai/CLIP.git
    """

    def __init__(
        self,
        model: str = "ViT-B/32",
        device: str = "cpu",
        normalize: bool = True,
        **kwargs,
    ):
        self.model_name = model
        self.device = device
        self._normalize = normalize
        self._model = None
        self._preprocess = None

    def _load(self):
        if self._model is None:
            try:
                import clip
                import torch
            except ImportError:
                raise ImportError(
                    "clip and torch required for CLIP backend."
                )
            self._model, self._preprocess = clip.load(self.model_name, device=self.device)
            self._model.eval()

    def extract(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> FeatureVector:
        import torch
        from PIL import Image

        self._load()

        if bbox is not None:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            image = image[y1:y2, x1:x2]

        pil_image = Image.fromarray(image)
        image_input = self._preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model.encode_image(image_input)
            features = features.cpu().numpy().flatten().astype(np.float32)

        fv = FeatureVector(
            features=features, dim=len(features),
            model=f"clip:{self.model_name}", normalized=False,
        )
        if self._normalize:
            fv = fv.normalize()
        return fv

    def get_feature_dim(self) -> int:
        dims = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768, "RN50": 1024}
        return dims.get(self.model_name, 512)
