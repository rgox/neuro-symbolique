"""DINOv2 feature extractor backend (requires torch)."""

from typing import Optional, Tuple
import numpy as np

from nesy.core.registry import register_plugin
from nesy.perception.base import FeatureExtractorBase
from nesy.perception.feature_extraction import FeatureVector


@register_plugin(FeatureExtractorBase, "dinov2")
class DINOv2Extractor(FeatureExtractorBase):
    """
    DINOv2 self-supervised feature extractor.

    Requires: pip install torch
    """

    def __init__(
        self,
        model: str = "dinov2_vitb14",
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
                import torch
                import torchvision.transforms as transforms
            except ImportError:
                raise ImportError("torch required for DINOv2 backend.")
            self._model = torch.hub.load("facebookresearch/dinov2", self.model_name)
            self._model.eval()
            self._model.to(self.device)
            self._preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

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
            features = self._model(image_input).cpu().numpy().flatten().astype(np.float32)

        fv = FeatureVector(features=features, dim=len(features), model=self.model_name, normalized=False)
        if self._normalize:
            fv = fv.normalize()
        return fv

    def get_feature_dim(self) -> int:
        dims = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}
        return dims.get(self.model_name, 768)
