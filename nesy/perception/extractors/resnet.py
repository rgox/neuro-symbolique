"""ResNet feature extractor backend (requires torchvision)."""

from typing import Optional, Tuple
import numpy as np

from nesy.core.registry import register_plugin
from nesy.perception.base import FeatureExtractorBase
from nesy.perception.feature_extraction import FeatureVector


@register_plugin(FeatureExtractorBase, "resnet")
class ResNetExtractor(FeatureExtractorBase):
    """
    ResNet-50 feature extractor.

    Requires: pip install torch torchvision
    """

    def __init__(
        self,
        device: str = "cpu",
        normalize: bool = True,
        **kwargs,
    ):
        self.device = device
        self._normalize = normalize
        self._model = None
        self._preprocess = None

    def _load(self):
        if self._model is None:
            try:
                import torch
                import torchvision.transforms as transforms
                from torchvision.models import resnet50, ResNet50_Weights
            except ImportError:
                raise ImportError("torch and torchvision required for ResNet backend.")
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
            self._model = torch.nn.Sequential(*list(model.children())[:-1])
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

        fv = FeatureVector(features=features, dim=len(features), model="resnet50", normalized=False)
        if self._normalize:
            fv = fv.normalize()
        return fv

    def get_feature_dim(self) -> int:
        return 2048
