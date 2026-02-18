"""Faster R-CNN object detector backend (requires torchvision)."""

from typing import List, Optional, Set
import numpy as np

from nesy.core.registry import register_plugin
from nesy.perception.base import ObjectDetectorBase
from nesy.perception.object_detection import Detection, COCO_NAMES


@register_plugin(ObjectDetectorBase, "faster_rcnn")
class FasterRCNNDetector(ObjectDetectorBase):
    """
    Faster R-CNN detector using torchvision.

    Requires: pip install torch torchvision
    """

    def __init__(
        self,
        model: str = "fasterrcnn_resnet50_fpn",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "cpu",
        class_filter: Optional[List[str]] = None,
        **kwargs,
    ):
        self.model_name = model
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.class_filter: Optional[Set[str]] = (
            set(class_filter) if class_filter else None
        )
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                import torch
                from torchvision.models.detection import (
                    fasterrcnn_resnet50_fpn,
                    FasterRCNN_ResNet50_FPN_Weights,
                )
            except ImportError:
                raise ImportError(
                    "torch and torchvision required for Faster R-CNN backend."
                )
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self._model = fasterrcnn_resnet50_fpn(weights=weights)
            self._model.eval()
            self._model.to(self.device)
        return self._model

    def detect(
        self,
        image: np.ndarray,
        return_embeddings: bool = False,
    ) -> List[Detection]:
        import torch

        model = self._load_model()
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            predictions = model([img_tensor])[0]

        detections = []
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < self.confidence_threshold:
                continue
            cls_id = int(label) - 1  # COCO labels are 1-indexed
            cls_name = (
                COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES)
                else f"class_{cls_id}"
            )
            if self.class_filter and cls_name not in self.class_filter:
                continue
            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=float(score),
                bbox=tuple(box),
                image_size=(image.shape[1], image.shape[0]),
            ))
        return detections

    def get_class_names(self) -> List[str]:
        return COCO_NAMES
