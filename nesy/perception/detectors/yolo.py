"""YOLO object detector backend (requires ultralytics)."""

from typing import List, Optional, Set
import numpy as np

from nesy.core.registry import register_plugin
from nesy.perception.base import ObjectDetectorBase
from nesy.perception.object_detection import Detection, COCO_NAMES


@register_plugin(ObjectDetectorBase, "yolo")
class YOLODetector(ObjectDetectorBase):
    """
    YOLO detector using the ultralytics package.

    Requires: pip install ultralytics
    """

    def __init__(
        self,
        model: str = "yolov8n",
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
                from ultralytics import YOLO
            except ImportError:
                raise ImportError(
                    "ultralytics package required for YOLO backend. "
                    "Install: pip install ultralytics"
                )
            self._model = YOLO(self.model_name)
            self._model.to(self.device)
        return self._model

    def detect(
        self,
        image: np.ndarray,
        return_embeddings: bool = False,
    ) -> List[Detection]:
        model = self._load_model()
        results = model(image, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                if conf < self.confidence_threshold:
                    continue
                cls_name = (
                    COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES)
                    else f"class_{cls_id}"
                )
                if self.class_filter and cls_name not in self.class_filter:
                    continue

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=tuple(box),
                    image_size=(image.shape[1], image.shape[0]),
                ))
        return detections

    def get_class_names(self) -> List[str]:
        return COCO_NAMES
