"""Mock object detector for testing and development."""

from typing import List, Optional, Set
import numpy as np

from nesy.core.registry import register_plugin
from nesy.perception.base import ObjectDetectorBase
from nesy.perception.object_detection import Detection, COCO_NAMES


@register_plugin(ObjectDetectorBase, "mock")
class MockDetector(ObjectDetectorBase):
    """
    Mock detector returning deterministic fake detections.

    Useful for testing pipelines without loading real models.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        class_filter: Optional[List[str]] = None,
        **kwargs,
    ):
        self.confidence_threshold = confidence_threshold
        self.class_filter: Optional[Set[str]] = (
            set(class_filter) if class_filter else None
        )

    def detect(
        self,
        image: np.ndarray,
        return_embeddings: bool = False,
    ) -> List[Detection]:
        h, w = image.shape[:2]
        detections = [
            Detection(
                class_id=56, class_name="chair", confidence=0.95,
                bbox=(100, 100, 300, 400), image_size=(w, h),
            ),
            Detection(
                class_id=41, class_name="cup", confidence=0.88,
                bbox=(350, 150, 450, 250), image_size=(w, h),
            ),
        ]
        if self.class_filter:
            detections = [d for d in detections if d.class_name in self.class_filter]
        return [d for d in detections if d.confidence >= self.confidence_threshold]

    def get_class_names(self) -> List[str]:
        return COCO_NAMES
