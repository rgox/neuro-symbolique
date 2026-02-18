"""Object detection backends."""

from nesy.perception.detectors.mock import MockDetector
from nesy.perception.detectors.yolo import YOLODetector
from nesy.perception.detectors.faster_rcnn import FasterRCNNDetector

__all__ = ["MockDetector", "YOLODetector", "FasterRCNNDetector"]
