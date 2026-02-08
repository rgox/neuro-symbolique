"""
Object Detection Module.

This module provides object detection capabilities for the neuro-symbolic platform.
It serves as the neural component of the perception pipeline, detecting objects in
images/video and extracting their bounding boxes, classes, and confidence scores.

Architecture:
    - Abstract ObjectDetector interface
    - Multiple backend support (YOLO, Faster R-CNN, etc.)
    - Configurable detection parameters
    - Integration with HAL for NPU execution
    - Output format compatible with scene graph

For MVP, we provide a simple mock detector for testing and a torchvision-based
detector using pretrained models.

Example:
    >>> detector = TorchvisionDetector(model_name="fasterrcnn_resnet50_fpn")
    >>> image = load_image("kitchen.jpg")
    >>> detections = detector.detect(image, confidence_threshold=0.5)
    >>> for det in detections:
    ...     print(f"{det.class_name} at {det.bbox} (conf: {det.confidence})")
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import time

from nesy.core.telemetry import TelemetryLogger, EventType


class ObjectClass(Enum):
    """Common object classes (COCO dataset categories)."""
    PERSON = "person"
    BICYCLE = "bicycle"
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    AIRPLANE = "airplane"
    BUS = "bus"
    TRAIN = "train"
    TRUCK = "truck"
    BOAT = "boat"

    # Furniture
    CHAIR = "chair"
    COUCH = "couch"
    BED = "bed"
    DINING_TABLE = "dining table"
    TOILET = "toilet"

    # Kitchen items
    CUP = "cup"
    FORK = "fork"
    KNIFE = "knife"
    SPOON = "spoon"
    BOWL = "bowl"
    BOTTLE = "bottle"
    WINE_GLASS = "wine glass"

    # Electronics
    TV = "tv"
    LAPTOP = "laptop"
    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    CELL_PHONE = "cell phone"

    # Animals
    DOG = "dog"
    CAT = "cat"
    BIRD = "bird"

    # Other common objects
    BOOK = "book"
    CLOCK = "clock"
    VASE = "vase"
    SCISSORS = "scissors"

    UNKNOWN = "unknown"


@dataclass
class Detection:
    """
    Single object detection result.

    Attributes:
        bbox: Bounding box [x_min, y_min, x_max, y_max] in pixels
        class_id: Integer class ID
        class_name: Human-readable class name
        confidence: Detection confidence score [0, 1]
        mask: Optional segmentation mask
        features: Optional feature vector/embedding
        metadata: Additional metadata (model info, etc.)
    """
    bbox: np.ndarray  # [x_min, y_min, x_max, y_max]
    class_id: int
    class_name: str
    confidence: float
    mask: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            "bbox": self.bbox.tolist(),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "has_mask": self.mask is not None,
            "has_features": self.features is not None,
            "metadata": self.metadata or {},
        }

    def get_center(self) -> np.ndarray:
        """Get center point of bounding box."""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        ])

    def get_area(self) -> float:
        """Get area of bounding box."""
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


class ObjectDetector(ABC):
    """
    Abstract base class for object detectors.

    All object detection models should inherit from this class and implement
    the detect() method.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        max_detections: int = 100,
        logger: Optional[TelemetryLogger] = None,
    ):
        """
        Initialize object detector.

        Args:
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
            max_detections: Maximum number of detections to return
            logger: Optional telemetry logger
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.logger = logger

        # Statistics
        self.stats = {
            "total_images": 0,
            "total_detections": 0,
            "avg_inference_time_ms": 0.0,
        }

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Detect objects in an image.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            confidence_threshold: Optional override for confidence threshold

        Returns:
            List of Detection objects

        Raises:
            RuntimeError: If detection fails
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get detector capabilities (classes, input size, etc.)."""
        pass

    def _update_statistics(self, num_detections: int, inference_time_ms: float):
        """Update detection statistics."""
        self.stats["total_images"] += 1
        self.stats["total_detections"] += num_detections

        # Running average
        n = self.stats["total_images"]
        old_avg = self.stats["avg_inference_time_ms"]
        self.stats["avg_inference_time_ms"] = (
            old_avg * (n - 1) + inference_time_ms
        ) / n


class MockDetector(ObjectDetector):
    """
    Mock object detector for testing.

    This detector generates random detections for testing purposes.
    It's useful for development when real detection models are not available.
    """

    def __init__(
        self,
        num_objects: int = 5,
        image_size: Tuple[int, int] = (640, 480),
        **kwargs,
    ):
        """
        Initialize mock detector.

        Args:
            num_objects: Number of random objects to generate
            image_size: Expected image size (width, height)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)
        self.num_objects = num_objects
        self.image_size = image_size

        if self.logger:
            self.logger.info(
                "MockDetector initialized",
                event_type=EventType.DEVICE,
                data={"num_objects": num_objects},
            )

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """Generate random detections."""
        start_time = time.time()

        threshold = confidence_threshold or self.confidence_threshold
        h, w = image.shape[:2]

        detections = []

        # Generate random detections
        common_classes = [
            (1, "cup"),
            (2, "table"),
            (3, "chair"),
            (4, "bottle"),
            (5, "bowl"),
        ]

        for i in range(self.num_objects):
            # Random bbox
            x1 = np.random.randint(0, w - 50)
            y1 = np.random.randint(0, h - 50)
            x2 = x1 + np.random.randint(30, 100)
            y2 = y1 + np.random.randint(30, 100)

            # Clip to image bounds
            x2 = min(x2, w)
            y2 = min(y2, h)

            # Random class
            class_id, class_name = common_classes[i % len(common_classes)]

            # Random confidence above threshold
            confidence = threshold + np.random.rand() * (1.0 - threshold)

            detection = Detection(
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidence),
                metadata={"detector": "mock"},
            )

            detections.append(detection)

        inference_time = (time.time() - start_time) * 1000
        self._update_statistics(len(detections), inference_time)

        if self.logger:
            self.logger.debug(
                f"MockDetector: {len(detections)} detections",
                event_type=EventType.PERCEPTION,
                data={"num_detections": len(detections)},
            )

        return detections

    def get_capabilities(self) -> Dict[str, Any]:
        """Get mock detector capabilities."""
        return {
            "detector_type": "mock",
            "num_classes": 5,
            "classes": ["cup", "table", "chair", "bottle", "bowl"],
            "image_size": self.image_size,
            "supports_segmentation": False,
        }


class TorchvisionDetector(ObjectDetector):
    """
    Object detector using torchvision pretrained models.

    Supports models like:
    - Faster R-CNN ResNet50 FPN
    - RetinaNet ResNet50 FPN
    - SSD300 VGG16
    - FCOS ResNet50 FPN

    Uses COCO pretrained weights (80 classes).
    """

    # COCO class names
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
        'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
        'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(
        self,
        model_name: str = "fasterrcnn_resnet50_fpn",
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize torchvision detector.

        Args:
            model_name: Model name (e.g., "fasterrcnn_resnet50_fpn")
            device: Device to run on ("cpu" or "cuda")
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.device = device
        self.model = None

        # Load model lazily
        self._load_model()

        if self.logger:
            self.logger.info(
                f"TorchvisionDetector initialized: {model_name}",
                event_type=EventType.DEVICE,
                data={"model": model_name, "device": device},
            )

    def _load_model(self):
        """Load pretrained model from torchvision."""
        try:
            import torch
            import torchvision
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn,
                FasterRCNN_ResNet50_FPN_Weights,
            )

            # Load model with pretrained weights
            if self.model_name == "fasterrcnn_resnet50_fpn":
                weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
                self.model = fasterrcnn_resnet50_fpn(weights=weights)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            self.model.eval()
            self.model.to(self.device)

            self.torch_device = torch.device(self.device)

        except ImportError as e:
            if self.logger:
                self.logger.error(
                    f"Failed to load torchvision: {e}",
                    event_type=EventType.ERROR,
                )
            raise RuntimeError(
                "PyTorch/torchvision not available. "
                "Install with: pip install torch torchvision"
            )

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """Detect objects using torchvision model."""
        import torch

        start_time = time.time()
        threshold = confidence_threshold or self.confidence_threshold

        # Preprocess image
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.torch_device)

        # Inference
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]

        # Post-process
        detections = []

        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < threshold:
                continue

            class_name = self.COCO_CLASSES[label] if label < len(self.COCO_CLASSES) else "unknown"

            detection = Detection(
                bbox=box.astype(np.float32),
                class_id=int(label),
                class_name=class_name,
                confidence=float(score),
                metadata={
                    "detector": self.model_name,
                    "device": self.device,
                },
            )

            detections.append(detection)

            if len(detections) >= self.max_detections:
                break

        inference_time = (time.time() - start_time) * 1000
        self._update_statistics(len(detections), inference_time)

        if self.logger:
            self.logger.info(
                f"Detected {len(detections)} objects in {inference_time:.2f}ms",
                event_type=EventType.PERCEPTION,
                data={"num_detections": len(detections)},
            )

        return detections

    def get_capabilities(self) -> Dict[str, Any]:
        """Get torchvision detector capabilities."""
        return {
            "detector_type": "torchvision",
            "model_name": self.model_name,
            "num_classes": len(self.COCO_CLASSES),
            "classes": self.COCO_CLASSES,
            "device": self.device,
            "supports_segmentation": False,
        }


def load_detector(
    detector_type: str = "mock",
    **kwargs,
) -> ObjectDetector:
    """
    Factory function to load a detector.

    Args:
        detector_type: Type of detector ("mock", "torchvision", etc.)
        **kwargs: Additional arguments for detector

    Returns:
        ObjectDetector instance

    Example:
        >>> detector = load_detector("mock", num_objects=10)
        >>> detector = load_detector("torchvision", model_name="fasterrcnn_resnet50_fpn")
    """
    if detector_type == "mock":
        return MockDetector(**kwargs)
    elif detector_type == "torchvision":
        return TorchvisionDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def non_maximum_suppression(
    detections: List[Detection],
    iou_threshold: float = 0.5,
) -> List[Detection]:
    """
    Apply non-maximum suppression to detections.

    Args:
        detections: List of detections
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

    keep = []

    while len(detections) > 0:
        # Keep highest confidence detection
        best = detections[0]
        keep.append(best)
        detections = detections[1:]

        # Remove overlapping detections
        filtered = []
        for det in detections:
            iou = compute_iou(best.bbox, det.bbox)
            if iou < iou_threshold:
                filtered.append(det)

        detections = filtered

    return keep


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bbox [x1, y1, x2, y2]
        bbox2: Second bbox [x1, y1, x2, y2]

    Returns:
        IoU value [0, 1]
    """
    # Intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return float(intersection / union) if union > 0 else 0.0
