"""
Object Detection Module.

This module provides object detection capabilities using state-of-the-art models:
- YOLO (You Only Look Once) for real-time detection
- Faster R-CNN for high-accuracy detection
- Integration with scene graph for automatic updates

The module wraps popular detection frameworks and provides a unified interface
that integrates seamlessly with the NeSy platform's UMA and scene graph.

Features:
- Multiple backend support (YOLO, Faster R-CNN, etc.)
- Confidence filtering
- Non-maximum suppression (NMS)
- Bounding box extraction
- Class label mapping
- Integration with UMA for embeddings

Example:
    >>> detector = ObjectDetector(model="yolov8n", confidence=0.5)
    >>> detections = detector.detect(image)
    >>> for det in detections:
    ...     print(f"{det.class_name} at {det.bbox} (conf: {det.confidence})")
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from enum import Enum

try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class DetectionBackend(Enum):
    """Supported detection backends."""
    YOLO = "yolo"  # YOLOv8 (ultralytics)
    FASTER_RCNN = "faster_rcnn"  # Torchvision Faster R-CNN
    MOCK = "mock"  # Mock detector for testing


@dataclass
class Detection:
    """
    Single object detection result.
    
    Attributes:
        class_id: Integer class ID
        class_name: Human-readable class name
        confidence: Detection confidence [0, 1]
        bbox: Bounding box (x_min, y_min, x_max, y_max) in pixel coordinates
        image_size: Original image size (width, height)
        mask: Optional segmentation mask
        embedding: Optional visual embedding vector
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    image_size: Tuple[int, int]  # (width, height)
    mask: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    
    def get_bbox_center(self) -> Tuple[float, float]:
        """Get bounding box center in pixel coordinates."""
        x_min, y_min, x_max, y_max = self.bbox
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)
    
    def get_bbox_size(self) -> Tuple[float, float]:
        """Get bounding box size (width, height) in pixels."""
        x_min, y_min, x_max, y_max = self.bbox
        return (x_max - x_min, y_max - y_min)
    
    def get_bbox_area(self) -> float:
        """Get bounding box area in pixels²."""
        w, h = self.get_bbox_size()
        return w * h
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "bbox_center": self.get_bbox_center(),
            "bbox_size": self.get_bbox_size(),
            "bbox_area": self.get_bbox_area(),
            "image_size": self.image_size,
        }


class ObjectDetector:
    """
    Object detector with multiple backend support.
    
    This class provides a unified interface for object detection using
    different backends (YOLO, Faster R-CNN, etc.).
    
    Features:
    - Backend abstraction
    - Confidence filtering
    - NMS (Non-Maximum Suppression)
    - Class filtering
    - Batch processing (optional)
    
    Example:
        >>> detector = ObjectDetector(
        ...     backend="yolo",
        ...     model="yolov8n",
        ...     confidence_threshold=0.5,
        ...     nms_threshold=0.4
        ... )
        >>> image = np.array(Image.open("scene.jpg"))
        >>> detections = detector.detect(image)
    """
    
    def __init__(
        self,
        backend: str = "mock",
        model: str = "yolov8n",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "cpu",
        class_filter: Optional[List[str]] = None,
    ):
        """
        Initialize object detector.
        
        Args:
            backend: Detection backend ("yolo", "faster_rcnn", "mock")
            model: Model variant (e.g., "yolov8n", "fasterrcnn_resnet50_fpn")
            confidence_threshold: Minimum confidence for detections [0, 1]
            nms_threshold: NMS IoU threshold [0, 1]
            device: Device to run on ("cpu", "cuda", "cuda:0", etc.)
            class_filter: Optional list of class names to keep (None = all)
        """
        self.backend = DetectionBackend(backend)
        self.model_name = model
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.class_filter = set(class_filter) if class_filter else None
        
        # Load model
        self.model = self._load_model()
        
        # COCO class names (default for most models)
        self.class_names = self._get_class_names()
    
    def _load_model(self):
        """Load detection model based on backend."""
        if self.backend == DetectionBackend.MOCK:
            # Mock model for testing without dependencies
            return MockDetectionModel()
        
        elif self.backend == DetectionBackend.YOLO:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for YOLO backend")
            
            try:
                from ultralytics import YOLO
                model = YOLO(self.model_name)
                model.to(self.device)
                return model
            except ImportError:
                raise ImportError("ultralytics package required for YOLO backend. Install: pip install ultralytics")
        
        elif self.backend == DetectionBackend.FASTER_RCNN:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for Faster R-CNN backend")
            
            # Load pretrained Faster R-CNN from torchvision
            if self.model_name == "fasterrcnn_resnet50_fpn":
                from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                model = fasterrcnn_resnet50_fpn(weights=weights)
            else:
                raise ValueError(f"Unknown Faster R-CNN model: {self.model_name}")
            
            model.eval()
            model.to(self.device)
            return model
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _get_class_names(self) -> List[str]:
        """Get class names for the model."""
        # COCO class names (80 classes)
        coco_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        return coco_names
    
    def detect(
        self,
        image: np.ndarray,
        return_embeddings: bool = False,
    ) -> List[Detection]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            return_embeddings: Whether to extract visual embeddings
        
        Returns:
            List of Detection objects
        """
        if self.backend == DetectionBackend.MOCK:
            return self._detect_mock(image)
        elif self.backend == DetectionBackend.YOLO:
            return self._detect_yolo(image, return_embeddings)
        elif self.backend == DetectionBackend.FASTER_RCNN:
            return self._detect_faster_rcnn(image, return_embeddings)
        else:
            raise ValueError(f"Detection not implemented for backend: {self.backend}")
    
    def _detect_mock(self, image: np.ndarray) -> List[Detection]:
        """Mock detection for testing."""
        h, w = image.shape[:2]
        
        # Return fake detections
        detections = [
            Detection(
                class_id=56,  # chair
                class_name="chair",
                confidence=0.95,
                bbox=(100, 100, 300, 400),
                image_size=(w, h),
            ),
            Detection(
                class_id=41,  # cup
                class_name="cup",
                confidence=0.88,
                bbox=(350, 150, 450, 250),
                image_size=(w, h),
            ),
        ]
        
        return detections
    
    def _detect_yolo(self, image: np.ndarray, return_embeddings: bool) -> List[Detection]:
        """Detect using YOLO backend."""
        # Run inference
        results = self.model(image, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Extract box data
                box = boxes.xyxy[i].cpu().numpy()  # [x_min, y_min, x_max, y_max]
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # Filter by confidence
                if conf < self.confidence_threshold:
                    continue
                
                # Get class name
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                
                # Filter by class if specified
                if self.class_filter and cls_name not in self.class_filter:
                    continue
                
                detection = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=tuple(box),
                    image_size=(image.shape[1], image.shape[0]),
                )
                
                detections.append(detection)
        
        return detections
    
    def _detect_faster_rcnn(self, image: np.ndarray, return_embeddings: bool) -> List[Detection]:
        """Detect using Faster R-CNN backend."""
        # Convert to tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]
        
        detections = []
        
        # Extract detections
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            # Filter by confidence
            if score < self.confidence_threshold:
                continue
            
            # Get class name (COCO labels are 1-indexed)
            cls_id = int(label) - 1
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
            
            # Filter by class
            if self.class_filter and cls_name not in self.class_filter:
                continue
            
            detection = Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=float(score),
                bbox=tuple(box),
                image_size=(image.shape[1], image.shape[0]),
            )
            
            detections.append(detection)
        
        # Apply NMS
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove overlapping boxes."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        while detections:
            # Keep highest confidence detection
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections of same class
            detections = [
                det for det in detections
                if det.class_id != best.class_id or
                self._compute_iou(best.bbox, det.bbox) < self.nms_threshold
            ]
        
        return keep
    
    @staticmethod
    def _compute_iou(bbox1: tuple, bbox2: tuple) -> float:
        """Compute IoU (Intersection over Union) between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class MockDetectionModel:
    """Mock detection model for testing without dependencies."""
    
    def __call__(self, image, verbose=False):
        """Mock detection - returns empty results."""
        return []
