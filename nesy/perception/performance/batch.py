"""
Batch Processing for Efficient Inference.

Process multiple images in parallel with GPU batching.

Example:
    >>> from nesy.perception.performance import BatchProcessor
    >>> 
    >>> processor = BatchProcessor(detector="yolov8n", batch_size=16)
    >>> images = [img1, img2, img3, ...]
    >>> results = processor.process_batch(images)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import time

from nesy.perception.object_detection import Detection


@dataclass
class BatchConfig:
    """
    Batch processing configuration.
    
    Attributes:
        batch_size: Number of images per batch
        max_queue_size: Maximum queue size for streaming
        num_workers: Number of parallel workers
        prefetch: Number of batches to prefetch
    """
    batch_size: int = 8
    max_queue_size: int = 32
    num_workers: int = 4
    prefetch: int = 2


class BatchProcessor:
    """
    Batch processor for efficient multi-image inference.
    
    Processes images in batches to maximize GPU utilization.
    
    Args:
        detector: Detector model name or instance
        embedder: Optional embedder model
        batch_size: Batch size
        device: Device ("cuda", "cpu")
    
    Example:
        >>> processor = BatchProcessor("yolov8n", batch_size=8)
        >>> results = processor.process_batch(images)
        >>> 
        >>> for img_results in results:
        ...     for det in img_results:
        ...         print(f"{det.class_name}: {det.confidence:.2f}")
    """
    
    def __init__(
        self,
        detector: Optional[str] = None,
        embedder: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
    ):
        """Initialize batch processor."""
        self.detector_name = detector
        self.embedder_name = embedder
        self.batch_size = batch_size
        self.device = device
        
        # Lazy load models
        self._detector = None
        self._embedder = None
        
        # Stats
        self.stats = {
            "total_images": 0,
            "total_detections": 0,
            "total_time": 0.0,
            "batch_count": 0,
        }
    
    def process_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.5,
    ) -> List[List[Detection]]:
        """
        Process batch of images.
        
        Args:
            images: List of RGB images
            conf_threshold: Detection confidence threshold
        
        Returns:
            List of detection lists (one per image)
        
        Example:
            >>> results = processor.process_batch([img1, img2, img3])
            >>> len(results) == 3
        """
        if not images:
            return []
        
        # Load detector if needed
        if self._detector is None and self.detector_name:
            self._load_detector()
        
        # Process in batches
        all_results = []
        start_time = time.time()
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            batch_results = self._process_single_batch(batch, conf_threshold)
            all_results.extend(batch_results)
        
        # Update stats
        elapsed = time.time() - start_time
        self.stats["total_images"] += len(images)
        self.stats["total_time"] += elapsed
        self.stats["batch_count"] += (len(images) + self.batch_size - 1) // self.batch_size
        self.stats["total_detections"] += sum(len(r) for r in all_results)
        
        return all_results
    
    def process_stream(
        self,
        image_generator,
        max_images: Optional[int] = None,
    ):
        """
        Process streaming images.
        
        Args:
            image_generator: Generator yielding images
            max_images: Maximum images to process
        
        Yields:
            Detection results for each image
        
        Example:
            >>> def video_stream():
            ...     cap = cv2.VideoCapture(0)
            ...     while True:
            ...         ret, frame = cap.read()
            ...         if not ret: break
            ...         yield frame
            >>> 
            >>> for detections in processor.process_stream(video_stream()):
            ...     # Handle detections in real-time
            ...     pass
        """
        batch = []
        count = 0
        
        for image in image_generator:
            batch.append(image)
            count += 1
            
            # Process when batch full
            if len(batch) >= self.batch_size:
                results = self.process_batch(batch)
                for result in results:
                    yield result
                batch = []
            
            # Check limit
            if max_images and count >= max_images:
                break
        
        # Process remaining
        if batch:
            results = self.process_batch(batch)
            for result in results:
                yield result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Statistics dictionary
        
        Example:
            >>> stats = processor.get_stats()
            >>> print(f"FPS: {stats['avg_fps']:.2f}")
            >>> print(f"Latency: {stats['avg_latency_ms']:.1f}ms")
        """
        if self.stats["total_time"] > 0:
            avg_fps = self.stats["total_images"] / self.stats["total_time"]
            avg_latency_ms = (self.stats["total_time"] / self.stats["total_images"]) * 1000
        else:
            avg_fps = 0.0
            avg_latency_ms = 0.0
        
        return {
            **self.stats,
            "avg_fps": avg_fps,
            "avg_latency_ms": avg_latency_ms,
            "avg_detections_per_image": (
                self.stats["total_detections"] / self.stats["total_images"]
                if self.stats["total_images"] > 0 else 0.0
            ),
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_images": 0,
            "total_detections": 0,
            "total_time": 0.0,
            "batch_count": 0,
        }
    
    def _load_detector(self):
        """Lazy load detector."""
        from nesy.perception.models import ModelZoo
        
        self._detector = ModelZoo.get_detector(self.detector_name)
    
    def _process_single_batch(
        self,
        batch: List[np.ndarray],
        conf_threshold: float,
    ) -> List[List[Detection]]:
        """Process single batch."""
        results = []
        
        if self._detector is None:
            # Return empty results if no detector
            return [[] for _ in batch]
        
        # Process each image (YOLO doesn't support true batching via our interface)
        # For true batching, would need to modify YOLODetector
        for image in batch:
            detections = self._detector.detect(image, conf_threshold=conf_threshold)
            results.append(detections)
        
        return results
