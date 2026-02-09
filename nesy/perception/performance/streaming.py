"""
Streaming Pipeline for Real-Time Perception.

Async pipeline for processing video streams with minimal latency.

Example:
    >>> from nesy.perception.performance import StreamingPipeline
    >>> 
    >>> pipeline = StreamingPipeline(detector="yolov8n")
    >>> 
    >>> # Process video stream
    >>> for frame, detections in pipeline.stream_from_video("video.mp4"):
    ...     # Handle detections
    ...     pass
"""

from typing import Iterator, Tuple, Optional, Callable
import numpy as np
import time


class StreamingPipeline:
    """
    Streaming perception pipeline for real-time processing.
    
    Minimizes latency through async processing and frame skipping.
    
    Args:
        detector: Detector model
        max_fps: Maximum processing FPS (skip frames if needed)
        buffer_size: Frame buffer size
    
    Example:
        >>> pipeline = StreamingPipeline("yolov8n", max_fps=30)
        >>> 
        >>> def frame_callback(detections):
        ...     print(f"Found {len(detections)} objects")
        >>> 
        >>> pipeline.run(video_source=0, callback=frame_callback)
    """
    
    def __init__(
        self,
        detector: Optional[str] = None,
        max_fps: float = 30.0,
        buffer_size: int = 2,
    ):
        """Initialize streaming pipeline."""
        self.detector_name = detector
        self.max_fps = max_fps
        self.buffer_size = buffer_size
        
        # Stats
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = None
        
        # Lazy load
        self._detector = None
    
    def stream_from_generator(
        self,
        frame_generator: Iterator[np.ndarray],
        conf_threshold: float = 0.5,
    ) -> Iterator[Tuple[np.ndarray, list]]:
        """
        Stream from frame generator.
        
        Args:
            frame_generator: Generator yielding frames
            conf_threshold: Detection threshold
        
        Yields:
            (frame, detections) tuples
        
        Example:
            >>> def frames():
            ...     while True:
            ...         yield get_frame()
            >>> 
            >>> for frame, dets in pipeline.stream_from_generator(frames()):
            ...     display(frame, dets)
        """
        if self._detector is None and self.detector_name:
            self._load_detector()
        
        self.start_time = time.time()
        min_frame_time = 1.0 / self.max_fps
        last_process_time = 0.0
        
        for frame in frame_generator:
            current_time = time.time()
            
            # Skip frame if too soon
            if current_time - last_process_time < min_frame_time:
                self.dropped_frames += 1
                continue
            
            # Process frame
            detections = []
            if self._detector:
                detections = self._detector.detect(frame, conf_threshold=conf_threshold)
            
            self.frame_count += 1
            last_process_time = current_time
            
            yield (frame, detections)
    
    def get_stats(self) -> dict:
        """Get streaming statistics."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        else:
            fps = 0.0
        
        return {
            "frames_processed": self.frame_count,
            "frames_dropped": self.dropped_frames,
            "fps": fps,
            "drop_rate": self.dropped_frames / (self.frame_count + self.dropped_frames)
            if (self.frame_count + self.dropped_frames) > 0 else 0.0,
        }
    
    def _load_detector(self):
        """Lazy load detector."""
        from nesy.perception.models import ModelZoo
        self._detector = ModelZoo.get_detector(self.detector_name)
