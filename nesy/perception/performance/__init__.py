"""
Performance Optimization for Real-Time Perception.

Batch processing, GPU acceleration, and streaming for production.

Features:
    - Batch image processing
    - GPU tensor operations
    - Async streaming pipeline
    - Performance profiling

Example:
    >>> from nesy.perception.performance import BatchProcessor
    >>> 
    >>> processor = BatchProcessor(detector="yolov8n", batch_size=8)
    >>> results = processor.process_batch(images)
"""

from nesy.perception.performance.batch import BatchProcessor, BatchConfig
from nesy.perception.performance.streaming import StreamingPipeline

__all__ = [
    "BatchProcessor",
    "BatchConfig",
    "StreamingPipeline",
]
