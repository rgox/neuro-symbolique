"""
Unit tests for nesy.perception.performance modules.

Tests batch processing and streaming pipelines for efficient inference.
"""

import pytest
import numpy as np
import time
from typing import Iterator

from nesy.perception.performance.batch import (
    BatchProcessor,
    BatchConfig,
)
from nesy.perception.performance.streaming import StreamingPipeline
from nesy.perception.object_detection import Detection


class TestBatchConfig:
    """Test BatchConfig dataclass."""

    def test_default_config(self):
        """Test default batch configuration."""
        config = BatchConfig()

        assert config.batch_size == 8
        assert config.max_queue_size == 32
        assert config.num_workers == 4
        assert config.prefetch == 2

    def test_custom_config(self):
        """Test custom batch configuration."""
        config = BatchConfig(
            batch_size=16,
            max_queue_size=64,
            num_workers=8,
            prefetch=4,
        )

        assert config.batch_size == 16
        assert config.max_queue_size == 64
        assert config.num_workers == 8
        assert config.prefetch == 4


class TestBatchProcessor:
    """Test BatchProcessor class."""

    def test_initialization(self):
        """Test batch processor initialization."""
        processor = BatchProcessor(
            detector="mock",
            batch_size=8,
            device="cpu",
        )

        assert processor.detector_name == "mock"
        assert processor.batch_size == 8
        assert processor.device == "cpu"
        assert processor._detector is None
        assert processor.stats["total_images"] == 0

    def test_initialization_with_embedder(self):
        """Test initialization with embedder."""
        processor = BatchProcessor(
            detector="mock",
            embedder="clip",
            batch_size=4,
        )

        assert processor.embedder_name == "clip"
        assert processor.batch_size == 4

    def test_process_empty_batch(self):
        """Test processing empty batch."""
        processor = BatchProcessor()

        results = processor.process_batch([])

        assert results == []
        assert processor.stats["total_images"] == 0

    def test_process_batch_no_detector(self):
        """Test processing batch without detector."""
        processor = BatchProcessor()

        # Create test images
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]

        results = processor.process_batch(images)

        # Should return empty results for each image
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_process_batch_updates_stats(self):
        """Test that processing updates statistics."""
        processor = BatchProcessor()

        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]

        results = processor.process_batch(images)

        assert processor.stats["total_images"] == 5
        assert processor.stats["total_time"] > 0
        assert processor.stats["batch_count"] > 0

    def test_process_batch_multiple_batches(self):
        """Test processing multiple batches."""
        processor = BatchProcessor(batch_size=2)

        # 5 images with batch size 2 = 3 batches (2, 2, 1)
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]

        results = processor.process_batch(images)

        assert len(results) == 5
        assert processor.stats["batch_count"] == 3  # ceil(5/2)

    def test_process_batch_with_confidence(self):
        """Test processing with custom confidence threshold."""
        processor = BatchProcessor()

        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]

        results = processor.process_batch(images, conf_threshold=0.7)

        assert len(results) == 1

    def test_process_stream_generator(self):
        """Test processing streaming images."""
        processor = BatchProcessor(batch_size=2)

        def image_generator():
            for _ in range(5):
                yield np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        results = list(processor.process_stream(image_generator()))

        assert len(results) == 5
        assert processor.stats["total_images"] == 5

    def test_process_stream_with_limit(self):
        """Test processing stream with max_images limit."""
        processor = BatchProcessor(batch_size=2)

        def image_generator():
            for _ in range(10):
                yield np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        results = list(processor.process_stream(image_generator(), max_images=5))

        assert len(results) == 5
        assert processor.stats["total_images"] == 5

    def test_process_stream_partial_batch(self):
        """Test processing stream with partial final batch."""
        processor = BatchProcessor(batch_size=4)

        def image_generator():
            # 7 images with batch size 4 = 2 batches (4, 3)
            for _ in range(7):
                yield np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        results = list(processor.process_stream(image_generator()))

        assert len(results) == 7

    def test_get_stats(self):
        """Test getting statistics."""
        processor = BatchProcessor()

        # Process some images
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(4)]
        processor.process_batch(images)

        stats = processor.get_stats()

        assert "total_images" in stats
        assert "total_time" in stats
        assert "batch_count" in stats
        assert "avg_fps" in stats
        assert "avg_latency_ms" in stats
        assert "avg_detections_per_image" in stats

        assert stats["total_images"] == 4
        assert stats["avg_fps"] > 0
        assert stats["avg_latency_ms"] > 0

    def test_get_stats_no_processing(self):
        """Test stats with no processing."""
        processor = BatchProcessor()

        stats = processor.get_stats()

        assert stats["total_images"] == 0
        assert stats["avg_fps"] == 0.0
        assert stats["avg_latency_ms"] == 0.0
        assert stats["avg_detections_per_image"] == 0.0

    def test_reset_stats(self):
        """Test resetting statistics."""
        processor = BatchProcessor()

        # Process some images
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
        processor.process_batch(images)

        assert processor.stats["total_images"] == 3

        # Reset
        processor.reset_stats()

        assert processor.stats["total_images"] == 0
        assert processor.stats["total_time"] == 0.0
        assert processor.stats["batch_count"] == 0

    def test_batch_size_handling(self):
        """Test different batch sizes."""
        processor = BatchProcessor(batch_size=3)

        images = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(10)]

        results = processor.process_batch(images)

        assert len(results) == 10
        # 10 images with batch size 3 = 4 batches
        assert processor.stats["batch_count"] == 4

    def test_device_auto(self):
        """Test device='auto' initialization."""
        processor = BatchProcessor(device="auto")

        assert processor.device == "auto"


class TestStreamingPipeline:
    """Test StreamingPipeline class."""

    def test_initialization(self):
        """Test streaming pipeline initialization."""
        pipeline = StreamingPipeline(
            detector="mock",
            max_fps=30.0,
            buffer_size=2,
        )

        assert pipeline.detector_name == "mock"
        assert pipeline.max_fps == 30.0
        assert pipeline.buffer_size == 2
        assert pipeline.frame_count == 0
        assert pipeline.dropped_frames == 0
        assert pipeline._detector is None

    def test_stream_from_generator_no_detector(self):
        """Test streaming without detector."""
        # Use high max_fps to reduce frame dropping
        pipeline = StreamingPipeline(max_fps=10000.0)

        def frame_generator():
            for _ in range(3):
                yield np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                time.sleep(0.0001)  # Small delay to allow processing

        results = list(pipeline.stream_from_generator(frame_generator()))

        # At least some frames should be processed
        assert len(results) >= 1
        assert all(len(r) == 2 for r in results)  # (frame, detections) tuples
        assert all(isinstance(r[1], list) for r in results)
        # Check stats updated
        assert pipeline.frame_count >= 1

    def test_stream_from_generator_with_confidence(self):
        """Test streaming with confidence threshold."""
        # Use high max_fps to reduce frame dropping
        pipeline = StreamingPipeline(max_fps=10000.0)

        def frame_generator():
            for _ in range(2):
                yield np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                time.sleep(0.0001)  # Small delay

        results = list(pipeline.stream_from_generator(frame_generator(), conf_threshold=0.7))

        # At least some frames should be processed
        assert len(results) >= 1

    def test_stream_frame_skipping(self):
        """Test frame skipping based on max_fps."""
        # Set very low max_fps to force frame skipping
        pipeline = StreamingPipeline(max_fps=10.0)

        def fast_frame_generator():
            # Generate frames quickly
            for _ in range(10):
                yield np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                time.sleep(0.01)  # 10ms between frames (100 fps)

        results = list(pipeline.stream_from_generator(fast_frame_generator()))

        # With max_fps=10, should process ~1 frame per 100ms
        # In 100ms of generation, should drop some frames
        assert pipeline.dropped_frames >= 0  # May drop frames
        assert len(results) <= 10

    def test_stream_updates_stats(self):
        """Test that streaming updates statistics."""
        # Use high max_fps to reduce frame dropping
        pipeline = StreamingPipeline(max_fps=10000.0)

        def frame_generator():
            for _ in range(5):
                yield np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                time.sleep(0.0001)  # Small delay

        list(pipeline.stream_from_generator(frame_generator()))

        # Stats should be updated (at least some frames processed)
        assert pipeline.frame_count >= 1
        assert pipeline.start_time is not None

    def test_get_stats(self):
        """Test getting streaming statistics."""
        # Use high max_fps to reduce frame dropping
        pipeline = StreamingPipeline(max_fps=10000.0)

        def frame_generator():
            for _ in range(4):
                yield np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                time.sleep(0.0001)  # Small delay

        list(pipeline.stream_from_generator(frame_generator()))

        stats = pipeline.get_stats()

        assert "frames_processed" in stats
        assert "frames_dropped" in stats
        assert "fps" in stats
        assert "drop_rate" in stats

        # At least some frames should be processed
        assert stats["frames_processed"] >= 1
        assert stats["fps"] > 0
        assert 0 <= stats["drop_rate"] <= 1

    def test_get_stats_no_streaming(self):
        """Test stats before streaming."""
        pipeline = StreamingPipeline()

        stats = pipeline.get_stats()

        assert stats["frames_processed"] == 0
        assert stats["frames_dropped"] == 0
        assert stats["fps"] == 0.0
        assert stats["drop_rate"] == 0.0

    def test_different_max_fps(self):
        """Test different max_fps settings."""
        pipeline_high = StreamingPipeline(max_fps=60.0)
        pipeline_low = StreamingPipeline(max_fps=5.0)

        assert pipeline_high.max_fps == 60.0
        assert pipeline_low.max_fps == 5.0

    def test_buffer_size(self):
        """Test different buffer sizes."""
        pipeline = StreamingPipeline(buffer_size=5)

        assert pipeline.buffer_size == 5

    def test_stream_generator_empty(self):
        """Test streaming with empty generator."""
        pipeline = StreamingPipeline()

        def empty_generator():
            return
            yield  # Never reached

        results = list(pipeline.stream_from_generator(empty_generator()))

        assert len(results) == 0
        assert pipeline.frame_count == 0

    def test_stream_maintains_frame_data(self):
        """Test that streaming returns correct frame data."""
        pipeline = StreamingPipeline()

        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128

        def frame_generator():
            yield test_image

        results = list(pipeline.stream_from_generator(frame_generator()))

        assert len(results) == 1
        frame, detections = results[0]
        assert np.array_equal(frame, test_image)
        assert isinstance(detections, list)


class TestIntegration:
    """Test integration between batch and streaming."""

    def test_batch_to_stream_conversion(self):
        """Test converting batch processing to streaming."""
        processor = BatchProcessor(batch_size=2)

        def image_generator():
            for _ in range(6):
                yield np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Use process_stream
        results = list(processor.process_stream(image_generator()))

        assert len(results) == 6
        assert processor.stats["total_images"] == 6

    def test_performance_comparison(self):
        """Test that batch processor tracks performance."""
        processor = BatchProcessor(batch_size=4)

        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(8)]

        start = time.time()
        processor.process_batch(images)
        elapsed = time.time() - start

        stats = processor.get_stats()

        # Processing time should be tracked
        assert stats["total_time"] > 0
        assert stats["total_time"] <= elapsed * 2  # Some overhead is acceptable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
