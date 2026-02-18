"""
Tests for Full Neural-Symbolic Pipeline.

Tests initialization, image processing, querying, and statistics.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from nesy.pipeline.full_pipeline import NeuralSymbolicPipeline


class TestPipelineInit:
    """Test pipeline initialization."""

    def test_default_init(self):
        """Test pipeline with default parameters."""
        pipeline = NeuralSymbolicPipeline()

        assert pipeline.scene_graph is not None
        assert pipeline.codebook is not None
        assert pipeline.grounding is not None
        assert pipeline.reasoning is not None
        assert pipeline.perception is not None
        assert pipeline.uma is not None

    def test_init_no_uma(self):
        """Test pipeline without UMA."""
        pipeline = NeuralSymbolicPipeline(use_uma=False)

        assert pipeline.uma is None
        assert pipeline.grounding_cache is None

    def test_init_custom_params(self):
        """Test pipeline with custom parameters."""
        pipeline = NeuralSymbolicPipeline(
            detector_backend="mock",
            feature_backend="mock",
            vsa_dim=5000,
            vsa_similarity_threshold=0.3,
        )

        assert pipeline.codebook.dim == 5000


class TestPipelineProcessImage:
    """Test image processing."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline."""
        return NeuralSymbolicPipeline(
            detector_backend="mock",
            feature_backend="mock",
        )

    def test_process_image(self, pipeline):
        """Test processing a mock image."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = pipeline.process_image(image)

        assert "detections" in result
        assert "num_nodes" in result
        assert "num_edges" in result
        assert "num_facts" in result
        assert "processing_time" in result
        assert isinstance(result["processing_time"], float)
        assert result["processing_time"] > 0

    def test_process_image_no_vsa(self, pipeline):
        """Test processing without VSA grounding."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = pipeline.process_image(image, ground_vsa=False)

        assert "detections" in result

    def test_process_image_no_reasoning_sync(self, pipeline):
        """Test processing without reasoning sync."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = pipeline.process_image(image, sync_reasoning=False)

        assert result["num_facts"] == 0

    def test_process_with_camera_pose(self, pipeline):
        """Test processing with camera pose."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        camera_pose = np.array([1.0, 2.0, 3.0])

        result = pipeline.process_image(image, camera_pose=camera_pose)

        assert "detections" in result


class TestPipelineQuery:
    """Test pipeline querying."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with some data."""
        p = NeuralSymbolicPipeline(
            detector_backend="mock",
            feature_backend="mock",
        )
        # Process an image to populate scene graph
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        p.process_image(image)
        return p

    def test_query_find_all(self, pipeline):
        """Test 'find all' query."""
        result = pipeline.query("Find all cups")

        assert isinstance(result, list)

    def test_query_in_location(self, pipeline):
        """Test 'in the' query."""
        result = pipeline.query("What's in the kitchen?")

        assert isinstance(result, list)

    def test_query_on_object(self, pipeline):
        """Test 'on the' query."""
        result = pipeline.query("What's on the table?")

        assert isinstance(result, list)

    def test_query_no_match(self, pipeline):
        """Test unmatched query."""
        result = pipeline.query("How many objects?")

        assert result == []


class TestPipelineReasoning:
    """Test pipeline reasoning methods."""

    @pytest.fixture
    def pipeline(self):
        """Create fresh pipeline."""
        return NeuralSymbolicPipeline(
            detector_backend="mock",
            feature_backend="mock",
        )

    def test_find_all(self, pipeline):
        """Test find_all method."""
        result = pipeline.find_all("cup")

        assert isinstance(result, list)

    def test_find_in_location(self, pipeline):
        """Test find_in_location method."""
        result = pipeline.find_in_location("kitchen_1")

        assert isinstance(result, list)

    def test_infer(self, pipeline):
        """Test infer method."""
        result = pipeline.infer("on('cup1', 'table1')")

        assert isinstance(result, (bool, type(None)))

    def test_find_similar_to(self, pipeline):
        """Test find_similar_to method."""
        result = pipeline.find_similar_to("cup_1", min_similarity=0.5)

        assert isinstance(result, list)


class TestPipelineStatistics:
    """Test pipeline statistics and lifecycle."""

    def test_get_statistics(self):
        """Test pipeline statistics."""
        pipeline = NeuralSymbolicPipeline()

        stats = pipeline.get_statistics()

        assert "scene_graph" in stats
        assert "reasoning" in stats
        assert "codebook_size" in stats

    def test_reset(self):
        """Test pipeline reset."""
        pipeline = NeuralSymbolicPipeline()

        # Process image to add data
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pipeline.process_image(image)

        # Reset
        pipeline.reset()

        stats = pipeline.get_statistics()
        assert stats["scene_graph"]["total_nodes"] == 0

    def test_repr(self):
        """Test pipeline string representation."""
        pipeline = NeuralSymbolicPipeline()

        repr_str = repr(pipeline)

        assert "NeuralSymbolicPipeline" in repr_str
        assert "nodes=" in repr_str
