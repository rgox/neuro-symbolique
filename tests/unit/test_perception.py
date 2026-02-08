"""
Unit tests for Perception module.

Tests cover:
- Object detection (mock backend)
- Feature extraction (mock backend)
- Perception pipeline
- Scene graph integration
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nesy.perception import (
    ObjectDetector,
    Detection,
    DetectionBackend,
    FeatureExtractor,
    FeatureVector,
    FeatureBackend,
    PerceptionPipeline,
)
from nesy.world_model import SceneGraph, ObjectLayer, LayerType
from nesy.core.memory import UMA, DeviceType


class TestObjectDetection:
    """Test object detection module."""
    
    def test_create_detector_mock(self):
        """Test creating a mock detector."""
        detector = ObjectDetector(backend="mock")
        assert detector.backend == DetectionBackend.MOCK
    
    def test_detect_mock(self):
        """Test detection with mock backend."""
        detector = ObjectDetector(backend="mock", confidence_threshold=0.5)
        
        # Create fake image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Detect
        detections = detector.detect(image)
        
        assert isinstance(detections, list)
        assert len(detections) > 0
        
        # Check detection properties
        det = detections[0]
        assert isinstance(det, Detection)
        assert det.confidence >= 0.0 and det.confidence <= 1.0
        assert len(det.bbox) == 4
        assert det.class_name in ["chair", "cup"]
    
    def test_detection_bbox_operations(self):
        """Test Detection bbox operations."""
        det = Detection(
            class_id=56,
            class_name="chair",
            confidence=0.95,
            bbox=(100, 100, 300, 400),
            image_size=(640, 480),
        )
        
        # Test bbox center
        center = det.get_bbox_center()
        assert center == (200.0, 250.0)
        
        # Test bbox size
        size = det.get_bbox_size()
        assert size == (200.0, 300.0)
        
        # Test bbox area
        area = det.get_bbox_area()
        assert area == 60000.0
    
    def test_detection_to_dict(self):
        """Test converting detection to dictionary."""
        det = Detection(
            class_id=56,
            class_name="chair",
            confidence=0.95,
            bbox=(100, 100, 300, 400),
            image_size=(640, 480),
        )
        
        det_dict = det.to_dict()
        
        assert "class_name" in det_dict
        assert "confidence" in det_dict
        assert "bbox" in det_dict
        assert "bbox_center" in det_dict
        assert det_dict["class_name"] == "chair"


class TestFeatureExtraction:
    """Test feature extraction module."""
    
    def test_create_extractor_mock(self):
        """Test creating a mock feature extractor."""
        extractor = FeatureExtractor(backend="mock")
        assert extractor.backend == FeatureBackend.MOCK
        assert extractor.feature_dim == 512
    
    def test_extract_features_mock(self):
        """Test feature extraction with mock backend."""
        extractor = FeatureExtractor(backend="mock", normalize=True)
        
        # Create fake image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Extract features
        features = extractor.extract(image)
        
        assert isinstance(features, FeatureVector)
        assert features.dim == 512
        assert len(features.features) == 512
        assert features.normalized == True
    
    def test_extract_with_bbox(self):
        """Test feature extraction from image crop."""
        extractor = FeatureExtractor(backend="mock")
        
        # Create fake image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Extract from bbox crop
        features = extractor.extract(image, bbox=(100, 100, 300, 400))
        
        assert isinstance(features, FeatureVector)
        assert features.dim == 512
    
    def test_feature_normalization(self):
        """Test feature vector normalization."""
        features = np.random.randn(512).astype(np.float32)
        feat_vec = FeatureVector(
            features=features,
            dim=512,
            model="mock",
            normalized=False,
        )
        
        # Normalize
        norm_feat_vec = feat_vec.normalize()
        
        assert norm_feat_vec.normalized == True
        # Check L2 norm is 1.0
        norm = np.linalg.norm(norm_feat_vec.features)
        assert abs(norm - 1.0) < 1e-5
    
    def test_cosine_similarity(self):
        """Test cosine similarity between feature vectors."""
        feat1 = FeatureVector(
            features=np.ones(512, dtype=np.float32),
            dim=512,
            model="mock",
        )
        feat2 = FeatureVector(
            features=np.ones(512, dtype=np.float32) * 0.5,
            dim=512,
            model="mock",
        )
        
        # Compute similarity
        sim = feat1.cosine_similarity(feat2)
        
        # Should be 1.0 (same direction)
        assert abs(sim - 1.0) < 1e-5
    
    def test_extract_batch(self):
        """Test batch feature extraction."""
        extractor = FeatureExtractor(backend="mock")
        
        # Create batch of images
        images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        # Extract batch
        features_list = extractor.extract_batch(images)
        
        assert len(features_list) == 3
        for feat in features_list:
            assert isinstance(feat, FeatureVector)


class TestPerceptionPipeline:
    """Test perception pipeline."""
    
    def test_create_pipeline(self):
        """Test creating a perception pipeline."""
        sg = SceneGraph()
        uma = UMA(device=DeviceType.CPU)
        
        pipeline = PerceptionPipeline(
            scene_graph=sg,
            uma=uma,
            detector_backend="mock",
            feature_backend="mock",
        )
        
        assert pipeline is not None
        assert pipeline.scene_graph is sg
        assert pipeline.uma is uma
    
    def test_process_image(self):
        """Test processing an image through the pipeline."""
        sg = SceneGraph()
        uma = UMA(device=DeviceType.CPU)
        
        pipeline = PerceptionPipeline(
            scene_graph=sg,
            uma=uma,
            detector_backend="mock",
            feature_backend="mock",
        )
        
        # Create fake image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process
        result = pipeline.process_image(image)
        
        assert len(result.objects_added) > 0
        assert len(result.detections) > 0
        assert result.processing_time > 0
    
    def test_scene_graph_update(self):
        """Test that pipeline updates scene graph."""
        sg = SceneGraph()
        uma = UMA(device=DeviceType.CPU)
        
        pipeline = PerceptionPipeline(
            scene_graph=sg,
            uma=uma,
            detector_backend="mock",
            feature_backend="mock",
        )
        
        # Initial empty
        assert len(sg.get_nodes(LayerType.L1)) == 0
        
        # Process image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process_image(image)
        
        # Should have added objects
        objects = sg.get_nodes(LayerType.L1)
        assert len(objects) > 0
        assert len(objects) == len(result.objects_added)
    
    def test_embedding_storage(self):
        """Test that embeddings are stored in UMA."""
        sg = SceneGraph()
        uma = UMA(device=DeviceType.CPU)
        
        pipeline = PerceptionPipeline(
            scene_graph=sg,
            uma=uma,
            detector_backend="mock",
            feature_backend="mock",
        )
        
        # Process image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process_image(image)
        
        # Should have stored embeddings
        assert len(result.embeddings_stored) > 0
        
        # Check embeddings exist in UMA
        for emb_key in result.embeddings_stored:
            assert uma.exists(emb_key)
            buffer = uma.get(emb_key)
            assert buffer.shape[0] == 512  # Mock feature dim
    
    def test_spatial_relations(self):
        """Test spatial relations computation."""
        sg = SceneGraph()
        uma = UMA(device=DeviceType.CPU)
        
        pipeline = PerceptionPipeline(
            scene_graph=sg,
            uma=uma,
            detector_backend="mock",
            feature_backend="mock",
            compute_relations=True,
        )
        
        # Process image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process_image(image)
        
        # May have computed relations (depends on object positions)
        assert result.spatial_relations >= 0
    
    def test_object_tracking(self):
        """Test object tracking across frames."""
        sg = SceneGraph()
        uma = UMA(device=DeviceType.CPU)
        
        pipeline = PerceptionPipeline(
            scene_graph=sg,
            uma=uma,
            detector_backend="mock",
            feature_backend="mock",
            track_objects=True,
        )
        
        # Process first frame
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pipeline.process_image(image)
        
        # Check tracked objects
        tracked = pipeline.get_tracked_objects()
        assert len(tracked) > 0
        
        # Clear tracking
        pipeline.clear_tracking()
        assert len(pipeline.get_tracked_objects()) == 0
    
    def test_pipeline_statistics(self):
        """Test getting pipeline statistics."""
        sg = SceneGraph()
        uma = UMA(device=DeviceType.CPU)
        
        pipeline = PerceptionPipeline(
            scene_graph=sg,
            uma=uma,
            detector_backend="mock",
            feature_backend="mock",
        )
        
        stats = pipeline.get_statistics()
        
        assert "detector" in stats
        assert "feature_extractor" in stats
        assert stats["detector"]["backend"] == "mock"
        assert stats["feature_extractor"]["backend"] == "mock"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
