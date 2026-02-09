"""
Tests for Model Zoo and Real Models.

Tests YOLO detector, CLIP embedder, and model zoo infrastructure.
"""

import pytest
import numpy as np

from nesy.perception.models import ModelZoo, ModelConfig, YOLODetector, CLIPEmbedder


class TestModelZoo:
    """Test model zoo infrastructure."""
    
    def test_register_and_get(self):
        """Test registering and getting models."""
        # Create dummy loader
        def dummy_loader(**kwargs):
            return "dummy_model"
        
        # Register
        ModelZoo.register("test_model", ModelConfig(
            name="test_model",
            model_type="detector",
            loader=dummy_loader,
        ))
        
        # Get
        model = ModelZoo.get("test_model")
        assert model == "dummy_model"
        
        # Should be cached
        model2 = ModelZoo.get("test_model")
        assert model2 is model
    
    def test_list_models(self):
        """Test listing registered models."""
        models = ModelZoo.list_models()
        assert isinstance(models, list)
        assert len(models) > 0  # Pre-registered models
        
        # Filter by type
        detectors = ModelZoo.list_models(model_type="detector")
        assert "yolov8n" in detectors
    
    def test_clear_cache(self):
        """Test clearing model cache."""
        def dummy_loader(**kwargs):
            return object()  # Unique instance
        
        ModelZoo.register("test_clear", ModelConfig(
            name="test_clear",
            model_type="detector",
            loader=dummy_loader,
        ))
        
        model1 = ModelZoo.get("test_clear")
        ModelZoo.clear_cache("test_clear")
        model2 = ModelZoo.get("test_clear")
        
        # Should be different instances
        assert model1 is not model2


class TestYOLODetector:
    """Test YOLO detector."""
    
    @pytest.fixture
    def detector(self):
        """Create YOLO detector (will use mock if ultralytics not installed)."""
        try:
            return YOLODetector(model_name="yolov8n")
        except ImportError:
            pytest.skip("ultralytics not installed")
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.model_name == "yolov8n"
        assert detector.conf_threshold == 0.25
        assert detector._model is None  # Lazy load
    
    def test_detect_loads_model(self, detector):
        """Test that detect() triggers model loading."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # This will load model
        detections = detector.detect(image)
        
        # Model should be loaded now
        assert detector._model is not None
        assert isinstance(detections, list)
    
    def test_coco_classes(self):
        """Test COCO class names."""
        assert len(YOLODetector.COCO_CLASSES) == 80
        assert YOLODetector.COCO_CLASSES[0] == "person"
        assert YOLODetector.COCO_CLASSES[56] == "chair"
        assert YOLODetector.COCO_CLASSES[41] == "cup"


class TestCLIPEmbedder:
    """Test CLIP embedder."""
    
    @pytest.fixture
    def embedder(self):
        """Create CLIP embedder (will use mock if transformers not installed)."""
        try:
            return CLIPEmbedder(model_name="clip-vit-b32")
        except ImportError:
            pytest.skip("transformers not installed")
    
    def test_initialization(self, embedder):
        """Test embedder initialization."""
        assert embedder.model_name == "clip-vit-b32"
        assert embedder._model is None  # Lazy load
    
    def test_encode_image_loads_model(self, embedder):
        """Test that encode_image() triggers model loading."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # This will load model
        features = embedder.encode_image(image)
        
        # Model should be loaded
        assert embedder._model is not None
        assert features.dim > 0
        assert features.model == "clip-vit-b32"
        assert features.source == "image"
    
    def test_encode_text(self, embedder):
        """Test text encoding."""
        text = "a red cup on the table"
        
        features = embedder.encode_text(text)
        
        assert features.dim > 0
        assert features.source == "text"
    
    def test_encode_text_batch(self, embedder):
        """Test batch text encoding."""
        texts = ["a cat", "a dog", "a bird"]
        
        features_list = embedder.encode_text(texts)
        
        assert len(features_list) == 3
        assert all(f.dim > 0 for f in features_list)
    
    def test_compute_similarity(self, embedder):
        """Test similarity computation."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        img_emb = embedder.encode_image(image)
        text_emb = embedder.encode_text("a photo")
        
        similarity = embedder.compute_similarity(img_emb, text_emb)
        
        assert 0 <= similarity <= 1
    
    def test_zero_shot_classify(self, embedder):
        """Test zero-shot classification."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        classes = ["cat", "dog", "bird"]
        
        probs = embedder.zero_shot_classify(image, classes)
        
        assert len(probs) == 3
        assert all(0 <= p <= 1 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 0.01  # Sum to 1
