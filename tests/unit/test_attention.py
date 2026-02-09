"""
Tests for Active Perception (Attention & Saliency).

Tests saliency detection, ROI proposals, and attention mechanisms.
"""

import pytest
import numpy as np

# Check if cv2 available
try:
    from nesy.perception.attention import (
        SaliencyDetector,
        SaliencyMethod,
        AttentionMechanism,
        ROI,
        ROIProposal,
    )
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    pytestmark = pytest.mark.skip(reason="OpenCV (cv2) not installed")


class TestSaliencyDetector:
    """Test saliency detection."""
    
    @pytest.fixture
    def detector(self):
        """Create saliency detector."""
        return SaliencyDetector(method=SaliencyMethod.SPECTRAL)
    
    @pytest.fixture
    def test_image(self):
        """Create test image."""
        # Simple test image: bright square on dark background
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[200:280, 270:370] = 255  # Bright square
        return img
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.method == SaliencyMethod.SPECTRAL
        assert detector.gaussian_blur == 5
        assert detector.threshold == 128
    
    def test_compute_saliency_spectral(self, detector, test_image):
        """Test spectral saliency."""
        saliency = detector.compute_saliency(test_image)
        
        # Should return grayscale map
        assert saliency.shape == test_image.shape[:2]
        assert saliency.dtype == np.uint8
        assert saliency.min() >= 0
        assert saliency.max() <= 255
    
    def test_get_top_points(self, detector, test_image):
        """Test extracting salient points."""
        saliency = detector.compute_saliency(test_image)
        points = detector.get_top_points(saliency, k=5)
        
        # Should return list of points
        assert isinstance(points, list)
        assert len(points) <= 5
        
        # Each point should be (x, y)
        for x, y in points:
            assert 0 <= x < test_image.shape[1]
            assert 0 <= y < test_image.shape[0]
    
    def test_get_salient_regions(self, detector, test_image):
        """Test extracting salient regions."""
        saliency = detector.compute_saliency(test_image)
        regions = detector.get_salient_regions(saliency, min_area=50)
        
        # Should return list of bboxes
        assert isinstance(regions, list)
        
        # Each region should be (x, y, w, h)
        for x, y, w, h in regions:
            assert w * h >= 50
    
    def test_edge_based_saliency(self, test_image):
        """Test edge-based method."""
        detector = SaliencyDetector(method=SaliencyMethod.EDGES)
        saliency = detector.compute_saliency(test_image)
        
        assert saliency.shape == test_image.shape[:2]
    
    def test_contrast_based_saliency(self, test_image):
        """Test contrast-based method."""
        detector = SaliencyDetector(method=SaliencyMethod.CONTRAST)
        saliency = detector.compute_saliency(test_image)
        
        assert saliency.shape == test_image.shape[:2]
    
    def test_visualize(self, detector, test_image):
        """Test saliency visualization."""
        saliency = detector.compute_saliency(test_image)
        overlay = detector.visualize(test_image, saliency, alpha=0.5)
        
        # Should return RGB image
        assert overlay.shape == test_image.shape
        assert overlay.dtype == np.uint8


class TestROI:
    """Test ROI dataclass."""
    
    def test_creation(self):
        """Test creating ROI."""
        roi = ROI(bbox=(10, 20, 100, 80), score=0.8)
        
        assert roi.bbox == (10, 20, 100, 80)
        assert roi.score == 0.8
        assert roi.label is None
    
    def test_area(self):
        """Test area computation."""
        roi = ROI(bbox=(0, 0, 100, 50), score=1.0)
        assert roi.area == 5000
    
    def test_center(self):
        """Test center computation."""
        roi = ROI(bbox=(0, 0, 100, 80), score=1.0)
        cx, cy = roi.center
        assert cx == 50
        assert cy == 40
    
    def test_extract_region(self):
        """Test extracting region from image."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        roi = ROI(bbox=(100, 100, 200, 150), score=1.0)
        
        crop = roi.extract_region(image)
        assert crop.shape == (150, 200, 3)
    
    def test_iou(self):
        """Test IoU computation."""
        roi1 = ROI(bbox=(0, 0, 100, 100), score=1.0)
        roi2 = ROI(bbox=(50, 50, 100, 100), score=1.0)
        
        iou = roi1.iou(roi2)
        
        # Overlapping region: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU = 2500 / 17500 ≈ 0.143
        assert 0.14 <= iou <= 0.15
    
    def test_iou_no_overlap(self):
        """Test IoU with no overlap."""
        roi1 = ROI(bbox=(0, 0, 50, 50), score=1.0)
        roi2 = ROI(bbox=(100, 100, 50, 50), score=1.0)
        
        iou = roi1.iou(roi2)
        assert iou == 0.0


class TestAttentionMechanism:
    """Test attention mechanism."""
    
    @pytest.fixture
    def attention(self):
        """Create attention mechanism."""
        return AttentionMechanism()
    
    @pytest.fixture
    def test_image(self):
        """Create test image."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_initialization(self, attention):
        """Test attention initialization."""
        assert attention.saliency_detector is not None
        assert attention.config is not None
    
    def test_propose_regions_saliency(self, attention, test_image):
        """Test saliency-based proposals."""
        rois = attention.propose_regions(
            test_image,
            top_k=5,
            use_saliency=True,
            use_grid=False
        )
        
        # Should return list of ROIs
        assert isinstance(rois, list)
        assert len(rois) <= 5
        
        if len(rois) > 0:
            assert isinstance(rois[0], ROI)
            assert 0 <= rois[0].score <= 1
    
    def test_propose_regions_grid(self, attention, test_image):
        """Test grid-based proposals."""
        rois = attention.propose_regions(
            test_image,
            top_k=10,
            use_saliency=False,
            use_grid=True
        )
        
        assert isinstance(rois, list)
        assert len(rois) > 0
    
    def test_adaptive_grid(self, attention, test_image):
        """Test adaptive grid generation."""
        rois = attention.adaptive_grid(test_image, num_regions=9)
        
        # Should return 9 regions (3x3 grid)
        assert len(rois) == 9
        
        # Each should have grid metadata
        for roi in rois:
            assert "grid_cell" in roi.metadata
    
    def test_focus_window(self, attention, test_image):
        """Test focus window creation."""
        center = (320, 240)  # Image center
        roi = attention.focus_window(test_image, center, window_size=(128, 128))
        
        assert isinstance(roi, ROI)
        assert roi.score == 1.0
        assert roi.label == "focus_window"
    
    def test_nms(self, attention):
        """Test non-maximum suppression."""
        # Create overlapping ROIs
        rois = [
            ROI(bbox=(0, 0, 100, 100), score=0.9),
            ROI(bbox=(50, 50, 100, 100), score=0.7),  # Overlaps with first
            ROI(bbox=(200, 200, 100, 100), score=0.8),  # No overlap
        ]
        
        filtered = attention._nms(rois, iou_threshold=0.3)
        
        # Should keep first (highest score) and third (no overlap)
        assert len(filtered) == 2
        assert filtered[0].score == 0.9
        assert filtered[1].score == 0.8
