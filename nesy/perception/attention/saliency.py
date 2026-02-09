"""
Saliency Detection for Active Perception.

Generates saliency maps to identify visually important regions.

Methods:
    - Spectral residual
    - Edge-based saliency
    - Contrast-based saliency
    - Deep learning (optional)

Example:
    >>> from nesy.perception.attention import SaliencyDetector
    >>> 
    >>> detector = SaliencyDetector(method="spectral")
    >>> saliency_map = detector.compute_saliency(image)
    >>> 
    >>> # Get top salient points
    >>> points = detector.get_top_points(saliency_map, k=10)
"""

from enum import Enum
from typing import List, Tuple, Optional
import numpy as np
import cv2


class SaliencyMethod(Enum):
    """Saliency detection methods."""
    SPECTRAL = "spectral"  # Spectral residual
    FINE_GRAINED = "fine_grained"  # Fine-grained
    STATIC = "static"  # Static spectral
    EDGES = "edges"  # Edge-based
    CONTRAST = "contrast"  # Contrast-based


class SaliencyDetector:
    """
    Saliency map generator for attention-based perception.
    
    Identifies visually salient regions for focused processing.
    
    Args:
        method: Saliency detection method
        gaussian_blur: Blur kernel size for smoothing
        threshold: Saliency threshold [0, 255]
    
    Example:
        >>> detector = SaliencyDetector(method="spectral")
        >>> saliency = detector.compute_saliency(image)
        >>> hot_spots = detector.get_top_points(saliency, k=5)
    """
    
    def __init__(
        self,
        method: SaliencyMethod = SaliencyMethod.SPECTRAL,
        gaussian_blur: int = 5,
        threshold: int = 128,
    ):
        """Initialize saliency detector."""
        self.method = method
        self.gaussian_blur = gaussian_blur
        self.threshold = threshold
        
        # Lazy load OpenCV saliency
        self._cv_saliency = None
    
    def compute_saliency(
        self,
        image: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute saliency map for image.
        
        Args:
            image: RGB image (H, W, 3)
            normalize: Normalize to [0, 255]
        
        Returns:
            Saliency map (H, W) in [0, 255]
        
        Example:
            >>> saliency = detector.compute_saliency(image)
            >>> # High values = salient regions
        """
        if self.method == SaliencyMethod.SPECTRAL:
            saliency = self._spectral_residual(image)
        elif self.method == SaliencyMethod.FINE_GRAINED:
            saliency = self._fine_grained(image)
        elif self.method == SaliencyMethod.EDGES:
            saliency = self._edge_based(image)
        elif self.method == SaliencyMethod.CONTRAST:
            saliency = self._contrast_based(image)
        else:
            # Default to spectral
            saliency = self._spectral_residual(image)
        
        # Normalize
        if normalize and saliency.max() > 0:
            saliency = (saliency / saliency.max() * 255).astype(np.uint8)
        
        # Smooth
        if self.gaussian_blur > 0:
            saliency = cv2.GaussianBlur(
                saliency,
                (self.gaussian_blur, self.gaussian_blur),
                0
            )
        
        return saliency
    
    def get_top_points(
        self,
        saliency_map: np.ndarray,
        k: int = 10,
        min_distance: int = 20,
    ) -> List[Tuple[int, int]]:
        """
        Extract top-k salient points.
        
        Args:
            saliency_map: Saliency map
            k: Number of points to extract
            min_distance: Minimum distance between points
        
        Returns:
            List of (x, y) coordinates
        
        Example:
            >>> points = detector.get_top_points(saliency, k=5)
            >>> for x, y in points:
            ...     cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        """
        # Threshold
        _, binary = cv2.threshold(
            saliency_map,
            self.threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        # Find local maxima
        points = []
        temp_map = saliency_map.copy()
        
        for _ in range(k):
            # Find max
            y, x = np.unravel_index(temp_map.argmax(), temp_map.shape)
            value = temp_map[y, x]
            
            if value < self.threshold:
                break
            
            points.append((int(x), int(y)))
            
            # Suppress neighborhood
            y1 = max(0, y - min_distance)
            y2 = min(temp_map.shape[0], y + min_distance)
            x1 = max(0, x - min_distance)
            x2 = min(temp_map.shape[1], x + min_distance)
            temp_map[y1:y2, x1:x2] = 0
        
        return points
    
    def get_salient_regions(
        self,
        saliency_map: np.ndarray,
        min_area: int = 100,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Extract salient regions as bounding boxes.
        
        Args:
            saliency_map: Saliency map
            min_area: Minimum region area
        
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        # Threshold
        _, binary = cv2.threshold(
            saliency_map,
            self.threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract bounding boxes
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))
        
        return regions
    
    def _spectral_residual(self, image: np.ndarray) -> np.ndarray:
        """Spectral residual saliency (Hou & Zhang 2007)."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # FFT
        fft = np.fft.fft2(gray.astype(np.float32))
        amplitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Log amplitude
        log_amplitude = np.log(amplitude + 1)
        
        # Spectral residual
        avg_filter = cv2.blur(log_amplitude, (3, 3))
        spectral_residual = log_amplitude - avg_filter
        
        # Inverse FFT
        saliency = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase)))
        
        return saliency.astype(np.uint8)
    
    def _fine_grained(self, image: np.ndarray) -> np.ndarray:
        """Fine-grained saliency using OpenCV."""
        try:
            if self._cv_saliency is None:
                self._cv_saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            
            success, saliency = self._cv_saliency.computeSaliency(image)
            if success:
                return (saliency * 255).astype(np.uint8)
        except:
            pass
        
        # Fallback to spectral
        return self._spectral_residual(image)
    
    def _edge_based(self, image: np.ndarray) -> np.ndarray:
        """Edge-based saliency."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to create regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        saliency = cv2.dilate(edges, kernel, iterations=2)
        
        return saliency
    
    def _contrast_based(self, image: np.ndarray) -> np.ndarray:
        """Contrast-based saliency."""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Compute contrast for each channel
        saliency = np.zeros(image.shape[:2], dtype=np.float32)
        
        for i in range(3):
            channel = lab[:, :, i].astype(np.float32)
            mean = cv2.blur(channel, (31, 31))
            contrast = np.abs(channel - mean)
            saliency += contrast
        
        return saliency.astype(np.uint8)
    
    def visualize(
        self,
        image: np.ndarray,
        saliency_map: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay saliency map on image.
        
        Args:
            image: Original RGB image
            saliency_map: Saliency map
            alpha: Overlay transparency
        
        Returns:
            Visualization image
        """
        # Create heatmap
        heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
