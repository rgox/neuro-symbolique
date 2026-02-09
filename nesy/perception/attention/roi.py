"""
Region of Interest (ROI) Proposals for Active Perception.

Proposes high-value regions for focused processing based on:
- Saliency maps
- Object detections
- Semantic importance
- Geometric constraints

Example:
    >>> from nesy.perception.attention import AttentionMechanism
    >>> 
    >>> attention = AttentionMechanism()
    >>> proposals = attention.propose_regions(image, top_k=5)
    >>> 
    >>> for roi in proposals:
    ...     crop = roi.extract_region(image)
    ...     # Process crop at high resolution
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from nesy.perception.attention.saliency import SaliencyDetector


@dataclass
class ROI:
    """
    Region of Interest.
    
    Attributes:
        bbox: Bounding box (x, y, w, h)
        score: Importance score [0, 1]
        label: Optional semantic label
        metadata: Additional information
    """
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    score: float
    label: Optional[str] = None
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def area(self) -> int:
        """Compute ROI area."""
        _, _, w, h = self.bbox
        return w * h
    
    @property
    def center(self) -> Tuple[int, int]:
        """Compute ROI center."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
    def extract_region(self, image: np.ndarray) -> np.ndarray:
        """
        Extract ROI from image.
        
        Args:
            image: Source image
        
        Returns:
            Cropped region
        """
        x, y, w, h = self.bbox
        return image[y:y+h, x:x+w]
    
    def iou(self, other: 'ROI') -> float:
        """
        Compute Intersection over Union with another ROI.
        
        Args:
            other: Another ROI
        
        Returns:
            IoU score [0, 1]
        """
        x1, y1, w1, h1 = self.bbox
        x2, y2, w2, h2 = other.bbox
        
        # Intersection
        x_int = max(x1, x2)
        y_int = max(y1, y2)
        w_int = min(x1 + w1, x2 + w2) - x_int
        h_int = min(y1 + h1, y2 + h2) - y_int
        
        if w_int <= 0 or h_int <= 0:
            return 0.0
        
        intersection = w_int * h_int
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class ROIProposal:
    """
    ROI proposal configuration.
    
    Attributes:
        min_size: Minimum ROI size (width, height)
        max_size: Maximum ROI size
        aspect_ratio_range: (min, max) aspect ratio
        nms_threshold: NMS IoU threshold
    """
    min_size: Tuple[int, int] = (32, 32)
    max_size: Tuple[int, int] = (512, 512)
    aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)
    nms_threshold: float = 0.5


class AttentionMechanism:
    """
    Attention-based region proposal mechanism.
    
    Proposes ROIs based on saliency, context, and task requirements.
    
    Args:
        saliency_detector: Saliency detector instance
        proposal_config: ROI proposal configuration
    
    Example:
        >>> attention = AttentionMechanism()
        >>> rois = attention.propose_regions(image, top_k=10)
        >>> 
        >>> for roi in rois:
        ...     print(f"ROI at {roi.center}: score={roi.score:.2f}")
    """
    
    def __init__(
        self,
        saliency_detector: Optional[SaliencyDetector] = None,
        proposal_config: Optional[ROIProposal] = None,
    ):
        """Initialize attention mechanism."""
        self.saliency_detector = saliency_detector or SaliencyDetector()
        self.config = proposal_config or ROIProposal()
    
    def propose_regions(
        self,
        image: np.ndarray,
        top_k: int = 10,
        use_saliency: bool = True,
        use_grid: bool = False,
    ) -> List[ROI]:
        """
        Propose regions of interest.
        
        Args:
            image: Input image
            top_k: Number of proposals
            use_saliency: Use saliency-based proposals
            use_grid: Use grid-based proposals
        
        Returns:
            List of ROI proposals sorted by score
        
        Example:
            >>> rois = attention.propose_regions(image, top_k=5)
            >>> best_roi = rois[0]
        """
        proposals = []
        
        # Saliency-based proposals
        if use_saliency:
            saliency_rois = self._saliency_proposals(image, k=top_k)
            proposals.extend(saliency_rois)
        
        # Grid-based proposals
        if use_grid:
            grid_rois = self._grid_proposals(image, grid_size=3)
            proposals.extend(grid_rois)
        
        # If no proposals, return full image
        if not proposals:
            h, w = image.shape[:2]
            proposals.append(ROI(
                bbox=(0, 0, w, h),
                score=1.0,
                label="full_image"
            ))
        
        # Non-maximum suppression
        proposals = self._nms(proposals, self.config.nms_threshold)
        
        # Sort by score and return top_k
        proposals.sort(key=lambda r: r.score, reverse=True)
        return proposals[:top_k]
    
    def adaptive_grid(
        self,
        image: np.ndarray,
        saliency_map: Optional[np.ndarray] = None,
        num_regions: int = 9,
    ) -> List[ROI]:
        """
        Generate adaptive grid based on saliency.
        
        High-saliency areas get finer grid cells.
        
        Args:
            image: Input image
            saliency_map: Optional pre-computed saliency
            num_regions: Target number of regions
        
        Returns:
            List of adaptive ROIs
        """
        h, w = image.shape[:2]
        
        # Compute saliency if needed
        if saliency_map is None:
            saliency_map = self.saliency_detector.compute_saliency(image)
        
        # Simple adaptive: divide into quadrants, subdivide high-saliency quadrants
        rois = []
        
        # Top-level grid
        grid_h, grid_w = 3, 3
        cell_h, cell_w = h // grid_h, w // grid_w
        
        for i in range(grid_h):
            for j in range(grid_w):
                y, x = i * cell_h, j * cell_w
                
                # Compute average saliency
                cell_saliency = saliency_map[y:y+cell_h, x:x+cell_w]
                avg_saliency = cell_saliency.mean() / 255.0
                
                rois.append(ROI(
                    bbox=(x, y, cell_w, cell_h),
                    score=avg_saliency,
                    label=f"grid_{i}_{j}",
                    metadata={"grid_cell": (i, j)}
                ))
        
        return rois
    
    def focus_window(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        window_size: Tuple[int, int] = (256, 256),
    ) -> ROI:
        """
        Create focused window around center point.
        
        Args:
            image: Input image
            center: (x, y) center point
            window_size: (width, height) of window
        
        Returns:
            ROI centered at point
        """
        h, w = image.shape[:2]
        win_w, win_h = window_size
        
        cx, cy = center
        
        # Compute bbox
        x = max(0, cx - win_w // 2)
        y = max(0, cy - win_h // 2)
        x = min(x, w - win_w)
        y = min(y, h - win_h)
        
        return ROI(
            bbox=(x, y, win_w, win_h),
            score=1.0,
            label="focus_window",
            metadata={"center": center}
        )
    
    def _saliency_proposals(
        self,
        image: np.ndarray,
        k: int = 10,
    ) -> List[ROI]:
        """Generate proposals from saliency map."""
        # Compute saliency
        saliency = self.saliency_detector.compute_saliency(image)
        
        # Get salient regions
        regions = self.saliency_detector.get_salient_regions(saliency)
        
        proposals = []
        for bbox in regions:
            x, y, w, h = bbox
            
            # Filter by size
            if w < self.config.min_size[0] or h < self.config.min_size[1]:
                continue
            if w > self.config.max_size[0] or h > self.config.max_size[1]:
                continue
            
            # Filter by aspect ratio
            aspect = w / h if h > 0 else 1.0
            if aspect < self.config.aspect_ratio_range[0] or aspect > self.config.aspect_ratio_range[1]:
                continue
            
            # Compute score from avg saliency
            region_saliency = saliency[y:y+h, x:x+w]
            score = region_saliency.mean() / 255.0
            
            proposals.append(ROI(
                bbox=bbox,
                score=score,
                label="saliency",
                metadata={"source": "saliency"}
            ))
        
        return proposals
    
    def _grid_proposals(
        self,
        image: np.ndarray,
        grid_size: int = 3,
    ) -> List[ROI]:
        """Generate regular grid proposals."""
        h, w = image.shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        
        proposals = []
        for i in range(grid_size):
            for j in range(grid_size):
                x, y = j * cell_w, i * cell_h
                
                proposals.append(ROI(
                    bbox=(x, y, cell_w, cell_h),
                    score=0.5,  # Uniform score
                    label=f"grid_{i}_{j}",
                    metadata={"source": "grid"}
                ))
        
        return proposals
    
    def _nms(
        self,
        rois: List[ROI],
        iou_threshold: float = 0.5,
    ) -> List[ROI]:
        """Non-maximum suppression for ROIs."""
        if not rois:
            return []
        
        # Sort by score
        rois = sorted(rois, key=lambda r: r.score, reverse=True)
        
        keep = []
        while rois:
            # Keep best
            best = rois.pop(0)
            keep.append(best)
            
            # Remove overlapping
            rois = [r for r in rois if best.iou(r) < iou_threshold]
        
        return keep
