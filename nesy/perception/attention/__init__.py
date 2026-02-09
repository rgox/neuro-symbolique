"""
Active Perception Module.

Implements attention-based perception for efficient processing:
- Attention mechanisms: Focus on salient regions
- Saliency maps: Identify important areas
- ROI proposals: Select regions of interest
- Adaptive processing: Variable resolution

Example:
    >>> from nesy.perception.attention import AttentionMechanism, SaliencyDetector
    >>> 
    >>> # Generate saliency map
    >>> saliency = SaliencyDetector()
    >>> heatmap = saliency.compute_saliency(image)
    >>> 
    >>> # Get top ROIs
    >>> attention = AttentionMechanism()
    >>> rois = attention.propose_regions(image, top_k=5)
"""

from nesy.perception.attention.saliency import SaliencyDetector, SaliencyMethod
from nesy.perception.attention.roi import AttentionMechanism, ROI, ROIProposal

__all__ = [
    "SaliencyDetector",
    "SaliencyMethod",
    "AttentionMechanism",
    "ROI",
    "ROIProposal",
]
