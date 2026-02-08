"""
Pipeline Module - Unified Neural-Symbolic Pipeline.

Provides high-level interfaces for complete neural-symbolic processing.

Example:
    >>> from nesy.pipeline import NeuralSymbolicPipeline
    >>> 
    >>> pipeline = NeuralSymbolicPipeline()
    >>> result = pipeline.process_image(image)
    >>> answer = pipeline.query("What's in the kitchen?")
"""

from nesy.pipeline.full_pipeline import NeuralSymbolicPipeline

__all__ = ['NeuralSymbolicPipeline']
