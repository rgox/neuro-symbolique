"""
VSA (Vector-Symbolic Architecture) Module.

Hyperdimensional computing for neural-symbolic grounding.

This module provides:
- HyperVector: Core hypervector operations (bind, bundle, permute)
- VSACodebook: Symbol-to-hypervector mapping
- NeuralGrounding: Project neural embeddings to VSA space
- GroundingCache: UMA-backed storage for grounded hypervectors

Example:
    >>> from nesy.reasoning.vsa import (
    ...     HyperVector, VSACodebook, NeuralGrounding
    ... )
    >>> 
    >>> # Create codebook
    >>> codebook = VSACodebook(dim=10000)
    >>> 
    >>> # Ground neural embedding
    >>> grounding = NeuralGrounding(codebook=codebook)
    >>> hv = grounding.ground_embedding(
    ...     embedding=neural_features,
    ...     attributes={"color": "red"},
    ...     object_class="cup"
    ... )
"""

from nesy.reasoning.vsa.hypervector import (
    HyperVector,
    HyperVectorType,
    HyperVectorConfig,
    cosine_similarity_batch
)

from nesy.reasoning.vsa.codebook import VSACodebook

from nesy.reasoning.vsa.grounding import (
    NeuralGrounding,
    GroundingCache
)

__all__ = [
    'HyperVector',
    'HyperVectorType',
    'HyperVectorConfig',
    'cosine_similarity_batch',
    'VSACodebook',
    'NeuralGrounding',
    'GroundingCache',
]
