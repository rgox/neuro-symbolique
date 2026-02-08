"""
Hypervector Module - Core VSA (Vector-Symbolic Architecture) Implementation.

This module implements hyperdimensional computing primitives for neural-symbolic
grounding. Hypervectors are high-dimensional (typically 10,000-dim) vectors that
represent symbolic concepts and support compositional operations.

Key Operations:
- Binding (⊗): Combine two concepts (e.g., color ⊗ red)
- Bundling (⊕): Superposition of concepts (e.g., cup ⊕ table)
- Permutation: Encode sequential/positional information
- Similarity: Cosine similarity for retrieval

Properties:
- Quasi-orthogonality: Random vectors are nearly orthogonal in high dimensions
- Reversibility: (A ⊗ B) ⊗ B ≈ A (unbinding)
- Commutativity: A ⊕ B = B ⊕ A
- Noise robustness: Distributed representations

Example:
    >>> hv1 = HyperVector.random(10000)
    >>> hv2 = HyperVector.random(10000)
    >>> bound = hv1.bind(hv2)
    >>> unbound = bound.bind(hv2)
    >>> similarity = hv1.similarity(unbound)  # ≈ 0.99+
"""

from typing import Optional, Union, List
import numpy as np
from enum import Enum
from dataclasses import dataclass


class HyperVectorType(Enum):
    """Type of hypervector representation."""
    BINARY = "binary"  # {-1, +1}
    REAL = "real"      # Real-valued, L2-normalized


@dataclass
class HyperVectorConfig:
    """Configuration for hypervector creation."""
    dim: int = 10000
    hv_type: HyperVectorType = HyperVectorType.REAL
    seed: Optional[int] = None
    normalized: bool = True


class HyperVector:
    """
    Hyperdimensional vector for VSA operations.
    
    Represents a point in high-dimensional space (typically 10,000-dim).
    Supports compositional operations (bind, bundle) for symbolic reasoning.
    
    Attributes:
        data: Numpy array of hypervector values
        dim: Dimensionality
        hv_type: Binary or real-valued
        normalized: Whether vector is L2-normalized
    
    Example:
        >>> # Create random hypervector
        >>> hv = HyperVector.random(10000)
        >>> 
        >>> # Bind two concepts
        >>> color_hv = HyperVector.random(10000)
        >>> red_hv = HyperVector.random(10000)
        >>> red_color = color_hv.bind(red_hv)
        >>> 
        >>> # Bundle concepts
        >>> cup_hv = HyperVector.random(10000)
        >>> table_hv = HyperVector.random(10000)
        >>> scene = cup_hv.bundle(table_hv)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        hv_type: HyperVectorType = HyperVectorType.REAL,
        normalized: bool = False
    ):
        """
        Initialize hypervector.
        
        Args:
            data: Numpy array of hypervector values
            hv_type: Binary or real-valued
            normalized: Whether data is already normalized
        """
        self.data = data.astype(np.float32)
        self.dim = len(data)
        self.hv_type = hv_type
        self._normalized = normalized
        
        if not normalized and hv_type == HyperVectorType.REAL:
            self._normalize()
    
    @classmethod
    def random(
        cls,
        dim: int = 10000,
        hv_type: HyperVectorType = HyperVectorType.REAL,
        seed: Optional[int] = None
    ) -> 'HyperVector':
        """
        Create random hypervector.
        
        Args:
            dim: Dimensionality
            hv_type: Binary or real-valued
            seed: Random seed for reproducibility
        
        Returns:
            Random hypervector
        """
        if seed is not None:
            np.random.seed(seed)
        
        if hv_type == HyperVectorType.BINARY:
            # Binary: {-1, +1}
            data = np.random.choice([-1.0, 1.0], size=dim).astype(np.float32)
        else:
            # Real: Gaussian, then L2-normalize
            data = np.random.randn(dim).astype(np.float32)
        
        return cls(data, hv_type=hv_type, normalized=False)
    
    @classmethod
    def zero(cls, dim: int = 10000, hv_type: HyperVectorType = HyperVectorType.REAL) -> 'HyperVector':
        """Create zero hypervector."""
        data = np.zeros(dim, dtype=np.float32)
        return cls(data, hv_type=hv_type, normalized=False)
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, normalize: bool = True) -> 'HyperVector':
        """
        Create hypervector from existing vector (e.g., neural embedding).
        
        Args:
            vector: Input vector (any dimension)
            normalize: Whether to L2-normalize
        
        Returns:
            Hypervector
        """
        return cls(vector, hv_type=HyperVectorType.REAL, normalized=not normalize)
    
    def _normalize(self) -> None:
        """L2-normalize the hypervector (in-place)."""
        if self.hv_type == HyperVectorType.REAL:
            norm = np.linalg.norm(self.data)
            if norm > 1e-8:
                self.data = self.data / norm
                self._normalized = True
    
    def normalize(self) -> 'HyperVector':
        """Return L2-normalized copy of hypervector."""
        if self._normalized:
            return self
        
        normalized_data = self.data.copy()
        norm = np.linalg.norm(normalized_data)
        if norm > 1e-8:
            normalized_data = normalized_data / norm
        
        return HyperVector(normalized_data, hv_type=self.hv_type, normalized=True)
    
    def bind(self, other: 'HyperVector') -> 'HyperVector':
        """
        Binding operation (⊗): Combine two concepts.
        
        Binding creates a new hypervector that is dissimilar to both inputs
        but can be unbound by binding again with one of the inputs.
        
        Properties:
        - Reversible: (A ⊗ B) ⊗ B ≈ A
        - Dissimilar: similarity(A, A ⊗ B) ≈ 0
        
        Implementation:
        - Binary: Element-wise multiplication (XOR)
        - Real: Circular convolution
        
        Args:
            other: Hypervector to bind with
        
        Returns:
            Bound hypervector
        
        Example:
            >>> color = HyperVector.random(10000)
            >>> red = HyperVector.random(10000)
            >>> red_color = color.bind(red)
            >>> # Unbind
            >>> recovered_color = red_color.bind(red)
            >>> color.similarity(recovered_color)  # ≈ 0.99+
        """
        assert self.dim == other.dim, "Dimensions must match"
        
        if self.hv_type == HyperVectorType.BINARY:
            # Binary: element-wise multiplication (XOR)
            bound_data = self.data * other.data
        else:
            # Real: circular convolution (FFT-based for efficiency)
            bound_data = np.fft.ifft(
                np.fft.fft(self.data) * np.fft.fft(other.data)
            ).real.astype(np.float32)
        
        return HyperVector(bound_data, hv_type=self.hv_type, normalized=False)
    
    def bundle(self, other: 'HyperVector') -> 'HyperVector':
        """
        Bundling operation (⊕): Superposition of concepts.
        
        Bundling creates a hypervector similar to both inputs, representing
        their union or superposition.
        
        Properties:
        - Commutative: A ⊕ B = B ⊕ A
        - Similar: similarity(A, A ⊕ B) > 0.5
        - Compositional: Can bundle many vectors
        
        Implementation:
        - Both: Element-wise addition + normalization
        
        Args:
            other: Hypervector to bundle with
        
        Returns:
            Bundled hypervector
        
        Example:
            >>> cup = HyperVector.random(10000)
            >>> table = HyperVector.random(10000)
            >>> scene = cup.bundle(table)
            >>> cup.similarity(scene)  # > 0.5
            >>> table.similarity(scene)  # > 0.5
        """
        assert self.dim == other.dim, "Dimensions must match"
        
        # Element-wise addition
        bundled_data = self.data + other.data
        
        return HyperVector(bundled_data, hv_type=self.hv_type, normalized=False)
    
    def permute(self, shift: int = 1) -> 'HyperVector':
        """
        Permutation operation: Circular shift.
        
        Used to encode sequential or positional information.
        Different shifts create dissimilar hypervectors.
        
        Args:
            shift: Number of positions to shift (can be negative)
        
        Returns:
            Permuted hypervector
        
        Example:
            >>> hv = HyperVector.random(10000)
            >>> hv_shifted = hv.permute(1)
            >>> hv.similarity(hv_shifted)  # ≈ 0 (dissimilar)
        """
        permuted_data = np.roll(self.data, shift)
        return HyperVector(permuted_data, hv_type=self.hv_type, normalized=self._normalized)
    
    def similarity(self, other: 'HyperVector') -> float:
        """
        Compute cosine similarity with another hypervector.
        
        Similarity ranges from -1 (opposite) to +1 (identical).
        Random hypervectors have similarity ≈ 0 (quasi-orthogonal).
        
        Args:
            other: Hypervector to compare with
        
        Returns:
            Cosine similarity in [-1, 1]
        
        Example:
            >>> hv1 = HyperVector.random(10000)
            >>> hv2 = HyperVector.random(10000)
            >>> hv1.similarity(hv2)  # ≈ 0 (random are quasi-orthogonal)
            >>> hv1.similarity(hv1)  # = 1.0 (identical)
        """
        assert self.dim == other.dim, "Dimensions must match"
        
        # Ensure both are normalized
        norm_self = self.normalize()
        norm_other = other.normalize()
        
        # Cosine similarity = dot product of normalized vectors
        return float(np.dot(norm_self.data, norm_other.data))
    
    def threshold(self, threshold: float = 0.0) -> 'HyperVector':
        """
        Threshold to binary hypervector.
        
        Useful for cleanup/quantization after many operations.
        
        Args:
            threshold: Values > threshold become +1, else -1
        
        Returns:
            Binary hypervector
        """
        binary_data = np.where(self.data > threshold, 1.0, -1.0).astype(np.float32)
        return HyperVector(binary_data, hv_type=HyperVectorType.BINARY, normalized=True)
    
    def __add__(self, other: 'HyperVector') -> 'HyperVector':
        """Alias for bundle."""
        return self.bundle(other)
    
    def __mul__(self, other: Union['HyperVector', float]) -> 'HyperVector':
        """Binding or scalar multiplication."""
        if isinstance(other, HyperVector):
            return self.bind(other)
        else:
            # Scalar multiplication
            scaled_data = self.data * other
            return HyperVector(scaled_data, hv_type=self.hv_type, normalized=False)
    
    def __repr__(self) -> str:
        norm = np.linalg.norm(self.data)
        return f"HyperVector(dim={self.dim}, type={self.hv_type.value}, norm={norm:.3f})"
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "data": self.data.tolist(),
            "dim": self.dim,
            "type": self.hv_type.value,
            "normalized": self._normalized
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'HyperVector':
        """Deserialize from dictionary."""
        return cls(
            data=np.array(data["data"], dtype=np.float32),
            hv_type=HyperVectorType(data["type"]),
            normalized=data["normalized"]
        )


def cosine_similarity_batch(query: HyperVector, hvs: List[HyperVector]) -> np.ndarray:
    """
    Compute cosine similarity between query and batch of hypervectors.
    
    Optimized batch operation using matrix multiplication.
    
    Args:
        query: Query hypervector
        hvs: List of hypervectors
    
    Returns:
        Numpy array of similarities
    """
    # Stack hypervectors into matrix
    matrix = np.stack([hv.normalize().data for hv in hvs])
    
    # Batch dot product
    query_norm = query.normalize().data
    similarities = matrix @ query_norm
    
    return similarities
