"""
Neural Grounding Layer - Project Neural Embeddings to VSA Space.

This module implements the grounding layer that bridges neural perception
and symbolic reasoning by projecting neural embeddings (e.g., from CLIP,
ResNet) into hyperdimensional VSA space.

The grounding process:
1. Project neural embedding (512-dim) → VSA space (10,000-dim)
2. Bind symbolic attributes (class, color, etc.)
3. Create composite hypervector representing grounded concept

Example:
    >>> grounding = NeuralGrounding(neural_dim=512, vsa_dim=10000, codebook=codebook)
    >>> 
    >>> # Neural features from CLIP
    >>> neural_embedding = np.random.randn(512)
    >>> 
    >>> # Ground to VSA
    >>> hv_grounded = grounding.ground_embedding(
    ...     embedding=neural_embedding,
    ...     attributes={"color": "red", "size": "large"},
    ...     object_class="cup"
    ... )
"""

from typing import Dict, Any, Optional
import numpy as np

from nesy.reasoning.vsa.hypervector import HyperVector, HyperVectorType
from nesy.reasoning.vsa.codebook import VSACodebook
from nesy.core.memory import UMA


class NeuralGrounding:
    """
    Ground neural embeddings in VSA hypervector space.
    
    Projects low-dimensional neural features to high-dimensional VSA space
    and binds them with symbolic attributes to create compositional
    representations.
    
    Architecture:
        Neural embedding (512-dim)
              ↓ Linear projection
        VSA space (10,000-dim)
              ↓ Bind attributes
        Grounded hypervector
    
    Attributes:
        neural_dim: Dimension of input neural embeddings
        vsa_dim: Dimension of hypervector space
        codebook: VSA codebook for symbol encoding
        projection_matrix: Random projection matrix (neural_dim → vsa_dim)
        uma: Optional UMA for storing projections
    
    Example:
        >>> codebook = VSACodebook(dim=10000)
        >>> grounding = NeuralGrounding(
        ...     neural_dim=512,
        ...     vsa_dim=10000,
        ...     codebook=codebook
        ... )
        >>> 
        >>> # Ground neural embedding
        >>> embedding = np.random.randn(512)
        >>> hv = grounding.ground_embedding(
        ...     embedding=embedding,
        ...     attributes={"color": "red"},
        ...     object_class="cup"
        ... )
    """
    
    def __init__(
        self,
        neural_dim: int = 512,
        vsa_dim: int = 10000,
        codebook: VSACodebook = None,
        uma: Optional[UMA] = None,
        projection_seed: int = 42
    ):
        """
        Initialize neural grounding layer.
        
        Args:
            neural_dim: Dimension of neural embeddings (e.g., 512 for CLIP)
            vsa_dim: Dimension of VSA hypervectors
            codebook: VSA codebook (created if None)
            uma: Optional UMA for storing projections
            projection_seed: Seed for random projection matrix
        """
        self.neural_dim = neural_dim
        self.vsa_dim = vsa_dim
        self.uma = uma
        
        # Create codebook if not provided
        if codebook is None:
            codebook = VSACodebook(dim=vsa_dim)
        self.codebook = codebook
        
        # Initialize random projection matrix (neural_dim → vsa_dim)
        # Using Gaussian random projection
        np.random.seed(projection_seed)
        self.projection_matrix = np.random.randn(
            neural_dim, vsa_dim
        ).astype(np.float32) / np.sqrt(neural_dim)
        
        # Normalize columns (each VSA dimension)
        self.projection_matrix = self.projection_matrix / (
            np.linalg.norm(self.projection_matrix, axis=0, keepdims=True) + 1e-8
        )
    
    def _project_neural(self, embedding: np.ndarray) -> HyperVector:
        """
        Project neural embedding to VSA space.
        
        Uses random projection matrix to map low-dimensional neural features
        to high-dimensional VSA space.
        
        Args:
            embedding: Neural embedding (neural_dim,)
        
        Returns:
            Hypervector in VSA space
        """
        assert len(embedding) == self.neural_dim, \
            f"Expected embedding dim {self.neural_dim}, got {len(embedding)}"
        
        # Project: embedding @ projection_matrix
        projected = embedding @ self.projection_matrix
        
        # Create hypervector
        hv = HyperVector.from_vector(projected, normalize=True)
        
        return hv
    
    def ground_embedding(
        self,
        embedding: np.ndarray,
        attributes: Dict[str, Any],
        object_class: str,
        position: Optional[np.ndarray] = None
    ) -> HyperVector:
        """
        Ground neural embedding with symbolic attributes.
        
        Creates composite hypervector:
            hv_grounded = hv_class
                        ⊕ hv_features
                        ⊕ hv_position (if provided)
                        ⊕ (hv_key ⊗ hv_value) for each attribute
        
        Args:
            embedding: Neural embedding (e.g., from CLIP, ResNet)
            attributes: Dictionary of symbolic attributes (color, size, etc.)
            object_class: Object class name (e.g., "cup", "table")
            position: Optional 3D position [x, y, z]
        
        Returns:
            Grounded hypervector combining neural and symbolic information
        
        Example:
            >>> hv = grounding.ground_embedding(
            ...     embedding=clip_features,
            ...     attributes={"color": "red", "material": "ceramic"},
            ...     object_class="cup",
            ...     position=np.array([1.5, 2.0, 0.8])
            ... )
        """
        # 1. Project neural embedding to VSA space
        hv_features = self._project_neural(embedding)
        
        # 2. Encode object class
        hv_class = self.codebook.encode(object_class)
        
        # 3. Encode position if provided
        if position is not None:
            hv_position = self.codebook.encode_position(position)
        else:
            hv_position = None
        
        # 4. Encode attributes
        hv_attributes = self.codebook.encode_attributes(attributes)
        
        # 5. Bundle all components
        hv_grounded = hv_class.bundle(hv_features)
        
        if hv_position is not None:
            hv_grounded = hv_grounded.bundle(hv_position)
        
        if attributes:
            hv_grounded = hv_grounded.bundle(hv_attributes)
        
        return hv_grounded
    
    def unbind_class(self, hv_grounded: HyperVector) -> str:
        """
        Decode object class from grounded hypervector.
        
        Args:
            hv_grounded: Grounded hypervector
        
        Returns:
            Most likely class name
        
        Example:
            >>> class_name = grounding.unbind_class(hv_grounded)
            >>> # → "cup"
        """
        # Decode to find most similar symbol
        symbols = self.codebook.decode(hv_grounded, top_k=1)
        
        if symbols:
            return symbols[0][0]
        else:
            return "unknown"
    
    def unbind_attribute(
        self,
        hv_grounded: HyperVector,
        attribute_key: str
    ) -> Optional[str]:
        """
        Decode specific attribute from grounded hypervector.
        
        Unbinds attribute by binding with key and decoding result.
        
        Args:
            hv_grounded: Grounded hypervector
            attribute_key: Attribute to extract (e.g., "color", "size")
        
        Returns:
            Attribute value or None if not found
        
        Example:
            >>> color = grounding.unbind_attribute(hv_grounded, "COLOR")
            >>> # → "red"
        """
        # Get key hypervector
        hv_key = self.codebook.encode(attribute_key)
        
        # Unbind: hv_grounded ⊗ hv_key should be similar to hv_value
        hv_unbound = hv_grounded.bind(hv_key)
        
        # Decode
        symbols = self.codebook.decode(hv_unbound, top_k=3)
        
        # Filter out the key itself
        for symbol, sim in symbols:
            if symbol != attribute_key and not symbol.isupper():
                return symbol
        
        return None
    
    def similarity(
        self,
        hv1: HyperVector,
        hv2: HyperVector
    ) -> float:
        """
        Compute similarity between two grounded hypervectors.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
        
        Returns:
            Cosine similarity in [-1, 1]
        """
        return hv1.similarity(hv2)
    
    def __repr__(self) -> str:
        return (
            f"NeuralGrounding(neural_dim={self.neural_dim}, "
            f"vsa_dim={self.vsa_dim}, codebook_size={self.codebook.size()})"
        )


class GroundingCache:
    """
    Cache for grounded hypervectors with UMA integration.
    
    Stores frequently used grounded hypervectors in UMA for efficient
    zero-copy sharing across modules.
    
    Attributes:
        uma: Unified Memory Architecture
        cache: Dictionary mapping keys to UMA buffer keys
    """
    
    def __init__(self, uma: UMA):
        """
        Initialize grounding cache.
        
        Args:
            uma: Unified Memory Architecture instance
        """
        self.uma = uma
        self.cache: Dict[str, str] = {}  # object_id → uma_key
    
    def store(
        self,
        object_id: str,
        hv: HyperVector
    ) -> str:
        """
        Store grounded hypervector in UMA.
        
        Args:
            object_id: Unique identifier for object
            hv: Hypervector to store
        
        Returns:
            UMA key for stored hypervector
        """
        uma_key = f"vsa_grounded_{object_id}"
        
        # Allocate in UMA if not exists
        if not self.uma.exists(uma_key):
            from nesy.core.memory import DataType, DeviceType
            
            self.uma.allocate(
                key=uma_key,
                shape=(hv.dim,),
                dtype=DataType.FLOAT32,
                device=DeviceType.CPU,
                metadata={"type": "vsa_grounded", "object_id": object_id}
            )
        
        # Store hypervector data
        buffer = self.uma.get(uma_key)
        buffer.data[:] = hv.data
        
        # Cache mapping
        self.cache[object_id] = uma_key
        
        return uma_key
    
    def retrieve(self, object_id: str) -> Optional[HyperVector]:
        """
        Retrieve grounded hypervector from UMA.
        
        Args:
            object_id: Unique identifier for object
        
        Returns:
            Hypervector or None if not found
        """
        if object_id not in self.cache:
            return None
        
        uma_key = self.cache[object_id]
        
        if not self.uma.exists(uma_key):
            return None
        
        buffer = self.uma.get(uma_key)
        hv = HyperVector.from_vector(buffer.data, normalize=False)
        
        return hv
    
    def clear(self) -> None:
        """Clear cache and free UMA buffers."""
        for uma_key in self.cache.values():
            if self.uma.exists(uma_key):
                self.uma.free(uma_key)
        
        self.cache.clear()
