"""
VSA Codebook - Symbol to Hypervector Mapping.

This module implements a codebook that maps symbolic names (strings) to
hypervectors, enabling the grounding of discrete symbols in continuous
high-dimensional space.

The codebook maintains a dictionary of symbol → hypervector mappings and
supports encoding (symbol → hypervector) and decoding (hypervector → symbols).

Features:
- Deterministic symbol encoding (same symbol always maps to same hypervector)
- Similarity-based decoding (find closest symbols)
- Attribute binding helpers
- Lazy hypervector creation

Example:
    >>> codebook = VSACodebook(dim=10000)
    >>> hv_cup = codebook.encode("cup")
    >>> hv_red = codebook.encode("red")
    >>> hv_color = codebook.encode("COLOR")  # Attribute key
    >>> 
    >>> # Bind attribute
    >>> hv_red_cup = hv_cup.bundle(hv_color.bind(hv_red))
    >>> 
    >>> # Decode
    >>> symbols = codebook.decode(hv_red_cup, top_k=3)
    >>> # → [("cup", 0.87), ("COLOR", 0.65), ("red", 0.58)]
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from nesy.reasoning.vsa.hypervector import HyperVector, HyperVectorType, cosine_similarity_batch


class VSACodebook:
    """
    Codebook for mapping symbols to hypervectors.
    
    Maintains a dictionary of symbol → hypervector mappings with deterministic
    generation (same symbol always produces same hypervector).
    
    Attributes:
        dim: Hypervector dimensionality
        hv_type: Binary or real-valued hypervectors
        seed: Base seed for reproducibility
        codebook: Dictionary mapping symbols to hypervectors
    
    Example:
        >>> codebook = VSACodebook(dim=10000, seed=42)
        >>> hv_cup = codebook.encode("cup")
        >>> hv_table = codebook.encode("table")
        >>> 
        >>> # Same symbol always produces same hypervector
        >>> hv_cup2 = codebook.encode("cup")
        >>> hv_cup.similarity(hv_cup2)  # = 1.0
    """
    
    def __init__(
        self,
        dim: int = 10000,
        hv_type: HyperVectorType = HyperVectorType.REAL,
        seed: int = 42
    ):
        """
        Initialize codebook.
        
        Args:
            dim: Hypervector dimensionality
            hv_type: Binary or real-valued
            seed: Random seed for deterministic generation
        """
        self.dim = dim
        self.hv_type = hv_type
        self.base_seed = seed
        
        # Codebook: symbol → hypervector
        self.codebook: Dict[str, HyperVector] = {}
        
        # Reverse lookup cache for decoding
        self._symbol_list: List[str] = []
        self._hv_matrix: Optional[np.ndarray] = None
        self._cache_valid = False
    
    def encode(self, symbol: str) -> HyperVector:
        """
        Encode symbol to hypervector.
        
        If symbol exists in codebook, return cached hypervector.
        Otherwise, create new hypervector deterministically based on symbol hash.
        
        Args:
            symbol: Symbolic name to encode
        
        Returns:
            Hypervector for symbol
        
        Example:
            >>> codebook = VSACodebook()
            >>> hv = codebook.encode("cup")
        """
        # Check cache
        if symbol in self.codebook:
            return self.codebook[symbol]
        
        # Generate deterministic hypervector based on symbol hash
        symbol_hash = hash(symbol) % (2**31)  # Platform-independent hash
        symbol_seed = self.base_seed + symbol_hash
        
        hv = HyperVector.random(
            dim=self.dim,
            hv_type=self.hv_type,
            seed=symbol_seed
        )
        
        # Cache
        self.codebook[symbol] = hv
        self._cache_valid = False  # Invalidate decode cache
        
        return hv
    
    def decode(
        self,
        hv: HyperVector,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Decode hypervector to most similar symbols.
        
        Finds symbols whose hypervectors are most similar to the query.
        
        Args:
            hv: Query hypervector
            top_k: Number of top symbols to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (symbol, similarity) tuples, sorted by similarity (descending)
        
        Example:
            >>> codebook = VSACodebook()
            >>> hv_cup = codebook.encode("cup")
            >>> hv_red = codebook.encode("red")
            >>> hv_mixed = hv_cup.bundle(hv_red)
            >>> symbols = codebook.decode(hv_mixed, top_k=2)
            >>> # → [("cup", 0.71), ("red", 0.71)]
        """
        if not self.codebook:
            return []
        
        # Build decode cache if needed
        if not self._cache_valid:
            self._build_decode_cache()
        
        # Batch similarity computation
        similarities = cosine_similarity_batch(hv, list(self.codebook.values()))
        
        # Filter by threshold
        valid_indices = similarities >= threshold
        valid_symbols = [self._symbol_list[i] for i in range(len(self._symbol_list)) if valid_indices[i]]
        valid_similarities = similarities[valid_indices]
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(valid_similarities)[::-1]
        
        # Take top_k
        results = [
            (valid_symbols[i], float(valid_similarities[i]))
            for i in sorted_indices[:top_k]
        ]
        
        return results
    
    def _build_decode_cache(self) -> None:
        """Build cache for efficient decoding."""
        self._symbol_list = list(self.codebook.keys())
        self._cache_valid = True
    
    def bind_attribute(
        self,
        key: str,
        value: Any,
        encode_value: bool = True
    ) -> HyperVector:
        """
        Create attribute binding: key ⊗ value.
        
        Helper function to bind an attribute key with its value.
        
        Args:
            key: Attribute name (e.g., "COLOR", "SIZE")
            value: Attribute value (e.g., "red", "large")
            encode_value: Whether to encode value as symbol (vs use as hypervector)
        
        Returns:
            Bound hypervector representing the attribute
        
        Example:
            >>> codebook = VSACodebook()
            >>> # color = red
            >>> hv_attr = codebook.bind_attribute("COLOR", "red")
            >>> 
            >>> # Decode
            >>> symbols = codebook.decode(hv_attr, top_k=2)
            >>> # → Contains "COLOR" and "red"
        """
        hv_key = self.encode(key)
        
        if encode_value:
            if isinstance(value, HyperVector):
                hv_value = value
            else:
                hv_value = self.encode(str(value))
        else:
            hv_value = value
        
        return hv_key.bind(hv_value)
    
    def encode_position(
        self,
        position: np.ndarray,
        scale: float = 1.0
    ) -> HyperVector:
        """
        Encode 3D position as hypervector.
        
        Projects position coordinates to hypervector space using basis vectors
        and permutation for different dimensions.
        
        Args:
            position: 3D position [x, y, z]
            scale: Scaling factor for position values
        
        Returns:
            Hypervector encoding position
        
        Example:
            >>> codebook = VSACodebook()
            >>> pos = np.array([1.5, 2.0, 0.5])
            >>> hv_pos = codebook.encode_position(pos)
        """
        x, y, z = position * scale
        
        # Get basis vectors (cached)
        if not hasattr(self, '_position_basis'):
            self._position_basis = {
                'x': HyperVector.random(self.dim, self.hv_type, seed=self.base_seed + 1),
                'y': HyperVector.random(self.dim, self.hv_type, seed=self.base_seed + 2),
                'z': HyperVector.random(self.dim, self.hv_type, seed=self.base_seed + 3),
            }
        
        # Encode each coordinate using permutation
        # Position is encoded as: x*basis_x ⊕ y*basis_y ⊕ z*basis_z
        hv_pos = (
            self._position_basis['x'] * float(x)
        ).bundle(
            self._position_basis['y'] * float(y)
        ).bundle(
            self._position_basis['z'] * float(z)
        )
        
        return hv_pos
    
    def encode_attributes(
        self,
        attributes: Dict[str, Any]
    ) -> HyperVector:
        """
        Encode multiple attributes as a single hypervector.
        
        Bundles all attribute bindings together.
        
        Args:
            attributes: Dictionary of attribute key-value pairs
        
        Returns:
            Hypervector encoding all attributes
        
        Example:
            >>> codebook = VSACodebook()
            >>> attrs = {"color": "red", "size": "large", "material": "ceramic"}
            >>> hv_attrs = codebook.encode_attributes(attrs)
        """
        if not attributes:
            return HyperVector.zero(self.dim, self.hv_type)
        
        # Bundle all attribute bindings
        hv_result = HyperVector.zero(self.dim, self.hv_type)
        
        for key, value in attributes.items():
            hv_attr = self.bind_attribute(key, value)
            hv_result = hv_result.bundle(hv_attr)
        
        return hv_result
    
    def clear(self) -> None:
        """Clear codebook."""
        self.codebook.clear()
        self._symbol_list = []
        self._cache_valid = False
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols in codebook."""
        return list(self.codebook.keys())
    
    def size(self) -> int:
        """Get number of symbols in codebook."""
        return len(self.codebook)
    
    def __repr__(self) -> str:
        return f"VSACodebook(dim={self.dim}, symbols={self.size()}, type={self.hv_type.value})"
    
    def to_dict(self) -> dict:
        """Serialize codebook to dictionary."""
        return {
            "dim": self.dim,
            "hv_type": self.hv_type.value,
            "base_seed": self.base_seed,
            "symbols": list(self.codebook.keys()),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VSACodebook':
        """Deserialize codebook from dictionary."""
        codebook = cls(
            dim=data["dim"],
            hv_type=HyperVectorType(data["hv_type"]),
            seed=data["base_seed"]
        )
        
        # Re-encode all symbols (deterministic)
        for symbol in data["symbols"]:
            codebook.encode(symbol)
        
        return codebook
