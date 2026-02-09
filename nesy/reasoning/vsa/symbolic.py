"""
VSA Symbolic Operations Module.

Advanced hypervector operations for symbolic reasoning:
- Composition: bind, bundle, permute
- Analogical reasoning: A:B :: C:?
- Pattern matching: template similarity
- Role-filler binding: structured representations

Example:
    >>> from nesy.reasoning.vsa.symbolic import VSAComposer, AnalogyEngine
    >>> 
    >>> # Compose concepts
    >>> composer = VSAComposer(dim=10000)
    >>> red_cup = composer.bind(color_hv, cup_hv)
    >>> 
    >>> # Analogical reasoning
    >>> analogy = AnalogyEngine(dim=10000)
    >>> result = analogy.solve("cat", "kitten", "dog")  # dog:puppy
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from nesy.reasoning.vsa.hypervector import HyperVector


class VSAComposer:
    """
    VSA composition operations.
    
    Implements bind, bundle, and permute for concept composition.
    
    Example:
        >>> composer = VSAComposer(dim=10000)
        >>> 
        >>> # Bind: role-filler (RED * CUP)
        >>> red_cup = composer.bind(red_hv, cup_hv)
        >>> 
        >>> # Bundle: superposition (CUP + PLATE + FORK)
        >>> tableware = composer.bundle([cup_hv, plate_hv, fork_hv])
        >>> 
        >>> # Permute: sequence (WORD1 ⊕ WORD2 ⊕ WORD3)
        >>> sentence = composer.permute([word1, word2, word3])
    """
    
    def __init__(self, dim: int = 10000):
        """Initialize composer with dimension."""
        self.dim = dim
    
    def bind(self, hv1: HyperVector, hv2: HyperVector) -> HyperVector:
        """
        Bind two hypervectors (role-filler binding).
        
        Uses element-wise multiplication (XOR for binary).
        
        Args:
            hv1: First hypervector (e.g., attribute)
            hv2: Second hypervector (e.g., object)
        
        Returns:
            Bound hypervector
        
        Example:
            >>> # RED * CUP = red_cup
            >>> red_cup = composer.bind(red_hv, cup_hv)
            >>> 
            >>> # Unbind: RED * red_cup ≈ CUP
            >>> retrieved = composer.bind(red_hv, red_cup)
            >>> retrieved.similarity(cup_hv)  # → ~1.0
        """
        # Element-wise multiplication
        result = hv1.data * hv2.data
        
        # Normalize
        result_hv = HyperVector.from_vector(result, normalize=True)
        
        return result_hv
    
    def bundle(self, hvs: List[HyperVector], weights: Optional[List[float]] = None) -> HyperVector:
        """
        Bundle multiple hypervectors (superposition).
        
        Uses element-wise addition with optional weights.
        
        Args:
            hvs: List of hypervectors to bundle
            weights: Optional weights for each vector
        
        Returns:
            Bundled hypervector
        
        Example:
            >>> # Tableware = CUP + PLATE + FORK
            >>> tableware = composer.bundle([cup, plate, fork])
            >>> 
            >>> # Check membership
            >>> tableware.similarity(cup)  # → ~0.6 (present in bundle)
        """
        if not hvs:
            raise ValueError("Cannot bundle empty list")
        
        if weights is None:
            weights = [1.0] * len(hvs)
        
        if len(weights) != len(hvs):
            raise ValueError("Weights must match number of hypervectors")
        
        # Weighted sum
        result = np.zeros(self.dim)
        for hv, w in zip(hvs, weights):
            result += w * hv.data
        
        # Normalize
        result_hv = HyperVector.from_vector(result, normalize=True)
        
        return result_hv
    
    def permute(self, hvs: List[HyperVector], forward: bool = True) -> HyperVector:
        """
        Permute hypervector (for sequences).
        
        Applies rotation for position encoding.
        
        Args:
            hvs: List of hypervectors in sequence
            forward: If True, left rotation; if False, right rotation
        
        Returns:
            Permuted sequence representation
        
        Example:
            >>> # Encode sentence: THE + CAT⊕ + SAT⊕⊕
            >>> sentence = composer.permute([the, cat, sat])
        """
        if not hvs:
            raise ValueError("Cannot permute empty list")
        
        result = np.zeros(self.dim)
        
        for i, hv in enumerate(hvs):
            # Rotate by i positions
            rotated = np.roll(hv.data, i if forward else -i)
            result += rotated
        
        # Normalize
        result_hv = HyperVector.from_vector(result, normalize=True)
        
        return result_hv
    
    def unbind(self, bound: HyperVector, known: HyperVector) -> HyperVector:
        """
        Unbind to retrieve component.
        
        Args:
            bound: Bound hypervector (A * B)
            known: Known component (A)
        
        Returns:
            Retrieved component (≈ B)
        
        Example:
            >>> red_cup = composer.bind(red, cup)
            >>> retrieved_cup = composer.unbind(red_cup, red)
            >>> retrieved_cup.similarity(cup)  # → ~1.0
        """
        # Binding is its own inverse for multiplication
        return self.bind(bound, known)


class AnalogyEngine:
    """
    Analogical reasoning with hypervectors.
    
    Solves analogies of form A:B :: C:?
    
    Example:
        >>> engine = AnalogyEngine(dim=10000)
        >>> 
        >>> # Cat:Kitten :: Dog:?
        >>> result = engine.solve("cat", "kitten", "dog", codebook)
        >>> # → "puppy" (most similar to dog + (kitten - cat))
    """
    
    def __init__(self, dim: int = 10000):
        """Initialize analogy engine."""
        self.dim = dim
        self.composer = VSAComposer(dim)
    
    def solve(
        self,
        a: str,
        b: str,
        c: str,
        codebook: Dict[str, HyperVector],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Solve analogy A:B :: C:?
        
        Computes: C + (B - A) and finds most similar in codebook.
        
        Args:
            a: First term (e.g., "cat")
            b: Second term (e.g., "kitten")
            c: Third term (e.g., "dog")
            codebook: Dict of concept names → hypervectors
            top_k: Number of results to return
        
        Returns:
            List of (concept, similarity) tuples
        
        Example:
            >>> results = engine.solve("cat", "kitten", "dog", codebook)
            >>> # → [("puppy", 0.92), ("pup", 0.88), ...]
        """
        # Get hypervectors
        hv_a = codebook.get(a)
        hv_b = codebook.get(b)
        hv_c = codebook.get(c)
        
        if hv_a is None or hv_b is None or hv_c is None:
            raise ValueError(f"Concepts not in codebook: {a}, {b}, {c}")
        
        # Compute analogy: C + (B - A)
        diff = hv_b.data - hv_a.data
        target = hv_c.data + diff
        target_hv = HyperVector.from_vector(target, normalize=True)
        
        # Find most similar in codebook
        similarities = []
        for concept, hv in codebook.items():
            if concept not in [a, b, c]:  # Exclude inputs
                sim = target_hv.similarity(hv)
                similarities.append((concept, sim))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def solve_with_vectors(
        self,
        hv_a: HyperVector,
        hv_b: HyperVector,
        hv_c: HyperVector,
        codebook: Dict[str, HyperVector],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Solve analogy with hypervector inputs directly.
        
        Args:
            hv_a: First hypervector
            hv_b: Second hypervector
            hv_c: Third hypervector
            codebook: Concept codebook
            top_k: Number of results
        
        Returns:
            List of (concept, similarity) tuples
        """
        # Compute C + (B - A)
        diff = hv_b.data - hv_a.data
        target = hv_c.data + diff
        target_hv = HyperVector.from_vector(target, normalize=True)
        
        # Find most similar
        similarities = [
            (concept, target_hv.similarity(hv))
            for concept, hv in codebook.items()
        ]
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class PatternMatcher:
    """
    Pattern matching with VSA templates.
    
    Matches structured patterns against observations.
    
    Example:
        >>> matcher = PatternMatcher()
        >>> 
        >>> # Define pattern: RED object ON TABLE
        >>> pattern = matcher.create_pattern({
        ...     "color": red_hv,
        ...     "relation": on_hv,
        ...     "location": table_hv
        ... })
        >>> 
        >>> # Match against scene
        >>> matches = matcher.match(pattern, scene_objects)
    """
    
    def __init__(self, dim: int = 10000):
        """Initialize pattern matcher."""
        self.dim = dim
        self.composer = VSAComposer(dim)
    
    def create_pattern(self, slots: Dict[str, HyperVector]) -> HyperVector:
        """
        Create pattern from role-filler slots.
        
        Args:
            slots: Dict of role → filler hypervectors
        
        Returns:
            Pattern hypervector
        
        Example:
            >>> pattern = matcher.create_pattern({
            ...     "shape": cube_hv,
            ...     "color": red_hv
            ... })
        """
        # Bind each role-filler and bundle
        bound_slots = []
        
        for role, filler in slots.items():
            # Create role hypervector (could be from codebook)
            role_hv = HyperVector.random(self.dim)
            bound = self.composer.bind(role_hv, filler)
            bound_slots.append(bound)
        
        # Bundle all slots
        pattern = self.composer.bundle(bound_slots)
        
        return pattern
    
    def match(
        self,
        pattern: HyperVector,
        candidates: List[HyperVector],
        threshold: float = 0.7
    ) -> List[Tuple[int, float]]:
        """
        Match pattern against candidates.
        
        Args:
            pattern: Pattern hypervector
            candidates: List of candidate hypervectors
            threshold: Minimum similarity threshold
        
        Returns:
            List of (index, similarity) for matches
        
        Example:
            >>> matches = matcher.match(pattern, scene_objects, threshold=0.8)
            >>> # → [(2, 0.92), (5, 0.84)]  # Objects 2 and 5 match
        """
        matches = []
        
        for i, candidate in enumerate(candidates):
            sim = pattern.similarity(candidate)
            if sim >= threshold:
                matches.append((i, sim))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches


class RoleFillerBinder:
    """
    Role-filler binding for structured representations.
    
    Represents structured objects with roles and fillers.
    
    Example:
        >>> binder = RoleFillerBinder()
        >>> 
        >>> # Represent "red cup on table"
        >>> obj = binder.bind_structure({
        ...     "type": cup_hv,
        ...     "color": red_hv,
        ...     "location": table_hv
        ... })
        >>> 
        >>> # Query: what color?
        >>> color = binder.query_role(obj, "color", codebook)
        >>> # → red_hv (most similar to red)
    """
    
    def __init__(self, dim: int = 10000, role_codebook: Optional[Dict[str, HyperVector]] = None):
        """
        Initialize role-filler binder.
        
        Args:
            dim: Hypervector dimension
            role_codebook: Optional codebook for role names
        """
        self.dim = dim
        self.composer = VSAComposer(dim)
        
        # Create or use role codebook
        if role_codebook:
            self.role_codebook = role_codebook
        else:
            # Create default roles
            self.role_codebook = {
                "type": HyperVector.random(dim),
                "color": HyperVector.random(dim),
                "size": HyperVector.random(dim),
                "location": HyperVector.random(dim),
                "relation": HyperVector.random(dim),
            }
    
    def bind_structure(self, bindings: Dict[str, HyperVector]) -> HyperVector:
        """
        Bind roles to fillers.
        
        Args:
            bindings: Dict of role_name → filler_hv
        
        Returns:
            Structured representation
        """
        bound_pairs = []
        
        for role_name, filler_hv in bindings.items():
            # Get role hypervector
            role_hv = self.role_codebook.get(role_name)
            if role_hv is None:
                # Create new role
                role_hv = HyperVector.random(self.dim)
                self.role_codebook[role_name] = role_hv
            
            # Bind role to filler
            bound = self.composer.bind(role_hv, filler_hv)
            bound_pairs.append(bound)
        
        # Bundle all bindings
        structure = self.composer.bundle(bound_pairs)
        
        return structure
    
    def query_role(
        self,
        structure: HyperVector,
        role_name: str,
        filler_codebook: Dict[str, HyperVector]
    ) -> Optional[Tuple[str, float]]:
        """
        Query structure for role filler.
        
        Args:
            structure: Structured hypervector
            role_name: Role to query
            filler_codebook: Codebook of possible fillers
        
        Returns:
            (filler_name, similarity) or None
        """
        # Get role hypervector
        role_hv = self.role_codebook.get(role_name)
        if role_hv is None:
            return None
        
        # Unbind to get filler
        retrieved = self.composer.unbind(structure, role_hv)
        
        # Find most similar in filler codebook
        best_match = None
        best_sim = -1.0
        
        for filler_name, filler_hv in filler_codebook.items():
            sim = retrieved.similarity(filler_hv)
            if sim > best_sim:
                best_sim = sim
                best_match = filler_name
        
        return (best_match, best_sim) if best_match else None
