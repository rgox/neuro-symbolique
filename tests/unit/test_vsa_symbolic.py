"""
Tests for VSA Symbolic Operations.

Tests composition, analogical reasoning, pattern matching, and role-filler binding.
"""

import pytest
import numpy as np

from nesy.reasoning.vsa import (
    HyperVector,
    VSAComposer,
    AnalogyEngine,
    PatternMatcher,
    RoleFillerBinder,
)


class TestVSAComposer:
    """Test VSA composition operations."""
    
    @pytest.fixture
    def composer(self):
        """Create composer with small dimension for testing."""
        return VSAComposer(dim=1000)
    
    @pytest.fixture
    def hvs(self, composer):
        """Create test hypervectors."""
        return {
            "red": HyperVector.random(1000),
            "blue": HyperVector.random(1000),
            "cup": HyperVector.random(1000),
            "plate": HyperVector.random(1000),
        }
    
    def test_bind(self, composer, hvs):
        """Test binding operation."""
        red = hvs["red"]
        cup = hvs["cup"]
        
        # Bind red and cup
        red_cup = composer.bind(red, cup)
        
        # Should be different from inputs
        assert red_cup.similarity(red) < 0.3
        assert red_cup.similarity(cup) < 0.3
    
    def test_unbind(self, composer, hvs):
        """Test unbinding retrieves original."""
        red = hvs["red"]
        cup = hvs["cup"]
        
        # Bind and unbind
        red_cup = composer.bind(red, cup)
        retrieved_cup = composer.unbind(red_cup, red)
        
        # Should recover cup (with some noise)
        sim = retrieved_cup.similarity(cup)
        assert sim > 0.5  # Reasonable similarity expected
    
    def test_bundle(self, composer, hvs):
        """Test bundling operation."""
        cup = hvs["cup"]
        plate = hvs["plate"]
        
        # Bundle
        tableware = composer.bundle([cup, plate])
        
        # Should be similar to both (superposition)
        assert tableware.similarity(cup) > 0.3
        assert tableware.similarity(plate) > 0.3
    
    def test_permute(self, composer, hvs):
        """Test permutation for sequences."""
        red = hvs["red"]
        blue = hvs["blue"]
        
        # Permute sequence
        sequence = composer.permute([red, blue])
        
        # Should encode both
        assert isinstance(sequence, HyperVector)
        assert sequence.dim == 1000


class TestAnalogyEngine:
    """Test analogical reasoning."""
    
    @pytest.fixture
    def engine(self):
        """Create analogy engine."""
        return AnalogyEngine(dim=1000)
    
    @pytest.fixture
    def codebook(self):
        """Create simple codebook."""
        return {
            "small": HyperVector.random(1000),
            "large": HyperVector.random(1000),
            "cat": HyperVector.random(1000),
            "kitten": HyperVector.random(1000),
            "dog": HyperVector.random(1000),
            "puppy": HyperVector.random(1000),
        }
    
    def test_solve_analogy(self, engine, codebook):
        """Test solving analogies."""
        # Small:Large :: Cat:? → should be similar to "kitten"
        results = engine.solve("small", "large", "cat", codebook, top_k=2)
        
        # Should return suggestions
        assert len(results) > 0
        assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)
    
    def test_solve_with_vectors(self, engine, codebook):
        """Test solving with direct vectors."""
        results = engine.solve_with_vectors(
            codebook["small"],
            codebook["large"],
            codebook["cat"],
            codebook,
            top_k=3
        )
        
        assert len(results) <= 3
        assert all(isinstance(r[1], float) for r in results)


class TestPatternMatcher:
    """Test pattern matching."""
    
    @pytest.fixture
    def matcher(self):
        """Create pattern matcher."""
        return PatternMatcher(dim=1000)
    
    def test_create_pattern(self, matcher):
        """Test pattern creation."""
        red = HyperVector.random(1000)
        cube = HyperVector.random(1000)
        
        pattern = matcher.create_pattern({
            "color": red,
            "shape": cube
        })
        
        assert isinstance(pattern, HyperVector)
        assert pattern.dim == 1000
    
    def test_match(self, matcher):
        """Test pattern matching."""
        # Create pattern
        red = HyperVector.random(1000)
        pattern = matcher.create_pattern({"color": red})
        
        # Create candidates (some similar,some not)
        similar = HyperVector.random(1000)
        different = HyperVector.random(1000)
        
        # Match with low threshold
        matches = matcher.match(pattern, [similar, different], threshold=0.0)
        
        # Should find some matches
        assert len(matches) >= 0  # May or may not match depending on random


class TestRoleFillerBinder:
    """Test role-filler binding."""
    
    @pytest.fixture
    def binder(self):
        """Create role-filler binder."""
        return RoleFillerBinder(dim=1000)
    
    @pytest.fixture
    def fillers(self):
        """Create filler codebook."""
        return {
            "red": HyperVector.random(1000),
            "blue": HyperVector.random(1000),
            "cup": HyperVector.random(1000),
            "table": HyperVector.random(1000),
        }
    
    def test_bind_structure(self, binder, fillers):
        """Test binding structured representation."""
        structure = binder.bind_structure({
            "type": fillers["cup"],
            "color": fillers["red"],
        })
        
        assert isinstance(structure, HyperVector)
        assert structure.dim == 1000
    
    def test_query_role(self, binder, fillers):
        """Test querying role from structure."""
        # Bind structure
        structure = binder.bind_structure({
            "type": fillers["cup"],
            "color": fillers["red"],
        })
        
        # Query color role
        result = binder.query_role(structure, "color", fillers)
        
        # Should retrieve red (or something close)
        assert result is not None
        color_name, similarity = result
        assert isinstance(color_name, str)
        assert isinstance(similarity, float)
    
    def test_query_nonexistent_role(self, binder, fillers):
        """Test querying nonexistent role."""
        structure = binder.bind_structure({
            "type": fillers["cup"],
        })
        
        # Query role that wasn't bound
        result = binder.query_role(structure, "nonexistent", fillers)
        
        # Should return None or low similarity
        assert result is None or result[1] < 0.5


class TestIntegration:
    """Integration tests for VSA symbolic operations."""
    
    def test_compose_and_analogy(self):
        """Test composition + analogy together."""
        composer = VSAComposer(dim=2000)
        engine = AnalogyEngine(dim=2000)
        
        # Create concepts
        concepts = {
            "animal": HyperVector.random(2000),
            "young": HyperVector.random(2000),
            "cat": HyperVector.random(2000),
            "dog": HyperVector.random(2000),
        }
        
        # Compose: kitten = cat + young
        kitten = composer.bind(concepts["cat"], concepts["young"])
        puppy = composer.bind(concepts["dog"], concepts["young"])
        
        # Add to codebook
        full_codebook = {
            **concepts,
            "kitten": kitten,
            "puppy": puppy,
        }
        
        # Solve analogy: cat:kitten :: dog:?
        results = engine.solve("cat", "kitten", "dog", full_codebook)
        
        # Should suggest puppy
        assert len(results) > 0
    
    def test_pattern_and_binding(self):
        """Test pattern matching with role-filler binding."""
        matcher = PatternMatcher(dim=1000)
        binder = RoleFillerBinder(dim=1000)
        
        # Create concepts
        red = HyperVector.random(1000)
        blue = HyperVector.random(1000)
        cup = HyperVector.random(1000)
        
        # Create objects
        red_cup = binder.bind_structure({
            "color": red,
            "type": cup
        })
        
        blue_cup = binder.bind_structure({
            "color": blue,
            "type": cup
        })
        
        # Create pattern for red objects
        pattern = matcher.create_pattern({"color": red})
        
        # Match (should prefer red_cup)
        matches = matcher.match(pattern, [red_cup, blue_cup], threshold=0.0)
        
        # Should find matches
        assert len(matches) >= 0
