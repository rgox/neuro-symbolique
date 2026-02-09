"""
Tests for Multi-Modal Perception.

Tests text-to-object search, visual grounding, and semantic queries.
"""

import pytest
import numpy as np

from nesy.perception.multimodal import MultiModalQuery, QueryResult
from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType
from nesy.perception.feature_extraction import FeatureVector


class TestMultiModalQuery:
    """Test multi-modal query functionality."""
    
    @pytest.fixture
    def scene_graph(self):
        """Create scene graph with test objects."""
        sg = SceneGraph()
        
        # Add some objects with fake CLIP embeddings
        # In reality these would be from CLIP, but we use random for testing
        np.random.seed(42)
        
        # Add cups with similar embeddings
        for i in range(3):
            node_id = f"cup_{i}"
            sg.add_node(
                node_id=node_id,
                layer=LayerType.L1,
                node_type=NodeType.OBJECT,
                position=np.array([i, 0, 0]),
                attributes={
                    "class": "cup",
                    "color": "red" if i == 0 else "blue",
                    "embedding": np.random.randn(512),  # Fake CLIP embedding
                }
            )
        
        # Add plates
        for i in range(2):
            node_id = f"plate_{i}"
            sg.add_node(
                node_id=node_id,
                layer=LayerType.L1,
                node_type=NodeType.OBJECT,
                position=np.array([i, 1, 0]),
                attributes={
                    "class": "plate",
                    "embedding": np.random.randn(512),
                }
            )
        
        return sg
    
    def test_initialization(self, scene_graph):
        """Test query initialization."""
        query = MultiModalQuery(scene_graph=scene_graph)
        
        assert query.scene_graph is scene_graph
        assert query.top_k == 5
        assert query._clip is None  # Lazy load
    
    def test_find_objects_basic(self, scene_graph):
        """Test basic object finding (may skip if CLIP not installed)."""
        query = MultiModalQuery(scene_graph=scene_graph)
        
        try:
            results = query.find_objects("cup", top_k=3)
            
            # Should return QueryResults
            assert isinstance(results, list)
            assert len(results) <= 3
            
            if len(results) > 0:
                assert isinstance(results[0], QueryResult)
                assert hasattr(results[0].node, "id")
                assert 0 <= results[0].similarity <= 1
        
        except ImportError:
            pytest.skip("CLIP not installed")
    
    def test_ground(self, scene_graph):
        """Test visual grounding."""
        query = MultiModalQuery(scene_graph=scene_graph)
        
        try:
            result = query.ground("a red cup")
            
            # May return None or a result
            if result is not None:
                assert isinstance(result, QueryResult)
                assert result.node.id in scene_graph.nodes
        
        except ImportError:
            pytest.skip("CLIP not installed")
    
    def test_find_similar(self, scene_graph):
        """Test finding similar objects."""
        query = MultiModalQuery(scene_graph=scene_graph)
        
        # Get reference cup
        cup_node = scene_graph.nodes["cup_0"]
        
        try:
            results = query.find_similar(cup_node, top_k=2)
            
            # Should find other cups
            assert isinstance(results, list)
            
            # Should not include reference itself
            if len(results) > 0:
                assert all(r.node.id != "cup_0" for r in results)
        
        except ImportError:
            pytest.skip("CLIP not installed")
    
    def test_semantic_search(self, scene_graph):
        """Test zero-shot semantic search."""
        query = MultiModalQuery(scene_graph=scene_graph)
        
        try:
            results = query.semantic_search(["cup", "plate", "fork"])
            
            # Should return dict
            assert isinstance(results, dict)
            assert "cup" in results
            assert "plate" in results
            assert "fork" in results
            
            # Each should be a list
            assert isinstance(results["cup"], list)
            assert isinstance(results["plate"], list)
        
        except ImportError:
            pytest.skip("CLIP not installed")
    
    def test_get_node_embedding_from_attributes(self, scene_graph):
        """Test getting embedding from node attributes."""
        query = MultiModalQuery(scene_graph=scene_graph)
        
        cup = scene_graph.nodes["cup_0"]
        emb = query._get_node_embedding(cup)
        
        # Should get embedding from attributes
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert len(emb) > 0
    
    def test_get_node_embedding_from_class_name(self, scene_graph):
        """Test generating embedding from class name."""
        # Add node without embedding
        scene_graph.add_node(
            node_id="test_obj",
            layer=LayerType.L1,
            node_type=NodeType.OBJECT,
            position=np.array([0, 0, 0]),
            attributes={"class": "chair"}
        )
        
        query = MultiModalQuery(scene_graph=scene_graph)
        node = scene_graph.nodes["test_obj"]
        
        try:
            emb = query._get_node_embedding(node)
            
            # Should generate from class name
            assert emb is not None
            assert isinstance(emb, np.ndarray)
        
        except ImportError:
            pytest.skip("CLIP not installed")


class TestQueryResult:
    """Test QueryResult dataclass."""
    
    def test_creation(self):
        """Test creating query result."""
        from nesy.world_model.scene_graph import Node
        
        node = Node(
            id="test",
            layer=LayerType.L1,
            node_type=NodeType.OBJECT,
            position=np.array([0, 0, 0]),
            attributes={"class": "cup"}
        )
        
        result = QueryResult(
            node=node,
            similarity=0.85,
            explanation="Test match"
        )
        
        assert result.node is node
        assert result.similarity == 0.85
        assert result.explanation == "Test match"
        assert result.metadata == {}
    
    def test_with_metadata(self):
        """Test query result with metadata."""
        from nesy.world_model.scene_graph import Node
        
        node = Node(
            id="test",
            layer=LayerType.L1,
            node_type=NodeType.OBJECT,
            position=np.array([0, 0, 0]),
            attributes={}
        )
        
        result = QueryResult(
            node=node,
            similarity=0.9,
            metadata={"query": "test", "method": "clip"}
        )
        
        assert result.metadata["query"] == "test"
        assert result.metadata["method"] == "clip"
