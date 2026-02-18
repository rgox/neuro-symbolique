"""
Tests for Natural Language Query Interface.

Tests all query patterns and LLM fallback.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from nesy.world_model.scene_graph import SceneGraph, LayerType, NodeType
from nesy.reasoning.logic import ReasoningEngine
from nesy.reasoning.nlquery import NLQueryInterface


class TestNLQueryInit:
    """Test NLQueryInterface initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        sg = SceneGraph()
        engine = ReasoningEngine(sg)
        nli = NLQueryInterface(engine)

        assert nli.engine is engine
        assert not nli.use_llm
        assert nli.llm_backend is None
        assert len(nli.patterns) == 5

    def test_init_with_llm(self):
        """Test initialization with LLM enabled."""
        sg = SceneGraph()
        engine = ReasoningEngine(sg)
        nli = NLQueryInterface(engine, use_llm=True, llm_backend="openai")

        assert nli.use_llm
        assert nli.llm_backend == "openai"


class TestNLQueryPatterns:
    """Test NL query pattern matching."""

    @pytest.fixture
    def setup(self):
        """Create scene with objects for querying."""
        sg = SceneGraph()

        # Add kitchen (location)
        kitchen = sg.add_node(
            LayerType.L1, NodeType.OBJECT, [0.0, 0.0, 0.0],
            attributes={"label": "kitchen", "class": "kitchen"},
            node_id="kitchen_1"
        )

        # Add cup (object in kitchen)
        cup = sg.add_node(
            LayerType.L1, NodeType.OBJECT, [1.0, 0.0, 0.0],
            attributes={"label": "cup", "class": "cup"},
            node_id="cup_1"
        )

        # Add table
        table = sg.add_node(
            LayerType.L1, NodeType.OBJECT, [2.0, 0.0, 0.0],
            attributes={"label": "table", "class": "table"},
            node_id="table_1"
        )

        engine = ReasoningEngine(sg)
        nli = NLQueryInterface(engine)

        return nli, engine, sg

    def test_find_all_query(self, setup):
        """Test 'Find all cups' pattern."""
        nli, engine, sg = setup

        # Mock find_all
        engine.find_all = MagicMock(return_value=["cup_1"])

        result = nli.query("Find all cups")

        engine.find_all.assert_called_once_with("cup")
        assert result == ["cup_1"]

    def test_in_location_query(self, setup):
        """Test 'What's in the kitchen?' pattern."""
        nli, engine, sg = setup

        # Mock find_in_location
        engine.find_in_location = MagicMock(return_value=["cup_1"])

        result = nli.query("What's in the kitchen?")

        assert isinstance(result, list)

    def test_in_location_not_found(self, setup):
        """Test query for non-existent location."""
        nli, engine, sg = setup

        result = nli.query("What's in the bathroom?")

        assert result == []

    def test_on_object_query(self, setup):
        """Test 'What's on the table?' pattern."""
        nli, engine, sg = setup

        # Mock query to return on-relations
        engine.query = MagicMock(return_value=[])

        result = nli.query("What's on the table?")

        assert isinstance(result, list)

    def test_on_object_not_found(self, setup):
        """Test on-query for non-existent object."""
        nli, engine, sg = setup

        result = nli.query("What's on the chair?")

        assert result == []

    def test_is_in_query(self, setup):
        """Test 'Is the cup in the kitchen?' pattern."""
        nli, engine, sg = setup

        # Mock infer
        engine.infer = MagicMock(return_value=True)

        result = nli.query("Is the cup in the kitchen?")

        # Result should be boolean-like
        assert isinstance(result, (bool, list))

    def test_is_in_not_found(self, setup):
        """Test is-in query with non-existent object."""
        nli, engine, sg = setup

        result = nli.query("Is the lamp in the kitchen?")

        assert result is False or result == []

    def test_where_is_query(self, setup):
        """Test 'Where is the cup?' pattern."""
        nli, engine, sg = setup

        # Mock query
        engine.query = MagicMock(return_value=[])

        result = nli.query("Where is the cup?")

        assert isinstance(result, list)

    def test_where_is_not_found(self, setup):
        """Test where-is for non-existent object."""
        nli, engine, sg = setup

        result = nli.query("Where is the lamp?")

        assert result == []

    def test_no_pattern_match(self, setup):
        """Test query with no matching pattern."""
        nli, engine, sg = setup

        result = nli.query("How old is the cup?")

        assert result == []

    def test_no_pattern_with_llm(self, setup):
        """Test fallback to LLM when no pattern matches."""
        nli, engine, sg = setup
        nli.use_llm = True

        result = nli.query("How old is the cup?")

        # LLM handler returns empty list (placeholder)
        assert result == []


class TestNLQueryCompilePatterns:
    """Test pattern compilation."""

    def test_compile_patterns(self):
        """Test that patterns compile correctly."""
        sg = SceneGraph()
        engine = ReasoningEngine(sg)
        nli = NLQueryInterface(engine)

        patterns = nli._compile_patterns()

        assert len(patterns) == 5
        for pattern, handler in patterns:
            assert hasattr(pattern, "search")
            assert callable(handler)

    def test_pattern_case_insensitive(self):
        """Test patterns are case insensitive."""
        sg = SceneGraph()
        engine = ReasoningEngine(sg)
        nli = NLQueryInterface(engine)

        # Mock find_all
        engine.find_all = MagicMock(return_value=[])

        # Various cases
        nli.query("FIND ALL cups")
        nli.query("find all cups")
        nli.query("Find All cups")

        assert engine.find_all.call_count == 3


class TestNLQueryHandlers:
    """Test individual handler methods."""

    @pytest.fixture
    def nli_with_data(self):
        """Create NLI with populated scene."""
        sg = SceneGraph()

        # Add objects
        sg.add_node(
            LayerType.L1, NodeType.OBJECT, [1.0, 1.0, 0.5],
            attributes={"label": "cup", "class": "cup"},
            node_id="cup_1"
        )
        sg.add_node(
            LayerType.L1, NodeType.OBJECT, [0.0, 0.0, 0.0],
            attributes={"label": "kitchen", "class": "kitchen"},
            node_id="kitchen_1"
        )

        engine = ReasoningEngine(sg)
        nli = NLQueryInterface(engine)

        return nli, engine, sg

    def test_handle_with_llm(self, nli_with_data):
        """Test LLM handler placeholder."""
        nli, engine, sg = nli_with_data

        result = nli._handle_with_llm("complex query")

        assert result == []

    def test_query_strip_whitespace(self, nli_with_data):
        """Test that query strips whitespace."""
        nli, engine, sg = nli_with_data

        engine.find_all = MagicMock(return_value=[])

        # Query with extra whitespace
        nli.query("  Find all cups  ")

        engine.find_all.assert_called_once()
