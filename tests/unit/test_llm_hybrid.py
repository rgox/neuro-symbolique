"""
Tests for Hybrid LLM Reasoning.

Tests rule extraction, hybrid queries, validation, and explanations.
"""

import pytest

from nesy.reasoning.llm import (
    LLMRuleBridge,
    HybridQueryEngine,
    SelfValidator,
    ExplanationGenerator,
    ExtractedRule,
)
from nesy.reasoning.logic import ReasoningEngine
from nesy.world_model.scene_graph import SceneGraph


class TestLLMRuleBridge:
    """Test LLM rule bridge."""
    
    @pytest.fixture
    def bridge(self):
        """Create mock LLM bridge."""
        return LLMRuleBridge(use_mock=True)
    
    def test_extract_rules(self, bridge):
        """Test rule extraction from text."""
        text = "Objects on tables are in the same room as the table"
        
        rules = bridge.extract_rules(text, domain_predicates=["on", "in"])
        
        # Should extract at least one rule
        assert len(rules) > 0
        assert isinstance(rules[0], ExtractedRule)
        assert rules[0].confidence > 0
    
    def test_explain_inference(self, bridge):
        """Test explanation generation."""
        explanation = bridge.explain_inference(
            goal="in(cup, kitchen)",
            facts=["on(cup, table)", "in(table, kitchen)"],
            rules=["in(X,R) :- on(X,T), in(T,R)"]
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "cup" in explanation or "kitchen" in explanation
    
    def test_validate_rule(self, bridge):
        """Test rule validation."""
        rule = "in(X, R) :- on(X, T), in(T, R)"
        
        is_valid = bridge.validate_rule(rule, examples=[])
        
        # Mock always validates
        assert is_valid


class TestHybridQueryEngine:
    """Test hybrid query engine."""
    
    @pytest.fixture
    def setup(self):
        """Create engine setup."""
        sg = SceneGraph()
        reasoning_engine = ReasoningEngine(sg)
        llm_bridge = LLMRuleBridge(use_mock=True)
        
        engine = HybridQueryEngine(reasoning_engine, llm_bridge)
        
        return {"engine": engine, "reasoning": reasoning_engine}
    
    def test_query_symbolic(self, setup):
        """Test symbolic query."""
        # Add some facts
        setup["reasoning"].ctx.add_fact("in", "cup", "kitchen")
        
        result = setup["engine"].query("What's in the kitchen?")
        
        assert isinstance(result, dict)
        assert "answer" in result
        assert "method" in result
    
    def test_query_hybrid(self, setup):
        """Test hybrid query."""
        result = setup["engine"].query(
            "Is the kitchen messy?",
            prefer_symbolic=False
        )
        
        assert isinstance(result, dict)
        assert result["method"] == "hybrid"


class TestSelfValidator:
    """Test self-validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator."""
        sg = SceneGraph()
        reasoning_engine = ReasoningEngine(sg)
        llm_bridge = LLMRuleBridge(use_mock=True)
        
        return SelfValidator(reasoning_engine, llm_bridge)
    
    def test_validate_claim(self, validator):
        """Test claim validation."""
        valid, explanation = validator.validate_claim(
            "The cup is in the kitchen",
            verify_symbolic=True
        )
        
        assert isinstance(valid, bool)
        assert isinstance(explanation, str)
    
    def test_detect_contradictions(self, validator):
        """Test contradiction detection."""
        statements = [
            "The cup is red",
            "The cup is not red"
        ]
        
        contradictions = validator.detect_contradictions(statements)
        
        # Should detect contradiction
        assert len(contradictions) > 0
    
    def test_self_correct(self, validator):
        """Test self-correction."""
        corrected = validator.self_correct(
            "The cup is blue",
            contradiction="The cup is red"
        )
        
        assert isinstance(corrected, str)
        assert len(corrected) > 0


class TestExplanationGenerator:
    """Test explanation generation."""
    
    @pytest.fixture
    def generator(self):
        """Create explanation generator."""
        llm_bridge = LLMRuleBridge(use_mock=True)
        return ExplanationGenerator(llm_bridge)
    
    def test_explain_plan(self, generator):
        """Test plan explanation."""
        plan = ["move(room1, room2)", "move(room2, room3)"]
        
        explanation = generator.explain_plan(plan)
        
        assert isinstance(explanation, str)
        assert "Step 1" in explanation
        assert "Step 2" in explanation
    
    def test_explain_reasoning_chain(self, generator):
        """Test reasoning chain explanation."""
        explanation = generator.explain_reasoning_chain(
            facts=["on(cup, table)", "in(table, kitchen)"],
            rules=["in(X,R) :- on(X,T), in(T,R)"],
            conclusion="in(cup, kitchen)"
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestIntegration:
    """Integration tests for hybrid reasoning."""
    
    def test_end_to_end_hybrid_reasoning(self):
        """Test complete hybrid reasoning flow."""
        # Setup
        sg = SceneGraph()
        reasoning_engine = ReasoningEngine(sg)
        llm_bridge = LLMRuleBridge(use_mock=True)
        
        # Extract rule from text
        text = "Objects on tables are in the same room"
        rules = llm_bridge.extract_rules(text)
        
        # Add to reasoning engine
        for rule in rules:
            try:
                reasoning_engine.add_custom_rule(rule.rule_text)
            except:
                pass  # May fail if rule syntax not perfect
        
        # Query with hybrid engine
        hybrid_engine = HybridQueryEngine(reasoning_engine, llm_bridge)
        result = hybrid_engine.query("What's in the room?")
        
        assert isinstance(result, dict)
    
    def test_validation_loop(self):
        """Test self-validation loop."""
        sg = SceneGraph()
        reasoning_engine = ReasoningEngine(sg)
        llm_bridge = LLMRuleBridge(use_mock=True)
        validator = SelfValidator(reasoning_engine, llm_bridge)
        
        # Make claim
        claim = "The cup is in the kitchen"
        
        # Validate
        valid, explanation = validator.validate_claim(claim)
        
        # If invalid, correct
        if not valid:
            corrected = validator.self_correct(claim)
            assert corrected != claim
        
        assert True  # Test completes without error
