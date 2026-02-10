"""
Tests for Knowledge Graph and Temporal Reasoning.
"""

import pytest
from nesy.world_model.knowledge_graph import KnowledgeGraph, ConceptNode, ConceptRelation
from nesy.reasoning.temporal import (
    TemporalReasoner, TemporalEvent, CausalRelation
)


class TestKnowledgeGraphOntology:
    """Test ontology hierarchy."""
    
    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_concept("thing")
        kg.add_concept("furniture", parent="thing")
        kg.add_concept("chair", parent="furniture")
        kg.add_concept("table", parent="furniture")
        kg.add_concept("armchair", parent="chair")
        kg.add_concept("food")
        kg.add_concept("fruit", parent="food")
        kg.add_concept("apple", parent="fruit")
        return kg
    
    def test_add_concept(self, kg):
        assert "chair" in kg.concepts
        assert kg.concepts["chair"].parent == "furniture"
    
    def test_is_a_direct(self, kg):
        assert kg.is_a("chair", "furniture")
    
    def test_is_a_transitive(self, kg):
        assert kg.is_a("chair", "thing")
        assert kg.is_a("armchair", "thing")
    
    def test_is_a_self(self, kg):
        assert kg.is_a("chair", "chair")
    
    def test_not_is_a(self, kg):
        assert not kg.is_a("chair", "food")
        assert not kg.is_a("furniture", "chair")
    
    def test_get_ancestors(self, kg):
        ancestors = kg.get_ancestors("armchair")
        assert "chair" in ancestors
        assert "furniture" in ancestors
        assert "thing" in ancestors
    
    def test_get_descendants(self, kg):
        desc = kg.get_descendants("furniture")
        assert "chair" in desc
        assert "table" in desc
        assert "armchair" in desc
    
    def test_get_children(self, kg):
        children = kg.get_children("furniture")
        assert "chair" in children
        assert "table" in children
        assert "armchair" not in children
    
    def test_get_siblings(self, kg):
        siblings = kg.get_siblings("chair")
        assert "table" in siblings
        assert "chair" not in siblings


class TestKnowledgeGraphProperties:
    """Test property management."""
    
    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_concept("furniture")
        kg.add_concept("chair", parent="furniture")
        kg.set_property("furniture", "has_legs", True)
        kg.set_property("chair", "typical_count", 4)
        return kg
    
    def test_direct_property(self, kg):
        assert kg.get_property("chair", "typical_count") == 4
    
    def test_inherited_property(self, kg):
        assert kg.get_property("chair", "has_legs") is True
    
    def test_no_inherit(self, kg):
        assert kg.get_property("chair", "has_legs", inherit=False) is None
    
    def test_missing_property(self, kg):
        assert kg.get_property("chair", "nonexistent") is None


class TestKnowledgeGraphInstances:
    """Test instance binding."""
    
    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_concept("furniture")
        kg.add_concept("chair", parent="furniture")
        kg.bind_instance("chair", "chair_1")
        kg.bind_instance("chair", "chair_2")
        return kg
    
    def test_bind_instance(self, kg):
        assert "chair_1" in kg.concepts["chair"].instances
    
    def test_get_instances(self, kg):
        instances = kg.get_instances("chair")
        assert len(instances) == 2
    
    def test_get_instances_descendants(self, kg):
        instances = kg.get_instances("furniture", include_descendants=True)
        assert len(instances) == 2


class TestKnowledgeGraphQuery:
    """Test SPARQL-like queries."""
    
    @pytest.fixture
    def kg(self):
        kg = KnowledgeGraph()
        kg.add_concept("furniture")
        kg.add_concept("chair", parent="furniture")
        kg.add_concept("table", parent="furniture")
        kg.set_property("chair", "material", "wood")
        kg.add_relation("leg", "table", ConceptRelation.PART_OF)
        return kg
    
    def test_query_is_a(self, kg):
        results = kg.query("SELECT ?x WHERE { ?x is_a furniture }")
        names = [r["?x"] for r in results]
        assert "chair" in names
        assert "table" in names
    
    def test_query_has_property(self, kg):
        results = kg.query("SELECT ?x WHERE { ?x has_property material }")
        names = [r["?x"] for r in results]
        assert "chair" in names
    
    def test_query_part_of(self, kg):
        results = kg.query("SELECT ?x WHERE { ?x part_of table }")
        names = [r["?x"] for r in results]
        assert "leg" in names


class TestKnowledgeGraphStats:
    """Test stats and export."""
    
    def test_stats(self):
        kg = KnowledgeGraph()
        kg.add_concept("a")
        kg.add_concept("b", parent="a")
        
        stats = kg.get_stats()
        assert stats["concepts"] == 2
        assert stats["max_depth"] == 1
    
    def test_to_dict(self):
        kg = KnowledgeGraph()
        kg.add_concept("a")
        d = kg.to_dict()
        assert "concepts" in d
        assert "a" in d["concepts"]
    
    def test_remove_concept(self):
        kg = KnowledgeGraph()
        kg.add_concept("a")
        kg.add_concept("b", parent="a")
        kg.remove_concept("b")
        assert "b" not in kg.concepts


class TestTemporalReasoner:
    """Test temporal reasoning."""
    
    @pytest.fixture
    def reasoner(self):
        r = TemporalReasoner()
        r.add_event("e1", "pick", 1.0, subject="cup")
        r.add_event("e2", "move", 2.0, subject="cup")
        r.add_event("e3", "place", 3.5, subject="cup")
        r.add_event("e4", "pick", 5.0, subject="plate")
        return r
    
    def test_add_event(self, reasoner):
        assert len(reasoner.events) == 4
    
    def test_timeline_sorted(self, reasoner):
        timestamps = [e.timestamp for e in reasoner.event_timeline]
        assert timestamps == sorted(timestamps)
    
    def test_events_in_range(self, reasoner):
        events = reasoner.get_events_in_range(1.0, 3.0)
        assert len(events) == 2
    
    def test_events_by_subject(self, reasoner):
        events = reasoner.get_events_by_subject("cup")
        assert len(events) == 3


class TestTemporalRelations:
    """Test Allen's interval algebra."""
    
    @pytest.fixture
    def reasoner(self):
        r = TemporalReasoner()
        r.add_event("a", "action", 1.0, duration=1.0)
        r.add_event("b", "action", 3.0, duration=1.0)
        r.add_event("c", "action", 1.0, duration=1.0)
        return r
    
    def test_before(self, reasoner):
        rel = reasoner.temporal_relation("a", "b")
        assert rel == "before"
    
    def test_after(self, reasoner):
        rel = reasoner.temporal_relation("b", "a")
        assert rel == "after"
    
    def test_equals(self, reasoner):
        rel = reasoner.temporal_relation("a", "c")
        assert rel == "equals"


class TestSequenceDetection:
    """Test pattern detection."""
    
    @pytest.fixture
    def reasoner(self):
        r = TemporalReasoner()
        r.add_event("e1", "pick", 1.0, subject="cup")
        r.add_event("e2", "move", 2.0, subject="cup")
        r.add_event("e3", "place", 3.0, subject="cup")
        r.add_event("e4", "pick", 5.0, subject="plate")
        r.add_event("e5", "move", 6.0, subject="plate")
        r.add_event("e6", "place", 7.0, subject="plate")
        return r
    
    def test_detect_sequence(self, reasoner):
        matches = reasoner.detect_sequence(["pick", "move", "place"])
        assert len(matches) >= 1
    
    def test_detect_sequence_subject(self, reasoner):
        matches = reasoner.detect_sequence(["pick", "move", "place"], subject="cup")
        assert len(matches) == 1
    
    def test_detect_repeated(self, reasoner):
        matches = reasoner.detect_repeated("pick", min_count=2, window=10.0)
        assert len(matches) >= 1


class TestCausalInference:
    """Test causal reasoning."""
    
    @pytest.fixture
    def reasoner(self):
        r = TemporalReasoner()
        r.add_event("e1", "pick", 1.0, subject="cup")
        r.add_event("e2", "move", 2.0, subject="cup")
        r.add_event("e3", "push", 10.0, subject="box")
        r.add_event("e4", "move", 10.5, subject="box")
        return r
    
    def test_infer_causality(self, reasoner):
        link = reasoner.infer_causality("e1", "e2")
        assert link is not None
        assert link.relation == CausalRelation.CAUSES
        assert link.confidence > 0.5
    
    def test_same_subject_higher_confidence(self, reasoner):
        link = reasoner.infer_causality("e1", "e2")
        assert link.confidence > 0.7
    
    def test_no_causality_reverse(self, reasoner):
        link = reasoner.infer_causality("e2", "e1")
        assert link is None
    
    def test_push_causes_move(self, reasoner):
        link = reasoner.infer_causality("e3", "e4")
        assert link is not None
    
    def test_causal_chain(self, reasoner):
        reasoner.infer_causality("e1", "e2")
        chain = reasoner.get_causal_chain("e1")
        assert len(chain) >= 1


class TestTemporalConstraints:
    """Test constraint checking."""
    
    def test_constraint_satisfied(self):
        r = TemporalReasoner()
        r.add_event("a", "pick", 1.0)
        r.add_event("b", "place", 3.0)
        r.add_constraint("a", "b", "before")
        
        results = r.check_constraints()
        assert results[0][1] is True
    
    def test_constraint_violated(self):
        r = TemporalReasoner()
        r.add_event("a", "pick", 5.0)
        r.add_event("b", "place", 1.0)
        r.add_constraint("a", "b", "before")
        
        results = r.check_constraints()
        assert results[0][1] is False


class TestTemporalPrediction:
    """Test event prediction."""
    
    def test_predict_next(self):
        r = TemporalReasoner()
        r.add_event("e1", "pick", 1.0)
        r.add_event("e2", "move", 2.0)
        
        predictions = r.predict_next(["pick"])
        assert len(predictions) > 0
        types = [p[0] for p in predictions]
        assert "move" in types or "hold" in types
    
    def test_predict_from_history(self):
        r = TemporalReasoner()
        r.add_event("e1", "pick", 1.0)
        r.add_event("e2", "move", 2.0)
        r.add_event("e3", "pick", 3.0)
        r.add_event("e4", "move", 4.0)
        
        predictions = r.predict_next()
        assert len(predictions) > 0


class TestTemporalStats:
    """Test stats."""
    
    def test_get_stats(self):
        r = TemporalReasoner()
        r.add_event("e1", "a", 1.0)
        r.add_event("e2", "b", 5.0)
        
        stats = r.get_stats()
        assert stats["total_events"] == 2
        assert stats["event_types"] == 2
        assert stats["timeline_span"] == 4.0
