"""Tests for Minalog — Pure Python Datalog Engine."""

import pytest
from nesy.reasoning.logic.minalog import (
    MinalogContext, Term, Atom, Comparison, Rule,
)


class TestTerm:
    def test_variable(self):
        t = Term("X", True)
        assert t.is_variable
        assert t.value == "X"
        assert "Var" in repr(t)

    def test_constant(self):
        t = Term("cup", False)
        assert not t.is_variable
        assert "Const" in repr(t)

    def test_equality(self):
        assert Term("X", True) == Term("X", True)
        assert Term("X", True) != Term("X", False)

    def test_hash(self):
        s = {Term("X", True), Term("X", True)}
        assert len(s) == 1


class TestAtom:
    def test_repr(self):
        a = Atom("edge", [Term("a", False), Term("b", False)])
        assert "edge" in repr(a)


class TestComparison:
    def test_gt(self):
        c = Comparison(Term("X", True), ">", Term(5, False))
        assert c.evaluate({"X": 10})
        assert not c.evaluate({"X": 3})

    def test_eq(self):
        c = Comparison(Term("X", True), "==", Term("hello", False))
        assert c.evaluate({"X": "hello"})

    def test_unbound_variable(self):
        c = Comparison(Term("X", True), ">", Term(5, False))
        assert not c.evaluate({})  # X not bound

    def test_right_variable(self):
        c = Comparison(Term("X", True), ">=", Term("Y", True))
        assert c.evaluate({"X": 10, "Y": 5})
        assert not c.evaluate({"X": 3, "Y": 5})

    def test_right_unbound(self):
        c = Comparison(Term(5, False), "<", Term("Y", True))
        assert not c.evaluate({})


class TestRule:
    def test_repr(self):
        head = Atom("path", [Term("X", True), Term("Z", True)])
        body = [Atom("edge", [Term("X", True), Term("Z", True)])]
        r = Rule(head, body)
        assert "path" in repr(r)
        assert "edge" in repr(r)


class TestMinalogContext:
    def test_add_relation(self):
        ctx = MinalogContext()
        ctx.add_relation("edge", ["String", "String"])
        assert "edge" in ctx.relations

    def test_add_facts(self):
        ctx = MinalogContext()
        ctx.add_facts("edge", [("a", "b"), ("b", "c")])
        assert ("a", "b") in ctx.facts["edge"]
        assert ("b", "c") in ctx.facts["edge"]

    def test_simple_query(self):
        ctx = MinalogContext()
        ctx.add_relation("edge", ["String", "String"])
        ctx.add_facts("edge", [("a", "b")])
        results = ctx.query("edge")
        assert ("a", "b") in results

    def test_relation_iterator(self):
        ctx = MinalogContext()
        ctx.add_facts("r", [("x", "y")])
        results = list(ctx.relation("r"))
        assert ("x", "y") in results

    def test_empty_relation(self):
        ctx = MinalogContext()
        assert list(ctx.relation("nonexistent")) == []

    def test_simple_rule(self):
        ctx = MinalogContext()
        ctx.add_relation("edge", ["String", "String"])
        ctx.add_relation("path", ["String", "String"])
        ctx.add_facts("edge", [("a", "b"), ("b", "c")])
        ctx.add_rule("path(X, Y) :- edge(X, Y)")
        ctx.run()
        paths = list(ctx.facts["path"])
        assert ("a", "b") in paths
        assert ("b", "c") in paths

    def test_transitive_closure(self):
        ctx = MinalogContext()
        ctx.add_relation("edge", ["String", "String"])
        ctx.add_relation("reachable", ["String", "String"])
        ctx.add_facts("edge", [("a", "b"), ("b", "c"), ("c", "d")])
        ctx.add_rule("reachable(X, Y) :- edge(X, Y)")
        ctx.add_rule("reachable(X, Z) :- edge(X, Y), reachable(Y, Z)")
        ctx.run()
        reach = list(ctx.facts["reachable"])
        assert ("a", "b") in reach
        assert ("a", "c") in reach
        assert ("a", "d") in reach

    def test_comparison_in_rule(self):
        ctx = MinalogContext()
        ctx.add_relation("score", ["String", "f32"])
        ctx.add_relation("high_score", ["String"])
        ctx.add_facts("score", [("alice", 95), ("bob", 40)])
        ctx.add_rule("high_score(X) :- score(X, S), S >= 50")
        ctx.run()
        high = list(ctx.facts["high_score"])
        assert ("alice",) in high
        assert ("bob",) not in high

    def test_constant_in_rule(self):
        ctx = MinalogContext()
        ctx.add_relation("object", ["String", "String"])
        ctx.add_relation("is_cup", ["String"])
        ctx.add_facts("object", [("obj1", "cup"), ("obj2", "table")])
        ctx.add_rule("is_cup(X) :- object(X, cup)")
        ctx.run()
        cups = list(ctx.facts["is_cup"])
        assert ("obj1",) in cups
        assert ("obj2",) not in cups

    def test_invalid_rule_no_head(self):
        ctx = MinalogContext()
        result = ctx._parse_rule("no_arrow_here")
        assert result is None

    def test_empty_rule(self):
        ctx = MinalogContext()
        result = ctx._parse_rule("")
        assert result is None

    def test_comment_in_rule(self):
        ctx = MinalogContext()
        result = ctx._parse_rule("% this is a comment")
        assert result is None

    def test_clear_facts(self):
        ctx = MinalogContext()
        ctx.add_facts("r", [("a",)])
        assert ctx.get_num_facts() == 1
        ctx.clear_facts()
        assert ctx.get_num_facts() == 0

    def test_get_num_rules(self):
        ctx = MinalogContext()
        ctx.add_rule("a(X) :- b(X)")
        assert ctx.get_num_rules() == 1

    def test_get_relations(self):
        ctx = MinalogContext()
        ctx.add_relation("edge", ["String", "String"])
        assert "edge" in ctx.get_relations()

    def test_parse_string_literal(self):
        ctx = MinalogContext()
        t = ctx._parse_term('"hello"')
        assert t.value == "hello"
        assert not t.is_variable

    def test_parse_single_quote_literal(self):
        ctx = MinalogContext()
        t = ctx._parse_term("'world'")
        assert t.value == "world"

    def test_parse_float(self):
        ctx = MinalogContext()
        t = ctx._parse_term("3.14")
        assert t.value == 3.14

    def test_parse_negative_int(self):
        ctx = MinalogContext()
        t = ctx._parse_term("-5")
        assert t.value == -5

    def test_parse_variable(self):
        ctx = MinalogContext()
        t = ctx._parse_term("X")
        assert t.is_variable

    def test_parse_underscore_variable(self):
        ctx = MinalogContext()
        t = ctx._parse_term("_unused")
        assert t.is_variable

    def test_parse_lowercase_constant(self):
        ctx = MinalogContext()
        t = ctx._parse_term("cup")
        assert not t.is_variable
        assert t.value == "cup"

    def test_parse_atom(self):
        ctx = MinalogContext()
        a = ctx._parse_atom("edge(a, b)")
        assert a.predicate == "edge"
        assert len(a.terms) == 2

    def test_parse_atom_invalid(self):
        ctx = MinalogContext()
        with pytest.raises(ValueError):
            ctx._parse_atom("invalid")

    def test_parse_comparison(self):
        ctx = MinalogContext()
        c = ctx._parse_comparison("X >= 0.8")
        assert c.op == ">="

    def test_parse_comparison_invalid(self):
        ctx = MinalogContext()
        with pytest.raises(ValueError):
            ctx._parse_comparison("X ~ Y")

    def test_max_iterations_safety(self):
        """Ensure the engine doesn't infinite loop."""
        ctx = MinalogContext()
        ctx.add_relation("r", ["String"])
        ctx.add_facts("r", [("a",)])
        # This rule always produces the same fact, so it converges immediately
        ctx.add_rule("r(a) :- r(a)")
        ctx.run()  # should not hang

    def test_unify_length_mismatch(self):
        ctx = MinalogContext()
        result = ctx._unify(
            [Term("X", True)],
            ("a", "b"),  # wrong arity
            {}
        )
        assert result is None

    def test_unify_variable_conflict(self):
        ctx = MinalogContext()
        result = ctx._unify(
            [Term("X", True), Term("X", True)],
            ("a", "b"),
            {}
        )
        assert result is None  # X can't be both "a" and "b"

    def test_unify_constant_mismatch(self):
        ctx = MinalogContext()
        result = ctx._unify(
            [Term("hello", False)],
            ("world",),
            {}
        )
        assert result is None

    def test_comparison_type_mismatch(self):
        """Comparisons with incompatible types should fail gracefully."""
        ctx = MinalogContext()
        ctx.add_relation("data", ["String", "String"])
        ctx.add_relation("out", ["String"])
        ctx.add_facts("data", [("a", "not_a_number")])
        ctx.add_rule("out(X) :- data(X, S), S > 5")
        ctx.run()
        assert list(ctx.facts.get("out", set())) == []
