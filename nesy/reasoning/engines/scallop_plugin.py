"""Scallop/Minalog logic engine plugin (wraps ScallopContext)."""

from typing import List, Tuple

from nesy.core.registry import register_plugin
from nesy.reasoning.base import LogicEngineBase
from nesy.reasoning.logic.scallop_context import ScallopContext


@register_plugin(LogicEngineBase, "scallop")
class ScallopEngine(LogicEngineBase):
    """
    Logic engine backed by Scallop (or Minalog fallback).

    This wraps the existing ScallopContext to conform to LogicEngineBase.
    """

    def __init__(self, provenance: str = "difftopkproofs", k: int = 3, **kwargs):
        self.ctx = ScallopContext(provenance=provenance, k=k)

    def add_relation(self, name: str, types: List[str]) -> None:
        self.ctx.add_relation(name, types)

    def add_fact(self, relation: str, *args) -> None:
        self.ctx.add_fact(relation, *args)

    def add_rule(self, rule: str) -> None:
        self.ctx.add_rule(rule)

    def query(self, relation: str) -> List[Tuple]:
        return self.ctx.query(relation)

    def clear_facts(self) -> None:
        self.ctx.clear_facts()

    def get_num_facts(self) -> int:
        return self.ctx.get_num_facts()

    def get_num_rules(self) -> int:
        return self.ctx.get_num_rules()

    def get_relations(self) -> List[str]:
        return self.ctx.get_relations()
