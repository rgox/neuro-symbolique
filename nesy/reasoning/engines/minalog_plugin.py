"""Pure-Python Minalog logic engine plugin."""

from typing import List, Tuple

from nesy.core.registry import register_plugin
from nesy.reasoning.base import LogicEngineBase
from nesy.reasoning.logic.minalog import MinalogContext


@register_plugin(LogicEngineBase, "minalog")
class MinalogEngine(LogicEngineBase):
    """
    Logic engine backed by Minalog (pure Python Datalog).

    No external dependencies. Good for testing and lightweight use.
    """

    def __init__(self, **kwargs):
        self.ctx = MinalogContext()

    def add_relation(self, name: str, types: List[str]) -> None:
        self.ctx.add_relation(name, types)

    def add_fact(self, relation: str, *args) -> None:
        self.ctx.add_facts(relation, [tuple(args)])

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
        rels = self.ctx.get_relations()
        if isinstance(rels, dict):
            return list(rels.keys())
        return rels
