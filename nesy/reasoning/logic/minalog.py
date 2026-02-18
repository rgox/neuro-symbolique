"""
Minimal Datalog Engine (Minalog).

A pure Python implementation of a subset of Datalog to support 
neuro-symbolic reasoning without external dependencies (like Scallop).

Features:
- Facts storage
- Rule parsing (Head :- Body)
- Bottom-up evaluation (semi-naive)
- Variable binding
- Basic comparisons (>, <, >=, <=, ==, !=)

Limitations:
- No negation in rule bodies (stratified negation not implemented)
- No complex aggregations
- Performance is not optimized for large datasets
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Set, Union, Callable, Optional
from collections import defaultdict
import operator

logger = logging.getLogger(__name__)


class Term:
    """Represents a term in a Datalog atom (constant or variable)."""
    def __init__(self, value: Any, is_variable: bool):
        self.value = value
        self.is_variable = is_variable

    def __repr__(self):
        return f"Var({self.value})" if self.is_variable else f"Const({self.value})"

    def __eq__(self, other):
        return (self.value == other.value) and (self.is_variable == other.is_variable)

    def __hash__(self):
        return hash((self.value, self.is_variable))


class Atom:
    """Represents a Datalog atom: predicate(term1, term2, ...)."""
    def __init__(self, predicate: str, terms: List[Term]):
        self.predicate = predicate
        self.terms = terms

    def __repr__(self):
        args = ", ".join([str(t.value) for t in self.terms])
        return f"{self.predicate}({args})"


class Comparison:
    """Represents a comparison: Var op Value or Var op Var."""
    def __init__(self, left: Term, op: str, right: Term):
        self.left = left
        self.op = op
        self.right = right
        
        self.ops = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
        }

    def evaluate(self, bindings: Dict[str, Any]) -> bool:
        if self.left.is_variable:
            if self.left.value not in bindings:
                return False  # Unbound variable in comparison
            l_val = bindings[self.left.value]
        else:
            l_val = self.left.value

        if self.right.is_variable:
            if self.right.value not in bindings:
                return False  # Unbound variable
            r_val = bindings[self.right.value]
        else:
            r_val = self.right.value

        return self.ops[self.op](l_val, r_val)


class Rule:
    """Represents a Datalog rule: Head :- Body."""
    def __init__(self, head: Atom, body: List[Union[Atom, Comparison]]):
        self.head = head
        self.body = body

    def __repr__(self):
        return f"{self.head} :- {', '.join(map(str, self.body))}"


class MinalogContext:
    """
    Minimal Datalog Context replacement for Scallop.
    """
    def __init__(self, provenance: str = "unit", k: int = 3):
        self.facts: Dict[str, Set[Tuple]] = defaultdict(set)
        self.rules: List[Rule] = []
        self.relations: Dict[str, List[str]] = {}
        self.provenance = provenance
        self.k = k

    def add_relation(self, name: str, types: List[str]) -> None:
        """Define a relation (schema)."""
        self.relations[name] = types

    def add_facts(self, relation: str, facts: List[Tuple]) -> None:
        """Add facts to a relation."""
        # Convert all to standard python types
        normalized_facts = []
        for fact in facts:
            norm_fact = tuple(fact)
            normalized_facts.append(norm_fact)
        
        self.facts[relation].update(normalized_facts)

    def add_rule(self, rule_str: str) -> None:
        """Parse and add a rule."""
        parsed = self._parse_rule(rule_str)
        if parsed:
            self.rules.append(parsed)

    def query(self, relation: str) -> List[Tuple]:
        """Execute inference and return results for a relation."""
        self.run()
        return list(self.facts.get(relation, set()))

    def relation(self, relation: str):
        """Iterator over relation (compatibility)."""
        return iter(self.facts.get(relation, set()))

    def run(self) -> None:
        """Run fixed-point evaluation."""
        changed = True
        iterations = 0
        max_iterations = 100  # Safety break
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for rule in self.rules:
                new_facts = self._evaluate_rule(rule)
                if new_facts:
                    before_len = len(self.facts[rule.head.predicate])
                    self.facts[rule.head.predicate].update(new_facts)
                    if len(self.facts[rule.head.predicate]) > before_len:
                        changed = True

    def _evaluate_rule(self, rule: Rule) -> Set[Tuple]:
        """Evaluate a single rule against current facts."""
        # This is a naive backtracking implementation
        # Find all valid bindings for variables in the body
        
        # Separate atoms and comparisons
        atoms = [b for b in rule.body if isinstance(b, Atom)]
        comparisons = [b for b in rule.body if isinstance(b, Comparison)]
        
        bindings_list = [{}]  # Start with empty binding
        
        for atom in atoms:
            predicate = atom.predicate
            terms = atom.terms
            
            # Facts for this predicate
            current_facts = self.facts.get(predicate, set())
            
            new_bindings_list = []
            for bindings in bindings_list:
                for fact in current_facts:
                    # Try to unify fact with atom using current bindings
                    new_bind = self._unify(terms, fact, bindings)
                    if new_bind is not None:
                        new_bindings_list.append(new_bind)
            
            bindings_list = new_bindings_list
            if not bindings_list:
                break
        
        # Filter by comparisons
        valid_bindings = []
        for bindings in bindings_list:
            valid = True
            for comp in comparisons:
                try:
                    if not comp.evaluate(bindings):
                        valid = False
                        break
                except Exception:
                    valid = False  # Evaluation failed (e.g. type mismatch)
                    break
            if valid:
                valid_bindings.append(bindings)
        
        # Generate head facts
        new_facts = set()
        for bindings in valid_bindings:
            head_fact = []
            for term in rule.head.terms:
                if term.is_variable:
                    head_fact.append(bindings.get(term.value))
                else:
                    head_fact.append(term.value)
            new_facts.add(tuple(head_fact))
            
        return new_facts

    def _unify(self, terms: List[Term], fact: Tuple, bindings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Unify atom terms with a fact tuple given existing bindings."""
        if len(terms) != len(fact):
            return None
            
        new_bindings = bindings.copy()
        
        for term, val in zip(terms, fact):
            if term.is_variable:
                if term.value in new_bindings:
                    if new_bindings[term.value] != val:
                        return None  # Variable conflict
                else:
                    new_bindings[term.value] = val
            else:
                if term.value != val:
                    return None  # Constant mismatch
                    
        return new_bindings

    def _parse_rule(self, rule_str: str) -> Optional[Rule]:
        # Remove comments
        if '%' in rule_str:
            rule_str = rule_str.split('%')[0]
        rule_str = rule_str.strip()
        if not rule_str:
            return None
            
        if ":-" not in rule_str:
            logger.warning(f"Ignoring invalid rule (no :-): {rule_str}")
            return None
            
        head_str, body_str = rule_str.split(":-", 1)
        head = self._parse_atom(head_str.strip())
        
        # Split body by comma, ignoring commas inside parens
        body_parts = []
        current = ""
        level = 0
        for char in body_str:
            if char == ',' and level == 0:
                body_parts.append(current.strip())
                current = ""
            else:
                current += char
                if char == '(': level += 1
                elif char == ')': level -= 1
        if current.strip():
            body_parts.append(current.strip())
            
        body = []
        for part in body_parts:
            # Check for comparison operators
            is_comp = False
            for op in [">=", "<=", "==", "!=", ">", "<"]:
                if op in part and not re.match(r"^\w+\s*\(", part): # Not an atom
                    body.append(self._parse_comparison(part))
                    is_comp = True
                    break
            
            if not is_comp:
                body.append(self._parse_atom(part))
                
        return Rule(head, body)

    def _parse_term(self, term_str: str) -> Term:
        term_str = term_str.strip()
        # Number
        try:
            if '.' in term_str:
                return Term(float(term_str), False)
            if term_str.isdigit() or (term_str.startswith('-') and term_str[1:].isdigit()):
                return Term(int(term_str), False)
        except ValueError:
            pass
            
        # String literal "..." or '...'
        if (term_str.startswith('"') and term_str.endswith('"')) or \
           (term_str.startswith("'") and term_str.endswith("'")):
            return Term(term_str[1:-1], False)
            
        # Variable (Uppercased or starting with _)
        if term_str and (term_str[0].isupper() or term_str.startswith('_')):
            return Term(term_str, True)
            
        # Unquoted string constant (standard Datalog atom)
        return Term(term_str, False)

    def _parse_atom(self, atom_str: str) -> Atom:
        match = re.match(r"^(\w+)\s*\((.*)\)$", atom_str)
        if not match:
            raise ValueError(f"Invalid atom format: {atom_str}")
            
        pred = match.group(1)
        args_str = match.group(2)
        
        # Split args by comma respecting quotes? Simple split for now
        # Assuming no commas in args for MVP
        terms = [self._parse_term(t) for t in args_str.split(',')]
        
        return Atom(pred, terms)

    def _parse_comparison(self, comp_str: str) -> Comparison:
        # Sort ops by length desc to match >= before >
        ops = sorted([">=", "<=", "==", "!=", ">", "<"], key=len, reverse=True)
        for op in ops:
            if op in comp_str:
                left, right = comp_str.split(op, 1)
                return Comparison(self._parse_term(left), op, self._parse_term(right))
        raise ValueError(f"Invalid comparison: {comp_str}")

    def clear_facts(self) -> None:
        self.facts.clear()
        
    def get_num_facts(self) -> int:
        return sum(len(f) for f in self.facts.values())

    def get_num_rules(self) -> int:
        return len(self.rules)
    
    def get_relations(self) -> Dict[str, List[str]]:
        return self.relations

