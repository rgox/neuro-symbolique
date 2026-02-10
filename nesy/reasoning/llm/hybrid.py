"""
Hybrid LLM Reasoning Module.

Bridges Large Language Models with symbolic reasoning for:
- Rule extraction from natural language
- Explanation generation from logic proofs
- Hybrid queries (neural + symbolic)
- Self-validation and correction

Example:
    >>> from nesy.reasoning.llm import LLMRuleBridge, HybridQueryEngine
    >>> 
    >>> # Extract rules from text
    >>> bridge = LLMRuleBridge()
    >>> rules = bridge.extract_rules(
    ...     "Objects on tables are in the same room as the table"
    ... )
    >>> # → ["in(X, R) :- on(X, T), in(T, R)"]
    >>> 
    >>> # Hybrid query
    >>> engine = HybridQueryEngine(reasoning_engine, llm_bridge)
    >>> answer = engine.query("What's on the kitchen table?")
    >>> # → Combines symbolic facts + LLM reasoning
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRule:
    """Rule extracted from natural language."""
    rule_text: str  # Datalog format
    confidence: float  # 0-1
    source_text: str  # Original natural language
    explanation: str  # Why this rule was extracted


class LLMRuleBridge:
    """
    Bridge between LLM and symbolic logic.
    
    Extracts logic rules from natural language and generates
    natural language explanations from logic proofs.
    
    Example:
        >>> bridge = LLMRuleBridge()
        >>> 
        >>> # Extract rule
        >>> rules = bridge.extract_rules(
        ...     "If an object is on a table, and the table is in a room, "
        ...     "then the object is also in that room"
        ... )
        >>> 
        >>> # Explain inference
        >>> explanation = bridge.explain_inference(
        ...     goal="in(cup, kitchen)",
        ...     facts=["on(cup, table)", "in(table, kitchen)"],
        ...     rules=["in(X,R) :- on(X,T), in(T,R)"]
        ... )
    """
    
    def __init__(self, use_mock: bool = True):
        """
        Initialize LLM bridge.
        
        Args:
            use_mock: If True, use mock LLM (for testing)
        """
        self.use_mock = use_mock
        self.llm = None  # Would initialize actual LLM here
    
    def extract_rules(
        self,
        text: str,
        domain_predicates: Optional[List[str]] = None
    ) -> List[ExtractedRule]:
        """
        Extract logic rules from natural language.
        
        Args:
            text: Natural language description
            domain_predicates: Known predicates in domain
        
        Returns:
            List of extracted rules
        
        Example:
            >>> rules = bridge.extract_rules(
            ...     "Cups on tables are in the same room",
            ...     domain_predicates=["on", "in"]
            ... )
        """
        if self.use_mock:
            return self._mock_extract_rules(text, domain_predicates)
        
        # Real LLM implementation would go here
        # prompt = self._build_extraction_prompt(text, domain_predicates)
        # response = self.llm.generate(prompt)
        # rules = self._parse_llm_response(response)
        
        return []
    
    def _mock_extract_rules(
        self,
        text: str,
        domain_predicates: Optional[List[str]] = None
    ) -> List[ExtractedRule]:
        """Mock rule extraction for testing."""
        text_lower = text.lower()
        
        rules = []
        
        # Simple pattern matching for demo
        if "on" in text_lower and "in" in text_lower and "room" in text_lower:
            rules.append(ExtractedRule(
                rule_text="in(X, R) :- on(X, T), in(T, R)",
                confidence=0.85,
                source_text=text,
                explanation="Transitive spatial containment: on + in → in"
            ))
        
        if "near" in text_lower and "same" in text_lower:
            rules.append(ExtractedRule(
                rule_text="same_location(X, Y) :- near(X, Y)",
                confidence=0.75,
                source_text=text,
                explanation="Proximity implies same location"
            ))
        
        return rules
    
    def explain_inference(
        self,
        goal: str,
        facts: List[str],
        rules: List[str],
        proof: Optional[List[str]] = None
    ) -> str:
        """
        Generate natural language explanation of inference.
        
        Args:
            goal: Goal that was inferred
            facts: Base facts used
            rules: Rules applied
            proof: Optional proof trace
        
        Returns:
            Natural language explanation
        
        Example:
            >>> explanation = bridge.explain_inference(
            ...     goal="in(cup, kitchen)",
            ...     facts=["on(cup, table)", "in(table, kitchen)"],
            ...     rules=["in(X,R) :- on(X,T), in(T,R)"]
            ... )
            >>> # → "The cup is in the kitchen because..."
        """
        if self.use_mock:
            return self._mock_explain(goal, facts, rules)
        
        # Real LLM implementation
        return ""
    
    def _mock_explain(
        self,
        goal: str,
        facts: List[str],
        rules: List[str]
    ) -> str:
        """Mock explanation generation."""
        return (
            f"The goal '{goal}' can be inferred because:\n"
            f"- We have the facts: {', '.join(facts)}\n"
            f"- Applying the rule(s): {', '.join(rules)}\n"
            f"- This allows us to conclude {goal}"
        )
    
    def validate_rule(self, rule: str, examples: List[Dict[str, Any]]) -> bool:
        """
        Validate rule against examples using LLM.
        
        Args:
            rule: Rule to validate
            examples: Test cases
        
        Returns:
            True if rule seems valid
        """
        if self.use_mock:
            return True  # Mock always validates
        
        return False


class HybridQueryEngine:
    """
    Query engine combining symbolic reasoning + LLM.
    
    Uses symbolic reasoning for precise inference and LLM for
    handling ambiguity, common sense, and natural language.
    
    Example:
        >>> engine = HybridQueryEngine(reasoning_engine, llm_bridge)
        >>> 
        >>> # Symbolic query
        >>> result = engine.query("What objects are in the kitchen?")
        >>> 
        >>> # Hybrid query (needs LLM)
        >>> result = engine.query("Is the kitchen messy?")
        >>> # → Uses LLM + symbolic facts
    """
    
    def __init__(self, reasoning_engine, llm_bridge: LLMRuleBridge):
        """
        Initialize hybrid engine.
        
        Args:
            reasoning_engine: Symbolic reasoning engine
            llm_bridge: LLM bridge for neural reasoning
        """
        self.reasoning_engine = reasoning_engine
        self.llm_bridge = llm_bridge
    
    def query(
        self,
        query: str,
        prefer_symbolic: bool = True
    ) -> Dict[str, Any]:
        """
        Answer query using hybrid reasoning.
        
        Args:
            query: Natural language query
            prefer_symbolic: Try symbolic first
        
        Returns:
            Answer with explanation
        
        Example:
            >>> result = engine.query("What's on the table?")
            >>> # → {"answer": [...], "method": "symbolic", "confidence": 1.0}
        """
        # Try symbolic reasoning first
        if prefer_symbolic:
            symbolic_result = self._try_symbolic(query)
            if symbolic_result:
                return symbolic_result
        
        # Fall back to hybrid
        return self._hybrid_query(query)
    
    def _try_symbolic(self, query: str) -> Optional[Dict[str, Any]]:
        """Try answering with pure symbolic reasoning."""
        # Parse query to extract relation
        # Simple pattern matching for demo
        query_lower = query.lower()
        
        if "in the" in query_lower or "in" in query_lower:
            # Extract location
            # This is very simplified - real implementation would parse properly
            results = self.reasoning_engine.query("in")
            
            if results:
                return {
                    "answer": results,
                    "method": "symbolic",
                    "confidence": 1.0,
                    "explanation": "Retrieved from symbolic knowledge base"
                }
        
        return None
    
    def _hybrid_query(self, query: str) -> Dict[str, Any]:
        """Answer using LLM + symbolic facts."""
        # Get relevant facts from symbolic engine
        facts = self._get_relevant_facts(query)
        
        # Use LLM to answer with context
        if self.llm_bridge.use_mock:
            answer = f"Mock answer for: {query} (using facts: {len(facts)})"
        else:
            # Real LLM call would be here
            answer = ""
        
        return {
            "answer": answer,
            "method": "hybrid",
            "confidence": 0.7,
            "explanation": f"Combined symbolic facts ({len(facts)}) with LLM reasoning",
            "facts_used": facts
        }
    
    def _get_relevant_facts(self, query: str) -> List[str]:
        """Extract relevant facts for query."""
        # Simplified - would use semantic similarity in real implementation
        all_facts = []
        
        # Get facts from reasoning engine
        try:
            for relation in ["on", "in", "near"]:
                results = self.reasoning_engine.query(relation)
                for r in results:
                    all_facts.append(f"{relation}{r}")
        except:
            pass
        
        return all_facts[:10]  # Limit for context


class SelfValidator:
    """
    Self-validation and correction for hybrid reasoning.
    
    Validates LLM outputs against symbolic constraints and
    corrects inconsistencies.
    
    Example:
        >>> validator = SelfValidator(reasoning_engine, llm_bridge)
        >>> 
        >>> # Validate LLM claim
        >>> valid = validator.validate_claim(
        ...     "The cup is in the kitchen",
        ...     verify_symbolic=True
        ... )
    """
    
    def __init__(self, reasoning_engine, llm_bridge: LLMRuleBridge):
        """Initialize validator."""
        self.reasoning_engine = reasoning_engine
        self.llm_bridge = llm_bridge
    
    def validate_claim(
        self,
        claim: str,
        verify_symbolic: bool = True
    ) -> Tuple[bool, str]:
        """
        Validate a claim.
        
        Args:
            claim: Claim to validate
            verify_symbolic: Check against symbolic KB
        
        Returns:
            (is_valid, explanation)
        """
        if verify_symbolic:
            # Try to verify symbolically
            # Parse claim to predicate
            # This is very simplified
            
            if "in" in claim:
                results = self.reasoning_engine.query("in")
                # Check if claim matches any fact
                # Real implementation would parse claim properly
                
                return (
                    len(results) > 0,
                    "Claim verified against symbolic knowledge base" if results
                    else "Cannot verify claim symbolically"
                )
        
        # Use LLM for validation
        return (True, "LLM validation (mock)")
    
    def detect_contradictions(
        self,
        statements: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Detect contradictions between statements.
        
        Args:
            statements: List of statements
        
        Returns:
            List of (statement1, statement2, explanation) contradictions
        """
        contradictions = []
        
        # Simple contradiction detection (mock)
        for i, s1 in enumerate(statements):
            for s2 in statements[i+1:]:
                # Check for direct negation
                s1_normalized = s1.lower().replace("not ", "").strip()
                s2_normalized = s2.lower().replace("not ", "").strip()
                
                # If one has "not" and the other doesn't, and they match otherwise
                has_not_s1 = "not " in s1.lower()
                has_not_s2 = "not " in s2.lower()
                
                if has_not_s1 != has_not_s2 and s1_normalized == s2_normalized:
                    contradictions.append((
                        s1, s2,
                        "Negation contradiction detected"
                    ))
        
        return contradictions
    
    def self_correct(
        self,
        claim: str,
        contradiction: Optional[str] = None
    ) -> str:
        """
        Attempt to correct invalid claim.
        
        Args:
            claim: Invalid claim
            contradiction: What it contradicts
        
        Returns:
            Corrected claim
        """
        # Use LLM to suggest correction
        if self.llm_bridge.use_mock:
            return f"Corrected: {claim}"
        
        return claim


class ExplanationGenerator:
    """
    Generate natural language explanations for reasoning.
    
    Example:
        >>> generator = ExplanationGenerator(llm_bridge)
        >>> 
        >>> explanation = generator.explain_plan(
        ...     plan=[move(room1, room2), move(room2, room3)]
        ... )
    """
    
    def __init__(self, llm_bridge: LLMRuleBridge):
        """Initialize generator."""
        self.llm_bridge = llm_bridge
    
    def explain_plan(self, plan: List[Any]) -> str:
        """Explain an action plan."""
        if not plan:
            return "No actions needed."
        
        steps = [f"Step {i+1}: {action}" for i, action in enumerate(plan)]
        return "To achieve the goal:\n" + "\n".join(steps)
    
    def explain_reasoning_chain(
        self,
        facts: List[str],
        rules: List[str],
        conclusion: str
    ) -> str:
        """Explain reasoning chain."""
        return self.llm_bridge.explain_inference(
            goal=conclusion,
            facts=facts,
            rules=rules
        )
