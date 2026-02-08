"""
Natural Language Query Interface.

Translates natural language queries to symbolic logic using pattern matching
and optional LLM integration.

Example:
    >>> from nesy.reasoning.nlquery import NLQueryInterface
    >>> 
    >>> nli = NLQueryInterface(reasoning_engine=engine)
    >>> results = nli.query("What objects are in the kitchen?")
"""

from typing import List, Optional, Dict, Any
import re
import logging

from nesy.reasoning.logic import ReasoningEngine


logger = logging.getLogger(__name__)


class NLQueryInterface:
    """
    Natural language query interface for reasoning engine.
    
    Supports pattern matching for common query types.
    Can be extended with LLM for more complex queries.
    
    Supported patterns:
    - "What's in [location]?" → find_in_location
    - "Find all [class]" → find_all
    - "What's on [object]?" → spatial query
    - "Is [object] in [location]?" → inference
    - "Where is [object]?" → reverse location query
    
    Example:
        >>> nli = NLQueryInterface(engine)
        >>> nli.query("What's in the kitchen?")
        >>> # → ["cup1", "table1", ...]
    """
    
    def __init__(
        self,
        reasoning_engine: ReasoningEngine,
        use_llm: bool = False,
        llm_backend: Optional[str] = None
    ):
        """
        Initialize NL query interface.
        
        Args:
            reasoning_engine: Reasoning engine instance
            use_llm: Whether to use LLM for complex queries
            llm_backend: LLM backend ("openai", "local", etc.)
        """
        self.engine = reasoning_engine
        self.use_llm = use_llm
        self.llm_backend = llm_backend
        
        # Query patterns
        self.patterns = self._compile_patterns()
        
        logger.info(f"NLQueryInterface initialized (LLM: {use_llm})")
    
    def _compile_patterns(self) -> List[tuple]:
        """Compile regex patterns for query matching."""
        return [
            # "What's in the kitchen?"
            (re.compile(r"what'?s? in (?:the )?(\w+)", re.I),
             self._handle_in_location),
            
            # "Find all cups"
            (re.compile(r"find all (\w+)", re.I),
             self._handle_find_all),
            
            # "What's on the table?"
            (re.compile(r"what'?s? on (?:the )?(\w+)", re.I),
             self._handle_on_object),
            
            # "Is the cup in the kitchen?"
            (re.compile(r"is (?:the )?(\w+) in (?:the )?(\w+)", re.I),
             self._handle_is_in),
            
            # "Where is the cup?"
            (re.compile(r"where is (?:the )?(\w+)", re.I),
             self._handle_where_is),
        ]
    
    def query(self, query_str: str) -> List[Any]:
        """
        Process natural language query.
        
        Args:
            query_str: Natural language query
        
        Returns:
            Query results (list of object IDs or boolean)
        """
        query_str = query_str.strip()
        
        # Try pattern matching first
        for pattern, handler in self.patterns:
            match = pattern.search(query_str)
            if match:
                logger.debug(f"Matched pattern: {pattern.pattern}")
                return handler(match)
        
        # Fallback to LLM if available
        if self.use_llm:
            return self._handle_with_llm(query_str)
        
        logger.warning(f"No pattern matched for query: {query_str}")
        return []
    
    def _handle_in_location(self, match) -> List[str]:
        """Handle 'What's in the [location]?' queries."""
        location = match.group(1)
        
        # Find location nodes
        location_nodes = self.engine.scene_graph.query_by_attributes(class_=location)
        
        if not location_nodes:
            logger.warning(f"Location not found: {location}")
            return []
        
        return self.engine.find_in_location(location_nodes[0].id)
    
    def _handle_find_all(self, match) -> List[str]:
        """Handle 'Find all [class]' queries."""
        obj_class = match.group(1).rstrip('s')  # Remove plural
        return self.engine.find_all(obj_class)
    
    def _handle_on_object(self, match) -> List[str]:
        """Handle 'What's on the [object]?' queries."""
        target = match.group(1)
        
        # Find target nodes
        target_nodes = self.engine.scene_graph.query_by_attributes(class_=target)
        
        if not target_nodes:
            logger.warning(f"Object not found: {target}")
            return []
        
        # Query spatial relations
        on_relations = self.engine.query("on")
        
        return [src for src, dst in on_relations if dst == target_nodes[0].id]
    
    def _handle_is_in(self, match) -> bool:
        """Handle 'Is [object] in [location]?' queries."""
        obj = match.group(1)
        location = match.group(2)
        
        # Find nodes
        obj_nodes = self.engine.scene_graph.query_by_attributes(class_=obj)
        location_nodes = self.engine.scene_graph.query_by_attributes(class_=location)
        
        if not obj_nodes or not location_nodes:
            return False
        
        # Infer
        return self.engine.infer(f"in('{obj_nodes[0].id}', '{location_nodes[0].id}')")
    
    def _handle_where_is(self, match) -> List[str]:
        """Handle 'Where is the [object]?' queries."""
        obj = match.group(1)
        
        # Find object nodes
        obj_nodes = self.engine.scene_graph.query_by_attributes(class_=obj)
        
        if not obj_nodes:
            logger.warning(f"Object not found: {obj}")
            return []
        
        # Query 'in' relations
        in_relations = self.engine.query("in")
        
        # Find locations
        locations = [dst for src, dst in in_relations if src == obj_nodes[0].id]
        
        if locations:
            # Get location class names
            result = []
            for loc_id in locations:
                node = self.engine.scene_graph.get_node(loc_id)
                if node:
                    result.append(node.attributes.get("class", loc_id))
            return result
        
        return []
    
    def _handle_with_llm(self, query_str: str) -> List[Any]:
        """Handle query using LLM (placeholder for now)."""
        logger.warning("LLM integration not implemented yet")
        return []
