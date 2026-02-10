"""
Temporal Reasoning Engine.

Higher-level temporal reasoning on top of TemporalSceneGraph:
- Event pattern detection
- Causal inference
- Temporal constraint checking
- Future prediction
- Allen's interval algebra

Example:
    >>> from nesy.reasoning.temporal import TemporalReasoner
    >>> 
    >>> reasoner = TemporalReasoner(temporal_scene_graph)
    >>> events = reasoner.detect_patterns("object_moved")
    >>> causes = reasoner.infer_causality(event1, event2)
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class CausalRelation(Enum):
    """Types of causal relations."""
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    ENABLES = "enables"
    PREVENTS = "prevents"
    CORRELATES = "correlates"


@dataclass
class TemporalEvent:
    """A discrete event in time."""
    id: str
    event_type: str
    timestamp: float
    duration: float = 0.0
    subject: str = ""
    data: Dict = field(default_factory=dict)
    
    @property
    def end_time(self) -> float:
        return self.timestamp + self.duration


@dataclass
class TemporalConstraint:
    """Constraint between events."""
    event_a: str
    event_b: str
    relation: str  # "before", "after", "during", "meets"
    min_gap: float = 0.0
    max_gap: float = float('inf')


@dataclass
class CausalLink:
    """Causal link between events."""
    cause: str
    effect: str
    relation: CausalRelation
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)


class TemporalReasoner:
    """
    Temporal reasoning engine.
    
    Provides event pattern detection, causal inference,
    and temporal constraint management.
    
    Example:
        >>> reasoner = TemporalReasoner()
        >>> reasoner.add_event("e1", "pick", 1.0, subject="cup")
        >>> reasoner.add_event("e2", "move", 2.0, subject="cup")
        >>> reasoner.add_event("e3", "place", 3.0, subject="cup")
        >>> 
        >>> # Pattern detection
        >>> patterns = reasoner.detect_sequence(["pick", "move", "place"])
        >>> 
        >>> # Causal inference
        >>> causes = reasoner.infer_causality("e1", "e2")
    """
    
    def __init__(self):
        """Initialize temporal reasoner."""
        self.events: Dict[str, TemporalEvent] = {}
        self.constraints: List[TemporalConstraint] = []
        self.causal_links: List[CausalLink] = []
        
        # Pattern rules (event_type → possible effects)
        self.causal_rules: Dict[str, List[str]] = {
            "pick": ["move", "hold"],
            "move": ["place", "drop"],
            "push": ["move", "fall"],
            "open": ["access", "see_inside"],
            "close": ["block", "hide"],
        }
        
        # Event history for pattern matching
        self.event_timeline: List[TemporalEvent] = []
        
        logger.info("Temporal reasoner initialized")
    
    # --- Event Management ---
    
    def add_event(
        self,
        event_id: str,
        event_type: str,
        timestamp: float,
        duration: float = 0.0,
        subject: str = "",
        data: Optional[Dict] = None
    ) -> TemporalEvent:
        """Add temporal event."""
        event = TemporalEvent(
            id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            duration=duration,
            subject=subject,
            data=data or {}
        )
        self.events[event_id] = event
        
        # Insert in timeline (sorted by timestamp)
        self.event_timeline.append(event)
        self.event_timeline.sort(key=lambda e: e.timestamp)
        
        logger.debug(f"Added event: {event_id} ({event_type}) at t={timestamp:.2f}")
        return event
    
    def get_events_in_range(
        self,
        start: float,
        end: float,
        event_type: Optional[str] = None
    ) -> List[TemporalEvent]:
        """Get events within time range."""
        results = [
            e for e in self.event_timeline
            if start <= e.timestamp <= end
        ]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        return results
    
    def get_events_by_subject(self, subject: str) -> List[TemporalEvent]:
        """Get all events for a subject."""
        return [e for e in self.event_timeline if e.subject == subject]
    
    # --- Temporal Relations (Allen's Interval Algebra) ---
    
    def temporal_relation(self, event_a: str, event_b: str) -> str:
        """
        Determine temporal relation between events.
        
        Uses Allen's interval algebra:
        before, after, meets, met_by, overlaps, during, equals, etc.
        """
        a = self.events.get(event_a)
        b = self.events.get(event_b)
        
        if not a or not b:
            return "unknown"
        
        a_start, a_end = a.timestamp, a.end_time
        b_start, b_end = b.timestamp, b.end_time
        
        if a_end < b_start:
            return "before"
        elif a_start > b_end:
            return "after"
        elif abs(a_end - b_start) < 0.001:
            return "meets"
        elif abs(b_end - a_start) < 0.001:
            return "met_by"
        elif a_start < b_start and a_end > b_start and a_end < b_end:
            return "overlaps"
        elif a_start > b_start and a_end < b_end:
            return "during"
        elif abs(a_start - b_start) < 0.001 and abs(a_end - b_end) < 0.001:
            return "equals"
        elif a_start < b_start and a_end > b_end:
            return "contains"
        else:
            return "overlaps"
    
    # --- Constraints ---
    
    def add_constraint(
        self,
        event_a: str,
        event_b: str,
        relation: str,
        min_gap: float = 0.0,
        max_gap: float = float('inf')
    ):
        """Add temporal constraint."""
        constraint = TemporalConstraint(
            event_a=event_a,
            event_b=event_b,
            relation=relation,
            min_gap=min_gap,
            max_gap=max_gap
        )
        self.constraints.append(constraint)
    
    def check_constraints(self) -> List[Tuple[TemporalConstraint, bool, str]]:
        """
        Check all temporal constraints.
        
        Returns:
            List of (constraint, satisfied, reason)
        """
        results = []
        
        for constraint in self.constraints:
            a = self.events.get(constraint.event_a)
            b = self.events.get(constraint.event_b)
            
            if not a or not b:
                results.append((constraint, False, "Event not found"))
                continue
            
            # Check relation
            actual_rel = self.temporal_relation(constraint.event_a, constraint.event_b)
            rel_ok = actual_rel == constraint.relation
            
            # Check gap
            gap = abs(b.timestamp - a.timestamp)
            gap_ok = constraint.min_gap <= gap <= constraint.max_gap
            
            satisfied = rel_ok and gap_ok
            reason = "OK" if satisfied else f"Expected {constraint.relation}, got {actual_rel}, gap={gap:.2f}"
            
            results.append((constraint, satisfied, reason))
        
        return results
    
    # --- Pattern Detection ---
    
    def detect_sequence(
        self,
        pattern: List[str],
        subject: Optional[str] = None,
        max_gap: float = 10.0
    ) -> List[List[TemporalEvent]]:
        """
        Detect event sequence patterns.
        
        Args:
            pattern: Ordered list of event types to match
            subject: Optional subject filter
            max_gap: Max time gap between consecutive events
        
        Returns:
            List of matching event sequences
        """
        matches = []
        events = self.event_timeline
        
        if subject:
            events = [e for e in events if e.subject == subject]
        
        # Sliding window pattern matching
        for i in range(len(events)):
            if events[i].event_type != pattern[0]:
                continue
            
            sequence = [events[i]]
            pattern_idx = 1
            
            for j in range(i + 1, len(events)):
                if pattern_idx >= len(pattern):
                    break
                
                # Check gap
                if events[j].timestamp - sequence[-1].timestamp > max_gap:
                    break
                
                if events[j].event_type == pattern[pattern_idx]:
                    if not subject or events[j].subject == sequence[0].subject:
                        sequence.append(events[j])
                        pattern_idx += 1
            
            if len(sequence) == len(pattern):
                matches.append(sequence)
        
        return matches
    
    def detect_repeated(
        self,
        event_type: str,
        min_count: int = 2,
        window: float = 60.0
    ) -> List[List[TemporalEvent]]:
        """Detect repeated events within time window."""
        matches = []
        events = [e for e in self.event_timeline if e.event_type == event_type]
        
        i = 0
        while i < len(events):
            group = [events[i]]
            j = i + 1
            while j < len(events) and events[j].timestamp - events[i].timestamp <= window:
                group.append(events[j])
                j += 1
            
            if len(group) >= min_count:
                matches.append(group)
            
            i = j if j > i + 1 else i + 1
        
        return matches
    
    # --- Causal Inference ---
    
    def infer_causality(
        self,
        event_a: str,
        event_b: str,
        max_delay: float = 5.0
    ) -> Optional[CausalLink]:
        """
        Infer causal relation between events.
        
        Uses temporal proximity + domain rules for inference.
        """
        a = self.events.get(event_a)
        b = self.events.get(event_b)
        
        if not a or not b:
            return None
        
        # Must be temporally ordered
        if a.timestamp >= b.timestamp:
            return None
        
        # Check delay
        delay = b.timestamp - a.timestamp
        if delay > max_delay:
            return None
        
        # Check domain rules
        possible_effects = self.causal_rules.get(a.event_type, [])
        
        if b.event_type in possible_effects:
            confidence = max(0.5, 1.0 - delay / max_delay)
            
            # Same subject increases confidence
            if a.subject and a.subject == b.subject:
                confidence = min(1.0, confidence + 0.2)
            
            link = CausalLink(
                cause=event_a,
                effect=event_b,
                relation=CausalRelation.CAUSES,
                confidence=confidence,
                evidence=[f"rule: {a.event_type}→{b.event_type}", f"delay: {delay:.2f}s"]
            )
            self.causal_links.append(link)
            return link
        
        return None
    
    def get_causal_chain(self, event_id: str) -> List[CausalLink]:
        """Get causal chain starting from event."""
        chain = []
        
        # Forward chain
        current = event_id
        visited = set()
        
        while current and current not in visited:
            visited.add(current)
            
            # Find effects
            for link in self.causal_links:
                if link.cause == current:
                    chain.append(link)
                    current = link.effect
                    break
            else:
                break
        
        return chain
    
    # --- Prediction ---
    
    def predict_next(
        self, 
        recent_events: Optional[List[str]] = None,
        n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Predict next likely events based on patterns.
        
        Args:
            recent_events: Recent event types (or use last N)
            n: Number of predictions
        
        Returns:
            List of (event_type, probability) predictions
        """
        if recent_events is None:
            recent = self.event_timeline[-3:] if len(self.event_timeline) >= 3 else self.event_timeline
            recent_events = [e.event_type for e in recent]
        
        if not recent_events:
            return []
        
        # Use causal rules for prediction
        last_type = recent_events[-1]
        predictions = []
        
        possible = self.causal_rules.get(last_type, [])
        for i, effect in enumerate(possible[:n]):
            prob = 1.0 / (i + 1)  # Simple decreasing probability
            predictions.append((effect, prob))
        
        # Also check historical patterns
        type_counts: Dict[str, int] = {}
        for i in range(len(self.event_timeline) - 1):
            if self.event_timeline[i].event_type == last_type:
                next_type = self.event_timeline[i + 1].event_type
                type_counts[next_type] = type_counts.get(next_type, 0) + 1
        
        total = sum(type_counts.values()) or 1
        for event_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:n]:
            prob = count / total
            # Don't duplicate
            if event_type not in [p[0] for p in predictions]:
                predictions.append((event_type, prob))
        
        return predictions[:n]
    
    # --- Stats ---
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reasoner statistics."""
        return {
            "total_events": len(self.events),
            "event_types": len(set(e.event_type for e in self.events.values())),
            "constraints": len(self.constraints),
            "causal_links": len(self.causal_links),
            "timeline_span": (
                self.event_timeline[-1].timestamp - self.event_timeline[0].timestamp
                if len(self.event_timeline) >= 2 else 0.0
            )
        }
