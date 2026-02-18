"""
Message Bus - Semantic Pub/Sub Communication.

Provides an abstract message bus and a local in-memory implementation
for inter-component communication in the NeSy platform.

This is the foundational middleware layer. Future versions will add:
- ROS2 bridge (nesy.middleware.ros2)
- Agent Communication Protocol (nesy.middleware.acp)
- Model Context Protocol (nesy.middleware.mcp)

Example:
    >>> bus = LocalMessageBus()
    >>> received = []
    >>> bus.subscribe("perception/detections", received.append)
    >>> bus.publish("perception/detections", {"objects": ["cup", "table"]})
    >>> assert len(received) == 1
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time
import logging
import re

from nesy.core.registry import register_plugin

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    A message on the bus.

    Attributes:
        topic: Topic string (e.g., "perception/detections")
        payload: Message data (any serializable object)
        timestamp: Creation time (seconds since epoch)
        sender: Optional sender identifier
        metadata: Optional extra metadata
    """
    topic: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    sender: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageBus(ABC):
    """
    Abstract message bus interface.

    All middleware implementations (local, ROS2, ACP, MCP) must
    implement this interface for interoperability.
    """

    @abstractmethod
    def publish(self, topic: str, payload: Any, sender: Optional[str] = None) -> None:
        """Publish a message to a topic."""
        ...

    @abstractmethod
    def subscribe(self, topic: str, callback: Callable[[Message], None]) -> None:
        """Subscribe to a topic with a callback."""
        ...

    @abstractmethod
    def unsubscribe(self, topic: str, callback: Callable[[Message], None]) -> None:
        """Remove a subscription."""
        ...

    @abstractmethod
    def get_topics(self) -> List[str]:
        """List all active topics."""
        ...


@register_plugin(MessageBus, "local")
class LocalMessageBus(MessageBus):
    """
    In-memory synchronous message bus.

    Messages are delivered immediately (synchronously) to all subscribers
    when published. Supports wildcard topic matching with '*'.

    Example:
        >>> bus = LocalMessageBus()
        >>>
        >>> results = []
        >>> bus.subscribe("scene/*", lambda msg: results.append(msg))
        >>> bus.publish("scene/update", {"nodes": 5})
        >>> bus.publish("scene/query", {"query": "cups"})
        >>> assert len(results) == 2
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Message], None]]] = defaultdict(list)
        self._message_count: int = 0
        self._history: List[Message] = []
        self._max_history: int = 1000

    def publish(self, topic: str, payload: Any, sender: Optional[str] = None) -> None:
        """Publish a message — delivered synchronously to matching subscribers."""
        msg = Message(topic=topic, payload=payload, sender=sender)
        self._message_count += 1

        # Store in history
        if len(self._history) < self._max_history:
            self._history.append(msg)

        # Deliver to exact topic subscribers
        delivered = 0
        for sub_topic, callbacks in self._subscribers.items():
            if self._topic_matches(sub_topic, topic):
                for cb in callbacks:
                    try:
                        cb(msg)
                        delivered += 1
                    except Exception as e:
                        logger.error(f"Subscriber error on {topic}: {e}")

        logger.debug(f"Published to '{topic}' — {delivered} subscriber(s)")

    def subscribe(self, topic: str, callback: Callable[[Message], None]) -> None:
        """Subscribe a callback to a topic (supports '*' wildcards)."""
        self._subscribers[topic].append(callback)
        logger.debug(f"Subscribed to '{topic}'")

    def unsubscribe(self, topic: str, callback: Callable[[Message], None]) -> None:
        """Remove a specific callback from a topic."""
        if topic in self._subscribers:
            try:
                self._subscribers[topic].remove(callback)
                if not self._subscribers[topic]:
                    del self._subscribers[topic]
            except ValueError:
                pass  # Callback not found — no-op

    def get_topics(self) -> List[str]:
        """List all topics that have at least one subscriber."""
        return list(self._subscribers.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get bus statistics."""
        return {
            "total_messages": self._message_count,
            "active_topics": len(self._subscribers),
            "total_subscribers": sum(len(cbs) for cbs in self._subscribers.values()),
            "history_size": len(self._history),
        }

    def get_history(self, topic: Optional[str] = None, limit: int = 50) -> List[Message]:
        """Get message history, optionally filtered by topic."""
        if topic is None:
            return self._history[-limit:]
        return [m for m in self._history if self._topic_matches(topic, m.topic)][-limit:]

    def clear(self) -> None:
        """Clear all subscribers and history."""
        self._subscribers.clear()
        self._history.clear()
        self._message_count = 0

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        """Check if a topic matches a subscription pattern (supports '*' wildcard)."""
        if pattern == topic:
            return True
        if "*" in pattern:
            regex = "^" + re.escape(pattern).replace(r"\*", "[^/]*") + "$"
            return bool(re.match(regex, topic))
        return False
