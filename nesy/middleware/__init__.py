"""
Middleware Module - Semantic Communication Layer.

Provides a message bus for inter-component communication.
"""

from nesy.middleware.bus import MessageBus, LocalMessageBus, Message

__all__ = [
    "MessageBus",
    "LocalMessageBus",
    "Message",
]
