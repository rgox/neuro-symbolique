"""Tests for Middleware Message Bus."""

import pytest
from nesy.middleware.bus import MessageBus, LocalMessageBus, Message


class TestMessage:
    def test_message_creation(self):
        msg = Message(topic="test/topic", payload={"key": "value"})
        assert msg.topic == "test/topic"
        assert msg.payload == {"key": "value"}
        assert msg.timestamp > 0
        assert msg.sender is None
        assert msg.metadata == {}

    def test_message_with_sender(self):
        msg = Message(topic="t", payload=42, sender="agent-1")
        assert msg.sender == "agent-1"

    def test_message_with_metadata(self):
        msg = Message(topic="t", payload=None, metadata={"priority": "high"})
        assert msg.metadata["priority"] == "high"


class TestLocalMessageBus:
    def test_publish_subscribe(self):
        bus = LocalMessageBus()
        received = []
        bus.subscribe("test/topic", lambda msg: received.append(msg))
        bus.publish("test/topic", {"data": 1})
        assert len(received) == 1
        assert received[0].payload == {"data": 1}
        assert received[0].topic == "test/topic"

    def test_multiple_subscribers(self):
        bus = LocalMessageBus()
        r1, r2 = [], []
        bus.subscribe("events", r1.append)
        bus.subscribe("events", r2.append)
        bus.publish("events", "hello")
        assert len(r1) == 1
        assert len(r2) == 1

    def test_no_subscribers(self):
        bus = LocalMessageBus()
        bus.publish("empty/topic", "data")  # should not raise

    def test_wildcard_subscribe(self):
        bus = LocalMessageBus()
        received = []
        bus.subscribe("scene/*", received.append)
        bus.publish("scene/update", {"nodes": 5})
        bus.publish("scene/query", {"q": "cups"})
        bus.publish("other/topic", "ignored")
        assert len(received) == 2

    def test_unsubscribe(self):
        bus = LocalMessageBus()
        received = []
        cb = lambda msg: received.append(msg)
        bus.subscribe("topic", cb)
        bus.publish("topic", "a")
        assert len(received) == 1
        bus.unsubscribe("topic", cb)
        bus.publish("topic", "b")
        assert len(received) == 1  # no new messages

    def test_unsubscribe_nonexistent(self):
        bus = LocalMessageBus()
        bus.unsubscribe("nope", lambda m: None)  # should not raise

    def test_unsubscribe_wrong_callback(self):
        bus = LocalMessageBus()
        bus.subscribe("topic", lambda m: None)
        bus.unsubscribe("topic", lambda m: None)  # different lambda, no-op

    def test_get_topics(self):
        bus = LocalMessageBus()
        bus.subscribe("a/b", lambda m: None)
        bus.subscribe("c/d", lambda m: None)
        topics = bus.get_topics()
        assert "a/b" in topics
        assert "c/d" in topics

    def test_statistics(self):
        bus = LocalMessageBus()
        bus.subscribe("t", lambda m: None)
        bus.publish("t", 1)
        bus.publish("t", 2)
        stats = bus.get_statistics()
        assert stats["total_messages"] == 2
        assert stats["active_topics"] == 1
        assert stats["total_subscribers"] == 1
        assert stats["history_size"] == 2

    def test_history(self):
        bus = LocalMessageBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        bus.publish("a", 3)
        history = bus.get_history()
        assert len(history) == 3
        filtered = bus.get_history(topic="a")
        assert len(filtered) == 2

    def test_history_limit(self):
        bus = LocalMessageBus()
        for i in range(10):
            bus.publish("t", i)
        assert len(bus.get_history(limit=3)) == 3

    def test_clear(self):
        bus = LocalMessageBus()
        bus.subscribe("t", lambda m: None)
        bus.publish("t", 1)
        bus.clear()
        assert bus.get_topics() == []
        assert bus.get_statistics()["total_messages"] == 0

    def test_publisher_sender(self):
        bus = LocalMessageBus()
        received = []
        bus.subscribe("t", received.append)
        bus.publish("t", "data", sender="perception")
        assert received[0].sender == "perception"

    def test_subscriber_error_doesnt_crash(self):
        bus = LocalMessageBus()
        def bad_callback(msg):
            raise ValueError("boom")
        bus.subscribe("t", bad_callback)
        bus.publish("t", "data")  # should not raise

    def test_topic_matches_exact(self):
        assert LocalMessageBus._topic_matches("a/b", "a/b")
        assert not LocalMessageBus._topic_matches("a/b", "a/c")

    def test_topic_matches_wildcard(self):
        assert LocalMessageBus._topic_matches("scene/*", "scene/update")
        assert not LocalMessageBus._topic_matches("scene/*", "other/update")
        assert not LocalMessageBus._topic_matches("scene/*", "scene/a/b")

    def test_abstract_interface(self):
        assert issubclass(LocalMessageBus, MessageBus)
