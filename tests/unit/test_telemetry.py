"""
Unit tests for nesy.core.telemetry module.

Tests the structured logging, event tracking, performance profiling,
and telemetry event management.
"""

import pytest
import tempfile
import time
import json
from pathlib import Path

from nesy.core.telemetry import (
    TelemetryLogger,
    TelemetryEvent,
    EventType,
    get_logger,
    configure_logging,
)


class TestTelemetryEvent:
    """Test TelemetryEvent dataclass."""

    def test_event_creation(self):
        """Test creating telemetry event."""
        event = TelemetryEvent(
            event_type=EventType.PERCEPTION,
            timestamp=time.time(),
            message="Test event",
            data={"key": "value"},
            level="info",
            source="test",
        )

        assert event.event_type == EventType.PERCEPTION
        assert event.message == "Test event"
        assert event.data == {"key": "value"}
        assert event.level == "info"
        assert event.source == "test"

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        timestamp = time.time()
        event = TelemetryEvent(
            event_type=EventType.DEVICE,
            timestamp=timestamp,
            message="Device event",
            data={"device": "NPU"},
        )

        event_dict = event.to_dict()
        assert isinstance(event_dict, dict)
        assert event_dict["event_type"] == "device"
        assert event_dict["message"] == "Device event"
        assert event_dict["data"] == {"device": "NPU"}
        assert "timestamp_iso" in event_dict

    def test_event_to_json(self):
        """Test converting event to JSON string."""
        event = TelemetryEvent(
            event_type=EventType.REASONING,
            timestamp=time.time(),
            message="Reasoning event",
        )

        json_str = event.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "reasoning"
        assert parsed["message"] == "Reasoning event"

    def test_event_with_duration(self):
        """Test event with performance duration."""
        event = TelemetryEvent(
            event_type=EventType.PERFORMANCE,
            timestamp=time.time(),
            message="Operation completed",
            duration_ms=123.45,
        )

        assert event.duration_ms == 123.45
        event_dict = event.to_dict()
        assert event_dict["duration_ms"] == 123.45


class TestTelemetryLogger:
    """Test TelemetryLogger class."""

    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = TelemetryLogger(source="test_component")

        assert logger.source == "test_component"
        assert logger.json_output is True
        assert logger.log_file is None
        assert len(logger.events) == 0

    def test_logger_with_file(self):
        """Test logger with file output."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            log_path = Path(f.name)

        logger = TelemetryLogger(
            source="test",
            log_file=log_path,
        )

        logger.info("Test message")

        # Check file was created
        assert log_path.exists()

        # Cleanup
        log_path.unlink()

    def test_debug_logging(self):
        """Test debug level logging."""
        logger = TelemetryLogger(source="test", log_level="debug")

        logger.debug("Debug message", data={"debug": True})

        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.level == "debug"
        assert event.message == "Debug message"
        assert event.data["debug"] is True

    def test_info_logging(self):
        """Test info level logging."""
        logger = TelemetryLogger(source="test")

        logger.info("Info message", event_type=EventType.PERCEPTION)

        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.level == "info"
        assert event.event_type == EventType.PERCEPTION

    def test_warning_logging(self):
        """Test warning level logging."""
        logger = TelemetryLogger(source="test")

        logger.warning("Warning message", data={"severity": "medium"})

        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.level == "warning"
        assert event.data["severity"] == "medium"

    def test_error_logging(self):
        """Test error level logging."""
        logger = TelemetryLogger(source="test")

        logger.error("Error occurred", event_type=EventType.ERROR)

        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.level == "error"
        assert event.event_type == EventType.ERROR

    def test_performance_logging(self):
        """Test performance event logging."""
        logger = TelemetryLogger(source="test")

        logger.perf("Operation completed", duration_ms=42.5, data={"op": "inference"})

        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == EventType.PERFORMANCE
        assert event.duration_ms == 42.5
        assert event.data["op"] == "inference"

    def test_profile_context_manager(self):
        """Test profiling context manager."""
        logger = TelemetryLogger(source="test")

        with logger.profile("test_operation"):
            time.sleep(0.01)  # Sleep for 10ms

        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == EventType.PERFORMANCE
        assert "test_operation completed" in event.message
        assert event.duration_ms >= 10.0  # Should be at least 10ms

    def test_profile_with_event_type(self):
        """Test profiling with custom event type."""
        logger = TelemetryLogger(source="test")

        with logger.profile("grounding", event_type=EventType.GROUNDING):
            pass

        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == EventType.PERFORMANCE

    def test_get_events_no_filter(self):
        """Test retrieving all events."""
        logger = TelemetryLogger(source="test")

        logger.info("Event 1")
        logger.debug("Event 2")
        logger.error("Event 3")

        events = logger.get_events()
        assert len(events) == 3

    def test_get_events_filter_by_type(self):
        """Test filtering events by type."""
        logger = TelemetryLogger(source="test")

        logger.info("Perception", event_type=EventType.PERCEPTION)
        logger.info("Reasoning", event_type=EventType.REASONING)
        logger.info("Device", event_type=EventType.DEVICE)

        perception_events = logger.get_events(event_type=EventType.PERCEPTION)
        assert len(perception_events) == 1
        assert perception_events[0].event_type == EventType.PERCEPTION

    def test_get_events_filter_by_level(self):
        """Test filtering events by log level."""
        logger = TelemetryLogger(source="test")

        logger.info("Info 1")
        logger.info("Info 2")
        logger.error("Error 1")

        error_events = logger.get_events(level="error")
        assert len(error_events) == 1
        assert error_events[0].level == "error"

    def test_get_events_filter_by_time(self):
        """Test filtering events by timestamp."""
        logger = TelemetryLogger(source="test")

        logger.info("Event 1")
        time.sleep(0.01)
        timestamp = time.time()
        time.sleep(0.01)
        logger.info("Event 2")

        recent_events = logger.get_events(since=timestamp)
        assert len(recent_events) == 1
        assert recent_events[0].message == "Event 2"

    def test_export_events(self):
        """Test exporting events to JSON file."""
        logger = TelemetryLogger(source="test")

        logger.info("Event 1", data={"value": 1})
        logger.info("Event 2", data={"value": 2})

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = Path(f.name)

        logger.export_events(export_path)

        # Check file exists and contains events
        assert export_path.exists()
        with open(export_path, 'r') as f:
            events_data = json.load(f)

        assert len(events_data) == 2
        assert events_data[0]["message"] == "Event 1"
        assert events_data[1]["message"] == "Event 2"

        # Cleanup
        export_path.unlink()

    def test_clear_events(self):
        """Test clearing event history."""
        logger = TelemetryLogger(source="test")

        logger.info("Event 1")
        logger.info("Event 2")
        assert len(logger.events) == 2

        logger.clear_events()
        assert len(logger.events) == 0

    def test_get_stats_no_events(self):
        """Test getting stats with no events."""
        logger = TelemetryLogger(source="test")

        stats = logger.get_stats()
        assert stats["total_events"] == 0

    def test_get_stats_with_events(self):
        """Test getting statistics."""
        logger = TelemetryLogger(source="test")

        logger.info("Event 1", event_type=EventType.PERCEPTION)
        logger.info("Event 2", event_type=EventType.PERCEPTION)
        logger.info("Event 3", event_type=EventType.REASONING)
        logger.perf("Operation", duration_ms=100.0)
        logger.perf("Operation", duration_ms=200.0)

        stats = logger.get_stats()

        assert stats["total_events"] == 5
        assert stats["event_counts"]["perception"] == 2
        assert stats["event_counts"]["reasoning"] == 1
        assert stats["event_counts"]["performance"] == 2

        # Check performance stats
        assert stats["performance"]["count"] == 2
        assert stats["performance"]["total_ms"] == 300.0
        assert stats["performance"]["avg_ms"] == 150.0
        assert stats["performance"]["min_ms"] == 100.0
        assert stats["performance"]["max_ms"] == 200.0

        # Check time range
        assert "time_range" in stats
        assert "start" in stats["time_range"]
        assert "end" in stats["time_range"]

    def test_log_levels(self):
        """Test different log levels."""
        logger_info = TelemetryLogger(source="test", log_level="info")
        logger_info.debug("Should be filtered")
        logger_info.info("Should be logged")

        # Debug should still be stored in events
        assert len(logger_info.events) == 2

    def test_json_output_disabled(self):
        """Test logger with JSON output disabled."""
        logger = TelemetryLogger(source="test", json_output=False)

        assert logger.json_output is False
        logger.info("Test message")
        assert len(logger.events) == 1

    def test_multiple_event_types(self):
        """Test logging different event types."""
        logger = TelemetryLogger(source="test")

        # Test all event types
        logger.info("Perception", event_type=EventType.PERCEPTION)
        logger.info("Reasoning", event_type=EventType.REASONING)
        logger.info("Planning", event_type=EventType.PLANNING)
        logger.info("Grounding", event_type=EventType.GROUNDING)
        logger.info("Graph", event_type=EventType.GRAPH)
        logger.info("Memory", event_type=EventType.MEMORY)
        logger.info("Device", event_type=EventType.DEVICE)
        logger.error("Error", event_type=EventType.ERROR)
        logger.perf("Perf", duration_ms=10.0)

        assert len(logger.events) == 9

        stats = logger.get_stats()
        assert stats["total_events"] == 9


class TestGlobalLoggerRegistry:
    """Test global logger registry functions."""

    def test_get_logger_creates_new(self):
        """Test get_logger creates new logger."""
        logger = get_logger("test_source_unique_1")

        assert isinstance(logger, TelemetryLogger)
        assert logger.source == "test_source_unique_1"

    def test_get_logger_returns_existing(self):
        """Test get_logger returns existing logger."""
        logger1 = get_logger("test_source_unique_2")
        logger1.info("Event from logger1")

        logger2 = get_logger("test_source_unique_2")

        # Should be the same instance
        assert logger1 is logger2
        assert len(logger2.events) == 1

    def test_get_logger_with_params(self):
        """Test get_logger with custom parameters."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            log_path = Path(f.name)

        logger = get_logger(
            "test_source_unique_3",
            log_level="debug",
            log_file=log_path,
            json_output=False,
        )

        assert logger.log_file == log_path
        assert logger.json_output is False

        # Cleanup
        log_path.unlink()

    def test_configure_logging(self):
        """Test configuring global logging settings."""
        # Create a logger first
        logger = get_logger("test_source_unique_4")

        # Configure logging globally
        configure_logging(log_level="debug", json_output=False)

        # Logger should be updated
        assert logger.json_output is False

    def test_configure_logging_updates_level(self):
        """Test configure_logging updates log level."""
        logger = get_logger("test_source_unique_5", log_level="info")

        # Change to debug
        configure_logging(log_level="debug")

        # Logger should now accept debug messages
        # (Level is checked by Python's logging module, not our code)
        logger.debug("Debug message")
        assert len(logger.events) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
