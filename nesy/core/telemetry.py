"""
Telemetry and Logging System for NeSy Platform.

This module provides structured logging with semantic annotations, performance
metrics, and event tracking. It's designed to support debugging complex
neuro-symbolic systems where understanding the interplay between neural and
symbolic components is critical.
"""

import logging
import json
import time
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from enum import Enum


class EventType(Enum):
    """Types of telemetry events."""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    PLANNING = "planning"
    GROUNDING = "grounding"  # Neural-symbolic grounding events
    GRAPH = "graph"  # Scene/knowledge graph events
    MEMORY = "memory"
    DEVICE = "device"  # NPU/SPU/CPU events
    ERROR = "error"
    PERFORMANCE = "performance"


@dataclass
class TelemetryEvent:
    """
    A telemetry event with structured data.

    Attributes:
        event_type: Type of event
        timestamp: Unix timestamp
        message: Human-readable message
        data: Structured event data
        level: Log level (debug, info, warning, error)
        source: Source component (e.g., 'npu', 'spu', 'scene_graph')
        duration_ms: Optional duration for performance events
    """
    event_type: EventType
    timestamp: float
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    level: str = "info"
    source: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["timestamp_iso"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class TelemetryLogger:
    """
    Central telemetry logger for the NeSy platform.

    This logger provides:
    - Structured logging with semantic annotations
    - JSON output for machine parsing
    - Performance profiling
    - Event filtering and aggregation

    Example:
        >>> logger = TelemetryLogger("my_component")
        >>> logger.info("Detected objects", data={"count": 5, "classes": ["cup", "table"]})
        >>>
        >>> # Profile code execution
        >>> with logger.profile("inference"):
        ...     result = model(input)
    """

    def __init__(
        self,
        source: str,
        log_level: str = "info",
        log_file: Optional[Path] = None,
        json_output: bool = True,
    ):
        """
        Initialize telemetry logger.

        Args:
            source: Source component name
            log_level: Minimum log level ('debug', 'info', 'warning', 'error')
            log_file: Optional file path for persistent logging
            json_output: If True, output structured JSON logs
        """
        self.source = source
        self.json_output = json_output
        self.log_file = log_file
        self.events: List[TelemetryEvent] = []

        # Setup Python logger
        self.logger = logging.getLogger(f"nesy.{source}")
        self.logger.setLevel(self._level_to_logging(log_level))

        # Console handler
        console_handler = logging.StreamHandler()
        if json_output:
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s [%(name)s] %(message)s'
            )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _level_to_logging(self, level: str) -> int:
        """Convert string level to logging constant."""
        mapping = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }
        return mapping.get(level.lower(), logging.INFO)

    def _log_event(self, event: TelemetryEvent) -> None:
        """Log an event."""
        self.events.append(event)

        # Format message
        if self.json_output:
            message = event.to_json()
        else:
            message = f"{event.message}"
            if event.data:
                message += f" | {event.data}"
            if event.duration_ms is not None:
                message += f" | {event.duration_ms:.2f}ms"

        # Log using Python logger
        log_method = getattr(self.logger, event.level, self.logger.info)
        log_method(message)

    def debug(
        self,
        message: str,
        event_type: EventType = EventType.DEVICE,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log debug event."""
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=time.time(),
            message=message,
            data=data or {},
            level="debug",
            source=self.source,
        )
        self._log_event(event)

    def info(
        self,
        message: str,
        event_type: EventType = EventType.DEVICE,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log info event."""
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=time.time(),
            message=message,
            data=data or {},
            level="info",
            source=self.source,
        )
        self._log_event(event)

    def warning(
        self,
        message: str,
        event_type: EventType = EventType.DEVICE,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log warning event."""
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=time.time(),
            message=message,
            data=data or {},
            level="warning",
            source=self.source,
        )
        self._log_event(event)

    def error(
        self,
        message: str,
        event_type: EventType = EventType.ERROR,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log error event."""
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=time.time(),
            message=message,
            data=data or {},
            level="error",
            source=self.source,
        )
        self._log_event(event)

    def perf(
        self,
        message: str,
        duration_ms: float,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log performance event."""
        event = TelemetryEvent(
            event_type=EventType.PERFORMANCE,
            timestamp=time.time(),
            message=message,
            data=data or {},
            level="info",
            source=self.source,
            duration_ms=duration_ms,
        )
        self._log_event(event)

    @contextmanager
    def profile(self, operation: str, event_type: EventType = EventType.PERFORMANCE):
        """
        Profile a code block.

        Args:
            operation: Name of the operation
            event_type: Type of event

        Example:
            >>> with logger.profile("inference"):
            ...     result = model(input)
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.perf(
                f"{operation} completed",
                duration_ms=duration_ms,
                data={"operation": operation},
            )

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        level: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[TelemetryEvent]:
        """
        Retrieve events with optional filtering.

        Args:
            event_type: Filter by event type
            level: Filter by log level
            since: Filter events after this timestamp

        Returns:
            List of filtered events
        """
        events = self.events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if level:
            events = [e for e in events if e.level == level]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events

    def export_events(self, path: Path) -> None:
        """
        Export all events to JSON file.

        Args:
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump([e.to_dict() for e in self.events], f, indent=2)

    def clear_events(self) -> None:
        """Clear all stored events."""
        self.events.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get telemetry statistics.

        Returns:
            Dictionary with event counts and performance metrics
        """
        if not self.events:
            return {"total_events": 0}

        # Event counts by type
        event_counts = {}
        for event in self.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Performance metrics
        perf_events = [e for e in self.events if e.duration_ms is not None]
        if perf_events:
            durations = [e.duration_ms for e in perf_events if e.duration_ms]
            perf_stats = {
                "count": len(durations),
                "total_ms": sum(durations),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
            }
        else:
            perf_stats = {}

        return {
            "total_events": len(self.events),
            "event_counts": event_counts,
            "performance": perf_stats,
            "time_range": {
                "start": datetime.fromtimestamp(self.events[0].timestamp).isoformat(),
                "end": datetime.fromtimestamp(self.events[-1].timestamp).isoformat(),
            },
        }


# Global logger registry
_logger_registry: Dict[str, TelemetryLogger] = {}


def get_logger(
    source: str,
    log_level: str = "info",
    log_file: Optional[Path] = None,
    json_output: bool = False,
) -> TelemetryLogger:
    """
    Get or create a telemetry logger for a source.

    Args:
        source: Source component name
        log_level: Minimum log level
        log_file: Optional file path for persistent logging
        json_output: If True, output structured JSON logs

    Returns:
        TelemetryLogger instance
    """
    if source not in _logger_registry:
        _logger_registry[source] = TelemetryLogger(
            source=source,
            log_level=log_level,
            log_file=log_file,
            json_output=json_output,
        )
    return _logger_registry[source]


def configure_logging(log_level: str = "info", json_output: bool = False) -> None:
    """
    Configure global logging settings.

    Args:
        log_level: Minimum log level for all loggers
        json_output: If True, output structured JSON logs
    """
    # Update existing loggers
    for logger in _logger_registry.values():
        logger.logger.setLevel(logger._level_to_logging(log_level))
        logger.json_output = json_output
