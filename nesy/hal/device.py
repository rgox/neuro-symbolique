"""
Device Abstraction Interface for Hardware Abstraction Layer (HAL).

This module defines the abstract interface that all devices (NPU, SPU, CPU) must implement.
It enables polymorphic device management and workload distribution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import time

from nesy.core.memory import UMA, MemoryBuffer, DeviceType
from nesy.core.telemetry import TelemetryLogger, EventType


class WorkloadType(Enum):
    """Types of computational workloads."""
    NEURAL_INFERENCE = "neural_inference"
    NEURAL_TRAINING = "neural_training"
    GRAPH_TRAVERSAL = "graph_traversal"
    LOGIC_EVALUATION = "logic_evaluation"
    SYMBOLIC_QUERY = "symbolic_query"
    GENERAL_COMPUTE = "general_compute"


@dataclass
class Workload:
    """
    Represents a computational workload to be executed on a device.

    Attributes:
        workload_type: Type of workload
        operation: The operation to execute (function, model, query, etc.)
        inputs: Input data (MemoryBuffer keys or direct data)
        outputs: Output buffer keys (will be allocated if not exist)
        metadata: Optional metadata (e.g., profiling info)
    """
    workload_type: WorkloadType
    operation: Any
    inputs: Dict[str, Any]
    outputs: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionResult:
    """
    Result of workload execution.

    Attributes:
        success: Whether execution succeeded
        outputs: Output data (MemoryBuffer keys or direct values)
        duration_ms: Execution duration in milliseconds
        memory_used_bytes: Memory used during execution
        device_type: Device that executed the workload
        metadata: Additional metadata (FLOPS, error messages, etc.)
    """
    success: bool
    outputs: Dict[str, Any]
    duration_ms: float
    memory_used_bytes: int
    device_type: DeviceType
    metadata: Optional[Dict[str, Any]] = None


class Device(ABC):
    """
    Abstract base class for all devices in the HAL.

    This interface defines the contract that NPU, SPU, and CPU must implement.
    It enables:
    - Polymorphic device management
    - Unified workload execution
    - Memory management via UMA
    - Performance profiling
    """

    def __init__(
        self,
        device_type: DeviceType,
        uma: UMA,
        logger: Optional[TelemetryLogger] = None,
    ):
        """
        Initialize device.

        Args:
            device_type: Type of this device
            uma: Unified Memory Architecture instance
            logger: Optional telemetry logger
        """
        self.device_type = device_type
        self.uma = uma
        self.logger = logger

        # Performance statistics
        self.total_executions = 0
        self.total_duration_ms = 0.0
        self.total_memory_bytes = 0

    @abstractmethod
    def execute(self, workload: Workload) -> ExecutionResult:
        """
        Execute a workload on this device.

        Args:
            workload: The workload to execute

        Returns:
            ExecutionResult with outputs and profiling data

        Raises:
            NotImplementedError: If workload type not supported
            RuntimeError: If execution fails
        """
        pass

    @abstractmethod
    def supports_workload(self, workload_type: WorkloadType) -> bool:
        """
        Check if this device supports a workload type.

        Args:
            workload_type: Type of workload

        Returns:
            True if supported, False otherwise
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get device capabilities and specifications.

        Returns:
            Dictionary describing device capabilities
        """
        pass

    def allocate_memory(
        self, key: str, shape: tuple, dtype: Any, metadata: Optional[Dict] = None
    ) -> MemoryBuffer:
        """
        Allocate memory buffer for this device.

        Args:
            key: Buffer identifier
            shape: Buffer shape
            dtype: Data type
            metadata: Optional metadata

        Returns:
            Allocated MemoryBuffer
        """
        return self.uma.allocate(
            key=key,
            shape=shape,
            dtype=dtype,
            device=self.device_type,
            metadata=metadata,
        )

    def get_memory(self, key: str) -> Optional[MemoryBuffer]:
        """
        Retrieve memory buffer.

        Args:
            key: Buffer identifier

        Returns:
            MemoryBuffer if found, None otherwise
        """
        return self.uma.get(key)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get device execution statistics.

        Returns:
            Dictionary with performance statistics
        """
        avg_duration = (
            self.total_duration_ms / self.total_executions
            if self.total_executions > 0
            else 0
        )
        avg_memory = (
            self.total_memory_bytes / self.total_executions
            if self.total_executions > 0
            else 0
        )

        return {
            "device_type": self.device_type.value,
            "total_executions": self.total_executions,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": avg_duration,
            "total_memory_bytes": self.total_memory_bytes,
            "avg_memory_bytes": avg_memory,
        }

    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self.total_executions = 0
        self.total_duration_ms = 0.0
        self.total_memory_bytes = 0

    def _log_execution(
        self, workload: Workload, result: ExecutionResult
    ) -> None:
        """
        Log execution to telemetry.

        Args:
            workload: The executed workload
            result: Execution result
        """
        if self.logger:
            self.logger.perf(
                f"{self.device_type.value} executed {workload.workload_type.value}",
                duration_ms=result.duration_ms,
                data={
                    "device": self.device_type.value,
                    "workload": workload.workload_type.value,
                    "success": result.success,
                    "memory_bytes": result.memory_used_bytes,
                },
            )

    def _update_statistics(self, result: ExecutionResult) -> None:
        """
        Update device statistics.

        Args:
            result: Execution result
        """
        self.total_executions += 1
        self.total_duration_ms += result.duration_ms
        self.total_memory_bytes += result.memory_used_bytes

    def __repr__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__} type={self.device_type.value}>"


class DevicePool:
    """
    Pool of devices for workload distribution.

    This class manages multiple devices and routes workloads to the appropriate
    device based on workload type and device availability.
    """

    def __init__(self, uma: UMA, logger: Optional[TelemetryLogger] = None):
        """
        Initialize device pool.

        Args:
            uma: Unified Memory Architecture
            logger: Optional telemetry logger
        """
        self.uma = uma
        self.logger = logger
        self.devices: Dict[DeviceType, Device] = {}

    def register_device(self, device: Device) -> None:
        """
        Register a device in the pool.

        Args:
            device: Device to register
        """
        self.devices[device.device_type] = device
        if self.logger:
            self.logger.info(
                f"Registered device: {device.device_type.value}",
                event_type=EventType.DEVICE,
                data={"capabilities": device.get_capabilities()},
            )

    def get_device(self, device_type: DeviceType) -> Optional[Device]:
        """
        Get a specific device.

        Args:
            device_type: Type of device

        Returns:
            Device if found, None otherwise
        """
        return self.devices.get(device_type)

    def find_device_for_workload(
        self, workload: Workload
    ) -> Optional[Device]:
        """
        Find the best device for a workload.

        Args:
            workload: Workload to execute

        Returns:
            Device that can execute this workload, None if no suitable device
        """
        # Priority order: NPU for neural, SPU for symbolic, CPU as fallback
        if workload.workload_type in [
            WorkloadType.NEURAL_INFERENCE,
            WorkloadType.NEURAL_TRAINING,
        ]:
            preferred_order = [DeviceType.NPU, DeviceType.CPU]
        elif workload.workload_type in [
            WorkloadType.GRAPH_TRAVERSAL,
            WorkloadType.LOGIC_EVALUATION,
            WorkloadType.SYMBOLIC_QUERY,
        ]:
            preferred_order = [DeviceType.SPU, DeviceType.CPU]
        else:
            preferred_order = [DeviceType.CPU, DeviceType.NPU, DeviceType.SPU]

        for device_type in preferred_order:
            device = self.devices.get(device_type)
            if device and device.supports_workload(workload.workload_type):
                return device

        return None

    def execute(self, workload: Workload) -> ExecutionResult:
        """
        Execute workload on the best available device.

        Args:
            workload: Workload to execute

        Returns:
            ExecutionResult

        Raises:
            RuntimeError: If no suitable device found
        """
        device = self.find_device_for_workload(workload)
        if device is None:
            raise RuntimeError(
                f"No device available for workload type: {workload.workload_type}"
            )

        return device.execute(workload)

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all devices.

        Returns:
            Dictionary mapping device types to statistics
        """
        return {
            device_type.value: device.get_statistics()
            for device_type, device in self.devices.items()
        }

    def reset_all_statistics(self) -> None:
        """Reset statistics for all devices."""
        for device in self.devices.values():
            device.reset_statistics()
