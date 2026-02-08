"""
Hardware Abstraction Layer (HAL) for NeSy Platform.

This module provides the hardware abstraction layer that simulates heterogeneous
computing devices for neuro-symbolic AI:
- NPU (Neural Processing Unit): Specialized for neural network inference
- SPU (Symbolic Processing Unit): Specialized for graph traversal and logic
- CPU: General orchestration and workload distribution
"""

from nesy.hal.device import (
    Device,
    DevicePool,
    Workload,
    WorkloadType,
    ExecutionResult,
)
from nesy.hal.npu.simulator import NPU
from nesy.hal.spu.simulator import SPU, Graph
from nesy.hal.cpu.orchestrator import CPUOrchestrator

__all__ = [
    # Base classes
    "Device",
    "DevicePool",
    "Workload",
    "WorkloadType",
    "ExecutionResult",
    # Devices
    "NPU",
    "SPU",
    "CPUOrchestrator",
    # Utilities
    "Graph",
]
