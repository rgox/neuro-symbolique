"""
Tests for Hardware Abstraction Layer (HAL).

Tests the device abstraction, NPU simulator, SPU simulator, and CPU orchestrator.
"""

import pytest
import numpy as np
import torch

from nesy.core.memory import UMA, DeviceType, DataType
from nesy.core.telemetry import TelemetryLogger
from nesy.hal.device import Device, Workload, WorkloadType, ExecutionResult
from nesy.hal.npu.simulator import NPU
from nesy.hal.spu.simulator import SPU, Graph
from nesy.hal.cpu.orchestrator import CPUOrchestrator, DevicePool


class TestGraph:
    """Test the SPU Graph data structure."""

    def test_graph_initialization(self):
        """Test graph initialization."""
        graph = Graph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_node(self):
        """Test adding nodes."""
        graph = Graph()
        graph.add_node("cup", attributes={"type": "object"})

        assert "cup" in graph.nodes
        assert graph.node_attributes["cup"] == {"type": "object"}

    def test_add_edge(self):
        """Test adding edges."""
        graph = Graph()
        graph.add_edge("cup", "on", "table", attributes={"confidence": 0.9})

        assert "cup" in graph.nodes
        assert "table" in graph.nodes
        assert ("on", "table") in graph.edges["cup"]
        assert ("on", "cup") in graph.reverse_edges["table"]
        assert graph.edge_attributes[("cup", "on", "table")] == {"confidence": 0.9}

    def test_get_neighbors(self):
        """Test getting neighbors."""
        graph = Graph()
        graph.add_edge("cup", "on", "table")
        graph.add_edge("cup", "near", "book")
        graph.add_edge("cup", "on", "desk")

        # All neighbors
        neighbors = graph.get_neighbors("cup")
        assert len(neighbors) == 3

        # Filtered by relation
        on_neighbors = graph.get_neighbors("cup", relation="on")
        assert len(on_neighbors) == 2
        assert "table" in on_neighbors
        assert "desk" in on_neighbors

    def test_get_neighbors_nonexistent(self):
        """Test getting neighbors of non-existent node."""
        graph = Graph()
        neighbors = graph.get_neighbors("nonexistent")
        assert neighbors == []


class TestNPU:
    """Test NPU simulator."""

    def test_npu_initialization(self):
        """Test NPU initialization."""
        uma = UMA(device=DeviceType.CPU)
        npu = NPU(uma=uma, device="cpu", precision="float32")

        assert npu.device_type == DeviceType.NPU
        assert npu.uma is uma
        assert npu.backend == "pytorch"
        assert npu.device_str == "cpu"
        assert npu.precision == "float32"
        assert npu.total_executions == 0

    def test_npu_cuda_fallback(self):
        """Test CUDA fallback to CPU when CUDA unavailable."""
        uma = UMA(device=DeviceType.CPU)
        logger = TelemetryLogger(source="test")

        npu = NPU(uma=uma, device="cuda", logger=logger)

        # Should fallback to CPU if CUDA not available
        if not torch.cuda.is_available():
            assert npu.device_str == "cpu"

    def test_npu_precision_settings(self):
        """Test NPU precision dtype mapping."""
        uma = UMA(device=DeviceType.CPU)

        # Test float32
        npu_fp32 = NPU(uma=uma, precision="float32")
        assert npu_fp32.default_dtype == torch.float32

        # Test float16
        npu_fp16 = NPU(uma=uma, precision="float16")
        assert npu_fp16.default_dtype == torch.float16

    def test_npu_execute_inference(self):
        """Test NPU neural inference execution."""
        uma = UMA(device=DeviceType.CPU)
        npu = NPU(uma=uma)

        # Create simple linear model
        model = torch.nn.Linear(10, 5)
        model.eval()

        # Create input tensor
        input_tensor = torch.randn(1, 10)

        # Create workload
        workload = Workload(
            workload_type=WorkloadType.NEURAL_INFERENCE,
            operation=model,
            inputs={"input": input_tensor},
        )

        # Execute
        result = npu.execute(workload)

        assert result.success
        assert "outputs" in result.outputs or "output_keys" in result.outputs
        assert result.duration_ms > 0
        assert result.device_type == DeviceType.NPU
        assert npu.total_executions == 1

    def test_npu_statistics(self):
        """Test NPU statistics."""
        uma = UMA(device=DeviceType.CPU)
        npu = NPU(uma=uma)

        # Check basic stats tracking
        assert npu.total_executions == 0
        assert npu.total_duration_ms == 0.0


class TestSPU:
    """Test SPU simulator."""

    def test_spu_initialization(self):
        """Test SPU initialization."""
        uma = UMA(device=DeviceType.CPU)
        spu = SPU(uma=uma)

        assert spu.device_type == DeviceType.SPU
        assert spu.uma is uma
        assert "default" in spu.graphs
        assert isinstance(spu.graphs["default"], Graph)
        assert spu.total_executions == 0

    def test_spu_graph_traversal(self):
        """Test SPU graph traversal execution."""
        uma = UMA(device=DeviceType.CPU)
        spu = SPU(uma=uma)

        # Add test graph
        graph = spu.get_graph("default")
        graph.add_edge("A", "connects_to", "B")
        graph.add_edge("B", "connects_to", "C")
        graph.add_edge("C", "connects_to", "D")

        # Create traversal workload
        workload = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="bfs",
            inputs={"start": "A", "target": "D"},
        )

        # Execute
        result = spu.execute(workload)

        assert result.success
        assert "path" in result.outputs
        assert result.device_type == DeviceType.SPU

    def test_spu_logic_evaluation(self):
        """Test SPU logic evaluation."""
        uma = UMA(device=DeviceType.CPU)
        spu = SPU(uma=uma)

        # Add facts
        graph = spu.get_graph("default")
        graph.add_node("socrates", attributes={"type": "person"})
        graph.add_node("human", attributes={"type": "category"})
        graph.add_edge("socrates", "is_a", "human")

        # Create logic workload
        workload = Workload(
            workload_type=WorkloadType.LOGIC_EVALUATION,
            operation="query",
            inputs={"pattern": {"?x": None, "is_a": "human"}},
        )

        # Execute
        result = spu.execute(workload)

        assert result.success
        assert result.device_type == DeviceType.SPU

    def test_spu_statistics(self):
        """Test SPU statistics."""
        uma = UMA(device=DeviceType.CPU)
        spu = SPU(uma=uma)

        # Check basic stats tracking
        assert spu.total_executions == 0
        assert spu.total_duration_ms == 0.0
        assert len(spu.graphs) == 1


class TestCPUOrchestrator:
    """Test CPU orchestrator."""

    def test_cpu_initialization(self):
        """Test CPU orchestrator initialization."""
        uma = UMA(device=DeviceType.CPU)
        npu = NPU(uma=uma)
        spu = SPU(uma=uma)

        device_pool = DevicePool(uma=uma)
        device_pool.register_device(npu)
        device_pool.register_device(spu)

        cpu = CPUOrchestrator(uma=uma, device_pool=device_pool)

        assert cpu.device_type == DeviceType.CPU
        assert cpu.device_pool is device_pool

    def test_cpu_dispatch_to_npu(self):
        """Test CPU dispatching workload to NPU."""
        uma = UMA(device=DeviceType.CPU)
        npu = NPU(uma=uma)
        spu = SPU(uma=uma)

        device_pool = DevicePool(uma=uma)
        device_pool.register_device(npu)
        device_pool.register_device(spu)

        cpu = CPUOrchestrator(uma=uma, device_pool=device_pool)

        # Create neural workload
        model = torch.nn.Linear(5, 3)
        workload = Workload(
            workload_type=WorkloadType.NEURAL_INFERENCE,
            operation=model,
            inputs={"input": torch.randn(1, 5)},
        )

        # Execute via CPU orchestrator
        result = cpu.execute(workload)

        assert result.success
        # Should be dispatched to NPU
        assert npu.total_executions == 1

    def test_cpu_dispatch_to_spu(self):
        """Test CPU dispatching workload to SPU."""
        uma = UMA(device=DeviceType.CPU)
        npu = NPU(uma=uma)
        spu = SPU(uma=uma)

        device_pool = DevicePool(uma=uma)
        device_pool.register_device(npu)
        device_pool.register_device(spu)

        cpu = CPUOrchestrator(uma=uma, device_pool=device_pool)

        # Add graph data
        spu.get_graph("default").add_edge("A", "rel", "B")

        # Create symbolic workload
        workload = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="bfs",
            inputs={"start": "A"},
        )

        # Execute via CPU orchestrator
        result = cpu.execute(workload)

        assert result.success
        # Should be dispatched to SPU
        assert spu.total_executions == 1

    def test_cpu_general_compute(self):
        """Test CPU handling general compute workload."""
        uma = UMA(device=DeviceType.CPU)
        npu = NPU(uma=uma)
        spu = SPU(uma=uma)

        device_pool = DevicePool(uma=uma)
        device_pool.register_device(npu)
        device_pool.register_device(spu)

        cpu = CPUOrchestrator(uma=uma, device_pool=device_pool)

        # Create general compute workload
        workload = Workload(
            workload_type=WorkloadType.GENERAL_COMPUTE,
            operation=lambda x: x * 2,
            inputs={"x": 5},
        )

        # Execute on CPU
        result = cpu.execute(workload)

        assert result.success
        assert result.outputs["result"] == 10
        assert result.device_type == DeviceType.CPU

    def test_cpu_statistics(self):
        """Test CPU statistics."""
        uma = UMA(device=DeviceType.CPU)
        npu = NPU(uma=uma)
        spu = SPU(uma=uma)

        device_pool = DevicePool(uma=uma)
        device_pool.register_device(npu)
        device_pool.register_device(spu)

        cpu = CPUOrchestrator(uma=uma, device_pool=device_pool)

        # Check basic stats tracking
        assert cpu.total_executions == 0
        assert cpu.total_duration_ms == 0.0


class TestDeviceIntegration:
    """Test integrated HAL functionality."""

    def test_multi_device_pipeline(self):
        """Test workload execution across multiple devices."""
        uma = UMA(device=DeviceType.CPU)
        npu = NPU(uma=uma)
        spu = SPU(uma=uma)

        device_pool = DevicePool(uma=uma)
        device_pool.register_device(npu)
        device_pool.register_device(spu)

        cpu = CPUOrchestrator(uma=uma, device_pool=device_pool)

        # Execute neural workload
        model = torch.nn.Linear(3, 2)
        neural_workload = Workload(
            workload_type=WorkloadType.NEURAL_INFERENCE,
            operation=model,
            inputs={"input": torch.randn(1, 3)},
        )
        neural_result = cpu.execute(neural_workload)
        assert neural_result.success

        # Execute symbolic workload
        spu.get_graph("default").add_edge("X", "rel", "Y")
        symbolic_workload = Workload(
            workload_type=WorkloadType.GRAPH_TRAVERSAL,
            operation="bfs",
            inputs={"start": "X"},
        )
        symbolic_result = cpu.execute(symbolic_workload)
        assert symbolic_result.success

        # Verify both devices were used
        assert npu.total_executions == 1
        assert spu.total_executions == 1

    def test_workload_with_uma(self):
        """Test workload execution with UMA memory sharing."""
        uma = UMA(device=DeviceType.CPU)
        npu = NPU(uma=uma)

        # Allocate input in UMA
        uma.allocate(
            key="input_tensor",
            shape=(1, 5),
            dtype=DataType.FLOAT32,
            device=DeviceType.NPU,
        )
        buffer = uma.get("input_tensor")
        buffer.data[:] = torch.randn(1, 5)

        # Execute workload using UMA buffer
        model = torch.nn.Linear(5, 3)
        workload = Workload(
            workload_type=WorkloadType.NEURAL_INFERENCE,
            operation=model,
            inputs={"input_key": "input_tensor"},
        )

        result = npu.execute(workload)
        assert result.success

    def test_performance_profiling(self):
        """Test performance profiling across devices."""
        uma = UMA(device=DeviceType.CPU)
        npu = NPU(uma=uma)

        # Execute multiple workloads
        model = torch.nn.Linear(5, 3)
        for _ in range(5):
            workload = Workload(
                workload_type=WorkloadType.NEURAL_INFERENCE,
                operation=model,
                inputs={"input": torch.randn(1, 5)},
            )
            npu.execute(workload)

        # Check stats
        assert npu.total_executions == 5
        assert npu.total_duration_ms > 0
