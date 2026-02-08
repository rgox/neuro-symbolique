"""
CPU Orchestrator for Workload Distribution.

This module implements the CPU component that orchestrates workload distribution
between NPU (neural) and SPU (symbolic) devices, manages the device pool, and
handles fallback execution for general compute tasks.
"""

from typing import Any, Dict, Optional, List
import time
from concurrent.futures import ThreadPoolExecutor, Future

from nesy.core.memory import UMA, DeviceType
from nesy.core.telemetry import TelemetryLogger, EventType
from nesy.hal.device import (
    Device,
    DevicePool,
    Workload,
    WorkloadType,
    ExecutionResult,
)


class CPUOrchestrator(Device):
    """
    CPU Orchestrator for workload distribution and general compute.

    The CPU acts as:
    1. **Orchestrator**: Routes workloads to NPU/SPU based on type
    2. **Fallback**: Executes general compute tasks not suited for NPU/SPU
    3. **Coordinator**: Manages multi-device pipelines and dependencies

    For the MVP, it uses sequential execution. Future versions will support:
    - Parallel execution across devices
    - Dataflow-based scheduling
    - Priority-based queueing
    """

    def __init__(
        self,
        uma: UMA,
        device_pool: DevicePool,
        num_workers: int = 4,
        logger: Optional[TelemetryLogger] = None,
    ):
        """
        Initialize CPU orchestrator.

        Args:
            uma: Unified Memory Architecture
            device_pool: Pool of available devices (NPU, SPU)
            num_workers: Number of worker threads for parallel execution
            logger: Optional telemetry logger
        """
        super().__init__(DeviceType.CPU, uma, logger)

        self.device_pool = device_pool
        self.num_workers = num_workers

        # Thread pool for parallel execution (future enhancement)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # Execution queue (for future priority scheduling)
        self.pending_workloads: List[Workload] = []

        if logger:
            logger.info(
                "CPU Orchestrator initialized",
                event_type=EventType.DEVICE,
                data={"num_workers": num_workers},
            )

    def supports_workload(self, workload_type: WorkloadType) -> bool:
        """CPU supports all workload types (as fallback)."""
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Get CPU orchestrator capabilities."""
        return {
            "num_workers": self.num_workers,
            "device_pool_size": len(self.device_pool.devices),
            "available_devices": [
                d.value for d in self.device_pool.devices.keys()
            ],
            "supported_workloads": "all",
        }

    def execute(self, workload: Workload) -> ExecutionResult:
        """
        Execute workload by routing to appropriate device.

        Strategy:
        1. Find best device for workload type
        2. If found, delegate execution
        3. If not found, execute on CPU as fallback
        4. Log and profile

        Args:
            workload: Workload to execute

        Returns:
            ExecutionResult from device or CPU fallback
        """
        start_time = time.time()

        if self.logger:
            self.logger.info(
                f"Orchestrating workload: {workload.workload_type.value}",
                event_type=EventType.DEVICE,
                data={
                    "workload_type": workload.workload_type.value,
                    "operation": str(workload.operation)[:50],
                },
            )

        try:
            # Try to find specialized device
            device = self.device_pool.find_device_for_workload(workload)

            if device and device.device_type != DeviceType.CPU:
                # Delegate to specialized device
                result = device.execute(workload)

                if self.logger:
                    self.logger.info(
                        f"Workload executed on {device.device_type.value}",
                        event_type=EventType.DEVICE,
                        data={
                            "device": device.device_type.value,
                            "duration_ms": result.duration_ms,
                            "success": result.success,
                        },
                    )

                return result
            else:
                # Execute on CPU as fallback
                if self.logger:
                    self.logger.warning(
                        f"No specialized device found, executing on CPU",
                        event_type=EventType.DEVICE,
                        data={"workload_type": workload.workload_type.value},
                    )

                return self._execute_on_cpu(workload)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            if self.logger:
                self.logger.error(
                    f"Orchestration failed: {str(e)}",
                    event_type=EventType.ERROR,
                    data={"workload": workload.workload_type.value},
                )

            return ExecutionResult(
                success=False,
                outputs={},
                duration_ms=duration_ms,
                memory_used_bytes=0,
                device_type=self.device_type,
                metadata={"error": str(e)},
            )

    def _execute_on_cpu(self, workload: Workload) -> ExecutionResult:
        """
        Execute workload on CPU (fallback or general compute).

        Args:
            workload: Workload to execute

        Returns:
            ExecutionResult
        """
        start_time = time.time()

        try:
            # Execute the operation
            operation = workload.operation
            inputs = workload.inputs

            if callable(operation):
                # If it's a function, call it with inputs
                outputs = operation(**inputs)
            else:
                # Generic execution (for MVP, just pass through)
                outputs = {"result": "CPU fallback execution", "inputs": inputs}

            duration_ms = (time.time() - start_time) * 1000

            result = ExecutionResult(
                success=True,
                outputs=outputs,
                duration_ms=duration_ms,
                memory_used_bytes=self.uma.memory_usage(),
                device_type=self.device_type,
                metadata={"fallback": True},
            )

            self._update_statistics(result)
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            return ExecutionResult(
                success=False,
                outputs={},
                duration_ms=duration_ms,
                memory_used_bytes=0,
                device_type=self.device_type,
                metadata={"error": str(e), "fallback": True},
            )

    def execute_pipeline(self, workloads: List[Workload]) -> List[ExecutionResult]:
        """
        Execute a pipeline of workloads sequentially.

        For MVP, this is sequential. Future versions will support:
        - Dependency analysis
        - Parallel execution of independent workloads
        - Dataflow optimization

        Args:
            workloads: List of workloads to execute in order

        Returns:
            List of ExecutionResults
        """
        results = []

        if self.logger:
            self.logger.info(
                f"Executing pipeline of {len(workloads)} workloads",
                event_type=EventType.DEVICE,
            )

        for i, workload in enumerate(workloads):
            if self.logger:
                self.logger.debug(
                    f"Pipeline step {i+1}/{len(workloads)}: {workload.workload_type.value}",
                    event_type=EventType.DEVICE,
                )

            result = self.execute(workload)
            results.append(result)

            # Stop pipeline if a step fails
            if not result.success:
                if self.logger:
                    self.logger.error(
                        f"Pipeline failed at step {i+1}",
                        event_type=EventType.ERROR,
                        data={"step": i+1, "total": len(workloads)},
                    )
                break

        return results

    def execute_parallel(self, workloads: List[Workload]) -> List[ExecutionResult]:
        """
        Execute multiple independent workloads in parallel.

        Args:
            workloads: List of independent workloads

        Returns:
            List of ExecutionResults (in same order as input)
        """
        if self.logger:
            self.logger.info(
                f"Executing {len(workloads)} workloads in parallel",
                event_type=EventType.DEVICE,
            )

        # Submit all workloads to thread pool
        futures: List[Future] = []
        for workload in workloads:
            future = self.executor.submit(self.execute, workload)
            futures.append(future)

        # Wait for all to complete
        results = [future.result() for future in futures]

        return results

    def get_device_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all devices in the pool.

        Returns:
            Dictionary mapping device types to statistics
        """
        return self.device_pool.get_all_statistics()

    def shutdown(self) -> None:
        """Shutdown orchestrator and thread pool."""
        if self.logger:
            self.logger.info("Shutting down CPU orchestrator", event_type=EventType.DEVICE)

        self.executor.shutdown(wait=True)

    def __del__(self):
        """Ensure clean shutdown."""
        self.shutdown()
