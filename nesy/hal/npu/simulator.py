"""
Neural Processing Unit (NPU) Simulator.

This module simulates a specialized neural processing unit by wrapping PyTorch
with profiling and performance tracking. It represents the neural component of
the neuro-symbolic architecture.
"""

from typing import Any, Dict, Optional, Callable
import time
import torch
import torch.nn as nn

from nesy.core.memory import UMA, DeviceType, DataType, MemoryBuffer
from nesy.core.telemetry import TelemetryLogger, EventType
from nesy.hal.device import Device, Workload, WorkloadType, ExecutionResult


class NPU(Device):
    """
    Neural Processing Unit simulator.

    The NPU specializes in:
    - Neural network inference
    - Dense linear algebra (GEMM operations)
    - High-throughput parallel computation
    - Low-precision arithmetic (FP16, INT8)

    It wraps PyTorch and provides:
    - Automatic device placement (CPU/CUDA)
    - Mixed precision support
    - Execution profiling (latency, memory, FLOPS)
    - Integration with UMA for zero-copy grounding
    """

    def __init__(
        self,
        uma: UMA,
        backend: str = "pytorch",
        device: str = "cpu",
        precision: str = "float32",
        logger: Optional[TelemetryLogger] = None,
    ):
        """
        Initialize NPU simulator.

        Args:
            uma: Unified Memory Architecture
            backend: Neural backend ('pytorch', 'jax', 'onnx')
            device: Compute device ('cpu', 'cuda', 'cuda:0', etc.)
            precision: Default precision ('float32', 'float16', 'bfloat16')
            logger: Optional telemetry logger
        """
        super().__init__(DeviceType.NPU, uma, logger)

        self.backend = backend
        self.device_str = device
        self.precision = precision

        # Setup PyTorch device
        if device.startswith("cuda") and not torch.cuda.is_available():
            if logger:
                logger.warning(
                    "CUDA requested but not available, falling back to CPU"
                )
            self.device_str = "cpu"

        self.torch_device = torch.device(self.device_str)

        # Precision settings
        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
        }
        self.default_dtype = self.dtype_map.get(precision, torch.float32)

        # Performance counters
        self.total_flops = 0
        self.total_params = 0

        if logger:
            logger.info(
                "NPU initialized",
                event_type=EventType.DEVICE,
                data={
                    "backend": backend,
                    "device": self.device_str,
                    "precision": precision,
                    "cuda_available": torch.cuda.is_available(),
                },
            )

    def supports_workload(self, workload_type: WorkloadType) -> bool:
        """Check if NPU supports this workload type."""
        return workload_type in [
            WorkloadType.NEURAL_INFERENCE,
            WorkloadType.NEURAL_TRAINING,
            WorkloadType.GENERAL_COMPUTE,  # Can do general tensor ops
        ]

    def get_capabilities(self) -> Dict[str, Any]:
        """Get NPU capabilities."""
        caps = {
            "backend": self.backend,
            "device": self.device_str,
            "precision": self.precision,
            "cuda_available": torch.cuda.is_available(),
            "supported_workloads": [
                WorkloadType.NEURAL_INFERENCE.value,
                WorkloadType.NEURAL_TRAINING.value,
            ],
        }

        if torch.cuda.is_available():
            caps["cuda_device_count"] = torch.cuda.device_count()
            caps["cuda_device_name"] = torch.cuda.get_device_name(0)

        return caps

    def execute(self, workload: Workload) -> ExecutionResult:
        """
        Execute neural workload.

        Args:
            workload: Workload containing model and inputs

        Returns:
            ExecutionResult with outputs and profiling

        Expected workload format:
            - operation: torch.nn.Module or callable
            - inputs: Dict with 'input' key containing tensor or buffer key
            - outputs: List of output buffer keys (optional)
        """
        if not self.supports_workload(workload.workload_type):
            raise NotImplementedError(
                f"NPU does not support {workload.workload_type}"
            )

        start_time = time.time()
        memory_before = self._get_memory_usage()

        try:
            # Extract model/operation
            model = workload.operation
            if isinstance(model, nn.Module):
                model = model.to(self.torch_device)
                model.eval()  # Inference mode by default

            # Prepare inputs
            inputs = self._prepare_inputs(workload.inputs)

            # Execute
            with torch.no_grad():  # Default to inference (no gradient tracking)
                if isinstance(model, nn.Module):
                    outputs = model(inputs)
                elif callable(model):
                    outputs = model(inputs)
                else:
                    raise ValueError(f"Invalid operation type: {type(model)}")

            # Store outputs in UMA
            output_keys = self._store_outputs(outputs, workload.outputs)

            duration_ms = (time.time() - start_time) * 1000
            memory_used = self._get_memory_usage() - memory_before

            result = ExecutionResult(
                success=True,
                outputs={"output_keys": output_keys, "outputs": outputs},
                duration_ms=duration_ms,
                memory_used_bytes=memory_used,
                device_type=self.device_type,
                metadata={
                    "device": self.device_str,
                    "dtype": str(self.default_dtype),
                },
            )

            # Update statistics
            self._update_statistics(result)
            self._log_execution(workload, result)

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            if self.logger:
                self.logger.error(
                    f"NPU execution failed: {str(e)}",
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

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Prepare inputs for execution.

        Args:
            inputs: Input data (tensor, buffer key, or numpy array)

        Returns:
            PyTorch tensor on correct device
        """
        # Handle different input formats
        if "input" in inputs:
            data = inputs["input"]
        elif "tensor" in inputs:
            data = inputs["tensor"]
        else:
            # Assume first value is the input
            data = next(iter(inputs.values()))

        # Convert to tensor if needed
        if isinstance(data, str):
            # It's a buffer key, retrieve from UMA
            buffer = self.uma.get(data)
            if buffer is None:
                raise ValueError(f"Buffer '{data}' not found in UMA")
            tensor = buffer.data
        elif isinstance(data, MemoryBuffer):
            tensor = data.data
        elif isinstance(data, torch.Tensor):
            tensor = data
        else:
            # Assume numpy array or list
            tensor = torch.tensor(data, dtype=self.default_dtype)

        # Move to device
        return tensor.to(self.torch_device)

    def _store_outputs(
        self, outputs: torch.Tensor, output_keys: Optional[list]
    ) -> list:
        """
        Store outputs in UMA.

        Args:
            outputs: Output tensor(s)
            output_keys: Desired output buffer keys

        Returns:
            List of output buffer keys
        """
        if output_keys is None:
            # Generate keys
            import uuid
            output_keys = [f"npu_output_{uuid.uuid4().hex[:8]}"]

        # Handle single tensor or tuple of tensors
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        stored_keys = []
        for i, (tensor, key) in enumerate(zip(outputs, output_keys)):
            # Store in UMA
            if isinstance(tensor, torch.Tensor):
                # Convert dtype to DataType enum
                if tensor.dtype == torch.float32:
                    dtype = DataType.FLOAT32
                elif tensor.dtype == torch.float16:
                    dtype = DataType.FLOAT16
                elif tensor.dtype == torch.int32:
                    dtype = DataType.INT32
                elif tensor.dtype == torch.int64:
                    dtype = DataType.INT64
                else:
                    dtype = DataType.FLOAT32

                buffer = self.uma.allocate(
                    key=key,
                    shape=tuple(tensor.shape),
                    dtype=dtype,
                    device=self.device_type,
                )
                buffer.data = tensor  # Store tensor directly
                stored_keys.append(key)

        return stored_keys

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        if self.device_str.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.torch_device)
        else:
            # For CPU, return UMA memory usage as proxy
            return self.uma.memory_usage()

    def execute_model(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        output_key: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Convenience method for executing a model.

        Args:
            model: PyTorch model
            input_tensor: Input tensor
            output_key: Optional output buffer key

        Returns:
            ExecutionResult
        """
        workload = Workload(
            workload_type=WorkloadType.NEURAL_INFERENCE,
            operation=model,
            inputs={"input": input_tensor},
            outputs=[output_key] if output_key else None,
        )
        return self.execute(workload)

    def profile_model(self, model: nn.Module, input_shape: tuple) -> Dict[str, Any]:
        """
        Profile a model (parameters, FLOPS, memory).

        Args:
            model: PyTorch model
            input_shape: Input tensor shape

        Returns:
            Profiling statistics
        """
        model = model.to(self.torch_device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.torch_device)

        # Measure forward pass
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        forward_time_ms = (time.time() - start) * 1000

        # Estimate FLOPs (rough estimate for linear layers)
        estimated_flops = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                estimated_flops += module.in_features * module.out_features

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "forward_time_ms": forward_time_ms,
            "estimated_flops": estimated_flops,
            "memory_params_mb": (total_params * 4) / (1024 * 1024),  # Assume FP32
            "device": self.device_str,
        }
