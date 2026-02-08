"""
NeSy: Universal Neuro-Symbolic AI Platform

This is the main entry point for the NeSy platform. It provides a high-level API
for building neuro-symbolic AI systems.

Example:
    >>> from nesy import NeSyPlatform
    >>> platform = NeSyPlatform.from_config('configs/minimal.yaml')
    >>> platform.perceive(image)
    >>> results = platform.reason("find all red cups")
"""

__version__ = "0.1.0"
__author__ = "NeSy Platform Contributors"
__all__ = [
    "NeSyPlatform",
    "UMA",
    "NeSyConfig",
    "load_config",
    "get_logger",
]

# Core infrastructure
from nesy.core.memory import UMA, DeviceType, DataType, MemoryBuffer
from nesy.core.config import (
    NeSyConfig,
    load_config,
    get_default_config,
    validate_config,
)
from nesy.core.telemetry import get_logger, TelemetryLogger, EventType

# Platform class (will be implemented in next steps)
from typing import Optional, Any, Dict, Union
from pathlib import Path


class NeSyPlatform:
    """
    High-level API for the NeSy platform.

    This is the main class that users interact with. It orchestrates all
    subsystems (HAL, World Model, Reasoning, etc.) and provides a simple
    interface for common tasks.

    Example:
        >>> # Minimal usage
        >>> platform = NeSyPlatform.from_config('configs/minimal.yaml')
        >>> platform.perceive(image)
        >>> results = platform.reason("find all cups")
        >>>
        >>> # Advanced usage
        >>> platform = NeSyPlatform(config)
        >>> platform.hal.npu.execute(model, input)
        >>> platform.world_model.scene_graph.query(...)
    """

    def __init__(self, config: Optional[NeSyConfig] = None):
        """
        Initialize NeSy platform.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or get_default_config()
        self.logger = get_logger("platform", log_level=self.config.log_level.value)

        # Validate configuration
        warnings = validate_config(self.config)
        for warning in warnings:
            self.logger.warning(warning)

        # Initialize core infrastructure
        self.logger.info("Initializing Unified Memory Architecture")
        device = self.config.hal.npu.get("device", "cpu")
        self.uma = UMA(device=device)

        # Initialize HAL (Hardware Abstraction Layer)
        self.logger.info("Initializing Hardware Abstraction Layer")
        self.hal = self._initialize_hal()

        # Initialize subsystems (placeholders for post-HAL implementation)
        self.world_model = None  # Will be initialized in World Model implementation
        self.reasoning = None  # Will be initialized in Reasoning implementation
        self.middleware = None  # Will be initialized in Middleware implementation
        self.perception = None  # Will be initialized in Perception implementation
        self.tools = None  # Will be initialized in Tools implementation

        self.logger.info("NeSy platform initialized successfully")

    def _initialize_hal(self):
        """Initialize Hardware Abstraction Layer with NPU, SPU, CPU."""
        from nesy.hal import DevicePool, NPU, SPU, CPUOrchestrator

        # Create device pool
        device_pool = DevicePool(self.uma, self.logger)

        # Initialize NPU (Neural Processing Unit)
        npu_config = self.config.hal.npu
        npu = NPU(
            uma=self.uma,
            backend=npu_config.get("backend", "pytorch"),
            device=npu_config.get("device", "cpu"),
            precision=npu_config.get("precision", "float32"),
            logger=self.logger,
        )
        device_pool.register_device(npu)

        # Initialize SPU (Symbolic Processing Unit)
        spu_config = self.config.hal.spu
        spu = SPU(
            uma=self.uma,
            mode=spu_config.get("mode", "simple"),
            graph_cache_size=spu_config.get("graph_cache_size", 1000),
            max_depth=spu_config.get("max_depth", 10),
            logger=self.logger,
        )
        device_pool.register_device(spu)

        # Initialize CPU Orchestrator
        cpu_config = self.config.hal.cpu
        cpu = CPUOrchestrator(
            uma=self.uma,
            device_pool=device_pool,
            num_workers=cpu_config.get("num_workers", 4),
            logger=self.logger,
        )
        device_pool.register_device(cpu)

        # Return HAL object with device references
        class HAL:
            def __init__(self, npu, spu, cpu, device_pool):
                self.npu = npu
                self.spu = spu
                self.cpu = cpu
                self.device_pool = device_pool

        return HAL(npu, spu, cpu, device_pool)

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "NeSyPlatform":
        """
        Create platform from configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            NeSyPlatform instance

        Example:
            >>> platform = NeSyPlatform.from_config('configs/minimal.yaml')
        """
        config = load_config(config_path)
        return cls(config)

    def perceive(self, sensor_data: Any) -> Dict[str, Any]:
        """
        Run perception pipeline on sensor data.

        Args:
            sensor_data: Raw sensor data (image, point cloud, etc.)

        Returns:
            Perception results (detected objects, features, etc.)

        Note:
            This is a placeholder. Full implementation in Phase 2.
        """
        self.logger.warning("Perception module not yet implemented")
        return {}

    def reason(self, query: str) -> Any:
        """
        Execute reasoning query.

        Args:
            query: Reasoning query (natural language or logic syntax)

        Returns:
            Query results

        Note:
            This is a placeholder. Full implementation in Phase 2.
        """
        self.logger.warning("Reasoning module not yet implemented")
        return None

    def visualize(self) -> None:
        """
        Open visualization interface.

        Note:
            This is a placeholder. Full implementation in Phase 2.
        """
        self.logger.warning("Visualization not yet implemented")

    def shutdown(self) -> None:
        """Clean shutdown of the platform."""
        self.logger.info("Shutting down NeSy platform")
        if self.uma:
            self.uma.clear_all()
        self.logger.info("Shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Convenience function for quick starts
def quick_start(config_path: str = "configs/minimal.yaml") -> NeSyPlatform:
    """
    Quick start with minimal configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Initialized NeSyPlatform

    Example:
        >>> platform = quick_start()
    """
    return NeSyPlatform.from_config(config_path)
