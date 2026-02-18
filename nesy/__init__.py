"""
NeSy: Universal Neuro-Symbolic AI Platform

This is the main entry point for the NeSy platform. It provides a high-level API
for building neuro-symbolic AI systems with full backend agnosticism.

All components (perception, reasoning, LLM, agents) are created via the
Registry plugin system and can be swapped by changing configuration.

Example:
    >>> from nesy import NeSyPlatform
    >>> platform = NeSyPlatform()
    >>> platform.perceive(image)
    >>> results = platform.reason("find all red cups")
"""

__version__ = "1.0.0"
__author__ = "NeSy Platform Contributors"
__all__ = [
    "NeSyPlatform",
    "UMA",
    "NeSyConfig",
    "load_config",
    "get_logger",
    "Registry",
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
from nesy.core.registry import Registry, register_plugin
from nesy.core.factory import ComponentFactory

# Platform class
from typing import Optional, Any, Dict, Union, List
from pathlib import Path


class NeSyPlatform:
    """
    High-level API for the NeSy platform.

    This is the main class that users interact with. It orchestrates all
    subsystems (HAL, Perception, Reasoning, Agents, etc.) using the
    Registry for backend-agnostic component creation.

    Example:
        >>> # Default (mock backends)
        >>> platform = NeSyPlatform()
        >>>
        >>> # With real backends
        >>> from nesy.core.config import NeSyConfig
        >>> config = NeSyConfig()
        >>> config.perception.object_detection["backend"] = "yolo"
        >>> platform = NeSyPlatform(config)
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

        # Initialize subsystems via Registry (lazy)
        self._perception_detector = None
        self._perception_extractor = None
        self._reasoning_engine = None
        self._llm_provider = None
        self._agent = None
        self._message_bus = None

        # World model (direct, not pluggable — it's our core data structure)
        self.world_model = None
        self.tools = None

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

    # --- Lazy component accessors via Registry ---

    @property
    def detector(self):
        """Object detector (lazy, created from config via Registry)."""
        if self._perception_detector is None:
            from nesy.perception.base import ObjectDetectorBase
            # Ensure plugins are loaded
            import nesy.perception.detectors  # noqa: F401
            config = dict(self.config.perception.object_detection)
            backend = config.pop("backend", "mock")
            self._perception_detector = Registry.create(
                ObjectDetectorBase, backend, **config
            )
        return self._perception_detector

    @property
    def extractor(self):
        """Feature extractor (lazy, created from config via Registry)."""
        if self._perception_extractor is None:
            from nesy.perception.base import FeatureExtractorBase
            import nesy.perception.extractors  # noqa: F401
            config = dict(self.config.perception.feature_extraction)
            backend = config.pop("backend", "mock")
            self._perception_extractor = Registry.create(
                FeatureExtractorBase, backend, **config
            )
        return self._perception_extractor

    @property
    def logic_engine(self):
        """Logic engine (lazy, created from config via Registry)."""
        if self._reasoning_engine is None:
            from nesy.reasoning.base import LogicEngineBase
            import nesy.reasoning.engines  # noqa: F401
            logic_config = self.config.reasoning.logic
            backend = logic_config.get("engine", "scallop")
            self._reasoning_engine = Registry.create(
                LogicEngineBase, backend,
                provenance=logic_config.get("provenance", "difftopkproofs"),
                k=logic_config.get("k", 3),
            )
        return self._reasoning_engine

    @property
    def llm(self):
        """LLM provider (lazy, created from config via Registry)."""
        if self._llm_provider is None:
            from nesy.reasoning.llm.base import LLMProviderBase
            import nesy.reasoning.llm.providers  # noqa: F401
            llm_config = self.config.reasoning.llm
            provider = llm_config.get("provider", "mock")
            self._llm_provider = Registry.create(
                LLMProviderBase, provider,
                model=llm_config.get("model", ""),
            )
        return self._llm_provider

    @property
    def message_bus(self):
        """Message bus (lazy)."""
        if self._message_bus is None:
            from nesy.middleware.bus import LocalMessageBus
            self._message_bus = LocalMessageBus()
        return self._message_bus

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "NeSyPlatform":
        """
        Create platform from configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            NeSyPlatform instance
        """
        config = load_config(config_path)
        return cls(config)

    def perceive(self, sensor_data: Any) -> Dict[str, Any]:
        """
        Run perception pipeline on sensor data.

        Uses the configured detector and extractor backends.

        Args:
            sensor_data: Raw sensor data (image as numpy array)

        Returns:
            Perception results (detected objects, features)
        """
        import numpy as np
        if not isinstance(sensor_data, np.ndarray):
            self.logger.warning("sensor_data should be a numpy array")
            return {}

        detections = self.detector.detect(sensor_data)
        features = self.extractor.extract(sensor_data)

        return {
            "detections": [d.to_dict() for d in detections],
            "features": {
                "dim": features.dim,
                "model": features.model,
            },
            "num_objects": len(detections),
        }

    def reason(self, query: str) -> Any:
        """
        Execute reasoning query via the logic engine.

        Args:
            query: Relation name to query

        Returns:
            Query results (list of tuples)
        """
        return self.logic_engine.query(query)

    def visualize(self) -> None:
        """Open visualization interface."""
        self.logger.warning("Visualization not yet implemented")

    def get_registry_summary(self) -> Dict[str, List[str]]:
        """
        Get a summary of all registered plugins.

        Returns:
            Dict mapping interface names to lists of plugin names.
        """
        return {
            iface.__name__: Registry.list_plugins(iface)
            for iface in Registry.get_all_interfaces()
        }

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
    """
    return NeSyPlatform.from_config(config_path)
