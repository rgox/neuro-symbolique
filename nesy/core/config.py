"""
Configuration Management System for NeSy Platform.

This module provides a hierarchical configuration system that supports:
- YAML-based configuration files
- Environment variable overrides
- Default configurations with user overrides
- Type validation and schema checking
"""

from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import yaml
import os
from dataclasses import dataclass, field, asdict
from enum import Enum


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class HALConfig:
    """Hardware Abstraction Layer configuration."""
    npu: Dict[str, Any] = field(default_factory=lambda: {
        "backend": "pytorch",
        "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        "precision": "float32",
    })
    spu: Dict[str, Any] = field(default_factory=lambda: {
        "mode": "simple",
        "graph_cache_size": 1000,
        "max_depth": 10,
    })
    cpu: Dict[str, Any] = field(default_factory=lambda: {
        "num_workers": os.cpu_count() or 4,
    })
    memory: Dict[str, Any] = field(default_factory=lambda: {
        "pool_enabled": True,
        "max_memory_mb": 4096,
    })


@dataclass
class WorldModelConfig:
    """World Model configuration."""
    scene_graph: Dict[str, Any] = field(default_factory=lambda: {
        "layers": ["L1", "L2", "L3"],
        "update_rate_hz": 10,
        "spatial_index": "octree",
    })
    knowledge_graph: Dict[str, Any] = field(default_factory=lambda: {
        "backend": "rdflib",
        "persist": False,
        "ontology_path": None,
    })
    vsa: Dict[str, Any] = field(default_factory=lambda: {
        "dim": 1024,
        "method": "hrr",  # Holographic Reduced Representations
        "seed": 42,
    })
    temporal: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "history_length": 100,
    })


@dataclass
class ReasoningConfig:
    """Reasoning engine configuration."""
    logic: Dict[str, Any] = field(default_factory=lambda: {
        "engine": "scallop",
        "provenance": "difftopkproofs",
        "k": 3,  # Top-k proofs
    })
    planning: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "planner": "pddl",
    })
    llm: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "model": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    })
    verification: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "check_safety": True,
    })


@dataclass
class MiddlewareConfig:
    """Middleware configuration."""
    semantic_pubsub: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "protocol": "zmq",
    })
    ros2: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "domain_id": 0,
    })
    mcp: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "port": 8080,
    })


@dataclass
class PerceptionConfig:
    """Perception configuration."""
    object_detection: Dict[str, Any] = field(default_factory=lambda: {
        "model": "yolov8",
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
    })
    feature_extraction: Dict[str, Any] = field(default_factory=lambda: {
        "model": "clip",
        "embedding_dim": 512,
    })


@dataclass
class ToolsConfig:
    """Development tools configuration."""
    debugger: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "trace_file": "trace.json",
        "log_level": "info",
    })
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        "backend": "rerun",
        "port": 9876,
        "auto_open": True,
    })
    simulator: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "engine": "pyreason",
    })


@dataclass
class NeSyConfig:
    """
    Top-level configuration for NeSy platform.

    This is the main configuration object that aggregates all subsystem configs.
    """
    hal: HALConfig = field(default_factory=HALConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    middleware: MiddlewareConfig = field(default_factory=MiddlewareConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    # Global settings
    log_level: LogLevel = LogLevel.INFO
    seed: int = 42
    profile: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output file path
        """
        path = Path(path)
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()
        # Convert enums to strings
        config_dict["log_level"] = self.log_level.value

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NeSyConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            NeSyConfig object
        """
        # Handle nested dataclasses
        config = cls()

        if "hal" in config_dict:
            config.hal = HALConfig(**config_dict["hal"])

        if "world_model" in config_dict:
            config.world_model = WorldModelConfig(**config_dict["world_model"])

        if "reasoning" in config_dict:
            config.reasoning = ReasoningConfig(**config_dict["reasoning"])

        if "middleware" in config_dict:
            config.middleware = MiddlewareConfig(**config_dict["middleware"])

        if "perception" in config_dict:
            config.perception = PerceptionConfig(**config_dict["perception"])

        if "tools" in config_dict:
            config.tools = ToolsConfig(**config_dict["tools"])

        # Global settings
        if "log_level" in config_dict:
            level = config_dict["log_level"]
            config.log_level = LogLevel(level) if isinstance(level, str) else level

        if "seed" in config_dict:
            config.seed = config_dict["seed"]

        if "profile" in config_dict:
            config.profile = config_dict["profile"]

        return config

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "NeSyConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Configuration file path

        Returns:
            NeSyConfig object

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict or {})

    @classmethod
    def merge(cls, base: "NeSyConfig", override: Dict[str, Any]) -> "NeSyConfig":
        """
        Merge a base configuration with overrides.

        Args:
            base: Base configuration
            override: Override dictionary

        Returns:
            New merged configuration
        """
        base_dict = base.to_dict()
        merged = _deep_merge(base_dict, override)
        return cls.from_dict(merged)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> NeSyConfig:
    """
    Load configuration with support for defaults, files, and overrides.

    Priority (highest to lowest):
    1. Explicit overrides dictionary
    2. Configuration file
    3. Default configuration

    Args:
        config_path: Optional path to YAML configuration file
        overrides: Optional dictionary of configuration overrides

    Returns:
        NeSyConfig object

    Example:
        >>> # Use defaults
        >>> config = load_config()
        >>>
        >>> # Load from file
        >>> config = load_config("configs/minimal.yaml")
        >>>
        >>> # Load from file with overrides
        >>> config = load_config(
        ...     "configs/minimal.yaml",
        ...     overrides={"hal": {"npu": {"device": "cpu"}}}
        ... )
    """
    # Start with defaults
    config = NeSyConfig()

    # Load from file if provided
    if config_path is not None:
        file_config = NeSyConfig.from_file(config_path)
        config = NeSyConfig.merge(config, file_config.to_dict())

    # Apply overrides
    if overrides:
        config = NeSyConfig.merge(config, overrides)

    return config


def get_default_config() -> NeSyConfig:
    """
    Get the default configuration.

    Returns:
        Default NeSyConfig object
    """
    return NeSyConfig()


def validate_config(config: NeSyConfig) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.

    Args:
        config: Configuration to validate

    Returns:
        List of validation messages (empty if valid)
    """
    messages = []

    # Check NPU device availability
    if config.hal.npu.get("device", "").startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                messages.append("CUDA device requested but not available, falling back to CPU")
        except ImportError:
            messages.append("PyTorch not installed, NPU will not work")

    # Check reasoning engine
    if config.reasoning.logic.get("engine") == "scallop":
        try:
            import scallop  # type: ignore
        except ImportError:
            messages.append("Scallop not installed, reasoning engine will not work")

    # Check visualization backend
    if config.tools.visualization.get("backend") == "rerun":
        try:
            import rerun  # type: ignore
        except ImportError:
            messages.append("Rerun not installed, visualization will not work")

    # Check LLM configuration
    if config.reasoning.llm.get("enabled"):
        if not config.reasoning.llm.get("api_key"):
            messages.append("LLM enabled but no API key configured")

    return messages
