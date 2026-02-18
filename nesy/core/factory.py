"""
Component Factory - Config-Driven Component Creation.

Creates platform components from YAML/dict configuration using the Registry.

Example:
    >>> config = {"backend": "mock", "confidence": 0.5}
    >>> detector = ComponentFactory.create_from_config(DetectorBase, config)
"""

from typing import Any, Dict, Optional, Type
from nesy.core.registry import Registry


class ComponentFactory:
    """
    Factory for creating components from configuration dictionaries.

    Reads a 'backend' key from config to look up the plugin in Registry,
    then passes remaining keys as constructor arguments.
    """

    @staticmethod
    def create(
        interface: type,
        config: Dict[str, Any],
        backend_key: str = "backend",
        **extra_kwargs,
    ) -> Any:
        """
        Create a component from config dict.

        Args:
            interface: Abstract interface to create
            config: Configuration dict (must contain backend_key)
            backend_key: Key in config that identifies the backend name
            **extra_kwargs: Additional kwargs merged into constructor call

        Returns:
            Instantiated component

        Raises:
            KeyError: If backend not found in config or registry
        """
        config = dict(config)  # Don't mutate original
        backend_name = config.pop(backend_key, None)
        if backend_name is None:
            raise KeyError(
                f"Config must contain '{backend_key}' key. Got: {list(config.keys())}"
            )

        # Merge extra kwargs (config values take precedence)
        kwargs = {**extra_kwargs, **config}
        return Registry.create(interface, backend_name, **kwargs)

    @staticmethod
    def create_with_default(
        interface: type,
        config: Optional[Dict[str, Any]] = None,
        default_backend: str = "mock",
        backend_key: str = "backend",
        **extra_kwargs,
    ) -> Any:
        """
        Create component with a default backend fallback.

        If config is None or missing the backend key, uses default_backend.

        Args:
            interface: Abstract interface
            config: Optional configuration dict
            default_backend: Fallback backend name
            backend_key: Key identifying backend in config
            **extra_kwargs: Additional kwargs

        Returns:
            Instantiated component
        """
        if config is None:
            config = {}
        config = dict(config)
        if backend_key not in config:
            config[backend_key] = default_backend
        return ComponentFactory.create(
            interface, config, backend_key=backend_key, **extra_kwargs
        )
