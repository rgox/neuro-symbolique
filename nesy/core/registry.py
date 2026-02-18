"""
Plugin Registry - Central Registration System for Backend-Agnostic Components.

Provides a generic registry pattern for discovering and instantiating
pluggable backends (detectors, feature extractors, reasoning engines, etc.)
without hard-coding dependencies.

Inspired by the HAL Device ABC pattern (nesy.hal.device).

Example:
    >>> from nesy.core.registry import Registry, register_plugin
    >>>
    >>> class DetectorBase(ABC):
    ...     @abstractmethod
    ...     def detect(self, image): ...
    >>>
    >>> @register_plugin(DetectorBase, "mock")
    ... class MockDetector(DetectorBase):
    ...     def detect(self, image):
    ...         return [{"class": "cup", "confidence": 0.9}]
    >>>
    >>> detector = Registry.create(DetectorBase, "mock")
"""

from typing import Dict, List, Type, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Registry:
    """
    Generic plugin registry.

    Maintains a mapping of interface → {name → implementation class} for
    every registered plugin. Supports creation, listing, and introspection.
    """

    _registries: Dict[type, Dict[str, type]] = {}

    @classmethod
    def register(cls, interface: type, name: str, implementation: type) -> None:
        """
        Register a plugin implementation.

        Args:
            interface: The abstract interface (e.g., DetectorBase)
            name: Plugin name (e.g., "mock", "yolo")
            implementation: The concrete class implementing the interface
        """
        if interface not in cls._registries:
            cls._registries[interface] = {}
        cls._registries[interface][name] = implementation
        logger.debug(f"Registered plugin '{name}' for {interface.__name__}")

    @classmethod
    def create(cls, interface: type, name: str, **kwargs) -> Any:
        """
        Create a plugin instance by name.

        Args:
            interface: The abstract interface
            name: Plugin name
            **kwargs: Arguments passed to the constructor

        Returns:
            Instance of the plugin

        Raises:
            KeyError: If plugin not found
        """
        impl_cls = cls.get_class(interface, name)
        return impl_cls(**kwargs)

    @classmethod
    def get_class(cls, interface: type, name: str) -> type:
        """
        Get the plugin class (without instantiating).

        Args:
            interface: The abstract interface
            name: Plugin name

        Returns:
            The plugin class

        Raises:
            KeyError: If plugin not found
        """
        if interface not in cls._registries:
            raise KeyError(
                f"No plugins registered for {interface.__name__}"
            )
        if name not in cls._registries[interface]:
            available = list(cls._registries[interface].keys())
            raise KeyError(
                f"Plugin '{name}' not found for {interface.__name__}. "
                f"Available: {available}"
            )
        return cls._registries[interface][name]

    @classmethod
    def list_plugins(cls, interface: type) -> List[str]:
        """
        List all registered plugins for an interface.

        Args:
            interface: The abstract interface

        Returns:
            List of plugin names
        """
        if interface not in cls._registries:
            return []
        return list(cls._registries[interface].keys())

    @classmethod
    def has_plugin(cls, interface: type, name: str) -> bool:
        """Check if a plugin is registered."""
        return (
            interface in cls._registries
            and name in cls._registries[interface]
        )

    @classmethod
    def clear(cls, interface: Optional[type] = None) -> None:
        """
        Clear registrations.

        Args:
            interface: Clear only this interface, or all if None
        """
        if interface is None:
            cls._registries.clear()
        elif interface in cls._registries:
            del cls._registries[interface]

    @classmethod
    def get_all_interfaces(cls) -> List[type]:
        """List all interfaces that have registered plugins."""
        return list(cls._registries.keys())


def register_plugin(interface: type, name: str):
    """
    Decorator to register a class as a plugin.

    Args:
        interface: The abstract interface this class implements
        name: Plugin name for lookup

    Example:
        >>> @register_plugin(DetectorBase, "mock")
        ... class MockDetector(DetectorBase):
        ...     pass
    """
    def decorator(cls):
        Registry.register(interface, name, cls)
        return cls
    return decorator
