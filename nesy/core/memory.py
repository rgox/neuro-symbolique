"""
Unified Memory Architecture (UMA) for Zero-Copy Neural-Symbolic Grounding.

This module provides the foundation for efficient data sharing between NPU (neural),
SPU (symbolic), and CPU components. The key innovation is zero-copy transfers that
enable neural embeddings and symbolic representations to share the same memory space.
"""

from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
from dataclasses import dataclass
from enum import Enum


class DeviceType(Enum):
    """Device types in the heterogeneous system."""
    CPU = "cpu"
    NPU = "npu"  # Neural Processing Unit (backed by PyTorch)
    SPU = "spu"  # Symbolic Processing Unit


class DataType(Enum):
    """Supported data types for shared buffers."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"


@dataclass
class MemoryBuffer:
    """
    A shared memory buffer accessible by multiple devices.

    Attributes:
        key: Unique identifier for this buffer
        shape: Shape of the tensor/array
        dtype: Data type
        device: Current device location
        data: The actual data (torch.Tensor)
        metadata: Optional metadata (e.g., symbolic annotations)
    """
    key: str
    shape: Tuple[int, ...]
    dtype: DataType
    device: DeviceType
    data: torch.Tensor
    metadata: Optional[Dict[str, Any]] = None

    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array (zero-copy when on CPU)."""
        if self.data.is_cuda:
            return self.data.cpu().numpy()
        return self.data.numpy()

    def size_bytes(self) -> int:
        """Return memory size in bytes."""
        return self.data.element_size() * self.data.nelement()


class UMA:
    """
    Unified Memory Architecture for zero-copy neural-symbolic grounding.

    The UMA manages a pool of shared memory buffers that can be accessed by
    NPU (neural networks), SPU (symbolic reasoner), and CPU without copying data.
    This is critical for efficient neural-symbolic integration.

    Key Features:
    - Zero-copy memory sharing between devices
    - Automatic dtype conversion when necessary
    - Metadata tagging for semantic annotations
    - Memory pooling for efficiency

    Example:
        >>> uma = UMA()
        >>> # Allocate buffer for neural embeddings
        >>> embedding_buffer = uma.allocate(
        ...     key="object_embeddings",
        ...     shape=(10, 512),  # 10 objects, 512-dim embeddings
        ...     dtype=DataType.FLOAT32,
        ...     device=DeviceType.NPU
        ... )
        >>> # SPU can access the same buffer without copying
        >>> symbolic_view = uma.get("object_embeddings")
    """

    def __init__(self, device: Union[str, DeviceType] = "cpu"):
        """
        Initialize Unified Memory Architecture.

        Args:
            device: Default device for allocations ('cpu', 'cuda', 'cuda:0', etc.)
                   Can be string or DeviceType enum
        """
        # Normalize device to string
        if isinstance(device, DeviceType):
            self.device = device.value
        else:
            self.device = device
        self._buffers: Dict[str, MemoryBuffer] = {}
        self._memory_pool: Dict[Tuple[Tuple[int, ...], DataType], list] = {}

    def allocate(
        self,
        key: str,
        shape: Tuple[int, ...],
        dtype: DataType = DataType.FLOAT32,
        device: DeviceType = DeviceType.CPU,
        metadata: Optional[Dict[str, Any]] = None,
        pool: bool = True,
    ) -> MemoryBuffer:
        """
        Allocate a shared memory buffer.

        Args:
            key: Unique identifier for this buffer
            shape: Shape of the tensor
            dtype: Data type
            device: Device to allocate on
            metadata: Optional metadata (e.g., symbolic tags)
            pool: Whether to use memory pooling for reuse

        Returns:
            MemoryBuffer object

        Raises:
            ValueError: If key already exists
        """
        if key in self._buffers:
            raise ValueError(f"Buffer with key '{key}' already exists")

        # Try to reuse from pool
        if pool and (shape, dtype) in self._memory_pool:
            pool_list = self._memory_pool[(shape, dtype)]
            if pool_list:
                data = pool_list.pop()
                data.zero_()  # Clear previous data
            else:
                data = self._create_tensor(shape, dtype, device)
        else:
            data = self._create_tensor(shape, dtype, device)

        buffer = MemoryBuffer(
            key=key,
            shape=shape,
            dtype=dtype,
            device=device,
            data=data,
            metadata=metadata or {},
        )

        self._buffers[key] = buffer
        return buffer

    def _create_tensor(
        self, shape: Tuple[int, ...], dtype: DataType, device: DeviceType
    ) -> torch.Tensor:
        """Create a PyTorch tensor with the specified properties."""
        torch_dtype = self._dtype_to_torch(dtype)
        # Convert device to string for PyTorch
        if device == DeviceType.NPU:
            torch_device = self.device  # Use configured device (already normalized to string)
        else:
            torch_device = "cpu"
        return torch.zeros(shape, dtype=torch_dtype, device=torch_device)

    def get(self, key: str) -> Optional[MemoryBuffer]:
        """
        Retrieve a buffer by key.

        Args:
            key: Buffer identifier

        Returns:
            MemoryBuffer if found, None otherwise
        """
        return self._buffers.get(key)

    def exists(self, key: str) -> bool:
        """
        Check if a buffer exists.

        Args:
            key: Buffer identifier

        Returns:
            True if buffer exists, False otherwise
        """
        return key in self._buffers

    def free(self, key: str, return_to_pool: bool = True) -> None:
        """
        Free a buffer.

        Args:
            key: Buffer identifier
            return_to_pool: If True, return tensor to pool for reuse
        """
        if key not in self._buffers:
            return

        buffer = self._buffers[key]

        if return_to_pool:
            pool_key = (buffer.shape, buffer.dtype)
            if pool_key not in self._memory_pool:
                self._memory_pool[pool_key] = []
            self._memory_pool[pool_key].append(buffer.data)

        del self._buffers[key]

    def zero_copy_transfer(
        self, key: str, src_device: DeviceType, dst_device: DeviceType
    ) -> MemoryBuffer:
        """
        Zero-copy transfer between devices.

        This is the core method for neural-symbolic grounding. When transferring
        from NPU to SPU (or vice versa), we return a VIEW of the same memory,
        not a copy. This enables the symbolic reasoner to directly access neural
        embeddings without serialization overhead.

        Args:
            key: Buffer identifier
            src_device: Source device
            dst_device: Destination device

        Returns:
            MemoryBuffer (same underlying data, updated metadata)

        Note:
            True zero-copy is only possible within the same physical memory space.
            Transfer from GPU to CPU may involve a copy.
        """
        buffer = self.get(key)
        if buffer is None:
            raise ValueError(f"Buffer '{key}' not found")

        # Same device → pure pointer return
        if src_device == dst_device:
            return buffer

        # NPU (GPU) → CPU requires copy
        if src_device == DeviceType.NPU and dst_device in [DeviceType.CPU, DeviceType.SPU]:
            if buffer.data.is_cuda:
                # Move to CPU (this involves a copy, but unavoidable with current hardware)
                buffer.data = buffer.data.cpu()
                buffer.device = dst_device
            return buffer

        # CPU/SPU ↔ SPU/CPU → zero-copy (same memory)
        if src_device in [DeviceType.CPU, DeviceType.SPU] and dst_device in [
            DeviceType.CPU,
            DeviceType.SPU,
        ]:
            buffer.device = dst_device
            return buffer

        # CPU/SPU → NPU
        if src_device in [DeviceType.CPU, DeviceType.SPU] and dst_device == DeviceType.NPU:
            if not buffer.data.is_cuda and torch.cuda.is_available():
                buffer.data = buffer.data.to(self.device)
                buffer.device = dst_device
            return buffer

        return buffer

    def share(self, key: str, new_key: str) -> MemoryBuffer:
        """
        Create a new buffer that shares data with an existing one (view).

        This is useful for creating multiple symbolic handles to the same neural data.

        Args:
            key: Original buffer key
            new_key: New buffer key

        Returns:
            New MemoryBuffer (view of original data)
        """
        original = self.get(key)
        if original is None:
            raise ValueError(f"Buffer '{key}' not found")

        if new_key in self._buffers:
            raise ValueError(f"Buffer with key '{new_key}' already exists")

        # Create view of the same data
        shared_buffer = MemoryBuffer(
            key=new_key,
            shape=original.shape,
            dtype=original.dtype,
            device=original.device,
            data=original.data,  # Same tensor, no copy
            metadata=original.metadata.copy() if original.metadata else {},
        )

        self._buffers[new_key] = shared_buffer
        return shared_buffer

    def annotate(self, key: str, metadata: Dict[str, Any]) -> None:
        """
        Add or update metadata annotations on a buffer.

        This is how symbolic information (e.g., "this embedding represents a cup")
        is attached to neural data.

        Args:
            key: Buffer identifier
            metadata: Metadata to add/update
        """
        buffer = self.get(key)
        if buffer is None:
            raise ValueError(f"Buffer '{key}' not found")

        if buffer.metadata is None:
            buffer.metadata = {}
        buffer.metadata.update(metadata)

    def list_buffers(self) -> Dict[str, Dict[str, Any]]:
        """
        List all buffers and their properties.

        Returns:
            Dictionary mapping keys to buffer info
        """
        return {
            key: {
                "shape": buf.shape,
                "dtype": buf.dtype.value,
                "device": buf.device.value,
                "size_bytes": buf.size_bytes(),
                "metadata": buf.metadata,
            }
            for key, buf in self._buffers.items()
        }

    def memory_usage(self) -> int:
        """Return total memory usage in bytes."""
        return sum(buf.size_bytes() for buf in self._buffers.values())

    def clear_pool(self) -> None:
        """Clear the memory pool."""
        self._memory_pool.clear()

    def clear_all(self) -> None:
        """Clear all buffers and pool."""
        self._buffers.clear()
        self._memory_pool.clear()

    @staticmethod
    def _dtype_to_torch(dtype: DataType) -> torch.dtype:
        """Convert DataType enum to torch dtype."""
        mapping = {
            DataType.FLOAT32: torch.float32,
            DataType.FLOAT16: torch.float16,
            DataType.INT32: torch.int32,
            DataType.INT64: torch.int64,
            DataType.UINT8: torch.uint8,
        }
        return mapping[dtype]

    @staticmethod
    def _torch_to_dtype(torch_dtype: torch.dtype) -> DataType:
        """Convert torch dtype to DataType enum."""
        mapping = {
            torch.float32: DataType.FLOAT32,
            torch.float16: DataType.FLOAT16,
            torch.int32: DataType.INT32,
            torch.int64: DataType.INT64,
            torch.uint8: DataType.UINT8,
        }
        return mapping.get(torch_dtype, DataType.FLOAT32)
