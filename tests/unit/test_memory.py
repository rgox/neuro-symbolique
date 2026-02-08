"""Unit tests for nesy.core.memory module."""

import pytest
import torch
import numpy as np
from nesy.core.memory import UMA, DeviceType, DataType, MemoryBuffer


class TestUMA:
    """Tests for Unified Memory Architecture."""

    def test_uma_initialization(self):
        """Test UMA can be initialized."""
        uma = UMA(device="cpu")
        assert uma.device == "cpu"
        assert len(uma._buffers) == 0

    def test_allocate_buffer(self):
        """Test buffer allocation."""
        uma = UMA()
        buffer = uma.allocate(
            key="test_buffer",
            shape=(10, 512),
            dtype=DataType.FLOAT32,
            device=DeviceType.CPU,
        )

        assert buffer.key == "test_buffer"
        assert buffer.shape == (10, 512)
        assert buffer.dtype == DataType.FLOAT32
        assert buffer.device == DeviceType.CPU
        assert buffer.data.shape == torch.Size([10, 512])

    def test_allocate_duplicate_key_raises(self):
        """Test that allocating with duplicate key raises error."""
        uma = UMA()
        uma.allocate(key="test", shape=(5,), dtype=DataType.FLOAT32)

        with pytest.raises(ValueError, match="already exists"):
            uma.allocate(key="test", shape=(5,), dtype=DataType.FLOAT32)

    def test_get_buffer(self):
        """Test retrieving a buffer."""
        uma = UMA()
        buffer = uma.allocate(key="test", shape=(5,), dtype=DataType.FLOAT32)

        retrieved = uma.get("test")
        assert retrieved is not None
        assert retrieved.key == "test"
        assert torch.equal(retrieved.data, buffer.data)

    def test_get_nonexistent_buffer(self):
        """Test retrieving non-existent buffer returns None."""
        uma = UMA()
        assert uma.get("nonexistent") is None

    def test_free_buffer(self):
        """Test freeing a buffer."""
        uma = UMA()
        uma.allocate(key="test", shape=(5,), dtype=DataType.FLOAT32)

        uma.free("test")
        assert uma.get("test") is None

    def test_zero_copy_transfer_same_device(self):
        """Test zero-copy transfer on same device returns same buffer."""
        uma = UMA()
        buffer = uma.allocate(key="test", shape=(10,), dtype=DataType.FLOAT32, device=DeviceType.CPU)

        transferred = uma.zero_copy_transfer("test", DeviceType.CPU, DeviceType.CPU)
        assert transferred.key == buffer.key
        assert torch.equal(transferred.data, buffer.data)

    def test_zero_copy_transfer_cpu_to_spu(self):
        """Test zero-copy transfer from CPU to SPU (should be true zero-copy)."""
        uma = UMA()
        buffer = uma.allocate(key="test", shape=(10,), dtype=DataType.FLOAT32, device=DeviceType.CPU)
        original_data_ptr = buffer.data.data_ptr()

        transferred = uma.zero_copy_transfer("test", DeviceType.CPU, DeviceType.SPU)
        assert transferred.device == DeviceType.SPU
        # Same underlying data (zero-copy)
        assert transferred.data.data_ptr() == original_data_ptr

    def test_share_buffer(self):
        """Test creating shared view of buffer."""
        uma = UMA()
        original = uma.allocate(key="original", shape=(5,), dtype=DataType.FLOAT32)
        original.data.fill_(42.0)

        shared = uma.share("original", "shared_copy")
        assert shared.key == "shared_copy"
        assert torch.equal(shared.data, original.data)

        # Modify shared, should affect original (same underlying memory)
        shared.data.fill_(100.0)
        assert torch.all(original.data == 100.0)

    def test_annotate_metadata(self):
        """Test adding metadata annotations."""
        uma = UMA()
        buffer = uma.allocate(key="test", shape=(5,), dtype=DataType.FLOAT32)

        uma.annotate("test", {"semantic_label": "cup", "confidence": 0.95})

        retrieved = uma.get("test")
        assert retrieved.metadata["semantic_label"] == "cup"
        assert retrieved.metadata["confidence"] == 0.95

    def test_list_buffers(self):
        """Test listing all buffers."""
        uma = UMA()
        uma.allocate(key="buffer1", shape=(10,), dtype=DataType.FLOAT32)
        uma.allocate(key="buffer2", shape=(20, 30), dtype=DataType.INT32)

        buffers = uma.list_buffers()
        assert len(buffers) == 2
        assert "buffer1" in buffers
        assert "buffer2" in buffers
        assert buffers["buffer1"]["shape"] == (10,)
        assert buffers["buffer2"]["shape"] == (20, 30)

    def test_memory_usage(self):
        """Test memory usage calculation."""
        uma = UMA()
        # Float32 = 4 bytes, 10 elements = 40 bytes
        uma.allocate(key="test", shape=(10,), dtype=DataType.FLOAT32)

        usage = uma.memory_usage()
        assert usage == 40

    def test_memory_pooling(self):
        """Test memory pooling for reuse."""
        uma = UMA()
        buffer1 = uma.allocate(key="test1", shape=(100,), dtype=DataType.FLOAT32, pool=True)
        data_ptr1 = buffer1.data.data_ptr()

        uma.free("test1", return_to_pool=True)

        # Allocate same shape/dtype, should reuse from pool
        buffer2 = uma.allocate(key="test2", shape=(100,), dtype=DataType.FLOAT32, pool=True)
        data_ptr2 = buffer2.data.data_ptr()

        assert data_ptr1 == data_ptr2  # Same tensor reused

    def test_clear_all(self):
        """Test clearing all buffers."""
        uma = UMA()
        uma.allocate(key="test1", shape=(10,), dtype=DataType.FLOAT32)
        uma.allocate(key="test2", shape=(20,), dtype=DataType.FLOAT32)

        uma.clear_all()
        assert len(uma._buffers) == 0
        assert len(uma._memory_pool) == 0

    def test_buffer_to_numpy(self):
        """Test converting buffer to numpy."""
        uma = UMA()
        buffer = uma.allocate(key="test", shape=(5,), dtype=DataType.FLOAT32)
        buffer.data.fill_(3.14)

        np_array = buffer.to_numpy()
        assert isinstance(np_array, np.ndarray)
        assert np.allclose(np_array, 3.14)

    def test_different_dtypes(self):
        """Test allocating buffers with different data types."""
        uma = UMA()

        for dtype, torch_dtype in [
            (DataType.FLOAT32, torch.float32),
            (DataType.FLOAT16, torch.float16),
            (DataType.INT32, torch.int32),
            (DataType.INT64, torch.int64),
            (DataType.UINT8, torch.uint8),
        ]:
            buffer = uma.allocate(
                key=f"test_{dtype.value}",
                shape=(10,),
                dtype=dtype
            )
            assert buffer.data.dtype == torch_dtype


class TestMemoryBuffer:
    """Tests for MemoryBuffer dataclass."""

    def test_buffer_size_bytes(self):
        """Test size calculation."""
        data = torch.zeros((10, 20), dtype=torch.float32)
        buffer = MemoryBuffer(
            key="test",
            shape=(10, 20),
            dtype=DataType.FLOAT32,
            device=DeviceType.CPU,
            data=data
        )

        # 10 * 20 * 4 bytes (float32) = 800 bytes
        assert buffer.size_bytes() == 800


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
