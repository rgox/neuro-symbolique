"""Unit tests for nesy.core.config module."""

import pytest
import tempfile
from pathlib import Path
import yaml
from nesy.core.config import (
    NeSyConfig,
    load_config,
    get_default_config,
    validate_config,
    LogLevel,
)


class TestNeSyConfig:
    """Tests for NeSyConfig."""

    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = NeSyConfig()
        assert config.hal is not None
        assert config.world_model is not None
        assert config.reasoning is not None
        assert config.log_level == LogLevel.INFO
        assert config.seed == 42

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = NeSyConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "hal" in config_dict
        assert "world_model" in config_dict
        assert "reasoning" in config_dict

    def test_config_save_and_load(self):
        """Test saving and loading configuration."""
        config = NeSyConfig()
        config.seed = 123
        config.hal.npu["device"] = "cuda"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = Path(f.name)
            config.save(config_path)

        # Load it back
        loaded_config = NeSyConfig.from_file(config_path)
        assert loaded_config.seed == 123
        assert loaded_config.hal.npu["device"] == "cuda"

        # Cleanup
        config_path.unlink()

    def test_config_from_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            NeSyConfig.from_file("nonexistent.yaml")

    def test_config_merge(self):
        """Test merging configurations."""
        base = NeSyConfig()
        base.seed = 42
        base.hal.npu["device"] = "cpu"

        override = {
            "seed": 99,
            "hal": {
                "npu": {
                    "device": "cuda"
                }
            }
        }

        merged = NeSyConfig.merge(base, override)
        assert merged.seed == 99
        assert merged.hal.npu["device"] == "cuda"

    def test_load_config_with_defaults(self):
        """Test load_config returns defaults when no file specified."""
        config = load_config()
        assert isinstance(config, NeSyConfig)
        assert config.seed == 42

    def test_load_config_with_overrides(self):
        """Test load_config with overrides."""
        config = load_config(
            overrides={"seed": 777, "log_level": "debug"}
        )
        assert config.seed == 777
        assert config.log_level == LogLevel.DEBUG

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        assert isinstance(config, NeSyConfig)
        assert config.seed == 42
        assert config.log_level == LogLevel.INFO

    def test_validate_config_valid(self):
        """Test validating a valid configuration."""
        config = NeSyConfig()
        config.hal.npu["device"] = "cpu"  # CPU always available

        messages = validate_config(config)
        # Should have at least a warning about Scallop not installed
        assert isinstance(messages, list)

    def test_validate_config_cuda_not_available(self):
        """Test validation warns if CUDA requested but not available."""
        config = NeSyConfig()
        config.hal.npu["device"] = "cuda"

        messages = validate_config(config)

        # Check if PyTorch is installed
        try:
            import torch
            if not torch.cuda.is_available():
                assert any("CUDA" in msg for msg in messages)
        except ImportError:
            assert any("PyTorch" in msg for msg in messages)

    def test_hal_config_defaults(self):
        """Test HAL configuration defaults."""
        config = NeSyConfig()
        assert "backend" in config.hal.npu
        assert config.hal.npu["backend"] == "pytorch"
        assert config.hal.spu["mode"] == "simple"
        assert config.hal.memory["pool_enabled"] is True

    def test_world_model_config_defaults(self):
        """Test World Model configuration defaults."""
        config = NeSyConfig()
        assert config.world_model.scene_graph["layers"] == ["L1", "L2", "L3"]
        assert config.world_model.vsa["dim"] == 1024
        assert config.world_model.vsa["method"] == "hrr"
        assert config.world_model.temporal["enabled"] is False

    def test_reasoning_config_defaults(self):
        """Test Reasoning configuration defaults."""
        config = NeSyConfig()
        assert config.reasoning.logic["engine"] == "scallop"
        assert config.reasoning.logic["k"] == 3
        assert config.reasoning.planning["enabled"] is False

    def test_config_from_dict_partial(self):
        """Test creating config from partial dictionary."""
        partial_dict = {
            "seed": 555,
            "hal": {
                "npu": {
                    "device": "cuda"
                }
            }
        }

        config = NeSyConfig.from_dict(partial_dict)
        assert config.seed == 555
        assert config.hal.npu["device"] == "cuda"
        # Other fields should have defaults
        assert config.world_model.vsa["dim"] == 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
