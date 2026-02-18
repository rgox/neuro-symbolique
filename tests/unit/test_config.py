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

    def test_config_from_dict_all_sections(self):
        """Test creating config from complete dictionary."""
        full_dict = {
            "hal": {"npu": {"device": "cuda"}},
            "world_model": {"vsa": {"dim": 2048}},
            "reasoning": {"logic": {"engine": "prolog"}},
            "middleware": {"ros2": {"enabled": True}},
            "perception": {"object_detection": {"confidence_threshold": 0.7}},
            "tools": {"debugger": {"enabled": False}},
            "log_level": "debug",
            "seed": 999,
            "profile": True,
        }

        config = NeSyConfig.from_dict(full_dict)
        assert config.hal.npu["device"] == "cuda"
        assert config.world_model.vsa["dim"] == 2048
        assert config.reasoning.logic["engine"] == "prolog"
        assert config.middleware.ros2["enabled"] is True
        assert config.perception.object_detection["confidence_threshold"] == 0.7
        assert config.tools.debugger["enabled"] is False
        assert config.log_level == LogLevel.DEBUG
        assert config.seed == 999
        assert config.profile is True

    def test_config_from_dict_log_level_enum(self):
        """Test config from dict with LogLevel enum."""
        config_dict = {"log_level": LogLevel.ERROR}
        config = NeSyConfig.from_dict(config_dict)
        assert config.log_level == LogLevel.ERROR

    def test_load_config_from_file_with_overrides(self):
        """Test loading config from file with overrides."""
        # Create a temp config file
        config = NeSyConfig()
        config.seed = 111

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = Path(f.name)
            config.save(config_path)

        # Load with overrides
        loaded = load_config(
            config_path=config_path,
            overrides={"seed": 222, "profile": True}
        )

        assert loaded.seed == 222  # Override wins
        assert loaded.profile is True

        # Cleanup
        config_path.unlink()

    def test_validate_config_llm_no_api_key(self):
        """Test validation warns when LLM enabled without API key."""
        config = NeSyConfig()
        config.reasoning.llm["enabled"] = True
        config.reasoning.llm["api_key"] = None

        messages = validate_config(config)
        assert any("API key" in msg for msg in messages)

    def test_middleware_config_defaults(self):
        """Test Middleware configuration defaults."""
        config = NeSyConfig()
        assert config.middleware.semantic_pubsub["enabled"] is False
        assert config.middleware.ros2["enabled"] is False
        assert config.middleware.ros2["domain_id"] == 0
        assert config.middleware.mcp["port"] == 8080

    def test_perception_config_defaults(self):
        """Test Perception configuration defaults."""
        config = NeSyConfig()
        assert config.perception.object_detection["model"] == "yolov8n"
        assert config.perception.object_detection["confidence_threshold"] == 0.5
        assert config.perception.object_detection["backend"] == "mock"
        assert config.perception.feature_extraction["model"] == "ViT-B/32"
        assert config.perception.feature_extraction["feature_dim"] == 512
        assert config.perception.feature_extraction["backend"] == "mock"

    def test_tools_config_defaults(self):
        """Test Tools configuration defaults."""
        config = NeSyConfig()
        assert config.tools.debugger["enabled"] is True
        assert config.tools.debugger["trace_file"] == "trace.json"
        assert config.tools.visualization["backend"] == "rerun"
        assert config.tools.visualization["auto_open"] is True
        assert config.tools.simulator["enabled"] is False

    def test_config_save_creates_parent_dirs(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "config.yaml"
            config = NeSyConfig()
            config.save(config_path)

            assert config_path.exists()
            assert config_path.parent.exists()

    def test_config_from_empty_file(self):
        """Test loading config from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = Path(f.name)
            f.write("")  # Empty file

        config = NeSyConfig.from_file(config_path)
        # Should create default config
        assert config.seed == 42
        assert config.log_level == LogLevel.INFO

        config_path.unlink()

    def test_deep_merge_nested(self):
        """Test deep merge with nested dictionaries."""
        from nesy.core.config import _deep_merge

        base = {
            "a": {"b": {"c": 1, "d": 2}},
            "e": 3
        }
        override = {
            "a": {"b": {"c": 99}},
            "f": 4
        }

        result = _deep_merge(base, override)

        # c should be overridden
        assert result["a"]["b"]["c"] == 99
        # d should remain
        assert result["a"]["b"]["d"] == 2
        # e should remain
        assert result["e"] == 3
        # f should be added
        assert result["f"] == 4

    def test_deep_merge_replace_non_dict(self):
        """Test deep merge replaces non-dict values."""
        from nesy.core.config import _deep_merge

        base = {"a": {"b": "old_value"}}
        override = {"a": {"b": "new_value"}}

        result = _deep_merge(base, override)
        assert result["a"]["b"] == "new_value"

    def test_all_log_levels(self):
        """Test all log level enum values."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"

    def test_config_profile_flag(self):
        """Test profile flag in config."""
        config = NeSyConfig()
        assert config.profile is False

        config.profile = True
        config_dict = config.to_dict()
        assert config_dict["profile"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
