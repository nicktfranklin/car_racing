"""
Tests for configuration classes.

Tests all configuration dataclasses and validation logic.
"""

import pytest

from world_models.config import (
    ControllerConfig,
    DataConfig,
    FSQVAEConfig,
    OptimizerConfig,
    WorldModelAgentConfig,
    WorldModelConfig,
)


class TestFSQVAEConfig:
    """Tests for FSQVAEConfig."""

    def test_default_config(self):
        """Test default FSQ-VAE configuration."""
        config = FSQVAEConfig()

        # Check basic attributes exist
        assert hasattr(config, "input_channels")
        assert hasattr(config, "input_height")
        assert hasattr(config, "input_width")
        assert hasattr(config, "latent_dim")
        assert hasattr(config, "fsq_levels")

        # Check reasonable defaults
        assert config.input_channels > 0
        assert config.input_height > 0
        assert config.input_width > 0
        assert config.latent_dim > 0

    def test_fsq_levels_valid(self):
        """Test that FSQ levels are valid."""
        config = FSQVAEConfig()

        # Levels should be a list of integers > 1
        assert isinstance(config.fsq_levels, list)
        assert len(config.fsq_levels) > 0
        assert all(isinstance(level, int) and level > 1 for level in config.fsq_levels)

    def test_codebook_size_computation(self):
        """Test that codebook size can be computed from levels."""
        config = FSQVAEConfig()

        # Codebook size should be product of levels
        from functools import reduce
        from operator import mul

        expected_size = reduce(mul, config.fsq_levels, 1)
        assert expected_size > 0


class TestWorldModelConfig:
    """Tests for WorldModelConfig."""

    def test_default_config(self):
        """Test default World Model configuration."""
        config = WorldModelConfig()

        # Check basic attributes exist
        assert hasattr(config, "state_dim")
        assert hasattr(config, "action_dim")
        assert hasattr(config, "num_layers")
        assert hasattr(config, "n_heads")
        assert hasattr(config, "hidden_size")

        # Check reasonable defaults
        assert config.state_dim > 0
        assert config.action_dim > 0
        assert config.num_layers > 0
        assert config.n_heads > 0
        assert config.hidden_size > 0

    def test_transformer_config_valid(self):
        """Test that transformer configuration is valid."""
        config = WorldModelConfig()

        # Embedding dimension should be divisible by number of heads
        if hasattr(config, "n_heads"):
            assert config.hidden_size % config.n_heads == 0


class TestControllerConfig:
    """Tests for ControllerConfig."""

    def test_default_config(self):
        """Test default Controller configuration."""
        config = ControllerConfig()

        # Check basic attributes exist
        assert hasattr(config, "state_dim")
        assert hasattr(config, "action_dim")
        assert hasattr(config, "hidden_sizes")

        # Check reasonable defaults
        assert config.state_dim > 0
        assert config.action_dim > 0
        assert len(config.hidden_sizes) > 0
        assert all(h > 0 for h in config.hidden_sizes)

    def test_action_dim_valid(self):
        """Test that action dimension matches CarRacing."""
        config = ControllerConfig()

        # CarRacing has 3 actions: steering, gas, brake
        assert config.action_dim == 3


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""

    def test_default_config(self):
        """Test default Optimizer configuration."""
        config = OptimizerConfig()

        # Check basic attributes exist
        assert hasattr(config, "optimizer")
        assert hasattr(config, "learning_rate")
        assert hasattr(config, "weight_decay")

        # Check reasonable defaults
        assert config.learning_rate > 0
        assert config.weight_decay >= 0

    def test_optimizer_choices(self):
        """Test that optimizer choice is valid."""
        config = OptimizerConfig()

        valid_optimizers = ["adam", "adamw", "sgd"]
        assert config.optimizer.lower() in valid_optimizers

    def test_scheduler_config(self):
        """Test scheduler configuration."""
        config = OptimizerConfig()

        if config.use_scheduler:
            assert hasattr(config, "scheduler")
            assert hasattr(config, "warmup_epochs")


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_config(self):
        """Test default Data configuration."""
        config = DataConfig()

        # Check basic attributes exist
        assert hasattr(config, "num_rollouts")
        assert hasattr(config, "max_episode_length")
        assert hasattr(config, "num_workers")

        # Check reasonable defaults
        assert config.num_rollouts > 0
        assert config.max_episode_length > 0

    def test_num_workers_valid(self):
        """Test that num_workers is valid."""
        config = DataConfig()

        # -1 means auto-detect, 0 means sequential, >0 means parallel
        assert config.num_workers >= -1


class TestWorldModelAgentConfig:
    """Tests for WorldModelAgentConfig."""

    def test_default_config(self):
        """Test default WorldModelAgent configuration."""
        config = WorldModelAgentConfig()

        # Check all sub-configs exist
        assert hasattr(config, "fsq_vae")
        assert hasattr(config, "world_model")
        assert hasattr(config, "controller")
        assert hasattr(config, "data")

        # Check they are correct types
        assert isinstance(config.fsq_vae, FSQVAEConfig)
        assert isinstance(config.world_model, WorldModelConfig)
        assert isinstance(config.controller, ControllerConfig)
        assert isinstance(config.data, DataConfig)

    def test_validation(self):
        """Test configuration validation."""
        config = WorldModelAgentConfig()

        # Should not raise any errors
        config.validate_consistency()

    def test_dimension_consistency(self):
        """Test that dimensions are consistent across configs."""
        config = WorldModelAgentConfig()
        config.validate_consistency()

        # Controller state_dim should match number of FSQ dimensions
        assert config.controller.state_dim == len(config.fsq_vae.fsq_levels)

        # Controller action_dim should match World Model action_dim
        assert config.controller.action_dim == config.world_model.action_dim

        # World model state_dim should match controller state_dim
        assert config.world_model.state_dim == config.controller.state_dim

    def test_fsq_codebook_size(self):
        """Test that FSQ codebook size is reasonable."""
        config = WorldModelAgentConfig()

        from functools import reduce
        from operator import mul

        codebook_size = reduce(mul, config.fsq_vae.fsq_levels, 1)

        # Should have a reasonable codebook size
        assert codebook_size > 0
        assert codebook_size < 1e10  # Not too large


class TestConfigModification:
    """Tests for modifying configurations."""

    def test_modify_fsq_vae_config(self):
        """Test modifying FSQ-VAE configuration."""
        config = FSQVAEConfig()

        # Modify a value
        original_latent_dim = config.latent_dim
        config.latent_dim = 256

        assert config.latent_dim == 256
        assert config.latent_dim != original_latent_dim

    def test_modify_world_model_config(self):
        """Test modifying World Model configuration."""
        config = WorldModelConfig()

        # Modify a value
        original_num_layers = config.num_layers
        config.num_layers = 8

        assert config.num_layers == 8
        assert config.num_layers != original_num_layers

    def test_modify_controller_config(self):
        """Test modifying Controller configuration."""
        config = ControllerConfig()

        # Modify a value
        original_hidden_sizes = config.hidden_sizes.copy()
        config.hidden_sizes = [256, 256, 128]

        assert config.hidden_sizes == [256, 256, 128]
        assert config.hidden_sizes != original_hidden_sizes

    def test_modify_optimizer_config(self):
        """Test modifying Optimizer configuration."""
        config = OptimizerConfig()

        # Modify learning rate
        config.learning_rate = 0.0001

        assert config.learning_rate == 0.0001

    def test_modify_and_validate(self):
        """Test that modified config can still validate."""
        config = WorldModelAgentConfig()

        # Modify consistently - change FSQ dimensions to 8
        new_fsq_levels = [8, 8, 8, 8, 8, 8, 8, 8]
        config.fsq_vae.fsq_levels = new_fsq_levels
        config.world_model.fsq_levels = new_fsq_levels
        config.world_model.state_dim = len(new_fsq_levels)
        config.controller.state_dim = len(new_fsq_levels)

        # Should still validate
        config.validate_consistency()

    def test_inconsistent_modification_fails(self):
        """Test that inconsistent modifications fail validation."""
        config = WorldModelAgentConfig()

        # Modify inconsistently - controller state_dim doesn't match len(fsq_levels)
        config.controller.state_dim = 10  # Doesn't match len(fsq_levels) = 4

        # Should fail validation
        with pytest.raises(ValueError):
            config.validate_consistency()


class TestConfigSerialization:
    """Tests for configuration serialization (if implemented)."""

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = WorldModelAgentConfig()

        # Try to access attributes
        assert hasattr(config.fsq_vae, "latent_dim")
        assert hasattr(config.world_model, "state_dim")
        assert hasattr(config.controller, "action_dim")

    def test_config_repr(self):
        """Test that config has string representation."""
        config = WorldModelAgentConfig()

        # Should be able to convert to string
        config_str = str(config)
        assert len(config_str) > 0
        assert isinstance(config_str, str)
