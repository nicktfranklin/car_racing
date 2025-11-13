"""
Integration tests for World Model components.

Tests the integration between FSQ-VAE, World Model, and Controller components
to ensure they work correctly together.
"""

import pytest
import torch

from world_models import (
    FSQVAE,
    Controller,
    WorldModel,
    WorldModelAgentConfig,
)


@pytest.fixture
def config():
    """Create default configuration for tests."""
    config = WorldModelAgentConfig()
    config.validate_consistency()
    return config


@pytest.fixture
def vae(config):
    """Create FSQ-VAE model."""
    return FSQVAE(config.fsq_vae)


@pytest.fixture
def world_model(config):
    """Create World Model."""
    return WorldModel(config.world_model)


@pytest.fixture
def controller(config):
    """Create standard Controller."""
    return Controller(config.controller)


class TestFSQVAE:
    """Tests for FSQ-VAE implementation."""

    def test_forward_pass(self, vae):
        """Test FSQ-VAE forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 64, 64)

        with torch.no_grad():
            x_recon, z, z_q, indices, tokens = vae(x)
            loss, loss_dict = vae.compute_loss(x, x_recon, z, z_q)

        # Check shapes
        assert x_recon.shape == x.shape
        assert z.shape[0] == batch_size
        assert z_q.shape[0] == batch_size

        # Check loss is computed
        assert loss.item() >= 0
        assert "total_loss" in loss_dict
        assert "recon_loss" in loss_dict

    def test_encoding_decoding(self, vae):
        """Test FSQ-VAE encoding and decoding."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 64, 64)

        with torch.no_grad():
            z_q_enc, indices, tokens = vae.encode(x)
            x_recon_dec = vae.decode(z_q_enc)

        # Check shapes
        assert indices.shape[0] == batch_size
        assert x_recon_dec.shape == x.shape

    def test_codebook_size(self, vae):
        """Test that codebook size is set correctly."""
        assert vae.quantizer.codebook_size > 0


class TestControllers:
    """Tests for Controller implementations."""

    def test_standard_controller(self, controller, config):
        """Test standard Controller forward pass."""
        batch_size = 4
        state = torch.randn(batch_size, config.controller.state_dim)

        with torch.no_grad():
            actions = controller(state)
            single_action = controller.get_action(
                torch.randn(config.controller.state_dim)
            )

        # Check shapes
        assert actions.shape == (batch_size, config.controller.action_dim)
        assert single_action.shape == (config.controller.action_dim,)

        # Check action ranges for CarRacing (steering, gas, brake)
        if config.controller.action_dim == 3:
            # Steering should be in [-1, 1]
            assert actions[:, 0].min() >= -1.0
            assert actions[:, 0].max() <= 1.0
            # Gas should be in [0, 1]
            assert actions[:, 1].min() >= 0.0
            assert actions[:, 1].max() <= 1.0
            # Brake should be in [0, 1]
            assert actions[:, 2].min() >= 0.0
            assert actions[:, 2].max() <= 1.0


class TestComponentIntegration:
    """Tests for integration between all components."""

    def test_config_validation(self, config):
        """Test that configuration validation works."""
        # Should not raise any errors
        config.validate_consistency()

        # Check that dimensions are set correctly
        assert config.controller.state_dim == len(config.fsq_vae.fsq_levels)
        assert config.controller.action_dim == config.world_model.action_dim
