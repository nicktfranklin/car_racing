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
    EvolutionaryController,
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


@pytest.fixture
def evo_controller(config):
    """Create EvolutionaryController."""
    return EvolutionaryController(config.controller)


class TestFSQVAE:
    """Tests for FSQ-VAE implementation."""

    def test_forward_pass(self, vae):
        """Test FSQ-VAE forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 64, 64)

        with torch.no_grad():
            x_recon, z, z_q = vae(x)
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
            z_q_enc, indices = vae.encode(x)
            x_recon_dec = vae.decode(z_q_enc)

        # Check shapes
        assert indices.shape[0] == batch_size
        assert x_recon_dec.shape == x.shape

    def test_codebook_size(self, vae):
        """Test that codebook size is set correctly."""
        assert vae.quantizer.codebook_size > 0


class TestWorldModel:
    """Tests for World Model implementation."""

    def test_forward_pass(self, world_model, config):
        """Test World Model forward pass."""
        batch_size = 4
        seq_len = 10

        state_indices = torch.randint(
            0, world_model.num_state_tokens, (batch_size, seq_len)
        )
        actions = torch.randn(batch_size, seq_len, config.world_model.action_dim)

        with torch.no_grad():
            next_state_logits, pred_rewards, pred_dones, hidden = world_model(
                state_indices, actions
            )

        # Check shapes
        assert next_state_logits.shape == (
            batch_size,
            seq_len,
            world_model.num_state_tokens,
        )
        assert pred_rewards.shape == (batch_size, seq_len, 1)
        assert pred_dones.shape == (batch_size, seq_len, 1)

    def test_loss_computation(self, world_model, config):
        """Test World Model loss computation."""
        batch_size = 4
        seq_len = 10

        state_indices = torch.randint(
            0, world_model.num_state_tokens, (batch_size, seq_len)
        )
        actions = torch.randn(batch_size, seq_len, config.world_model.action_dim)
        next_state_indices = torch.randint(
            0, world_model.num_state_tokens, (batch_size, seq_len)
        )
        rewards = torch.randn(batch_size, seq_len)
        dones = torch.randint(0, 2, (batch_size, seq_len))

        with torch.no_grad():
            loss, loss_dict = world_model.compute_loss(
                state_indices, actions, next_state_indices, rewards, dones
            )

        # Check loss is computed
        assert loss.item() >= 0
        assert "total_loss" in loss_dict
        assert "state_loss" in loss_dict
        assert "state_accuracy" in loss_dict

    def test_sampling(self, world_model, config):
        """Test World Model sampling."""
        state_indices = torch.randint(0, world_model.num_state_tokens, (1, 1))
        actions = torch.randn(1, 1, config.world_model.action_dim)

        with torch.no_grad():
            next_state_sample, reward_sample, done_sample, _ = (
                world_model.sample_next_state(state_indices, actions, temperature=1.0)
            )

        # Check shapes
        assert next_state_sample.shape == (1, 1, world_model.num_state_tokens)
        assert reward_sample.shape == (1, 1)
        assert done_sample.shape == (1, 1)


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

    def test_evolutionary_controller(self, evo_controller, config):
        """Test EvolutionaryController."""
        batch_size = 4
        state = torch.randn(batch_size, config.controller.state_dim)

        with torch.no_grad():
            actions = evo_controller(state)

        # Check shape
        assert actions.shape == (batch_size, config.controller.action_dim)

    def test_controller_mutation(self, evo_controller):
        """Test EvolutionaryController parameter mutation."""
        original_params = evo_controller.get_parameters_flat().clone()

        # Mutate with small noise
        evo_controller.mutate(0.1)
        new_params = evo_controller.get_parameters_flat()

        # Parameters should have changed
        param_change = torch.norm(new_params - original_params)
        assert param_change > 0

    def test_controller_parameters(self, evo_controller):
        """Test EvolutionaryController parameter extraction."""
        flat_params = evo_controller.get_parameters_flat()

        # Should have parameters
        assert flat_params.numel() > 0


class TestComponentIntegration:
    """Tests for integration between all components."""

    def test_full_pipeline(self, vae, world_model, evo_controller, config):
        """Test full pipeline: observation -> action -> next observation."""
        batch_size = 2
        obs = torch.randn(batch_size, 3, 64, 64)

        with torch.no_grad():
            # Encode observation
            z_q, state_indices = vae.encode(obs)

            # Get action from controller
            actions = evo_controller(z_q)
            actions = actions.unsqueeze(1)  # Add time dimension

            # Predict next state with world model
            next_state_logits, rewards, dones, _ = world_model(
                state_indices.unsqueeze(1), actions
            )

            # Sample next state
            next_state_probs = torch.softmax(next_state_logits, dim=-1)
            next_state_indices = torch.multinomial(next_state_probs.squeeze(1), 1)

            # Convert back to FSQ representation
            from world_models.models.world_model import indices_to_fsq

            next_z_q = indices_to_fsq(
                next_state_indices.squeeze(-1), config.fsq_vae.fsq_levels
            )

            # Decode next state
            next_obs = vae.decode(next_z_q)

        # Check all shapes match expectations
        assert z_q.shape[0] == batch_size
        assert state_indices.shape[0] == batch_size
        assert actions.shape[0] == batch_size
        assert rewards.shape[0] == batch_size
        assert next_obs.shape == obs.shape

    def test_config_validation(self, config):
        """Test that configuration validation works."""
        # Should not raise any errors
        config.validate_consistency()

        # Check that dimensions are set correctly
        assert config.controller.state_dim == config.fsq_vae.latent_dim
        assert config.controller.action_dim == config.world_model.action_dim
