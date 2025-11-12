"""
Tests for Controller implementations.

Tests the Controller and EvolutionaryController classes.
"""

import pytest
import torch

from world_models import Controller, EvolutionaryController
from world_models.config import ControllerConfig


@pytest.fixture
def config():
    """Create test configuration."""
    return ControllerConfig()


@pytest.fixture
def controller(config):
    """Create standard Controller."""
    return Controller(config)


@pytest.fixture
def evo_controller(config):
    """Create EvolutionaryController."""
    return EvolutionaryController(config)


class TestController:
    """Tests for standard Controller."""

    def test_forward_pass(self, controller, config):
        """Test Controller forward pass."""
        batch_size = 4
        state = torch.randn(batch_size, config.state_dim)

        with torch.no_grad():
            actions = controller(state)

        # Check output shape
        assert actions.shape == (batch_size, config.action_dim)

    def test_single_action(self, controller, config):
        """Test getting single action."""
        state = torch.randn(config.state_dim)

        with torch.no_grad():
            action = controller.get_action(state)

        # Check output shape
        assert action.shape == (config.action_dim,)

    def test_action_ranges_carracing(self, controller, config):
        """Test action ranges for CarRacing environment."""
        if config.action_dim != 3:
            pytest.skip("Test only applies to CarRacing (3 actions)")

        batch_size = 100
        state = torch.randn(batch_size, config.state_dim)

        with torch.no_grad():
            actions = controller(state)

        # Steering should be in [-1, 1]
        assert actions[:, 0].min() >= -1.0
        assert actions[:, 0].max() <= 1.0

        # Gas should be in [0, 1]
        assert actions[:, 1].min() >= 0.0
        assert actions[:, 1].max() <= 1.0

        # Brake should be in [0, 1]
        assert actions[:, 2].min() >= 0.0
        assert actions[:, 2].max() <= 1.0

    def test_deterministic_output(self, controller, config):
        """Test that controller produces deterministic output for same input."""
        state = torch.randn(config.state_dim)

        with torch.no_grad():
            action1 = controller.get_action(state)
            action2 = controller.get_action(state)

        # Same input should produce same output
        assert torch.allclose(action1, action2)


class TestEvolutionaryController:
    """Tests for EvolutionaryController."""

    def test_forward_pass(self, evo_controller, config):
        """Test EvolutionaryController forward pass."""
        batch_size = 4
        state = torch.randn(batch_size, config.state_dim)

        with torch.no_grad():
            actions = evo_controller(state)

        # Check output shape
        assert actions.shape == (batch_size, config.action_dim)

    def test_parameter_extraction(self, evo_controller):
        """Test extracting flat parameters."""
        flat_params = evo_controller.get_parameters_flat()

        # Should have parameters
        assert flat_params.numel() > 0
        assert len(flat_params.shape) == 1  # Should be 1D

    def test_parameter_setting(self, evo_controller):
        """Test setting parameters from flat vector."""
        # Get original parameters
        original_params = evo_controller.get_parameters_flat()

        # Create new random parameters
        new_params = torch.randn_like(original_params)

        # Set new parameters
        evo_controller.set_parameters_flat(new_params)

        # Verify they were set
        current_params = evo_controller.get_parameters_flat()
        assert torch.allclose(current_params, new_params)

    def test_mutation(self, evo_controller):
        """Test parameter mutation."""
        original_params = evo_controller.get_parameters_flat().clone()

        # Mutate with small noise
        noise_scale = 0.1
        evo_controller.mutate(noise_scale)

        # Get new parameters
        new_params = evo_controller.get_parameters_flat()

        # Parameters should have changed
        param_change = torch.norm(new_params - original_params)
        assert param_change > 0

        # Change should be roughly proportional to noise scale
        # (though this is stochastic so we just check it's non-zero)
        assert not torch.allclose(new_params, original_params)

    def test_mutation_scale(self, evo_controller):
        """Test that mutation scale affects parameter changes."""
        # Small mutation
        original_params = evo_controller.get_parameters_flat().clone()
        evo_controller.mutate(0.01)
        small_change = torch.norm(evo_controller.get_parameters_flat() - original_params)

        # Reset and do large mutation
        evo_controller.set_parameters_flat(original_params.clone())
        evo_controller.mutate(1.0)
        large_change = torch.norm(evo_controller.get_parameters_flat() - original_params)

        # Larger noise should generally produce larger changes
        # (statistical test - may occasionally fail)
        assert large_change > small_change * 0.5

    def test_clone(self, evo_controller):
        """Test cloning evolutionary controller."""
        # Get original parameters
        original_params = evo_controller.get_parameters_flat()

        # Clone
        cloned = evo_controller.clone()

        # Should have same parameters
        cloned_params = cloned.get_parameters_flat()
        assert torch.allclose(original_params, cloned_params)

        # Modifying clone shouldn't affect original
        cloned.mutate(0.5)
        assert not torch.allclose(
            evo_controller.get_parameters_flat(), cloned.get_parameters_flat()
        )

    def test_reproducible_output(self, evo_controller, config):
        """Test that same parameters produce same actions."""
        state = torch.randn(config.state_dim)

        # Get action
        with torch.no_grad():
            action1 = evo_controller.get_action(state)

        # Save and restore parameters
        params = evo_controller.get_parameters_flat()
        evo_controller.mutate(0.1)  # Change parameters
        evo_controller.set_parameters_flat(params)  # Restore

        # Get action again
        with torch.no_grad():
            action2 = evo_controller.get_action(state)

        # Should produce same output
        assert torch.allclose(action1, action2)


class TestControllerComparison:
    """Tests comparing Controller and EvolutionaryController."""

    def test_both_produce_valid_actions(self, controller, evo_controller, config):
        """Test that both controllers produce valid actions."""
        state = torch.randn(config.state_dim)

        with torch.no_grad():
            action_std = controller.get_action(state)
            action_evo = evo_controller.get_action(state)

        # Both should have correct shape
        assert action_std.shape == (config.action_dim,)
        assert action_evo.shape == (config.action_dim,)

    def test_parameter_counts(self, controller, evo_controller):
        """Test that evolutionary controller has parameters."""
        # Standard controller should have parameters
        std_params = sum(p.numel() for p in controller.parameters())
        assert std_params > 0

        # Evolutionary controller should have same architecture
        evo_params = evo_controller.get_parameters_flat().numel()
        assert evo_params == std_params
