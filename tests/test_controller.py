"""
Tests for Controller implementations.

Tests the Controller class.
"""

import pytest
import torch

from world_models import Controller
from world_models.config import ControllerConfig


@pytest.fixture
def config():
    """Create test configuration."""
    return ControllerConfig()


@pytest.fixture
def controller(config):
    """Create standard Controller."""
    return Controller(config)


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
