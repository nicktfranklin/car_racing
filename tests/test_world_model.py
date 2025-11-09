"""
Tests for the vanilla GPT-2 world model.

Tests include:
- Forward pass with mocked inputs
- Backward pass and gradient flow
- Token discretization/undiscretization
- Sequence creation
- Loss computation
"""

import pytest
import torch
import torch.nn as nn

from world_models.config import WorldModelAgentConfig, WorldModelConfig
from world_models.models.world_model import WorldModel


@pytest.fixture
def world_model_config():
    """Create a minimal world model configuration for testing."""
    config = WorldModelConfig(
        hidden_size=64,
        dropout=0.1,
        n_layers=2,
        n_heads=4,
        state_dim=4,
        action_dim=3,
        fsq_levels=[8, 8, 8, 4],
        learning_rate=0.001,
        sequence_length=8,
    )
    return config


@pytest.fixture
def world_model(world_model_config):
    """Create a world model instance."""
    model = WorldModel(world_model_config)
    model.eval()  # Set to eval mode for deterministic behavior
    return model


@pytest.fixture
def mock_batch():
    """Create mock batch data."""
    batch_size = 4
    seq_len = 8
    fsq_dim = 4
    action_dim = 3

    # FSQ tokens: values in [0, 7] for dims 0-2, [0, 3] for dim 3
    current_state_tokens = torch.randint(0, 8, (batch_size, seq_len, fsq_dim))
    current_state_tokens[:, :, 3] = torch.randint(0, 4, (batch_size, seq_len))  # Dim 3 only has 4 levels

    next_state_tokens = torch.randint(0, 8, (batch_size, seq_len, fsq_dim))
    next_state_tokens[:, :, 3] = torch.randint(0, 4, (batch_size, seq_len))

    # Continuous actions: steering in [-1, 1], gas/brake in [0, 1]
    actions = torch.randn(batch_size, seq_len, action_dim)
    actions[:, :, 0] = torch.tanh(actions[:, :, 0])  # Steering in [-1, 1]
    actions[:, :, 1:] = torch.sigmoid(actions[:, :, 1:])  # Gas/brake in [0, 1]

    # Rewards and dones
    rewards = torch.randn(batch_size, seq_len)
    dones = torch.randint(0, 2, (batch_size, seq_len)).float()

    return {
        "current_state_tokens": current_state_tokens,
        "next_state_tokens": next_state_tokens,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


class TestWorldModel:
    """Test suite for WorldModel."""

    def test_initialization(self, world_model, world_model_config):
        """Test that model initializes correctly."""
        assert isinstance(world_model, nn.Module)
        assert world_model.config == world_model_config
        assert world_model.VOCAB_SIZE == 32  # No SOS token
        assert world_model.fsq_dim == 4
        assert world_model.action_dim == 3
        assert world_model.tokens_per_timestep == 7

    def test_action_discretization(self, world_model, mock_batch):
        """Test action discretization and undiscretization."""
        actions = mock_batch["actions"]

        # Discretize
        action_tokens = world_model.discretize_actions(actions)

        # Check shapes
        assert action_tokens.shape == actions.shape
        assert action_tokens.dtype == torch.long

        # Check token ranges (no SOS, so ranges are 8-15, 16-23, 24-31)
        assert (action_tokens[:, :, 0] >= world_model.ACTION_STEERING_START).all()
        assert (action_tokens[:, :, 0] <= world_model.ACTION_STEERING_START + 7).all()
        assert (action_tokens[:, :, 1] >= world_model.ACTION_GAS_START).all()
        assert (action_tokens[:, :, 1] <= world_model.ACTION_GAS_START + 7).all()
        assert (action_tokens[:, :, 2] >= world_model.ACTION_BRAKE_START).all()
        assert (action_tokens[:, :, 2] <= world_model.ACTION_BRAKE_START + 7).all()

        # Undiscretize
        actions_reconstructed = world_model.undiscretize_actions(action_tokens)

        # Check shapes
        assert actions_reconstructed.shape == actions.shape
        assert actions_reconstructed.dtype == torch.float32

        # Check ranges
        assert (actions_reconstructed[:, :, 0] >= -1.0).all()
        assert (actions_reconstructed[:, :, 0] <= 1.0).all()
        assert (actions_reconstructed[:, :, 1] >= 0.0).all()
        assert (actions_reconstructed[:, :, 1] <= 1.0).all()
        assert (actions_reconstructed[:, :, 2] >= 0.0).all()
        assert (actions_reconstructed[:, :, 2] <= 1.0).all()

        # Check reconstruction error is bounded
        max_error = (actions - actions_reconstructed).abs().max()
        # Each bin is 1/7 wide, so max error should be around 1/(2*7) ≈ 0.07
        assert max_error < 0.15, f"Reconstruction error too large: {max_error}"

    def test_sequence_creation(self, world_model, mock_batch):
        """Test token sequence creation."""
        state_tokens = mock_batch["current_state_tokens"]
        actions = mock_batch["actions"]
        batch_size, seq_len, fsq_dim = state_tokens.shape

        # Create sequence
        token_ids = world_model.create_token_sequence(state_tokens, actions)

        # Check shape: seq_len * (4 FSQ + 3 actions) = seq_len * 7 (no SOS)
        expected_length = seq_len * world_model.tokens_per_timestep
        assert token_ids.shape == (batch_size, expected_length)
        assert token_ids.dtype == torch.long

        # Check token IDs are in valid range
        assert (token_ids >= 0).all()
        assert (token_ids < world_model.VOCAB_SIZE).all()

        # Check FSQ tokens are in range [0, 7]
        for t in range(seq_len):
            base_idx = t * world_model.tokens_per_timestep
            fsq_tokens = token_ids[:, base_idx:base_idx + fsq_dim]
            assert (fsq_tokens >= world_model.FSQ_TOKEN_START).all()
            assert (fsq_tokens <= world_model.FSQ_TOKEN_START + 7).all()

    def test_forward_pass(self, world_model, mock_batch):
        """Test forward pass produces correct output shapes."""
        state_tokens = mock_batch["current_state_tokens"]
        actions = mock_batch["actions"]
        batch_size, seq_len = state_tokens.shape[:2]

        # Forward pass
        logits, rewards, dones, past_kv = world_model(state_tokens, actions)

        # Check shapes (no SOS token)
        expected_seq_len = seq_len * world_model.tokens_per_timestep
        assert logits.shape == (batch_size, expected_seq_len, world_model.VOCAB_SIZE)
        assert rewards.shape == (batch_size, seq_len, 1)
        assert dones.shape == (batch_size, seq_len, 1)

        # Check outputs are finite
        assert torch.isfinite(logits).all()
        assert torch.isfinite(rewards).all()
        assert torch.isfinite(dones).all()

    def test_loss_computation(self, world_model, mock_batch):
        """Test loss computation."""
        world_model.train()  # Set to train mode

        current_state_tokens = mock_batch["current_state_tokens"]
        next_state_tokens = mock_batch["next_state_tokens"]
        actions = mock_batch["actions"]
        rewards = mock_batch["rewards"]
        dones = mock_batch["dones"]

        # Compute loss
        loss, loss_dict = world_model.compute_loss(
            current_state_tokens, next_state_tokens, actions, rewards, dones
        )

        # Check loss is scalar
        assert loss.shape == ()
        assert loss.dtype == torch.float32

        # Check loss is positive and finite
        assert loss.item() > 0
        assert torch.isfinite(loss).all()

        # Check loss dict
        assert "total_loss" in loss_dict
        assert "token_loss" in loss_dict
        assert "reward_loss" in loss_dict
        assert "done_loss" in loss_dict
        assert "token_accuracy" in loss_dict

        # Check all losses are positive
        assert loss_dict["total_loss"] > 0
        assert loss_dict["token_loss"] > 0
        assert loss_dict["reward_loss"] >= 0
        assert loss_dict["done_loss"] >= 0

        # Check accuracy is in [0, 1]
        assert 0 <= loss_dict["token_accuracy"] <= 1

    def test_backward_pass(self, world_model, mock_batch):
        """Test backward pass and gradient flow."""
        world_model.train()

        current_state_tokens = mock_batch["current_state_tokens"]
        next_state_tokens = mock_batch["next_state_tokens"]
        actions = mock_batch["actions"]
        rewards = mock_batch["rewards"]
        dones = mock_batch["dones"]

        # Zero gradients
        world_model.zero_grad()

        # Forward + backward
        loss, _ = world_model.compute_loss(
            current_state_tokens, next_state_tokens, actions, rewards, dones
        )
        loss.backward()

        # Check gradients exist and are finite for all parameters
        has_grad = 0
        total_params = 0
        grad_norms = []

        for name, param in world_model.named_parameters():
            total_params += 1
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
                has_grad += 1
                grad_norms.append(param.grad.norm().item())

        # Check that we have gradients for all parameters
        assert has_grad == total_params, f"Only {has_grad}/{total_params} parameters have gradients"

        # Check that gradients are not all zero
        total_grad_norm = sum(grad_norms)
        assert total_grad_norm > 0, "All gradients are zero"

        print(f"\nGradient check passed:")
        print(f"  - {has_grad} parameters with gradients")
        print(f"  - Total gradient norm: {total_grad_norm:.6f}")
        print(f"  - Mean gradient norm: {total_grad_norm/has_grad:.6f}")
        print(f"  - Max gradient norm: {max(grad_norms):.6f}")
        print(f"  - Min gradient norm: {min(grad_norms):.6f}")

    def test_gradient_flow_to_all_components(self, world_model, mock_batch):
        """Test that gradients flow to all model components."""
        world_model.train()

        current_state_tokens = mock_batch["current_state_tokens"]
        next_state_tokens = mock_batch["next_state_tokens"]
        actions = mock_batch["actions"]
        rewards = mock_batch["rewards"]
        dones = mock_batch["dones"]

        # Zero gradients
        world_model.zero_grad()

        # Forward + backward
        loss, _ = world_model.compute_loss(
            current_state_tokens, next_state_tokens, actions, rewards, dones
        )
        loss.backward()

        # Check specific components have gradients
        components = {
            "transformer": world_model.transformer,
            "reward_head": world_model.reward_head,
            "done_head": world_model.done_head,
        }

        for component_name, component in components.items():
            component_has_grad = False
            for name, param in component.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if param.grad.abs().sum() > 0:
                        component_has_grad = True
                        break
            assert component_has_grad, f"No gradients flowing to {component_name}"

    def test_sampling(self, world_model, mock_batch):
        """Test autoregressive sampling."""
        world_model.eval()

        # Take a single timestep from batch
        current_state_tokens = mock_batch["current_state_tokens"][:, :1, :]  # (batch, 1, fsq_dim)
        action = mock_batch["actions"][:, :1, :]  # (batch, 1, 3)

        # Sample next state
        next_state, reward, done, past_kv = world_model.sample_next_state(
            current_state_tokens, action, temperature=1.0
        )

        batch_size = current_state_tokens.shape[0]

        # Check shapes
        assert next_state.shape == (batch_size, 1, world_model.fsq_dim)
        assert reward.shape == (batch_size, 1, 1)
        assert done.shape == (batch_size, 1, 1)

        # Check sampled tokens are in valid range
        assert (next_state >= 0).all()
        assert (next_state[:, :, :3] < 8).all()  # First 3 dims: 0-7
        assert (next_state[:, :, 3] < 4).all()   # Last dim: 0-3

        # Check outputs are finite
        assert torch.isfinite(next_state.float()).all()
        assert torch.isfinite(reward).all()
        assert torch.isfinite(done).all()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
