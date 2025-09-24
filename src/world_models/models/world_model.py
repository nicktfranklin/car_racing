"""
LSTM-based world model with softmax over state tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

from ..config import WorldModelConfig


class WorldModel(nn.Module):
    """LSTM-based world model that predicts next state tokens."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        # Calculate total number of state tokens
        self.num_state_tokens = int(np.prod(config.fsq_levels))
        self.fsq_dim = len(config.fsq_levels)

        # Input embedding: combine current state indices and action
        self.state_embedding = nn.Embedding(
            num_embeddings=self.num_state_tokens,
            embedding_dim=config.hidden_size // 2
        )

        # Action projection
        self.action_projection = nn.Linear(
            config.action_dim,
            config.hidden_size // 2
        )

        # LSTM core
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )

        # Output heads for predicting next state tokens
        self.state_head = nn.Linear(config.hidden_size, self.num_state_tokens)

        # Reward prediction head
        self.reward_head = nn.Linear(config.hidden_size, 1)

        # Done prediction head
        self.done_head = nn.Linear(config.hidden_size, 1)

    def forward(self, state_indices: torch.Tensor, actions: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the world model.

        Args:
            state_indices: Tensor of shape (batch, seq_len) with state token indices
            actions: Tensor of shape (batch, seq_len, action_dim)
            hidden: Optional initial hidden state

        Returns:
            next_state_logits: Logits for next state tokens (batch, seq_len, num_tokens)
            rewards: Predicted rewards (batch, seq_len, 1)
            dones: Predicted done flags (batch, seq_len, 1)
            hidden: Final hidden state
        """
        batch_size, seq_len = state_indices.shape

        # Embed state indices
        state_emb = self.state_embedding(state_indices)  # (batch, seq_len, hidden_size//2)

        # Project actions
        action_emb = self.action_projection(actions)  # (batch, seq_len, hidden_size//2)

        # Combine state and action embeddings
        lstm_input = torch.cat([state_emb, action_emb], dim=-1)  # (batch, seq_len, hidden_size)

        # LSTM forward pass
        lstm_output, hidden = self.lstm(lstm_input, hidden)  # (batch, seq_len, hidden_size)

        # Predict next state tokens
        next_state_logits = self.state_head(lstm_output)  # (batch, seq_len, num_tokens)

        # Predict rewards and done flags
        rewards = self.reward_head(lstm_output)  # (batch, seq_len, 1)
        dones = self.done_head(lstm_output)  # (batch, seq_len, 1)

        return next_state_logits, rewards, dones, hidden

    def sample_next_state(self, state_indices: torch.Tensor, actions: torch.Tensor,
                         hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                         temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                           Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample next state from the model distribution.

        Args:
            state_indices: Current state indices (batch, 1)
            actions: Actions to take (batch, 1, action_dim)
            hidden: Current hidden state
            temperature: Sampling temperature

        Returns:
            next_state_indices: Sampled next state indices (batch, 1)
            rewards: Predicted rewards (batch, 1, 1)
            dones: Predicted done flags (batch, 1, 1)
            hidden: Updated hidden state
        """
        with torch.no_grad():
            next_state_logits, rewards, dones, hidden = self.forward(state_indices, actions, hidden)

            # Sample from categorical distribution
            if temperature > 0:
                probs = F.softmax(next_state_logits / temperature, dim=-1)
                next_state_indices = torch.multinomial(probs.squeeze(1), 1)
            else:
                # Greedy sampling
                next_state_indices = torch.argmax(next_state_logits, dim=-1)

            return next_state_indices, rewards, dones, hidden

    def compute_loss(self, state_indices: torch.Tensor, actions: torch.Tensor,
                    next_state_indices: torch.Tensor, rewards: torch.Tensor,
                    dones: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute world model training loss.

        Args:
            state_indices: Current state indices (batch, seq_len)
            actions: Actions taken (batch, seq_len, action_dim)
            next_state_indices: True next state indices (batch, seq_len)
            rewards: True rewards (batch, seq_len)
            dones: True done flags (batch, seq_len)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Forward pass
        next_state_logits, pred_rewards, pred_dones, _ = self.forward(state_indices, actions)

        # State prediction loss (cross-entropy)
        state_loss = F.cross_entropy(
            next_state_logits.reshape(-1, self.num_state_tokens),
            next_state_indices.reshape(-1),
            reduction='mean'
        )

        # Reward prediction loss (MSE)
        reward_loss = F.mse_loss(
            pred_rewards.squeeze(-1),
            rewards,
            reduction='mean'
        )

        # Done prediction loss (binary cross-entropy)
        done_loss = F.binary_cross_entropy_with_logits(
            pred_dones.squeeze(-1),
            dones.float(),
            reduction='mean'
        )

        # Combined loss
        total_loss = state_loss + reward_loss + done_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'state_loss': state_loss.item(),
            'reward_loss': reward_loss.item(),
            'done_loss': done_loss.item(),
        }

        # Calculate accuracy for state prediction
        with torch.no_grad():
            state_preds = torch.argmax(next_state_logits, dim=-1)
            state_accuracy = (state_preds == next_state_indices).float().mean()
            loss_dict['state_accuracy'] = state_accuracy.item()

        return total_loss, loss_dict

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0

    def detach_hidden(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detach hidden state from computation graph."""
        h, c = hidden
        return h.detach(), c.detach()


def indices_to_fsq(indices: torch.Tensor, levels: list) -> torch.Tensor:
    """Convert flat indices back to FSQ representation."""
    batch_size = indices.shape[0]
    device = indices.device
    fsq_dim = len(levels)

    # Initialize output
    fsq_repr = torch.zeros(batch_size, fsq_dim, device=device)

    # Convert from flat index to mixed radix representation
    remaining = indices.clone()
    for i in reversed(range(fsq_dim)):
        level = levels[i]
        if level > 1:
            fsq_repr[:, i] = remaining % level
            remaining = remaining // level

    # Convert to [-1, 1] range
    for i, level in enumerate(levels):
        if level > 1:
            fsq_repr[:, i] = (fsq_repr[:, i] * 2 / (level - 1)) - 1

    return fsq_repr


def fsq_to_indices(fsq_repr: torch.Tensor, levels: list) -> torch.Tensor:
    """Convert FSQ representation to flat indices."""
    batch_size = fsq_repr.shape[0]
    device = fsq_repr.device

    indices = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i, level in enumerate(levels):
        if level > 1:
            # Convert from [-1, 1] to [0, level-1]
            level_indices = ((fsq_repr[:, i] + 1) * (level - 1) / 2).round().long()
            level_indices = torch.clamp(level_indices, 0, level - 1)

            # Accumulate index (mixed radix)
            if i == 0:
                indices = level_indices
            else:
                indices = indices * level + level_indices

    return indices


if __name__ == "__main__":
    # Test the World Model implementation
    config = WorldModelConfig()
    model = WorldModel(config)

    batch_size = 4
    seq_len = 10

    # Generate random test data
    state_indices = torch.randint(0, model.num_state_tokens, (batch_size, seq_len))
    actions = torch.randn(batch_size, seq_len, config.action_dim)
    next_state_indices = torch.randint(0, model.num_state_tokens, (batch_size, seq_len))
    rewards = torch.randn(batch_size, seq_len)
    dones = torch.randint(0, 2, (batch_size, seq_len))

    # Test forward pass
    with torch.no_grad():
        next_state_logits, pred_rewards, pred_dones, hidden = model(state_indices, actions)
        loss, loss_dict = model.compute_loss(state_indices, actions, next_state_indices, rewards, dones)

    print(f"Input state indices shape: {state_indices.shape}")
    print(f"Input actions shape: {actions.shape}")
    print(f"Next state logits shape: {next_state_logits.shape}")
    print(f"Predicted rewards shape: {pred_rewards.shape}")
    print(f"Predicted dones shape: {pred_dones.shape}")
    print(f"Number of state tokens: {model.num_state_tokens}")
    print(f"Loss: {loss.item():.4f}")
    print(f"State accuracy: {loss_dict['state_accuracy']:.4f}")
    print("World Model test passed!")