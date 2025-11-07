"""
GPT-2 based transformer world model using HuggingFace transformers.

This is the default world model - a causal transformer that autoregressively
predicts next states, rewards, and dones given sequences of (state, action) pairs.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from ..config import WorldModelConfig


class WorldModel(nn.Module):
    """GPT-2 based world model that predicts next state tokens autoregressively."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # Calculate total number of state tokens (FSQ codebook size)
        self.num_state_tokens = int(np.prod(config.fsq_levels))
        self.fsq_dim = len(config.fsq_levels)

        # GPT-2 configuration (scaled for world modeling)
        # Using smaller dimensions than GPT-2 since we have:
        # - Small discrete state space (~1000-2000 codes)
        # - Short sequences (~50-100 timesteps)
        # - Simple dynamics (CarRacing)
        gpt_config = GPT2Config(
            vocab_size=1,  # Not used, we have custom embeddings
            n_positions=config.sequence_length * 2,  # States + actions interleaved
            n_embd=config.hidden_size,  # Match config (default 256)
            n_layer=getattr(config, 'n_layers', 6),  # 6 transformer layers
            n_head=getattr(config, 'n_heads', 8),  # 8 attention heads
            n_inner=config.hidden_size * 4,  # FFN dimension (standard 4x)
            activation_function='gelu_new',
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,  # Enable KV caching for fast sampling
        )

        # Core GPT-2 transformer
        self.transformer = GPT2Model(gpt_config)

        # Token embeddings (separate for states and actions)
        # We interleave state and action tokens: [s_0, a_0, s_1, a_1, ...]
        self.state_embedding = nn.Embedding(
            num_embeddings=self.num_state_tokens,
            embedding_dim=config.hidden_size
        )

        # Action embedding (continuous → discrete embedding space)
        self.action_embedding = nn.Linear(config.action_dim, config.hidden_size)

        # Token type embeddings (distinguish state vs action tokens)
        self.token_type_embedding = nn.Embedding(2, config.hidden_size)

        # Output heads
        self.state_head = nn.Linear(config.hidden_size, self.num_state_tokens)
        self.reward_head = nn.Linear(config.hidden_size, 1)
        self.done_head = nn.Linear(config.hidden_size, 1)

        # Layer norm for output heads
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)

    def create_interleaved_sequence(
        self,
        state_indices: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create interleaved sequence of state and action embeddings.

        Args:
            state_indices: (batch, seq_len) discrete state indices
            actions: (batch, seq_len, action_dim) continuous actions

        Returns:
            embeddings: (batch, seq_len*2, hidden_size) interleaved embeddings
            token_types: (batch, seq_len*2) token type IDs (0=state, 1=action)
        """
        batch_size, seq_len = state_indices.shape
        device = state_indices.device

        # Embed states and actions
        state_emb = self.state_embedding(state_indices)  # (batch, seq_len, hidden)
        action_emb = self.action_embedding(actions)  # (batch, seq_len, hidden)

        # Interleave: [s_0, a_0, s_1, a_1, s_2, a_2, ...]
        embeddings = torch.zeros(
            batch_size, seq_len * 2, self.config.hidden_size,
            device=device, dtype=state_emb.dtype
        )
        embeddings[:, 0::2] = state_emb  # Even positions: states
        embeddings[:, 1::2] = action_emb  # Odd positions: actions

        # Token type IDs
        token_types = torch.zeros(batch_size, seq_len * 2, device=device, dtype=torch.long)
        token_types[:, 0::2] = 0  # States
        token_types[:, 1::2] = 1  # Actions

        return embeddings, token_types

    def forward(
        self,
        state_indices: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple] = None,  # For compatibility with LSTM interface
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Forward pass through the transformer world model.

        Args:
            state_indices: (batch, seq_len) state token indices
            actions: (batch, seq_len, action_dim) continuous actions
            hidden: Optional past_key_values for fast generation (renamed for compatibility)

        Returns:
            next_state_logits: (batch, seq_len, num_tokens) logits for next states
            rewards: (batch, seq_len, 1) predicted rewards
            dones: (batch, seq_len, 1) predicted done flags
            hidden: Cached key-values (if use_cache=True)
        """
        batch_size, seq_len = state_indices.shape

        # Create interleaved sequence
        embeddings, token_types = self.create_interleaved_sequence(
            state_indices, actions
        )

        # Add token type embeddings
        token_type_emb = self.token_type_embedding(token_types)
        inputs_embeds = embeddings + token_type_emb

        # Forward through GPT-2 transformer
        # Note: GPT-2 handles position embeddings internally
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=hidden,
            use_cache=True if hidden is not None or self.training == False else False,
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq_len*2, hidden)
        past_key_values = outputs.past_key_values

        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Extract features after each (state, action) pair to predict next state
        # We want features at positions [1, 3, 5, ...] (after each action)
        # These will predict states at positions [2, 4, 6, ...] (next states)
        prediction_positions = hidden_states[:, 1::2]  # (batch, seq_len, hidden)

        # Predict next state logits
        next_state_logits = self.state_head(prediction_positions)

        # Predict rewards and dones
        rewards = self.reward_head(prediction_positions)
        dones = self.done_head(prediction_positions)

        return next_state_logits, rewards, dones, past_key_values

    def sample_next_state(
        self,
        state_indices: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Sample next state from the model distribution.

        Args:
            state_indices: Current state indices (batch, 1)
            actions: Actions to take (batch, 1, action_dim)
            hidden: Cached key-values for fast generation
            temperature: Sampling temperature

        Returns:
            next_state_indices: Sampled next state indices (batch, 1)
            rewards: Predicted rewards (batch, 1, 1)
            dones: Predicted done flags (batch, 1, 1)
            hidden: Updated cached key-values
        """
        with torch.no_grad():
            next_state_logits, rewards, dones, hidden = self.forward(
                state_indices, actions, hidden=hidden
            )

            # Sample from categorical distribution
            if temperature > 0:
                probs = F.softmax(next_state_logits / temperature, dim=-1)
                next_state_indices = torch.multinomial(probs.squeeze(1), 1)
            else:
                # Greedy sampling
                next_state_indices = torch.argmax(next_state_logits, dim=-1, keepdim=True)

            return next_state_indices, rewards, dones, hidden

    def compute_loss(
        self,
        state_indices: torch.Tensor,
        actions: torch.Tensor,
        next_state_indices: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
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
        next_state_logits, pred_rewards, pred_dones, _ = self.forward(
            state_indices, actions, hidden=None
        )

        # State prediction loss (cross-entropy)
        state_loss = F.cross_entropy(
            next_state_logits.reshape(-1, self.num_state_tokens),
            next_state_indices.reshape(-1),
            reduction='mean',
        )

        # Reward prediction loss (MSE)
        reward_loss = F.mse_loss(
            pred_rewards.squeeze(-1), rewards, reduction='mean'
        )

        # Done prediction loss (binary cross-entropy)
        done_loss = F.binary_cross_entropy_with_logits(
            pred_dones.squeeze(-1), dones.float(), reduction='mean'
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

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Optional[Tuple]:
        """Initialize hidden state (returns None for transformer)."""
        return None

    def detach_hidden(
        self, hidden: Optional[Tuple]
    ) -> Optional[Tuple]:
        """Detach hidden state (no-op for transformer with KV cache)."""
        return None


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
    # Test the Transformer World Model implementation
    from ..config import WorldModelConfig

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
    print("Testing forward pass...")
    with torch.no_grad():
        next_state_logits, pred_rewards, pred_dones, _ = model(
            state_indices, actions
        )
        loss, loss_dict = model.compute_loss(
            state_indices, actions, next_state_indices, rewards, dones
        )

    print(f"Input state indices shape: {state_indices.shape}")
    print(f"Input actions shape: {actions.shape}")
    print(f"Next state logits shape: {next_state_logits.shape}")
    print(f"Predicted rewards shape: {pred_rewards.shape}")
    print(f"Predicted dones shape: {pred_dones.shape}")
    print(f"Number of state tokens: {model.num_state_tokens}")
    print(f"Loss: {loss.item():.4f}")
    print(f"State accuracy: {loss_dict['state_accuracy']:.4f}")

    # Test sampling with KV caching
    print("\nTesting sampling with KV caching...")
    past_kv = None
    current_state = state_indices[:, 0:1]  # First state
    for t in range(5):
        action = actions[:, t:t+1]
        next_state, reward, done, past_kv = model.sample_next_state(
            current_state, action, hidden=past_kv, temperature=1.0
        )
        print(f"Step {t}: state={next_state[0].item()}, reward={reward[0].item():.3f}")
        current_state = next_state

    print("\nTransformer World Model test passed!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
