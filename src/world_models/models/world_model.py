"""
GPT-2 based transformer world model using HuggingFace transformers.

This is the default world model - a causal transformer that autoregressively
predicts next states, rewards, and dones given sequences of (state, action) pairs.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from world_models.config import WorldModelConfig


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
            n_layer=getattr(config, "n_layers", 6),  # 6 transformer layers
            n_head=getattr(config, "n_heads", 8),  # 8 attention heads
            n_inner=config.hidden_size * 4,  # FFN dimension (standard 4x)
            activation_function="gelu_new",
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,  # Enable KV caching for fast sampling
        )

        # Core GPT-2 transformer
        self.transformer = GPT2Model(gpt_config)

        # Token embeddings - one per FSQ dimension (autoregressive within state)
        # Sequence: [s_0^0, s_0^1, s_0^2, s_0^3, a_0, s_1^0, s_1^1, s_1^2, s_1^3, a_1, ...]
        self.fsq_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=level, embedding_dim=config.hidden_size)
                for level in config.fsq_levels
            ]
        )

        # Action embedding (continuous → discrete embedding space)
        self.action_embedding = nn.Linear(config.action_dim, config.hidden_size)

        # Token type embeddings (distinguish FSQ dim0, dim1, dim2, dim3, action)
        self.token_type_embedding = nn.Embedding(self.fsq_dim + 1, config.hidden_size)

        # Output heads - one per FSQ dimension
        self.fsq_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, level) for level in config.fsq_levels]
        )
        self.reward_head = nn.Linear(config.hidden_size, 1)
        self.done_head = nn.Linear(config.hidden_size, 1)

        # Layer norm for output heads
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)

    def create_interleaved_sequence(
        self,
        state_tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create autoregressive sequence with FSQ dimensions and actions interleaved.

        Sequence structure: [s_0^0, s_0^1, s_0^2, s_0^3, a_0, s_1^0, s_1^1, s_1^2, s_1^3, a_1, ...]

        Within each timestep, FSQ dimensions are predicted autoregressively:
        - s_t^0 depends on history + a_{t-1}
        - s_t^1 depends on history + a_{t-1} + s_t^0
        - s_t^2 depends on history + a_{t-1} + s_t^0 + s_t^1
        - s_t^3 depends on history + a_{t-1} + s_t^0 + s_t^1 + s_t^2

        Args:
            state_tokens: (batch, seq_len, fsq_dim) discrete tokens per FSQ dimension
            actions: (batch, seq_len, action_dim) continuous actions

        Returns:
            embeddings: (batch, seq_len*(fsq_dim+1), hidden_size) interleaved embeddings
            token_types: (batch, seq_len*(fsq_dim+1)) token type IDs (0-3=FSQ dims, 4=action)
        """
        batch_size, seq_len, fsq_dim = state_tokens.shape
        device = state_tokens.device
        tokens_per_timestep = fsq_dim + 1  # FSQ dims + action
        total_length = seq_len * tokens_per_timestep

        # Pre-allocate output tensors
        embeddings = torch.zeros(
            batch_size,
            total_length,
            self.config.hidden_size,
            device=device,
            dtype=torch.float32,
        )
        token_types = torch.zeros(
            batch_size, total_length, device=device, dtype=torch.long
        )

        # Embed actions once
        action_emb = self.action_embedding(actions)  # (batch, seq_len, hidden)

        # Interleave FSQ dimensions and actions
        for t in range(seq_len):
            base_idx = t * tokens_per_timestep

            # Embed each FSQ dimension with its corresponding embedding table
            for dim in range(fsq_dim):
                idx = base_idx + dim
                embeddings[:, idx] = self.fsq_embeddings[dim](state_tokens[:, t, dim])
                token_types[:, idx] = dim

            # Add action embedding
            idx = base_idx + fsq_dim
            embeddings[:, idx] = action_emb[:, t]
            token_types[:, idx] = fsq_dim  # Token type for action

        return embeddings, token_types

    def forward(
        self,
        state_tokens: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple] = None,  # For compatibility with LSTM interface
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Forward pass through the transformer world model with autoregressive FSQ prediction.

        Args:
            state_tokens: (batch, seq_len, fsq_dim) discrete tokens per FSQ dimension
            actions: (batch, seq_len, action_dim) continuous actions
            hidden: Optional past_key_values for fast generation (renamed for compatibility)

        Returns:
            next_fsq_logits: List of (batch, seq_len, level_i) logits for each FSQ dimension
            rewards: (batch, seq_len, 1) predicted rewards
            dones: (batch, seq_len, 1) predicted done flags
            hidden: Cached key-values (if use_cache=True)
        """
        batch_size, seq_len, fsq_dim = state_tokens.shape
        tokens_per_timestep = fsq_dim + 1  # FSQ dims + action

        # Create interleaved sequence
        embeddings, token_types = self.create_interleaved_sequence(
            state_tokens, actions
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

        hidden_states = (
            outputs.last_hidden_state
        )  # (batch, seq_len*(fsq_dim+1), hidden)
        past_key_values = outputs.past_key_values

        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Extract prediction positions for each FSQ dimension and rewards/dones
        # For each timestep t, we predict:
        #   s_t^0 from position (t*5 - 1) [after a_{t-1}]
        #   s_t^1 from position (t*5 + 0) [after s_t^0]
        #   s_t^2 from position (t*5 + 1) [after s_t^1]
        #   s_t^3 from position (t*5 + 2) [after s_t^2]
        #   rewards/dones from position (t*5 + 3) [after s_t^3]

        # Collect logits for each FSQ dimension
        next_fsq_logits = []
        for dim in range(fsq_dim):
            # Positions where this dimension should be predicted
            # dim=0: positions [4, 9, 14, ...] (after previous action)
            # dim=1: positions [5, 10, 15, ...] (after dim 0)
            # dim=2: positions [6, 11, 16, ...] (after dim 1)
            # dim=3: positions [7, 12, 17, ...] (after dim 2)
            if dim == 0:
                # First dimension predicted after action (position tokens_per_timestep-1, 2*tokens_per_timestep-1, ...)
                positions = torch.arange(
                    tokens_per_timestep - 1,
                    seq_len * tokens_per_timestep,
                    tokens_per_timestep,
                    device=hidden_states.device,
                )
            else:
                # Subsequent dimensions predicted after previous dimension
                positions = torch.arange(
                    tokens_per_timestep - 1 + dim,
                    seq_len * tokens_per_timestep,
                    tokens_per_timestep,
                    device=hidden_states.device,
                )

            # Extract features at these positions
            features = hidden_states[:, positions]  # (batch, seq_len, hidden)

            # Predict logits for this dimension
            logits = self.fsq_heads[dim](features)  # (batch, seq_len, level_i)
            next_fsq_logits.append(logits)

        # Predict rewards and dones after last FSQ dimension
        # Position after s_t^3: [8, 13, 18, ...]
        reward_positions = torch.arange(
            tokens_per_timestep - 1 + fsq_dim,
            seq_len * tokens_per_timestep,
            tokens_per_timestep,
            device=hidden_states.device,
        )
        reward_features = hidden_states[:, reward_positions]  # (batch, seq_len, hidden)

        rewards = self.reward_head(reward_features)
        dones = self.done_head(reward_features)

        return next_fsq_logits, rewards, dones, past_key_values

    def sample_next_state(
        self,
        current_state_tokens: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Sample next state from the model distribution autoregressively.

        Samples each FSQ dimension in order, conditioning on previously sampled dimensions.

        Args:
            current_state_tokens: Current state tokens (batch, 1, fsq_dim)
            actions: Actions to take (batch, 1, action_dim)
            hidden: Cached key-values for fast generation
            temperature: Sampling temperature

        Returns:
            next_state_tokens: Sampled next state tokens (batch, 1, fsq_dim)
            rewards: Predicted rewards (batch, 1, 1)
            dones: Predicted done flags (batch, 1, 1)
            hidden: Updated cached key-values
        """
        with torch.no_grad():
            # Get predictions for all FSQ dimensions
            next_fsq_logits, rewards, dones, hidden = self.forward(
                current_state_tokens, actions, hidden=hidden
            )

            # Sample each FSQ dimension
            sampled_tokens = []
            for dim, logits in enumerate(next_fsq_logits):
                # logits: (batch, 1, level_i)
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    sampled = torch.multinomial(probs.squeeze(1), 1)  # (batch, 1)
                else:
                    # Greedy sampling
                    sampled = torch.argmax(logits, dim=-1)  # (batch, 1)

                sampled_tokens.append(sampled)

            # Stack into (batch, 1, fsq_dim)
            next_state_tokens = torch.stack(
                sampled_tokens, dim=-1
            )  # (batch, 1, fsq_dim)

            return next_state_tokens, rewards, dones, hidden

    def compute_loss(
        self,
        current_state_tokens: torch.Tensor,
        actions: torch.Tensor,
        next_state_tokens: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute world model training loss with per-dimension FSQ prediction.

        Args:
            current_state_tokens: Current state tokens (batch, seq_len, fsq_dim)
            actions: Actions taken (batch, seq_len, action_dim)
            next_state_tokens: True next state tokens (batch, seq_len, fsq_dim)
            rewards: True rewards (batch, seq_len)
            dones: True done flags (batch, seq_len)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Forward pass
        next_fsq_logits, pred_rewards, pred_dones, _ = self.forward(
            current_state_tokens, actions, hidden=None
        )

        # FSQ dimension prediction losses (cross-entropy per dimension)
        fsq_losses = []
        fsq_accuracies = []
        for dim, (logits, level) in enumerate(
            zip(next_fsq_logits, self.config.fsq_levels)
        ):
            # logits: (batch, seq_len, level)
            # next_state_tokens[:, :, dim]: (batch, seq_len)
            loss = F.cross_entropy(
                logits.reshape(-1, level),
                next_state_tokens[:, :, dim].reshape(-1),
                reduction="mean",
            )
            fsq_losses.append(loss)

            # Calculate accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                accuracy = (preds == next_state_tokens[:, :, dim]).float().mean()
                fsq_accuracies.append(accuracy.item())

        # Total FSQ loss (average across dimensions)
        fsq_loss = torch.stack(fsq_losses).mean()

        # Reward prediction loss (MSE)
        reward_loss = F.mse_loss(pred_rewards.squeeze(-1), rewards, reduction="mean")

        # Done prediction loss (binary cross-entropy)
        done_loss = F.binary_cross_entropy_with_logits(
            pred_dones.squeeze(-1), dones.float(), reduction="mean"
        )

        # Combined loss
        total_loss = fsq_loss + reward_loss + done_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "fsq_loss": fsq_loss.item(),
            "reward_loss": reward_loss.item(),
            "done_loss": done_loss.item(),
        }

        # Add per-dimension losses and accuracies
        for dim in range(self.fsq_dim):
            loss_dict[f"fsq_dim{dim}_loss"] = fsq_losses[dim].item()
            loss_dict[f"fsq_dim{dim}_accuracy"] = fsq_accuracies[dim]

        # Overall FSQ accuracy (average across dimensions)
        loss_dict["fsq_accuracy"] = sum(fsq_accuracies) / len(fsq_accuracies)

        return total_loss, loss_dict

    def init_hidden(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        """Initialize hidden state (returns None for transformer)."""
        return None

    def detach_hidden(self, hidden: Optional[Tuple]) -> Optional[Tuple]:
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
        next_state_logits, pred_rewards, pred_dones, _ = model(state_indices, actions)
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
        action = actions[:, t : t + 1]
        next_state, reward, done, past_kv = model.sample_next_state(
            current_state, action, hidden=past_kv, temperature=1.0
        )
        print(f"Step {t}: state={next_state[0].item()}, reward={reward[0].item():.3f}")
        current_state = next_state

    print("\nTransformer World Model test passed!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
