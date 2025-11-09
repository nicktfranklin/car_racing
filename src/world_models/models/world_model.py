"""
Vanilla GPT-2 world model for CarRacing.

This is a pure autoregressive transformer that predicts token sequences.
Vocabulary (33 tokens):
  - Token 0: SOS (Start of Sequence)
  - Tokens 1-8: FSQ values (0-7)
  - Tokens 9-16: Discretized steering (8 bins)
  - Tokens 17-24: Discretized gas (8 bins)
  - Tokens 25-32: Discretized brake (8 bins)

Sequence structure:
  [SOS, s_0^0, s_0^1, s_0^2, s_0^3, a_0_steer, a_0_gas, a_0_brake, s_1^0, ...]
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from world_models.config import WorldModelConfig


class WorldModel(nn.Module):
    """Vanilla GPT-2 world model with 33-token vocabulary."""

    # Token vocabulary constants
    SOS_TOKEN = 0
    FSQ_TOKEN_START = 1  # Tokens 1-8 for FSQ values 0-7
    ACTION_STEERING_START = 9   # Tokens 9-16 for steering bins
    ACTION_GAS_START = 17       # Tokens 17-24 for gas bins
    ACTION_BRAKE_START = 25     # Tokens 25-32 for brake bins
    VOCAB_SIZE = 33
    NUM_ACTION_BINS = 8

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.fsq_dim = len(config.fsq_levels)
        self.action_dim = config.action_dim

        # Tokens per timestep: 4 FSQ + 3 actions = 7
        self.tokens_per_timestep = self.fsq_dim + self.action_dim

        # GPT-2 configuration
        gpt_config = GPT2Config(
            vocab_size=self.VOCAB_SIZE,
            n_positions=1024,  # Max sequence length
            n_embd=config.hidden_size,
            n_layer=getattr(config, "n_layers", 6),
            n_head=getattr(config, "n_heads", 8),
            n_inner=config.hidden_size * 4,
            activation_function="gelu_new",
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
        )

        # GPT-2 model with language modeling head
        self.transformer = GPT2LMHeadModel(gpt_config)

        # Reward and done prediction heads (predict from transformer hidden states)
        self.reward_head = nn.Linear(config.hidden_size, 1)
        self.done_head = nn.Linear(config.hidden_size, 1)

    def discretize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Discretize continuous actions into token IDs.

        Args:
            actions: (batch, seq_len, 3) continuous actions in [-1, 1] or [0, 1]

        Returns:
            action_tokens: (batch, seq_len, 3) discrete token IDs
        """
        batch_size, seq_len = actions.shape[:2]
        action_tokens = torch.zeros(batch_size, seq_len, 3, dtype=torch.long, device=actions.device)

        # Steering: [-1, 1] -> bins 0-7 -> tokens 9-16
        steering = actions[:, :, 0]
        steering_bins = torch.clamp(
            ((steering + 1.0) / 2.0 * self.NUM_ACTION_BINS).long(),
            0,
            self.NUM_ACTION_BINS - 1
        )
        action_tokens[:, :, 0] = self.ACTION_STEERING_START + steering_bins

        # Gas: [0, 1] -> bins 0-7 -> tokens 17-24
        gas = actions[:, :, 1]
        gas_bins = torch.clamp(
            (gas * self.NUM_ACTION_BINS).long(),
            0,
            self.NUM_ACTION_BINS - 1
        )
        action_tokens[:, :, 1] = self.ACTION_GAS_START + gas_bins

        # Brake: [0, 1] -> bins 0-7 -> tokens 25-32
        brake = actions[:, :, 2]
        brake_bins = torch.clamp(
            (brake * self.NUM_ACTION_BINS).long(),
            0,
            self.NUM_ACTION_BINS - 1
        )
        action_tokens[:, :, 2] = self.ACTION_BRAKE_START + brake_bins

        return action_tokens

    def undiscretize_actions(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert action token IDs back to continuous values using bin centers.

        Args:
            action_tokens: (batch, seq_len, 3) discrete token IDs

        Returns:
            actions: (batch, seq_len, 3) continuous actions
        """
        batch_size, seq_len = action_tokens.shape[:2]
        actions = torch.zeros(batch_size, seq_len, 3, dtype=torch.float32, device=action_tokens.device)

        # Steering: tokens 9-16 -> bins 0-7 -> [-1, 1]
        # Use bin centers: (bin + 0.5) / NUM_BINS
        steering_bins = action_tokens[:, :, 0] - self.ACTION_STEERING_START
        actions[:, :, 0] = ((steering_bins.float() + 0.5) / self.NUM_ACTION_BINS) * 2.0 - 1.0

        # Gas: tokens 17-24 -> bins 0-7 -> [0, 1]
        gas_bins = action_tokens[:, :, 1] - self.ACTION_GAS_START
        actions[:, :, 1] = (gas_bins.float() + 0.5) / self.NUM_ACTION_BINS

        # Brake: tokens 25-32 -> bins 0-7 -> [0, 1]
        brake_bins = action_tokens[:, :, 2] - self.ACTION_BRAKE_START
        actions[:, :, 2] = (brake_bins.float() + 0.5) / self.NUM_ACTION_BINS

        return actions

    def create_token_sequence(
        self,
        state_tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create interleaved token sequence with SOS prepended.

        Sequence structure:
          [SOS, s_0^0, s_0^1, s_0^2, s_0^3, a_0_steer, a_0_gas, a_0_brake, s_1^0, ...]

        Args:
            state_tokens: (batch, seq_len, fsq_dim) FSQ tokens (values 0-7)
            actions: (batch, seq_len, 3) continuous actions

        Returns:
            token_ids: (batch, 1 + seq_len * 7) token ID sequence
        """
        batch_size, seq_len = state_tokens.shape[:2]

        # Discretize actions
        action_tokens = self.discretize_actions(actions)  # (batch, seq_len, 3)

        # Convert FSQ tokens (0-7) to vocabulary IDs (1-8)
        state_token_ids = state_tokens + self.FSQ_TOKEN_START  # (batch, seq_len, 4)

        # Total sequence length: SOS + seq_len * (4 FSQ + 3 actions)
        total_length = 1 + seq_len * self.tokens_per_timestep

        # Pre-allocate token sequence
        token_ids = torch.zeros(batch_size, total_length, dtype=torch.long, device=state_tokens.device)

        # Add SOS token
        token_ids[:, 0] = self.SOS_TOKEN

        # Interleave states and actions
        for t in range(seq_len):
            base_idx = 1 + t * self.tokens_per_timestep

            # Add FSQ tokens
            token_ids[:, base_idx:base_idx + self.fsq_dim] = state_token_ids[:, t, :]

            # Add action tokens
            token_ids[:, base_idx + self.fsq_dim:base_idx + self.tokens_per_timestep] = action_tokens[:, t, :]

        return token_ids

    def forward(
        self,
        state_tokens: torch.Tensor,
        actions: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Forward pass through GPT-2.

        Args:
            state_tokens: (batch, seq_len, fsq_dim) FSQ tokens (0-7)
            actions: (batch, seq_len, 3) continuous actions
            past_key_values: Cached key-values for fast generation

        Returns:
            logits: (batch, total_seq_len, VOCAB_SIZE) next token logits
            rewards: (batch, seq_len, 1) predicted rewards
            dones: (batch, seq_len, 1) predicted done flags
            past_key_values: Updated cache
        """
        # Create token sequence
        token_ids = self.create_token_sequence(state_tokens, actions)  # (batch, 1 + seq_len * 7)

        # Forward through GPT-2
        outputs = self.transformer(
            input_ids=token_ids,
            past_key_values=past_key_values,
            use_cache=True if past_key_values is not None or not self.training else False,
        )

        logits = outputs.logits  # (batch, seq_len, VOCAB_SIZE)
        past_key_values = outputs.past_key_values

        # Extract hidden states for reward/done prediction
        # We predict reward/done after each complete timestep (after the brake action)
        hidden_states = self.transformer.transformer(input_ids=token_ids)[0]  # (batch, seq_len, hidden)

        # Positions after each brake action: 1 + t*7 + 6 for t in range(seq_len)
        seq_len = state_tokens.shape[1]
        reward_positions = [1 + t * self.tokens_per_timestep + self.tokens_per_timestep - 1
                           for t in range(seq_len)]
        reward_features = hidden_states[:, reward_positions, :]  # (batch, seq_len, hidden)

        rewards = self.reward_head(reward_features)  # (batch, seq_len, 1)
        dones = self.done_head(reward_features)      # (batch, seq_len, 1)

        return logits, rewards, dones, past_key_values

    def compute_loss(
        self,
        current_state_tokens: torch.Tensor,
        next_state_tokens: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute training loss.

        Args:
            current_state_tokens: (batch, seq_len, fsq_dim) current states
            next_state_tokens: (batch, seq_len, fsq_dim) next states (targets)
            actions: (batch, seq_len, 3) actions
            rewards: (batch, seq_len) rewards
            dones: (batch, seq_len) done flags

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of losses and metrics
        """
        batch_size, seq_len = current_state_tokens.shape[:2]

        # Create input sequence (without targets)
        input_ids = self.create_token_sequence(current_state_tokens, actions)  # (batch, 1 + seq_len * 7)

        # Create target sequence (shift left by 1 for next-token prediction)
        # Target: predict next_state_tokens and actions
        target_ids = self.create_token_sequence(next_state_tokens, actions)  # (batch, 1 + seq_len * 7)

        # Shift targets: predict position i+1 from position i
        # Input:  [SOS, s_0^0, s_0^1, ...]
        # Target: [s_0^0, s_0^1, s_0^2, ...]
        targets = target_ids[:, 1:].contiguous()  # Remove SOS from targets
        inputs = input_ids[:, :-1].contiguous()   # Remove last token from inputs

        # Forward pass
        outputs = self.transformer(input_ids=inputs)
        logits = outputs.logits  # (batch, seq_len-1, VOCAB_SIZE)

        # Token prediction loss (cross-entropy)
        token_loss = F.cross_entropy(
            logits.reshape(-1, self.VOCAB_SIZE),
            targets.reshape(-1),
            reduction="mean",
        )

        # Reward and done prediction
        # Extract hidden states at positions after each complete timestep
        # After removing first token (SOS), positions are: t*7 + 6 for t in range(seq_len)
        hidden_states = self.transformer.transformer(input_ids=inputs)[0]
        reward_positions = [t * self.tokens_per_timestep + self.tokens_per_timestep - 1
                           for t in range(seq_len)]
        reward_features = hidden_states[:, reward_positions, :]

        pred_rewards = self.reward_head(reward_features).squeeze(-1)
        pred_dones = self.done_head(reward_features).squeeze(-1)

        reward_loss = F.mse_loss(pred_rewards, rewards)
        done_loss = F.binary_cross_entropy_with_logits(pred_dones, dones.float())

        # Total loss
        total_loss = token_loss + reward_loss + done_loss

        # Compute accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == targets).float().mean()

        loss_dict = {
            "total_loss": total_loss.item(),
            "token_loss": token_loss.item(),
            "reward_loss": reward_loss.item(),
            "done_loss": done_loss.item(),
            "token_accuracy": accuracy.item(),
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def sample_next_state(
        self,
        current_state_tokens: torch.Tensor,
        action: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Sample next state autoregressively.

        Args:
            current_state_tokens: (batch, 1, fsq_dim) current state
            action: (batch, 1, 3) action to take
            past_key_values: Cached KV for fast generation
            temperature: Sampling temperature

        Returns:
            next_state_tokens: (batch, 1, fsq_dim) sampled next state
            reward: (batch, 1, 1) predicted reward
            done: (batch, 1, 1) predicted done
            past_key_values: Updated cache
        """
        batch_size = current_state_tokens.shape[0]
        device = current_state_tokens.device

        # Create input sequence
        input_ids = self.create_token_sequence(current_state_tokens, action)  # (batch, 1 + 7)

        # Sample next state tokens autoregressively
        sampled_tokens = []
        for dim in range(self.fsq_dim):
            # Forward through transformer
            outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]  # (batch, VOCAB_SIZE)
            past_key_values = outputs.past_key_values

            # Filter to valid FSQ tokens for this dimension
            # Dimensions 0-2: 8 levels (0-7), Dimension 3: 4 levels (0-3)
            num_levels = self.config.fsq_levels[dim]
            fsq_logits = logits[:, self.FSQ_TOKEN_START:self.FSQ_TOKEN_START + num_levels]

            # Sample
            if temperature > 0:
                probs = F.softmax(fsq_logits / temperature, dim=-1)
                token = torch.multinomial(probs, 1)  # (batch, 1)
            else:
                token = torch.argmax(fsq_logits, dim=-1, keepdim=True)

            # Convert back to FSQ value (0 to num_levels-1)
            sampled_tokens.append(token)

            # Append to input for next dimension (add FSQ_TOKEN_START offset)
            input_ids = torch.cat([input_ids, token + self.FSQ_TOKEN_START], dim=1)

        # Stack sampled tokens: [(batch, 1), (batch, 1), ...] -> (batch, fsq_dim) -> (batch, 1, fsq_dim)
        next_state_tokens = torch.cat(sampled_tokens, dim=1).unsqueeze(1)  # (batch, 1, fsq_dim)

        # Get reward/done predictions
        hidden_states = self.transformer.transformer(input_ids=input_ids)[0]
        last_hidden = hidden_states[:, -1:, :]
        reward = self.reward_head(last_hidden)
        done = self.done_head(last_hidden)

        return next_state_tokens, reward, done, past_key_values
