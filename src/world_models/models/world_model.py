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

from ..config import WorldModelConfig


class WorldModel(nn.Module):
    """Vanilla GPT-2 world model with 32-token vocabulary (no SOS)."""

    # Token vocabulary constants (no SOS token)
    FSQ_TOKEN_START = 0  # Tokens 0-7 for FSQ values 0-7
    ACTION_STEERING_START = 8  # Tokens 8-15 for steering bins
    ACTION_GAS_START = 16  # Tokens 16-23 for gas bins
    ACTION_BRAKE_START = 24  # Tokens 24-31 for brake bins
    VOCAB_SIZE = 32
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
        action_tokens = torch.zeros(
            batch_size, seq_len, 3, dtype=torch.long, device=actions.device
        )

        # Steering: [-1, 1] -> bins 0-7 -> tokens 8-15
        steering = actions[:, :, 0]
        steering_bins = torch.clamp(
            ((steering + 1.0) / 2.0 * self.NUM_ACTION_BINS).long(),
            0,
            self.NUM_ACTION_BINS - 1,
        )
        action_tokens[:, :, 0] = self.ACTION_STEERING_START + steering_bins

        # Gas: [0, 1] -> bins 0-7 -> tokens 16-23
        gas = actions[:, :, 1]
        gas_bins = torch.clamp(
            (gas * self.NUM_ACTION_BINS).long(), 0, self.NUM_ACTION_BINS - 1
        )
        action_tokens[:, :, 1] = self.ACTION_GAS_START + gas_bins

        # Brake: [0, 1] -> bins 0-7 -> tokens 24-31
        brake = actions[:, :, 2]
        brake_bins = torch.clamp(
            (brake * self.NUM_ACTION_BINS).long(), 0, self.NUM_ACTION_BINS - 1
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
        actions = torch.zeros(
            batch_size, seq_len, 3, dtype=torch.float32, device=action_tokens.device
        )

        # Steering: tokens 8-15 -> bins 0-7 -> [-1, 1]
        # Use bin centers: (bin + 0.5) / NUM_BINS
        steering_bins = action_tokens[:, :, 0] - self.ACTION_STEERING_START
        actions[:, :, 0] = (
            (steering_bins.float() + 0.5) / self.NUM_ACTION_BINS
        ) * 2.0 - 1.0

        # Gas: tokens 16-23 -> bins 0-7 -> [0, 1]
        gas_bins = action_tokens[:, :, 1] - self.ACTION_GAS_START
        actions[:, :, 1] = (gas_bins.float() + 0.5) / self.NUM_ACTION_BINS

        # Brake: tokens 24-31 -> bins 0-7 -> [0, 1]
        brake_bins = action_tokens[:, :, 2] - self.ACTION_BRAKE_START
        actions[:, :, 2] = (brake_bins.float() + 0.5) / self.NUM_ACTION_BINS

        return actions

    def create_token_sequence(
        self,
        state_tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create interleaved token sequence (no SOS token).

        Sequence structure:
          [s_0^0, s_0^1, s_0^2, s_0^3, a_0_steer, a_0_gas, a_0_brake, s_1^0, ...]

        Args:
            state_tokens: (batch, seq_len, fsq_dim) FSQ tokens (values 0-7)
            actions: (batch, seq_len, 3) continuous actions

        Returns:
            token_ids: (batch, seq_len * 7) token ID sequence
        """
        batch_size, seq_len = state_tokens.shape[:2]

        # Discretize actions
        action_tokens = self.discretize_actions(actions)  # (batch, seq_len, 3)

        # FSQ tokens are already in range 0-7, map to vocabulary IDs 0-7
        state_token_ids = state_tokens + self.FSQ_TOKEN_START  # (batch, seq_len, 4)

        # Total sequence length: seq_len * (4 FSQ + 3 actions)
        total_length = seq_len * self.tokens_per_timestep

        # Pre-allocate token sequence
        token_ids = torch.zeros(
            batch_size, total_length, dtype=torch.long, device=state_tokens.device
        )

        # Interleave states and actions
        for t in range(seq_len):
            base_idx = t * self.tokens_per_timestep

            # Add FSQ tokens
            token_ids[:, base_idx : base_idx + self.fsq_dim] = state_token_ids[:, t, :]

            # Add action tokens
            token_ids[
                :, base_idx + self.fsq_dim : base_idx + self.tokens_per_timestep
            ] = action_tokens[:, t, :]

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
        # Create token sequence (no SOS token)
        token_ids = self.create_token_sequence(
            state_tokens, actions
        )  # (batch, seq_len * 7)

        # Forward through GPT-2
        outputs = self.transformer(
            input_ids=token_ids,
            past_key_values=past_key_values,
            use_cache=(
                True if past_key_values is not None or not self.training else False
            ),
        )

        logits = outputs.logits  # (batch, seq_len, VOCAB_SIZE)
        past_key_values = outputs.past_key_values

        # Extract hidden states for reward/done prediction
        # We predict reward/done after each complete timestep (after the brake action)
        hidden_states = self.transformer.transformer(input_ids=token_ids)[
            0
        ]  # (batch, seq_len, hidden)

        # Positions after each brake action: t*7 + 6 for t in range(seq_len)
        seq_len = state_tokens.shape[1]
        reward_positions = [
            t * self.tokens_per_timestep + self.tokens_per_timestep - 1
            for t in range(seq_len)
        ]
        reward_features = hidden_states[
            :, reward_positions, :
        ]  # (batch, seq_len, hidden)

        rewards = self.reward_head(reward_features)  # (batch, seq_len, 1)
        dones = self.done_head(reward_features)  # (batch, seq_len, 1)

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
        Compute fully autoregressive training loss with masking.

        The sequence is: [s_0, a_0, s_1, a_1, s_2, a_2, ...]
        We predict each token from all previous tokens (standard GPT-2).

        Loss masking:
        - Mask first state (positions 0-3): no prior context
        - Mask all actions (positions 4-6, 11-13, ...): actions are inputs, not predictions
        - Compute loss on all other state tokens (positions 7-10, 14-17, ...)

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

        # Create full sequence: [s_0, a_0, s_1, a_1, ...]
        # We need s_0 through s_seq_len, so concatenate current and next states
        all_states = torch.cat(
            [current_state_tokens, next_state_tokens[:, -1:]], dim=1
        )  # (batch, seq_len+1, fsq_dim)
        token_ids = self.create_token_sequence(
            all_states[:, :-1], actions
        )  # (batch, seq_len * 7)

        # Standard autoregressive targets: predict token i+1 from tokens 0...i
        inputs = token_ids[:, :-1].contiguous()  # (batch, seq_len * 7 - 1)
        targets = token_ids[:, 1:].contiguous()  # (batch, seq_len * 7 - 1)

        # Create loss mask: 1 = compute loss, 0 = ignore
        mask = torch.ones_like(targets, dtype=torch.float32)

        # Mask first state (positions 0-3 in targets, which are positions 1-4 in original sequence)
        mask[:, : self.fsq_dim] = 0.0

        # Mask all action positions
        for t in range(seq_len):
            # Action positions in targets: t*7 + 3, t*7 + 4, t*7 + 5
            # (corresponding to positions t*7 + 4, t*7 + 5, t*7 + 6 in original sequence)
            action_start = t * self.tokens_per_timestep + self.fsq_dim
            if action_start < targets.shape[1]:
                mask[:, action_start : min(action_start + 3, targets.shape[1])] = 0.0

        # Forward pass
        outputs = self.transformer(input_ids=inputs)
        logits = outputs.logits  # (batch, seq_len-1, VOCAB_SIZE)

        # Token prediction loss with masking
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, self.VOCAB_SIZE),
            targets.reshape(-1),
            reduction="none",
        )
        loss_per_token = loss_per_token.reshape(batch_size, -1)

        # Apply mask and compute mean over non-masked tokens
        masked_loss = loss_per_token * mask
        token_loss = masked_loss.sum() / (mask.sum() + 1e-8)

        # Reward and done prediction (from positions after each complete (state, action) pair)
        hidden_states = self.transformer.transformer(input_ids=inputs)[0]
        # Reward positions: after each action sequence (t*7 + 5 for t in range(seq_len))
        reward_positions = [
            t * self.tokens_per_timestep + self.fsq_dim + 2
            for t in range(seq_len)
            if t * self.tokens_per_timestep + self.fsq_dim + 2 < hidden_states.shape[1]
        ]

        if len(reward_positions) > 0:
            reward_features = hidden_states[:, reward_positions, :]
            pred_rewards = self.reward_head(reward_features).squeeze(-1)
            pred_dones = self.done_head(reward_features).squeeze(-1)

            # Truncate targets if needed
            num_reward_preds = len(reward_positions)
            reward_loss = F.mse_loss(pred_rewards, rewards[:, :num_reward_preds])
            done_loss = F.binary_cross_entropy_with_logits(
                pred_dones, dones[:, :num_reward_preds].float()
            )
        else:
            reward_loss = torch.tensor(0.0, device=token_loss.device)
            done_loss = torch.tensor(0.0, device=token_loss.device)

        # Total loss
        total_loss = token_loss + reward_loss + done_loss

        # Compute accuracy (only on non-masked tokens)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == targets).float() * mask
            accuracy = correct.sum() / (mask.sum() + 1e-8)

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

        # Create input sequence (no SOS token)
        input_ids = self.create_token_sequence(
            current_state_tokens, action
        )  # (batch, 7)

        # Sample next state tokens autoregressively
        sampled_tokens = []
        for dim in range(self.fsq_dim):
            # Forward through transformer
            outputs = self.transformer(
                input_ids=input_ids, past_key_values=past_key_values
            )
            logits = outputs.logits[:, -1, :]  # (batch, VOCAB_SIZE)
            past_key_values = outputs.past_key_values

            # Filter to valid FSQ tokens for this dimension
            # Dimensions 0-2: 8 levels (0-7), Dimension 3: 4 levels (0-3)
            num_levels = self.config.fsq_levels[dim]
            fsq_logits = logits[
                :, self.FSQ_TOKEN_START : self.FSQ_TOKEN_START + num_levels
            ]

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
        next_state_tokens = torch.cat(sampled_tokens, dim=1).unsqueeze(
            1
        )  # (batch, 1, fsq_dim)

        # Get reward/done predictions
        hidden_states = self.transformer.transformer(input_ids=input_ids)[0]
        last_hidden = hidden_states[:, -1:, :]
        reward = self.reward_head(last_hidden)
        done = self.done_head(last_hidden)

        return next_state_tokens, reward, done, past_key_values
