"""
Controller training using PPO.

Note: VAE and World Model training have been migrated to PyTorch Lightning.
See lightning_training.py and training/train_vae.py, training/train_world_model.py
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from ...config import WorldModelAgentConfig
from ...models.controller import EvolutionaryController
from ...models.fsq_vae import FSQVAE
from ...models.world_model import WorldModel
from ...utils import get_logger, print_model_info

logger = get_logger("world_models")


class ControllerTrainer:
    """Trainer for controller using evolutionary strategy."""

    def __init__(
        self, vae: FSQVAE, world_model: WorldModel, config: WorldModelAgentConfig
    ):
        self.vae = vae
        self.world_model = world_model
        self.config = config
        # Use configured device, validate it's available
        device_str = config.training.device
        if device_str == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is not available on this system"
            )
        elif device_str == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS device requested but MPS is not available on this system"
            )
        self.device = torch.device(device_str)
        print(f"ControllerTrainer using device: {self.device}")

        # Move models to device and set to eval mode
        self.vae.to(self.device).eval()

        # Create controller
        self.controller = EvolutionaryController(config.controller).to(self.device)

        # PPO hyperparameters
        self.lr = 3e-4
        self.gamma = 0.99  # Discount factor
        self.clip_epsilon = 0.2  # PPO clip parameter
        self.num_epochs = 4  # Number of epochs per update
        self.batch_size = 64  # Minibatch size

        # Optimizer
        self.optimizer = optim.Adam(self.controller.parameters(), lr=self.lr)

        # Print controller architecture and parameters
        print_model_info(self.controller, "Controller (PPO)")
        print(f"Learning rate: {self.lr}")
        print(f"Gamma (discount): {self.gamma}")
        print(f"Clip epsilon: {self.clip_epsilon}")
        print(f"PPO epochs per update: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"State dimension: {config.controller.state_dim}")
        print(f"Action dimension: {config.controller.action_dim}")
        print(f"Controller log_std shape: {self.controller.log_std.shape}\n")

    def train(
        self, num_updates: int, rollout_length: int = 128, num_rollouts: int = 4
    ) -> Dict[str, List[float]]:
        """
        Train controller using PPO.

        Args:
            num_updates: Number of PPO update iterations
            rollout_length: Length of each rollout in the world model
            num_rollouts: Number of parallel rollouts per update

        Returns:
            history: Dictionary of training metrics
        """
        history = {"loss": [], "mean_reward": []}

        for update in range(num_updates):
            # Collect rollouts
            rollout_data = self._collect_rollouts(rollout_length, num_rollouts)

            # Compute returns
            returns = self._compute_returns(rollout_data["rewards"])

            # Flatten batch dimensions
            states = rollout_data["states"].view(-1, self.config.controller.state_dim)
            actions_raw = rollout_data["actions_raw"].view(
                -1, self.config.controller.action_dim
            )
            old_log_probs = rollout_data["log_probs"].view(-1)
            returns_flat = returns.view(-1)

            # PPO update loop
            for epoch in range(self.num_epochs):
                # Shuffle data
                indices = torch.randperm(states.size(0))

                # Mini-batch updates
                for start_idx in range(0, states.size(0), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, states.size(0))
                    batch_indices = indices[start_idx:end_idx]

                    # Get batch
                    batch_states = states[batch_indices]
                    batch_actions_raw = actions_raw[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns_flat[batch_indices]

                    # Evaluate log probs with current policy
                    new_log_probs = self.controller.evaluate_log_prob(
                        batch_states, batch_actions_raw
                    )

                    # Compute ratio
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)

                    # Clipped PPO loss
                    surr1 = ratio * batch_returns
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                        * batch_returns
                    )
                    loss = -torch.min(surr1, surr2).mean()

                    # Update
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.controller.parameters(), 0.5)
                    self.optimizer.step()

            # Track metrics
            history["loss"].append(loss.item())
            history["mean_reward"].append(
                rollout_data["rewards"].sum(dim=1).mean().item()
            )

            print(
                f"Update {update+1}/{num_updates}: "
                f"Loss = {loss.item():.4f}, "
                f"Mean Reward = {history['mean_reward'][-1]:.2f}"
            )

        return history

    def _collect_rollouts(
        self, rollout_length: int, num_rollouts: int
    ) -> Dict[str, torch.Tensor]:
        """
        Collect rollouts in the world model.

        Args:
            rollout_length: Length of each rollout
            num_rollouts: Number of rollouts to collect

        Returns:
            Dictionary containing states, actions_raw, log_probs, rewards
        """
        self.controller.eval()

        # Storage
        states = torch.zeros(
            (num_rollouts, rollout_length, self.config.controller.state_dim),
            device=self.device,
        )
        actions_raw = torch.zeros(
            (num_rollouts, rollout_length, self.config.controller.action_dim),
            device=self.device,
        )
        log_probs = torch.zeros((num_rollouts, rollout_length), device=self.device)
        rewards = torch.zeros((num_rollouts, rollout_length), device=self.device)

        with torch.no_grad():
            for rollout_idx in range(num_rollouts):
                # Random initial observation
                obs = torch.randn(1, 3, 64, 64, device=self.device)

                # Encode initial state
                z_q, indices, state_tokens = self.vae.encode(obs)

                # Rollout
                for step in range(rollout_length):
                    # Store state
                    state = z_q.squeeze(0)
                    states[rollout_idx, step] = state

                    # Get action from policy (store raw action before constraints)
                    mean = self.controller.network(state.unsqueeze(0))
                    std = torch.exp(self.controller.log_std).expand_as(mean)
                    action_raw_sample = mean + std * torch.randn_like(mean)
                    actions_raw[rollout_idx, step] = action_raw_sample.squeeze(0)

                    # Compute log prob
                    var = std**2
                    log_prob = (
                        -((action_raw_sample - mean) ** 2) / (2 * var)
                        - self.controller.log_std
                        - 0.5 * torch.log(2 * torch.tensor(torch.pi))
                    )
                    log_probs[rollout_idx, step] = log_prob.sum()

                    # Apply constraints for world model
                    # Move to CPU to avoid MPS indexing bugs, then back to device
                    action_raw_cpu = action_raw_sample.cpu()
                    action_cpu = self.controller._apply_constraints(action_raw_cpu)
                    action = action_cpu.to(self.device)

                    # Predict next state using world model
                    # sample_next_state expects: current_state_tokens (batch, 1, fsq_dim), action (batch, 1, 3)
                    state_tokens_batch = state_tokens.unsqueeze(1)  # (1, 1, fsq_dim)
                    action_batch = action.unsqueeze(1)  # (1, 1, action_dim)

                    # Move to CPU for world model (which stays on CPU to avoid MPS bugs)
                    action_batch_cpu = action_batch.cpu()
                    state_tokens_batch_cpu = state_tokens_batch.cpu()

                    next_state_tokens, reward_pred, done_pred, _ = (
                        self.world_model.sample_next_state(
                            state_tokens_batch_cpu, action_batch_cpu, temperature=1.0
                        )
                    )

                    # Move results back to main device
                    next_state_tokens = next_state_tokens.to(self.device)

                    # Store reward
                    rewards[rollout_idx, step] = reward_pred.squeeze()

                    # Convert next state tokens back to quantized z_q
                    # Tokens are discrete [0, level-1], convert to continuous [-1, 1]
                    levels_tensor = torch.tensor(
                        self.config.fsq_vae.fsq_levels,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    # Vectorized conversion: (token * 2 / (level - 1)) - 1
                    z_q = (next_state_tokens.float() * 2.0 / (levels_tensor - 1)) - 1.0

                    # Update state tokens for next iteration
                    # Squeeze to remove sequence dimension since next_state_tokens has shape (batch, 1, fsq_dim)
                    state_tokens = next_state_tokens.squeeze(1)  # (batch, fsq_dim)

                    # Check for early termination
                    done_prob = torch.sigmoid(done_pred).squeeze()
                    if done_prob > 0.5:
                        # Zero out remaining trajectory
                        rewards[rollout_idx, step + 1 :] = 0
                        break

        self.controller.train()

        return {
            "states": states,
            "actions_raw": actions_raw,
            "log_probs": log_probs,
            "rewards": rewards,
        }

    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute discounted returns.

        Args:
            rewards: Rewards tensor (num_rollouts, rollout_length)

        Returns:
            returns: Discounted returns (num_rollouts, rollout_length)
        """
        num_rollouts, rollout_length = rewards.shape
        returns = torch.zeros_like(rewards)

        # Compute returns backward through time
        for t in reversed(range(rollout_length)):
            if t == rollout_length - 1:
                returns[:, t] = rewards[:, t]
            else:
                returns[:, t] = rewards[:, t] + self.gamma * returns[:, t + 1]

        return returns

    def save_checkpoint(self, filepath: str):
        """Save controller checkpoint."""
        checkpoint = {
            "controller_state_dict": self.controller.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.controller,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load controller checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.controller.load_state_dict(checkpoint["controller_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
