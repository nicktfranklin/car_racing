"""
Controller network for the World Model agent.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ControllerConfig


class Controller(nn.Module):
    """Simple feedforward controller that maps state representations to actions."""

    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config

        # Build network layers
        layers = []
        input_size = config.state_dim

        for hidden_size in config.hidden_sizes:
            layers.extend(
                [
                    nn.Linear(input_size, hidden_size),
                    self._get_activation(config.activation),
                ]
            )
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, config.action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
            "elu": nn.ELU(inplace=True),
        }
        return activations.get(activation, nn.Tanh())

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the controller.

        Args:
            state: State representation tensor (batch, state_dim)

        Returns:
            actions: Action tensor (batch, action_dim)
        """
        actions = self.network(state)

        # For CarRacing environment, apply appropriate action constraints
        # Action 0: steering (-1 to 1)
        # Action 1: gas (0 to 1)
        # Action 2: brake (0 to 1)

        if self.config.action_dim == 3:
            actions = torch.stack(
                [
                    torch.tanh(actions[:, 0]),  # Steering: -1 to 1
                    torch.sigmoid(actions[:, 1]),  # Gas: 0 to 1
                    torch.sigmoid(actions[:, 2]),  # Brake: 0 to 1
                ],
                dim=1,
            )
        else:
            # Generic case - apply tanh to all actions
            actions = torch.tanh(actions)

        return actions

    def get_action(
        self, state: torch.Tensor, deterministic: bool = True, noise_scale: float = 0.1
    ) -> torch.Tensor:
        """
        Get action from the controller with optional exploration noise.

        Args:
            state: State representation (batch, state_dim) or (state_dim,)
            deterministic: If True, return deterministic action
            noise_scale: Scale of exploration noise

        Returns:
            action: Action tensor
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        with torch.no_grad():
            action = self.forward(state)

            if not deterministic:
                # Add exploration noise
                noise = torch.randn_like(action) * noise_scale
                action = action + noise

                # Re-apply constraints after adding noise
                if self.config.action_dim == 3:
                    action = torch.stack(
                        [
                            torch.clamp(action[:, 0], -1, 1),  # Steering
                            torch.clamp(action[:, 1], 0, 1),  # Gas
                            torch.clamp(action[:, 2], 0, 1),  # Brake
                        ],
                        dim=1,
                    )
                else:
                    action = torch.clamp(action, -1, 1)

        if squeeze_output:
            action = action.squeeze(0)

        return action


class EvolutionaryController(nn.Module):
    """
    Evolutionary controller that can be optimized using evolutionary strategies.
    This is closer to the original World Models paper approach.
    Can also be trained with PPO by using action distribution methods.
    """

    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config

        # Use a simpler linear network for evolutionary optimization
        self.network = nn.Linear(config.state_dim, config.action_dim)

        # Log std for Gaussian policy (used for PPO training)
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))

        # Initialize with small random weights
        nn.init.normal_(self.network.weight, 0, 0.1)
        nn.init.constant_(self.network.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the evolutionary controller."""
        actions = self.network(state)

        # Apply action constraints for CarRacing
        if self.config.action_dim == 3:
            actions = torch.stack(
                [
                    torch.tanh(actions[:, 0]),  # Steering: -1 to 1
                    torch.sigmoid(actions[:, 1]),  # Gas: 0 to 1
                    torch.sigmoid(actions[:, 2]),  # Brake: 0 to 1
                ],
                dim=1,
            )
        else:
            actions = torch.tanh(actions)

        return actions

    def get_parameters_flat(self) -> torch.Tensor:
        """Get all parameters as a flat tensor for evolutionary optimization."""
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        return torch.cat(params)

    def set_parameters_flat(self, flat_params: torch.Tensor):
        """Set parameters from a flat tensor."""
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param.data.copy_(flat_params[idx : idx + param_size].view(param.shape))
            idx += param_size

    def mutate(self, mutation_strength: float = 0.02):
        """Apply Gaussian mutation to parameters."""
        with torch.no_grad():
            for param in self.parameters():
                param += torch.randn_like(param) * mutation_strength

    def get_action_with_logprob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from Gaussian policy and return log probability.
        Used for PPO training.

        Args:
            state: State tensor (batch, state_dim) or (state_dim,)

        Returns:
            action: Sampled action with constraints applied (batch, action_dim)
            log_prob: Log probability of the action (batch,)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Get mean from network
        mean = self.network(state)
        std = torch.exp(self.log_std).expand_as(mean)

        # Sample from Gaussian
        action_raw = mean + std * torch.randn_like(mean)

        # Compute log probability
        var = std ** 2
        log_prob = -((action_raw - mean) ** 2) / (2 * var) - self.log_std - 0.5 * torch.log(2 * torch.tensor(torch.pi))
        log_prob = log_prob.sum(dim=-1)  # Sum over action dimensions

        # Apply constraints
        action = self._apply_constraints(action_raw)

        if squeeze_output:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

        return action, log_prob

    def evaluate_log_prob(self, state: torch.Tensor, action_raw: torch.Tensor) -> torch.Tensor:
        """
        Evaluate log probability of given actions.
        Used for PPO loss computation.

        Args:
            state: State tensor (batch, state_dim)
            action_raw: Raw actions BEFORE constraint application (batch, action_dim)

        Returns:
            log_prob: Log probabilities (batch,)
        """
        mean = self.network(state)
        std = torch.exp(self.log_std).expand_as(mean)
        var = std ** 2
        log_prob = -((action_raw - mean) ** 2) / (2 * var) - self.log_std - 0.5 * torch.log(2 * torch.tensor(torch.pi))
        return log_prob.sum(dim=-1)

    def _apply_constraints(self, action_raw: torch.Tensor) -> torch.Tensor:
        """Apply action constraints for CarRacing environment."""
        # Ensure action_raw has correct shape (batch, action_dim) or (batch, seq, action_dim)
        if action_raw.dim() == 1:
            action_raw = action_raw.unsqueeze(0)

        # Handle case where action_raw has wrong shape
        if action_raw.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"action_raw has wrong shape {action_raw.shape}, expected last dim to be {self.config.action_dim}"
            )

        if self.config.action_dim == 3:
            # Split and apply constraints separately to avoid MPS indexing bugs
            # Use slicing on the LAST dimension (action dim) regardless of 2D or 3D input
            action_raw = action_raw.contiguous()
            steering = torch.tanh(action_raw[..., 0:1])     # Steering: -1 to 1
            gas = torch.sigmoid(action_raw[..., 1:2])       # Gas: 0 to 1
            brake = torch.sigmoid(action_raw[..., 2:3])     # Brake: 0 to 1
            action = torch.cat([steering, gas, brake], dim=-1)  # Concatenate along last dim
        else:
            action = torch.tanh(action_raw)

        return action


def evaluate_controller(
    controller: nn.Module, env, num_episodes: int = 5, max_steps: int = 1000
) -> float:
    """
    Evaluate a controller in the environment.

    Args:
        controller: Controller network
        env: Gymnasium environment
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        average_return: Average return over episodes
    """
    controller.eval()
    total_return = 0.0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0

        # Convert observation to state representation (this would typically use the VAE encoder)
        # For now, we'll use a placeholder
        state = torch.randn(controller.config.state_dim)  # Placeholder

        for step in range(max_steps):
            with torch.no_grad():
                action = controller.get_action(state, deterministic=True)
                action_np = action.cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action_np)
            episode_return += reward

            if terminated or truncated:
                break

            # Update state (placeholder - would use VAE encoder in practice)
            state = torch.randn(controller.config.state_dim)

        total_return += episode_return

    return total_return / num_episodes


if __name__ == "__main__":
    # Test the Controller implementation
    config = ControllerConfig()

    # Test standard controller
    controller = Controller(config)
    batch_size = 4
    state = torch.randn(batch_size, config.state_dim)

    with torch.no_grad():
        actions = controller(state)
        action_single = controller.get_action(torch.randn(config.state_dim))

    print(f"Input state shape: {state.shape}")
    print(f"Output actions shape: {actions.shape}")
    print(f"Single action shape: {action_single.shape}")
    print(
        f"Action ranges - Steering: [{actions[:, 0].min():.3f}, {actions[:, 0].max():.3f}]"
    )
    print(
        f"Action ranges - Gas: [{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}]"
    )
    print(
        f"Action ranges - Brake: [{actions[:, 2].min():.3f}, {actions[:, 2].max():.3f}]"
    )

    # Test evolutionary controller
    evo_controller = EvolutionaryController(config)
    flat_params = evo_controller.get_parameters_flat()
    print(f"Evolutionary controller parameters: {flat_params.shape[0]}")

    # Test mutation
    original_params = flat_params.clone()
    evo_controller.mutate(0.1)
    new_params = evo_controller.get_parameters_flat()
    param_change = torch.norm(new_params - original_params)
    print(f"Parameter change after mutation: {param_change:.6f}")

    print("Controller test passed!")
