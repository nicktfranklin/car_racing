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
    """

    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config

        # Use a simpler linear network for evolutionary optimization
        self.network = nn.Linear(config.state_dim, config.action_dim)

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
