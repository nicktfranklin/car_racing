"""
Agent interfaces and implementations for World Model demonstration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn


class Agent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action given an observation."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for new episode."""
        pass

    @property
    def name(self) -> str:
        """Get agent name for display."""
        return self.__class__.__name__


class RandomAgent(Agent):
    """Random agent that samples actions uniformly."""

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get random action."""
        return self.action_space.sample()

    def reset(self) -> None:
        """Nothing to reset for random agent."""
        pass

    @property
    def name(self) -> str:
        return "Random Agent"


class HumanAgent(Agent):
    """Human agent controlled via keyboard."""

    def __init__(self):
        self.action = np.array([0.0, 0.0, 0.0])  # [steering, gas, brake]

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get current action state (updated by keyboard handler)."""
        return self.action.copy()

    def reset(self) -> None:
        """Reset to neutral action."""
        self.action = np.array([0.0, 0.0, 0.0])

    def update_action(self, steering: float = 0.0, gas: float = 0.0, brake: float = 0.0):
        """Update action from keyboard input."""
        self.action[0] = steering
        self.action[1] = gas
        self.action[2] = brake

    @property
    def name(self) -> str:
        return "Human Player"


class WorldModelAgent(Agent):
    """Agent using trained World Model components."""

    def __init__(self, vae_model, controller_model, device: str = "cpu"):
        self.vae = vae_model
        self.controller = controller_model
        self.device = torch.device(device)

        if self.vae is not None:
            self.vae.to(self.device).eval()
        if self.controller is not None:
            self.controller.to(self.device).eval()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action from World Model controller."""
        if self.vae is None or self.controller is None:
            # Fallback to random if models not loaded
            return np.random.uniform(-1, 1, 3)

        # Preprocess observation (resize from 96x96 to 64x64)
        from skimage.transform import resize
        obs_resized = resize(observation, (64, 64), anti_aliasing=True, preserve_range=True)
        obs_tensor = (
            torch.from_numpy(obs_resized.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            # Encode observation to latent state
            z_q, _ = self.vae.encode(obs_tensor)

            # Get action from controller
            action = self.controller(z_q.squeeze(0))
            return action.cpu().numpy()

    def reset(self) -> None:
        """Nothing to reset for World Model agent."""
        pass

    @property
    def name(self) -> str:
        return "World Model Agent"


class ConstantAgent(Agent):
    """Agent that performs a constant action (useful for debugging)."""

    def __init__(self, action: np.ndarray):
        self.constant_action = np.array(action)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Return constant action."""
        return self.constant_action.copy()

    def reset(self) -> None:
        """Nothing to reset for constant agent."""
        pass

    @property
    def name(self) -> str:
        return f"Constant Agent {self.constant_action}"


def create_agent(agent_type: str, **kwargs) -> Agent:
    """Factory function to create agents."""
    if agent_type.lower() == "random":
        action_space = kwargs.get("action_space")
        if action_space is None:
            raise ValueError("action_space required for random agent")
        return RandomAgent(action_space)

    elif agent_type.lower() == "human":
        return HumanAgent()

    elif agent_type.lower() == "world_model":
        vae_model = kwargs.get("vae_model")
        controller_model = kwargs.get("controller_model")
        device = kwargs.get("device", "cpu")
        return WorldModelAgent(vae_model, controller_model, device)

    elif agent_type.lower() == "constant":
        action = kwargs.get("action", [0.0, 0.0, 0.0])
        return ConstantAgent(action)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")