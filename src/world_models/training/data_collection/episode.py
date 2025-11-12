"""
Episode container for data collection.
"""

from typing import Tuple

import numpy as np


class Episode:
    """Container for episode data."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add_step(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool):
        """Add a step to the episode."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert lists to numpy arrays."""
        return (
            np.array(self.observations),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.dones),
        )

    def __len__(self) -> int:
        return len(self.observations)
