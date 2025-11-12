"""
Random agent for data collection.
"""

import numpy as np


class RandomAgent:
    """Random agent for data collection."""

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get random action."""
        return self.action_space.sample()
