"""
Worker functions for parallel episode collection.
"""

import gc
import os
import time
import warnings
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from skimage.transform import resize

from .agent import RandomAgent
from .episode import Episode


def collect_episodes_worker(
    args: Tuple[str, Optional[str], int, int, int],
) -> List[Episode]:
    """Worker function for parallel episode collection."""
    # Suppress all warnings in worker processes
    warnings.filterwarnings("ignore")
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

    env_name, render_mode, num_episodes, max_episode_length, worker_id = args

    # Create environment for this worker (None render mode for fastest collection)
    if render_mode is None:
        env = gym.make(env_name, max_episode_steps=max_episode_length)
    else:
        env = gym.make(
            env_name, render_mode=render_mode, max_episode_steps=max_episode_length
        )
    agent = RandomAgent(env.action_space)
    episodes = []

    # Set different random seed for each worker
    np.random.seed(worker_id * 1000 + int(time.time()) % 1000)

    for i in range(num_episodes):
        episode = Episode()
        obs, _ = env.reset()

        # Preprocess observation - keep as uint8 for efficient storage
        obs = resize(obs, (64, 64), anti_aliasing=True, preserve_range=True).astype(
            np.uint8
        )

        for step in range(max_episode_length):
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Preprocess next observation - keep as uint8
            next_obs = resize(
                next_obs, (64, 64), anti_aliasing=True, preserve_range=True
            ).astype(np.uint8)

            episode.add_step(obs, action, reward, terminated or truncated)
            obs = next_obs

            if terminated or truncated:
                break

        episodes.append(episode)

        # Cleanup after each episode to reduce memory pressure
        if (i + 1) % 10 == 0:
            gc.collect()

    env.close()
    return episodes
