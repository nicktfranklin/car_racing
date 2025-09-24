"""
Data collection system for World Model training.
"""

import os
import pickle
import numpy as np
import torch
import gymnasium as gym
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import h5py

from .config import DataConfig


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
            np.array(self.dones)
        )

    def __len__(self) -> int:
        return len(self.observations)


class RandomAgent:
    """Random agent for data collection."""

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get random action."""
        return self.action_space.sample()


class DataCollector:
    """Collects data from environment interactions."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.env = None

    def setup_env(self):
        """Setup the environment."""
        self.env = gym.make(
            self.config.env_name,
            render_mode=self.config.render_mode
        )
        print(f"Environment: {self.config.env_name}")
        print(f"Action space: {self.env.action_space}")
        print(f"Observation space: {self.env.observation_space}")

    def collect_random_episodes(self, num_episodes: int) -> List[Episode]:
        """Collect episodes using random actions."""
        if self.env is None:
            self.setup_env()

        agent = RandomAgent(self.env.action_space)
        episodes = []

        print(f"Collecting {num_episodes} random episodes...")
        for i in tqdm(range(num_episodes)):
            episode = self._collect_single_episode(agent)
            episodes.append(episode)

            if (i + 1) % 100 == 0:
                avg_length = np.mean([len(ep) for ep in episodes[-100:]])
                avg_return = np.mean([sum(ep.rewards) for ep in episodes[-100:]])
                print(f"Episodes {i+1-99}-{i+1}: Avg length = {avg_length:.1f}, Avg return = {avg_return:.2f}")

        return episodes

    def _collect_single_episode(self, agent) -> Episode:
        """Collect a single episode."""
        episode = Episode()
        obs, _ = self.env.reset()

        # Preprocess observation
        obs = self._preprocess_observation(obs)

        for step in range(self.config.max_episode_length):
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            # Preprocess next observation
            next_obs = self._preprocess_observation(next_obs)

            episode.add_step(obs, action, reward, terminated or truncated)

            obs = next_obs

            if terminated or truncated:
                break

        return episode

    def _preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """Preprocess observation."""
        # Convert to float and normalize to [0, 1]
        obs = obs.astype(np.float32) / 255.0

        # Resize if needed (CarRacing is already 96x96)
        # You could add resizing logic here if needed

        return obs

    def save_episodes(self, episodes: List[Episode], filename: str):
        """Save episodes to disk."""
        os.makedirs(self.config.data_dir, exist_ok=True)
        filepath = os.path.join(self.config.data_dir, filename)

        # Save as HDF5 for efficient storage
        with h5py.File(filepath, 'w') as f:
            for i, episode in enumerate(tqdm(episodes, desc="Saving episodes")):
                obs, actions, rewards, dones = episode.to_arrays()

                ep_group = f.create_group(f'episode_{i}')
                ep_group.create_dataset('observations', data=obs, compression='gzip')
                ep_group.create_dataset('actions', data=actions)
                ep_group.create_dataset('rewards', data=rewards)
                ep_group.create_dataset('dones', data=dones)

        print(f"Saved {len(episodes)} episodes to {filepath}")

    def load_episodes(self, filename: str) -> List[Episode]:
        """Load episodes from disk."""
        filepath = os.path.join(self.config.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        episodes = []
        with h5py.File(filepath, 'r') as f:
            for ep_name in tqdm(f.keys(), desc="Loading episodes"):
                ep_group = f[ep_name]

                episode = Episode()
                episode.observations = list(ep_group['observations'][:])
                episode.actions = list(ep_group['actions'][:])
                episode.rewards = list(ep_group['rewards'][:])
                episode.dones = list(ep_group['dones'][:])

                episodes.append(episode)

        print(f"Loaded {len(episodes)} episodes from {filepath}")
        return episodes

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset for training the world model with sequences."""

    def __init__(self, episodes: List[Episode], sequence_length: int,
                 include_initial_frame: bool = True):
        self.episodes = episodes
        self.sequence_length = sequence_length
        self.include_initial_frame = include_initial_frame

        # Build sequence indices
        self.sequences = []
        for ep_idx, episode in enumerate(episodes):
            max_start = len(episode) - sequence_length
            if max_start > 0:
                for start_idx in range(max_start):
                    self.sequences.append((ep_idx, start_idx))

        print(f"Created dataset with {len(self.sequences)} sequences from {len(episodes)} episodes")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start_idx = self.sequences[idx]
        episode = self.episodes[ep_idx]

        # Extract sequence
        end_idx = start_idx + self.sequence_length
        if self.include_initial_frame:
            # Include initial frame for VAE training
            obs_seq = np.array(episode.observations[start_idx:end_idx + 1])
        else:
            obs_seq = np.array(episode.observations[start_idx + 1:end_idx + 1])

        actions_seq = np.array(episode.actions[start_idx:end_idx])
        rewards_seq = np.array(episode.rewards[start_idx:end_idx])
        dones_seq = np.array(episode.dones[start_idx:end_idx])

        # Convert to tensors
        return {
            'observations': torch.from_numpy(obs_seq).float().permute(0, 3, 1, 2),  # (T, C, H, W)
            'actions': torch.from_numpy(actions_seq).float(),
            'rewards': torch.from_numpy(rewards_seq).float(),
            'dones': torch.from_numpy(dones_seq).bool(),
        }


class ImageDataset(torch.utils.data.Dataset):
    """Dataset for training VAE with individual images."""

    def __init__(self, episodes: List[Episode]):
        self.images = []

        # Collect all images
        for episode in episodes:
            for obs in episode.observations:
                self.images.append(obs)

        self.images = np.array(self.images)
        print(f"Created image dataset with {len(self.images)} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.images[idx]
        # Convert to tensor and permute to (C, H, W)
        return torch.from_numpy(img).float().permute(2, 0, 1)


if __name__ == "__main__":
    # Test data collection
    config = DataConfig()
    config.num_rollouts = 5  # Small number for testing
    config.max_episode_length = 100

    collector = DataCollector(config)

    # Collect data
    episodes = collector.collect_random_episodes(config.num_rollouts)

    # Print statistics
    lengths = [len(ep) for ep in episodes]
    returns = [sum(ep.rewards) for ep in episodes]

    print(f"\nData collection statistics:")
    print(f"Number of episodes: {len(episodes)}")
    print(f"Average episode length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Average return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Total steps: {sum(lengths)}")

    # Test saving/loading
    collector.save_episodes(episodes, "test_data.h5")
    loaded_episodes = collector.load_episodes("test_data.h5")
    assert len(loaded_episodes) == len(episodes)

    # Test datasets
    seq_dataset = SequenceDataset(episodes, sequence_length=10)
    img_dataset = ImageDataset(episodes)

    print(f"Sequence dataset size: {len(seq_dataset)}")
    print(f"Image dataset size: {len(img_dataset)}")

    # Test data loading
    if len(seq_dataset) > 0:
        sample_seq = seq_dataset[0]
        print(f"Sample sequence shapes:")
        for key, tensor in sample_seq.items():
            print(f"  {key}: {tensor.shape}")

    if len(img_dataset) > 0:
        sample_img = img_dataset[0]
        print(f"Sample image shape: {sample_img.shape}")

    collector.close()
    print("Data collection test passed!")