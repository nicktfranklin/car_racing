"""
Data collection system for World Model training.
"""

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import h5py
import numpy as np
import torch
from tqdm import tqdm

try:
    from .config import DataConfig
except ImportError:
    # For direct execution
    from config import DataConfig


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


class RandomAgent:
    """Random agent for data collection."""

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get random action."""
        return self.action_space.sample()


def collect_episodes_worker(args: Tuple[str, str, int, int, int]) -> List[Episode]:
    """Worker function for parallel episode collection."""
    env_name, render_mode, num_episodes, max_episode_length, worker_id = args

    # Create environment for this worker
    env = gym.make(env_name, render_mode=render_mode)
    agent = RandomAgent(env.action_space)
    episodes = []

    # Set different random seed for each worker
    np.random.seed(worker_id * 1000 + int(time.time()) % 1000)

    for i in range(num_episodes):
        episode = Episode()
        obs, _ = env.reset()

        # Preprocess observation
        from skimage.transform import resize
        obs = obs.astype(np.float32) / 255.0
        obs = resize(obs, (64, 64), anti_aliasing=True, preserve_range=True)

        for step in range(max_episode_length):
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Preprocess next observation
            next_obs = next_obs.astype(np.float32) / 255.0
            next_obs = resize(next_obs, (64, 64), anti_aliasing=True, preserve_range=True)

            episode.add_step(obs, action, reward, terminated or truncated)
            obs = next_obs

            if terminated or truncated:
                break

        episodes.append(episode)

    env.close()
    return episodes


class DataCollector:
    """Collects data from environment interactions."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.env = None

    def setup_env(self):
        """Setup the environment."""
        self.env = gym.make(self.config.env_name, render_mode=self.config.render_mode)
        print(f"Environment: {self.config.env_name}")
        print(f"Action space: {self.env.action_space}")
        print(f"Observation space: {self.env.observation_space}")

    def collect_random_episodes(self, num_episodes: int) -> List[Episode]:
        """Collect episodes using random actions (with optional parallelization)."""
        # Determine number of workers
        num_workers = self.config.num_workers
        if num_workers == -1:
            num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overload

        # Use single-threaded for small collections or when explicitly set to 1
        if num_episodes < 50 or num_workers == 1:
            return self._collect_episodes_sequential(num_episodes)
        else:
            return self._collect_episodes_parallel(num_episodes, num_workers)

    def _collect_episodes_sequential(self, num_episodes: int) -> List[Episode]:
        """Sequential episode collection (original method)."""
        if self.env is None:
            self.setup_env()

        agent = RandomAgent(self.env.action_space)
        episodes = []

        print(f"Collecting {num_episodes} random episodes (sequential)...")
        for i in tqdm(range(num_episodes)):
            episode = self._collect_single_episode(agent)
            episodes.append(episode)

            if (i + 1) % 100 == 0:
                avg_length = np.mean([len(ep) for ep in episodes[-100:]])
                avg_return = np.mean([sum(ep.rewards) for ep in episodes[-100:]])
                print(
                    f"Episodes {i+1-99}-{i+1}: Avg length = {avg_length:.1f}, Avg return = {avg_return:.2f}"
                )

        return episodes

    def _collect_episodes_parallel(
        self, num_episodes: int, num_workers: int
    ) -> List[Episode]:
        """Parallel episode collection using multiprocessing."""
        print(
            f"Collecting {num_episodes} episodes using {num_workers} parallel workers..."
        )

        # Calculate episodes per worker
        episodes_per_worker = num_episodes // num_workers
        remaining_episodes = num_episodes % num_workers

        # Create worker arguments
        worker_args = []
        for i in range(num_workers):
            episodes_for_this_worker = episodes_per_worker + (
                1 if i < remaining_episodes else 0
            )
            if episodes_for_this_worker > 0:
                args = (
                    self.config.env_name,
                    self.config.render_mode,
                    episodes_for_this_worker,
                    self.config.max_episode_length,
                    i,  # worker_id
                )
                worker_args.append(args)

        start_time = time.time()
        all_episodes = []

        # Use ProcessPoolExecutor for better control and progress tracking
        with ProcessPoolExecutor(max_workers=len(worker_args)) as executor:
            # Submit all jobs
            future_to_worker = {
                executor.submit(collect_episodes_worker, args): i
                for i, args in enumerate(worker_args)
            }

            # Collect results with progress bar
            with tqdm(total=len(worker_args), desc="Workers completed") as worker_pbar:
                for future in as_completed(future_to_worker):
                    worker_id = future_to_worker[future]
                    try:
                        episodes = future.result()
                        all_episodes.extend(episodes)
                        worker_pbar.set_postfix(
                            {"Episodes": len(all_episodes), "Worker": worker_id}
                        )
                        worker_pbar.update(1)
                    except Exception as exc:
                        print(f"Worker {worker_id} generated an exception: {exc}")

        elapsed = time.time() - start_time

        # Print statistics
        if all_episodes:
            lengths = [len(ep) for ep in all_episodes]
            returns = [sum(ep.rewards) for ep in all_episodes]
            print(f"\n📊 Collection completed in {elapsed:.1f}s:")
            print(f"  • Episodes: {len(all_episodes)}")
            print(f"  • Avg length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
            print(f"  • Avg return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
            print(f"  • Total steps: {sum(lengths):,}")
            print(f"  • Steps/second: {sum(lengths)/elapsed:.0f}")

        return all_episodes

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

        # Resize from 96x96 to 64x64
        from skimage.transform import resize
        obs = resize(obs, (64, 64), anti_aliasing=True, preserve_range=True)

        return obs

    def save_episodes(self, episodes: List[Episode], filename: str):
        """Save episodes to disk."""
        os.makedirs(self.config.data_dir, exist_ok=True)
        filepath = os.path.join(self.config.data_dir, filename)

        # Save as HDF5 for efficient storage
        with h5py.File(filepath, "w") as f:
            for i, episode in enumerate(tqdm(episodes, desc="Saving episodes")):
                obs, actions, rewards, dones = episode.to_arrays()

                ep_group = f.create_group(f"episode_{i}")
                ep_group.create_dataset("observations", data=obs, compression="gzip")
                ep_group.create_dataset("actions", data=actions)
                ep_group.create_dataset("rewards", data=rewards)
                ep_group.create_dataset("dones", data=dones)

        print(f"Saved {len(episodes)} episodes to {filepath}")

    def load_episodes(self, filename: str) -> List[Episode]:
        """Load episodes from disk."""
        filepath = os.path.join(self.config.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        episodes = []
        with h5py.File(filepath, "r") as f:
            for ep_name in tqdm(f.keys(), desc="Loading episodes"):
                ep_group = f[ep_name]

                episode = Episode()
                episode.observations = list(ep_group["observations"][:])
                episode.actions = list(ep_group["actions"][:])
                episode.rewards = list(ep_group["rewards"][:])
                episode.dones = list(ep_group["dones"][:])

                episodes.append(episode)

        print(f"Loaded {len(episodes)} episodes from {filepath}")
        return episodes

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset for training the world model with sequences."""

    def __init__(
        self,
        episodes: List[Episode],
        sequence_length: int,
        include_initial_frame: bool = True,
    ):
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

        print(
            f"Created dataset with {len(self.sequences)} sequences from {len(episodes)} episodes"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start_idx = self.sequences[idx]
        episode = self.episodes[ep_idx]

        # Extract sequence
        end_idx = start_idx + self.sequence_length
        if self.include_initial_frame:
            # Include initial frame for VAE training
            obs_seq = np.array(episode.observations[start_idx : end_idx + 1])
        else:
            obs_seq = np.array(episode.observations[start_idx + 1 : end_idx + 1])

        actions_seq = np.array(episode.actions[start_idx:end_idx])
        rewards_seq = np.array(episode.rewards[start_idx:end_idx])
        dones_seq = np.array(episode.dones[start_idx:end_idx])

        # Convert to tensors
        return {
            "observations": torch.from_numpy(obs_seq)
            .float()
            .permute(0, 3, 1, 2),  # (T, C, H, W)
            "actions": torch.from_numpy(actions_seq).float(),
            "rewards": torch.from_numpy(rewards_seq).float(),
            "dones": torch.from_numpy(dones_seq).bool(),
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


def test_parallel_performance():
    """Test parallel vs sequential collection performance."""
    print("🏃‍♂️ Testing Parallel Data Collection Performance")
    print("=" * 60)

    num_test_episodes = 100
    max_length = 200

    # Test sequential
    print("\n1️⃣ Sequential Collection:")
    config_seq = DataConfig()
    config_seq.num_rollouts = num_test_episodes
    config_seq.max_episode_length = max_length
    config_seq.num_workers = 1  # Force sequential

    collector_seq = DataCollector(config_seq)
    start_time = time.time()
    episodes_seq = collector_seq.collect_random_episodes(num_test_episodes)
    seq_time = time.time() - start_time
    collector_seq.close()

    # Test parallel
    print("\n2️⃣ Parallel Collection:")
    config_par = DataConfig()
    config_par.num_rollouts = num_test_episodes
    config_par.max_episode_length = max_length
    config_par.num_workers = -1  # Auto-detect workers

    collector_par = DataCollector(config_par)
    start_time = time.time()
    episodes_par = collector_par.collect_random_episodes(num_test_episodes)
    par_time = time.time() - start_time
    collector_par.close()

    # Compare results
    print(f"\n🏆 Performance Comparison:")
    print(f"  Sequential time: {seq_time:.1f}s")
    print(f"  Parallel time:   {par_time:.1f}s")
    print(f"  Speedup:         {seq_time/par_time:.2f}x")
    print(f"  Episodes collected: {len(episodes_seq)} vs {len(episodes_par)}")

    return episodes_par[:5]  # Return small subset for further testing


if __name__ == "__main__":
    # Test parallel performance
    episodes = test_parallel_performance()

    if not episodes:
        print("⚠️  No episodes collected, falling back to basic test")
        # Fallback to basic test
        config = DataConfig()
        config.num_rollouts = 5
        config.max_episode_length = 100
        collector = DataCollector(config)
        episodes = collector.collect_random_episodes(config.num_rollouts)
        collector.close()

    # Test data processing pipeline
    print(f"\n🧪 Testing Data Processing Pipeline:")

    # Print statistics
    lengths = [len(ep) for ep in episodes]
    returns = [sum(ep.rewards) for ep in episodes]

    print(f"  Episodes: {len(episodes)}")
    print(f"  Avg length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  Avg return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Total steps: {sum(lengths)}")

    # Test saving/loading
    config = DataConfig()
    collector = DataCollector(config)
    collector.save_episodes(episodes, "test_parallel_data.h5")
    loaded_episodes = collector.load_episodes("test_parallel_data.h5")
    assert len(loaded_episodes) == len(episodes)
    print(f"  ✅ Save/Load test passed")

    # Test datasets
    seq_dataset = SequenceDataset(episodes, sequence_length=10)
    img_dataset = ImageDataset(episodes)

    print(f"  Sequence dataset size: {len(seq_dataset)}")
    print(f"  Image dataset size: {len(img_dataset)}")

    # Test data loading
    if len(seq_dataset) > 0:
        sample_seq = seq_dataset[0]
        print(f"  Sample sequence shapes:")
        for key, tensor in sample_seq.items():
            print(f"    {key}: {tensor.shape}")

    if len(img_dataset) > 0:
        sample_img = img_dataset[0]
        print(f"  Sample image shape: {sample_img.shape}")

    print("\n✅ All tests passed! Parallel data collection is ready.")
