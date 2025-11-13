"""
Data collector for World Model training.
"""

import gc
import multiprocessing as mp
import os
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import List

# Suppress third-party library warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import gymnasium as gym
import h5py
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

from ...config import DataConfig, WorldModelAgentConfig
from ...utils import get_logger
from .agent import RandomAgent
from .episode import Episode
from .workers import collect_episodes_worker


class DataCollector:
    """Collects data from environment interactions."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.env = None

    def setup_env(self):
        """Setup the environment."""
        if self.config.render_mode is None:
            self.env = gym.make(
                self.config.env_name, max_episode_steps=self.config.max_episode_length
            )
        else:
            self.env = gym.make(
                self.config.env_name,
                render_mode=self.config.render_mode,
                max_episode_steps=self.config.max_episode_length,
            )
        print(f"Environment: {self.config.env_name}")
        print(f"Action space: {self.env.action_space}")
        print(f"Observation space: {self.env.observation_space}")

    def collect_random_episodes(
        self, num_episodes: int, data_file: str = None, checkpoint_every: int = 100
    ) -> List[Episode]:
        """Collect episodes using random actions with checkpointing support."""
        if data_file is None:
            # No checkpointing, collect all at once
            return self._collect_episodes_no_checkpoint(num_episodes, existing_count=0)

        # Check existing progress across all files
        existing_count = self.count_all_episodes(data_file)
        remaining = num_episodes - existing_count

        if remaining <= 0:
            return []

        # Collect in chunks with checkpointing (memory efficient - don't load existing episodes)
        episodes_collected = 0

        # Create single progress bar for all chunks
        with tqdm(
            total=num_episodes,
            initial=existing_count,
            desc="Collecting episodes",
            unit="ep",
            ncols=100,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            while episodes_collected < remaining:
                # Determine chunk size
                chunk_size = min(checkpoint_every, remaining - episodes_collected)

                # Collect chunk without its own progress bar
                chunk_episodes = self._collect_episodes_no_checkpoint_for_chunking(
                    chunk_size, pbar
                )

                # Save the count before clearing the episodes
                chunk_count = len(chunk_episodes)

                # Save to separate file (one file per chunk)
                file_idx = (existing_count + episodes_collected) // checkpoint_every
                chunk_file = self._get_chunk_filename(data_file, file_idx)
                self.save_episodes(chunk_episodes, chunk_file)

                # Explicit memory cleanup
                del chunk_episodes
                gc.collect()

                episodes_collected += chunk_count

        # Don't load all episodes into memory - they're already saved to disk
        return []

    def _collect_episodes_no_checkpoint_for_chunking(
        self, num_episodes: int, pbar
    ) -> List[Episode]:
        """Collect episodes and update the provided progress bar."""
        # Determine number of workers
        num_workers = self.config.num_workers
        if num_workers == -1:
            num_workers = min(mp.cpu_count(), 16)

        # Use single-threaded for very small collections or when explicitly set to 1
        if num_episodes < 20 or num_workers == 1:
            return self._collect_episodes_sequential_with_pbar(num_episodes, pbar)
        else:
            return self._collect_episodes_parallel_with_pbar(
                num_episodes, num_workers, pbar
            )

    def _collect_episodes_no_checkpoint(
        self, num_episodes: int, existing_count: int = 0, total_episodes: int = None
    ) -> List[Episode]:
        """Collect episodes without checkpointing (original method)."""
        # Determine number of workers
        num_workers = self.config.num_workers
        if num_workers == -1:
            num_workers = min(mp.cpu_count(), 16)  # Increased cap for faster collection

        # Use single-threaded for very small collections or when explicitly set to 1
        if num_episodes < 20 or num_workers == 1:
            return self._collect_episodes_sequential(
                num_episodes, existing_count, total_episodes
            )
        else:
            return self._collect_episodes_parallel(
                num_episodes, num_workers, existing_count, total_episodes
            )

    def _collect_episodes_sequential_with_pbar(
        self, num_episodes: int, pbar
    ) -> List[Episode]:
        """Sequential episode collection with external progress bar."""
        if self.env is None:
            self.setup_env()

        agent = RandomAgent(self.env.action_space)
        episodes = []

        print(f"Collecting {num_episodes} random episodes (sequential)...")
        for i in range(num_episodes):
            episode = self._collect_single_episode(agent)
            episodes.append(episode)
            pbar.update(1)

            if (i + 1) % 100 == 0:
                avg_length = np.mean([len(ep) for ep in episodes[-100:]])
                avg_return = np.mean([sum(ep.rewards) for ep in episodes[-100:]])
                print(
                    f"Episodes {i+1-99}-{i+1}: Avg length = {avg_length:.1f}, Avg return = {avg_return:.2f}"
                )

        return episodes

    def _collect_episodes_sequential(
        self, num_episodes: int, existing_count: int = 0, total_episodes: int = None
    ) -> List[Episode]:
        """Sequential episode collection (original method)."""
        if self.env is None:
            self.setup_env()

        agent = RandomAgent(self.env.action_space)
        episodes = []

        # Determine progress bar total: show remaining work, not total
        pbar_total = (
            (total_episodes - existing_count) if total_episodes else num_episodes
        )

        print(f"Collecting {num_episodes} random episodes (sequential)...")
        with tqdm(
            total=pbar_total,
            initial=0,
            desc=f"Collecting episodes ({existing_count} already done)",
        ) as pbar:
            for i in range(num_episodes):
                episode = self._collect_single_episode(agent)
                episodes.append(episode)
                pbar.update(1)

                if (i + 1) % 100 == 0:
                    avg_length = np.mean([len(ep) for ep in episodes[-100:]])
                    avg_return = np.mean([sum(ep.rewards) for ep in episodes[-100:]])
                    print(
                        f"Episodes {i+1-99}-{i+1}: Avg length = {avg_length:.1f}, Avg return = {avg_return:.2f}"
                    )

        return episodes

    def _collect_episodes_parallel_with_pbar(
        self, num_episodes: int, num_workers: int, pbar
    ) -> List[Episode]:
        """Parallel episode collection with external progress bar - memory efficient."""
        # OPTIMIZATION: Batch multiple episodes per worker to reduce process spawning overhead
        episodes_per_batch = getattr(self.config, "episodes_per_batch", 10)

        all_episodes = []

        # OPTIMIZATION: Increase max_in_flight for better throughput
        # Allow more tasks to be queued to keep workers busy
        max_in_flight = num_workers * 3

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}  # Map future -> (episode_id, batch_size)
            episode_id = 0

            # Submit initial batch
            while episode_id < min(max_in_flight * episodes_per_batch, num_episodes):
                # Calculate how many episodes this batch should collect
                remaining = num_episodes - episode_id
                batch_size = min(episodes_per_batch, remaining)

                args = (
                    self.config.env_name,
                    self.config.render_mode,
                    batch_size,  # Collect multiple episodes per worker
                    self.config.max_episode_length,
                    episode_id,  # worker_id
                )
                future = executor.submit(collect_episodes_worker, args)
                futures[future] = (episode_id, batch_size)
                episode_id += batch_size

            # Process results and submit new tasks as others complete
            while futures:
                # Wait for next task to complete
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    try:
                        episodes = future.result()
                        all_episodes.extend(episodes)
                        pbar.update(len(episodes))
                    except Exception as exc:
                        print(f"Worker generated an exception: {exc}")

                    # Remove completed future
                    _, batch_size = futures[future]
                    del futures[future]

                    # Submit new task if more episodes needed
                    if episode_id < num_episodes:
                        remaining = num_episodes - episode_id
                        batch_size = min(episodes_per_batch, remaining)

                        args = (
                            self.config.env_name,
                            self.config.render_mode,
                            batch_size,
                            self.config.max_episode_length,
                            episode_id,
                        )
                        new_future = executor.submit(collect_episodes_worker, args)
                        futures[new_future] = (episode_id, batch_size)
                        episode_id += batch_size

        return all_episodes

    def _collect_episodes_parallel(
        self,
        num_episodes: int,
        num_workers: int,
        existing_count: int = 0,
        total_episodes: int = None,
    ) -> List[Episode]:
        """Parallel episode collection using multiprocessing - memory efficient."""
        # OPTIMIZATION: Batch multiple episodes per worker to reduce process spawning overhead
        episodes_per_batch = getattr(self.config, "episodes_per_batch", 10)

        all_episodes = []

        # Determine progress bar total: show remaining work, not total
        pbar_total = (
            (total_episodes - existing_count) if total_episodes else num_episodes
        )

        # OPTIMIZATION: Increase max_in_flight for better throughput
        max_in_flight = num_workers * 3

        with tqdm(
            total=pbar_total,
            initial=0,
            desc=f"Collecting episodes ({existing_count} already done)",
        ) as episode_pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {}  # Map future -> (episode_id, batch_size)
                episode_id = 0

                # Submit initial batch
                while episode_id < min(
                    max_in_flight * episodes_per_batch, num_episodes
                ):
                    remaining = num_episodes - episode_id
                    batch_size = min(episodes_per_batch, remaining)

                    args = (
                        self.config.env_name,
                        self.config.render_mode,
                        batch_size,  # Collect multiple episodes per worker
                        self.config.max_episode_length,
                        episode_id,  # worker_id
                    )
                    future = executor.submit(collect_episodes_worker, args)
                    futures[future] = (episode_id, batch_size)
                    episode_id += batch_size

                # Process results and submit new tasks as others complete
                while futures:
                    # Wait for next task to complete
                    done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                    for future in done:
                        try:
                            episodes = future.result()
                            all_episodes.extend(episodes)
                            episode_pbar.update(len(episodes))
                        except Exception as exc:
                            print(f"Worker generated an exception: {exc}")

                        # Remove completed future
                        _, batch_size = futures[future]
                        del futures[future]

                        # Submit new task if more episodes needed
                        if episode_id < num_episodes:
                            remaining = num_episodes - episode_id
                            batch_size = min(episodes_per_batch, remaining)

                            args = (
                                self.config.env_name,
                                self.config.render_mode,
                                batch_size,
                                self.config.max_episode_length,
                                episode_id,
                            )
                            new_future = executor.submit(collect_episodes_worker, args)
                            futures[new_future] = (episode_id, batch_size)
                            episode_id += batch_size

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
        """Preprocess observation - keep as uint8 for efficient storage."""
        # Resize from 96x96 to 64x64 and keep as uint8 [0, 255]
        # Don't normalize to [0, 1] - that happens at load time
        obs = resize(obs, (64, 64), anti_aliasing=True, preserve_range=True).astype(
            np.uint8
        )

        return obs

    def save_episodes(self, episodes: List[Episode], filename: str):
        """Save episodes to disk with optimized uint8 compression."""
        os.makedirs(self.config.data_dir, exist_ok=True)
        filepath = os.path.join(self.config.data_dir, filename)

        # Save as HDF5 with optimized compression (silently)
        with h5py.File(filepath, "w") as f:
            for i, episode in enumerate(episodes):
                obs, actions, rewards, dones = episode.to_arrays()

                ep_group = f.create_group(f"episode_{i}")

                # Store observations as uint8 with gzip compression and optimal chunking
                # Images should already be uint8 from _preprocess_observation
                ep_group.create_dataset(
                    "observations",
                    data=obs,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, 64, 64, 3),  # Optimize for frame-level access
                )

                # Add compression to other datasets too
                ep_group.create_dataset("actions", data=actions, compression="gzip")
                ep_group.create_dataset("rewards", data=rewards, compression="gzip")
                ep_group.create_dataset("dones", data=dones, compression="gzip")

    def _get_chunk_filename(self, base_filename: str, chunk_idx: int) -> str:
        """Generate filename for a chunk."""
        # Remove extension if present
        if base_filename.endswith(".h5"):
            base_filename = base_filename[:-3]
        return f"{base_filename}_chunk_{chunk_idx:04d}.h5"

    def _parse_chunk_filename(self, filename: str) -> int:
        """Extract chunk index from filename. Returns -1 if not a chunk file."""
        import re

        match = re.search(r"_chunk_(\d+)\.h5$", filename)
        if match:
            return int(match.group(1))
        return -1

    def count_episodes(self, filename: str) -> int:
        """Count episodes in a single file."""
        filepath = os.path.join(self.config.data_dir, filename)

        if not os.path.exists(filepath):
            return 0

        try:
            with h5py.File(filepath, "r") as f:
                return len([key for key in f.keys() if key.startswith("episode_")])
        except Exception:
            return 0

    def count_all_episodes(self, base_filename: str) -> int:
        """Count episodes across all chunk files."""
        import glob

        # Remove extension if present
        if base_filename.endswith(".h5"):
            base_filename = base_filename[:-3]

        # Find all chunk files
        pattern = os.path.join(self.config.data_dir, f"{base_filename}_chunk_*.h5")
        chunk_files = glob.glob(pattern)

        total = 0
        for chunk_file in chunk_files:
            filename = os.path.basename(chunk_file)
            total += self.count_episodes(filename)

        return total

    def get_chunk_files(self, base_filename: str) -> List[str]:
        """Get list of all chunk files for a dataset."""
        import glob

        # Remove extension if present
        if base_filename.endswith(".h5"):
            base_filename = base_filename[:-3]

        # Find all chunk files
        pattern = os.path.join(self.config.data_dir, f"{base_filename}_chunk_*.h5")
        chunk_files = sorted(glob.glob(pattern))

        return [os.path.basename(f) for f in chunk_files]

    def load_episodes(self, base_filename: str) -> List[Episode]:
        """Load episodes from disk (supports both single files and chunked files)."""
        # Check if it's a chunked dataset
        chunk_files = self.get_chunk_files(base_filename)

        if chunk_files:
            # Load from multiple chunk files
            print(f"Found {len(chunk_files)} chunk files")
            all_episodes = []
            for chunk_file in chunk_files:
                episodes = self._load_single_file(chunk_file)
                all_episodes.extend(episodes)
            print(
                f"Loaded {len(all_episodes)} episodes total from {len(chunk_files)} files"
            )
            return all_episodes
        else:
            # Try loading single file (backward compatibility)
            return self._load_single_file(base_filename)

    def _load_single_file(self, filename: str) -> List[Episode]:
        """Load episodes from a single file."""
        filepath = os.path.join(self.config.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        episodes = []
        with h5py.File(filepath, "r") as f:
            for ep_name in tqdm(f.keys(), desc=f"Loading {filename}", leave=False):
                ep_group = f[ep_name]

                episode = Episode()
                episode.observations = list(ep_group["observations"][:])
                episode.actions = list(ep_group["actions"][:])
                episode.rewards = list(ep_group["rewards"][:])
                episode.dones = list(ep_group["dones"][:])

                episodes.append(episode)

        return episodes

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()


def collect_data(
    config: WorldModelAgentConfig, data_file: str, checkpoint_every: int = 100
):
    """Collect training data with checkpointing."""
    logger = get_logger("world_models")
    collector = DataCollector(config.data)

    logger.info(f"Collecting {config.data.num_rollouts} episodes with checkpointing...")
    collector.collect_random_episodes(
        config.data.num_rollouts, data_file=data_file, checkpoint_every=checkpoint_every
    )

    collector.close()
    logger.info("Data collection completed!")
