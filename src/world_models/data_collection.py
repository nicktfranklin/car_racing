"""
Data collection system for World Model training.
"""

import gc
import multiprocessing as mp
import os
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import h5py
import numpy as np
import torch
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

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


def collect_episodes_worker(
    args: Tuple[str, Optional[str], int, int, int],
) -> List[Episode]:
    """Worker function for parallel episode collection."""
    import os
    import warnings

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
        from skimage.transform import resize

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

        start_time = time.time()
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

        start_time = time.time()
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
        from skimage.transform import resize

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


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset for training the world model with sequences.

    Supports lazy loading from multiple HDF5 files to avoid loading all data into memory.
    """

    def __init__(
        self,
        episodes: List[Episode] = None,
        sequence_length: int = 10,
        include_initial_frame: bool = True,
        data_dir: str = None,
        chunk_files: List[str] = None,
    ):
        """
        Args:
            episodes: Pre-loaded episodes (legacy mode, loads all into memory)
            sequence_length: Length of sequences to extract
            include_initial_frame: Whether to include initial observation
            data_dir: Directory containing chunk files (for lazy loading)
            chunk_files: List of chunk filenames to load from (for lazy loading)
        """
        self.sequence_length = sequence_length
        self.include_initial_frame = include_initial_frame
        self.data_dir = data_dir
        self.chunk_files = chunk_files
        self.lazy_load = data_dir is not None and chunk_files is not None

        # Cache for HDF5 file handles (process-local for DataLoader workers)
        self._file_handles = {}

        if self.lazy_load:
            # Lazy loading mode: build index without loading episodes
            self._build_lazy_index()
        else:
            # Legacy mode: keep episodes in memory
            self.episodes = episodes if episodes is not None else []
            self._build_memory_index()

    def _build_memory_index(self):
        """Build sequence indices from episodes in memory."""
        self.sequences = []
        for ep_idx, episode in enumerate(self.episodes):
            max_start = len(episode) - self.sequence_length
            if max_start > 0:
                for start_idx in range(max_start):
                    self.sequences.append((ep_idx, start_idx, None))

        print(
            f"Created dataset with {len(self.sequences)} sequences from {len(self.episodes)} episodes"
        )

    def _build_lazy_index(self):
        """Build sequence indices by scanning HDF5 files without loading data."""
        self.sequences = []  # List of (file_idx, ep_idx_in_file, start_idx)

        total_episodes = 0
        for file_idx, chunk_file in enumerate(self.chunk_files):
            filepath = os.path.join(self.data_dir, chunk_file)
            with h5py.File(filepath, "r") as f:
                ep_names = sorted([k for k in f.keys() if k.startswith("episode_")])
                for ep_idx_in_file, ep_name in enumerate(ep_names):
                    ep_length = len(f[ep_name]["observations"])
                    max_start = ep_length - self.sequence_length
                    if max_start > 0:
                        for start_idx in range(max_start):
                            self.sequences.append((file_idx, ep_idx_in_file, start_idx))
                total_episodes += len(ep_names)

        print(
            f"Created lazy dataset with {len(self.sequences)} sequences from {total_episodes} episodes across {len(self.chunk_files)} files"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.lazy_load:
            return self._get_lazy_item(idx)
        else:
            return self._get_memory_item(idx)

    def _get_memory_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from episodes in memory."""
        ep_idx, start_idx, _ = self.sequences[idx]
        episode = self.episodes[ep_idx]

        # Extract sequence
        end_idx = start_idx + self.sequence_length
        if self.include_initial_frame:
            obs_seq = np.array(episode.observations[start_idx : end_idx + 1])
        else:
            obs_seq = np.array(episode.observations[start_idx + 1 : end_idx + 1])

        actions_seq = np.array(episode.actions[start_idx:end_idx])
        rewards_seq = np.array(episode.rewards[start_idx:end_idx])
        dones_seq = np.array(episode.dones[start_idx:end_idx])

        # Normalize uint8 [0, 255] to float32 [0, 1]
        obs_seq = obs_seq.astype(np.float32) / 255.0

        # Convert to tensors
        return {
            "observations": torch.from_numpy(obs_seq)
            .float()
            .permute(0, 3, 1, 2),  # (T, C, H, W)
            "actions": torch.from_numpy(actions_seq).float(),
            "rewards": torch.from_numpy(rewards_seq).float(),
            "dones": torch.from_numpy(dones_seq).bool(),
        }

    def _get_file_handle(self, file_idx: int) -> h5py.File:
        """Get cached file handle for the given file index.

        Uses process-level caching to avoid reopening files repeatedly.
        Each worker process maintains its own cache.
        """
        import os

        pid = os.getpid()  # Get current process ID for worker isolation
        cache_key = (pid, file_idx)

        if cache_key not in self._file_handles:
            filepath = os.path.join(self.data_dir, self.chunk_files[file_idx])
            # Open with minimal cache to prevent memory leaks
            self._file_handles[cache_key] = h5py.File(
                filepath,
                "r",
                rdcc_nbytes=16 * 1024 * 1024,  # 16MB cache per file (reduced from 512MB to prevent OOM)
                rdcc_nslots=5000,  # Reduced from 20000
            )

        return self._file_handles[cache_key]

    def _get_lazy_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by lazy loading from HDF5 file with caching."""
        file_idx, ep_idx_in_file, start_idx = self.sequences[idx]

        # Use cached file handle
        f = self._get_file_handle(file_idx)
        ep_names = sorted([k for k in f.keys() if k.startswith("episode_")])
        ep_name = ep_names[ep_idx_in_file]
        ep_group = f[ep_name]

        # Extract sequence
        end_idx = start_idx + self.sequence_length
        if self.include_initial_frame:
            obs_seq = ep_group["observations"][start_idx : end_idx + 1]
        else:
            obs_seq = ep_group["observations"][start_idx + 1 : end_idx + 1]

        actions_seq = ep_group["actions"][start_idx:end_idx]
        rewards_seq = ep_group["rewards"][start_idx:end_idx]
        dones_seq = ep_group["dones"][start_idx:end_idx]

        # Normalize uint8 [0, 255] to float32 [0, 1]
        # Note: HDF5 returns numpy arrays, no need for np.array() copy
        obs_seq = obs_seq.astype(np.float32) / 255.0

        # Convert to tensors (HDF5 already returns numpy arrays)
        return {
            "observations": torch.from_numpy(obs_seq)
            .float()
            .permute(0, 3, 1, 2),  # (T, C, H, W)
            "actions": torch.from_numpy(actions_seq).float(),
            "rewards": torch.from_numpy(rewards_seq).float(),
            "dones": torch.from_numpy(dones_seq).bool(),
        }

    def __del__(self):
        """Close all file handles when dataset is destroyed."""
        for handle in self._file_handles.values():
            try:
                handle.close()
            except:
                pass


class ImageDataset(torch.utils.data.Dataset):
    """Dataset for training VAE with individual images.

    Supports lazy loading from multiple HDF5 files to avoid loading all data into memory.
    """

    def __init__(
        self,
        episodes: List[Episode] = None,
        data_dir: str = None,
        chunk_files: List[str] = None,
        subsample_rate: int = 1,
    ):
        """
        Args:
            episodes: Pre-loaded episodes (legacy mode, loads all into memory)
            data_dir: Directory containing chunk files (for lazy loading)
            chunk_files: List of chunk filenames to load from (for lazy loading)
            subsample_rate: Only use every Nth image (default 1 = use all)
        """
        self.data_dir = data_dir
        self.chunk_files = chunk_files
        self.lazy_load = data_dir is not None and chunk_files is not None
        self.subsample_rate = subsample_rate

        # Cache for HDF5 file handles (process-local for DataLoader workers)
        # Note: initialized as empty dict, no lock needed since each worker has its own process
        self._file_handles = {}

        if self.lazy_load:
            # Lazy loading mode: build index without loading images
            self._build_lazy_index()
        else:
            # Legacy mode: load all images into memory
            self._build_memory_dataset(episodes if episodes is not None else [])

    def _build_memory_dataset(self, episodes: List[Episode]):
        """Build dataset from episodes in memory."""
        self.images = []
        for episode in episodes:
            for obs in episode.observations:
                self.images.append(obs)

        self.images = np.array(self.images)
        print(f"Created image dataset with {len(self.images)} images")

    def _build_lazy_index(self):
        """Build image index by scanning HDF5 files without loading data."""
        self.image_indices = []  # List of (file_idx, ep_idx_in_file, frame_idx)

        total_images = 0
        subsampled_images = 0
        for file_idx, chunk_file in enumerate(self.chunk_files):
            filepath = os.path.join(self.data_dir, chunk_file)
            with h5py.File(filepath, "r") as f:
                ep_names = sorted([k for k in f.keys() if k.startswith("episode_")])
                for ep_idx_in_file, ep_name in enumerate(ep_names):
                    num_frames = len(f[ep_name]["observations"])
                    for frame_idx in range(0, num_frames, self.subsample_rate):
                        self.image_indices.append((file_idx, ep_idx_in_file, frame_idx))
                        subsampled_images += 1
                    total_images += num_frames

        if self.subsample_rate > 1:
            print(
                f"Created lazy image dataset with {subsampled_images} images (subsampled from {total_images} at rate 1/{self.subsample_rate}) across {len(self.chunk_files)} files"
            )
        else:
            print(
                f"Created lazy image dataset with {total_images} images across {len(self.chunk_files)} files"
            )

    def __len__(self) -> int:
        if self.lazy_load:
            return len(self.image_indices)
        else:
            return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.lazy_load:
            return self._get_lazy_item(idx)
        else:
            return self._get_memory_item(idx)

    def _get_memory_item(self, idx: int) -> torch.Tensor:
        """Get item from images in memory."""
        img = self.images[idx]
        # Normalize uint8 [0, 255] to float32 [0, 1]
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).float().permute(2, 0, 1)

    def _get_file_handle(self, file_idx: int) -> h5py.File:
        """Get cached file handle for the given file index.

        Uses process-level caching to avoid reopening files repeatedly.
        Each worker process maintains its own cache.
        """
        import os

        pid = os.getpid()  # Get current process ID for worker isolation
        cache_key = (pid, file_idx)

        if cache_key not in self._file_handles:
            filepath = os.path.join(self.data_dir, self.chunk_files[file_idx])
            # Open with minimal cache to prevent memory leaks
            self._file_handles[cache_key] = h5py.File(
                filepath,
                "r",
                rdcc_nbytes=16 * 1024 * 1024,  # 16MB cache per file (reduced from 512MB to prevent OOM)
                rdcc_nslots=5000,  # Reduced from 20000
            )

        return self._file_handles[cache_key]

    def _get_lazy_item(self, idx: int) -> torch.Tensor:
        """Get item by lazy loading from HDF5 file with caching.

        Uses cached file handles to avoid reopening files on each access.
        """
        file_idx, ep_idx_in_file, frame_idx = self.image_indices[idx]

        # Use cached file handle
        f = self._get_file_handle(file_idx)
        ep_names = sorted([k for k in f.keys() if k.startswith("episode_")])
        ep_name = ep_names[ep_idx_in_file]
        img = f[ep_name]["observations"][frame_idx]

        # Normalize uint8 [0, 255] to float32 [0, 1]
        # Note: HDF5 returns numpy arrays, no need for np.array() copy
        img = img.astype(np.float32) / 255.0

        # Convert to tensor and permute to (C, H, W)
        return torch.from_numpy(img).float().permute(2, 0, 1)

    def __del__(self):
        """Close all file handles when dataset is destroyed."""
        for handle in self._file_handles.values():
            try:
                handle.close()
            except:
                pass


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
