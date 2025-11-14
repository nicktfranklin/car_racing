"""
Sequence-based datasets for World Model training.
"""

import os
import warnings
from typing import Dict, List

import h5py
import numpy as np
import torch

from ...training.data_collection import Episode

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


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

    def set_worker_shard(self, worker_id: int, num_workers: int):
        """Set worker sharding info for file handle caching.

        Does NOT rebuild index - all workers maintain the full index.
        Sharding only affects which files get cached (memory optimization).
        """
        self._worker_shard_info = (worker_id, num_workers)
        print(
            f"Worker {worker_id}/{num_workers} will cache files {[i for i in range(len(self.chunk_files)) if i % num_workers == worker_id][:5]}... ({len([i for i in range(len(self.chunk_files)) if i % num_workers == worker_id])} files)"
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
        """Get file handle with selective caching based on worker sharding.

        Files assigned to this worker are cached. Files not assigned are
        opened without caching (on-demand access).
        """
        import os

        pid = os.getpid()
        cache_key = (pid, file_idx)

        # Check if this file should be cached by this worker
        should_cache = True
        if hasattr(self, "_worker_shard_info") and self._worker_shard_info is not None:
            worker_id, num_workers = self._worker_shard_info
            should_cache = file_idx % num_workers == worker_id

        if should_cache:
            # Cache files belonging to this worker
            if cache_key not in self._file_handles:
                filepath = os.path.join(self.data_dir, self.chunk_files[file_idx])
                self._file_handles[cache_key] = h5py.File(
                    filepath,
                    "r",
                    rdcc_nbytes=4 * 1024 * 1024,  # 4MB cache per file
                    rdcc_nslots=1000,
                )
            return self._file_handles[cache_key]
        else:
            # Open without caching for files not assigned to this worker
            filepath = os.path.join(self.data_dir, self.chunk_files[file_idx])
            return h5py.File(filepath, "r", rdcc_nbytes=0, rdcc_nslots=1)

    def _get_lazy_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by lazy loading from HDF5 file with caching.

        Cached files (assigned to this worker) are kept open.
        Non-cached files are opened and closed immediately.
        """
        file_idx, ep_idx_in_file, start_idx = self.sequences[idx]

        # Check if this file should be cached
        should_cache = True
        if hasattr(self, "_worker_shard_info") and self._worker_shard_info is not None:
            worker_id, num_workers = self._worker_shard_info
            should_cache = file_idx % num_workers == worker_id

        # Get file handle
        f = self._get_file_handle(file_idx)
        try:
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
        finally:
            # Close non-cached files immediately
            if not should_cache:
                f.close()

    def __del__(self):
        """Close all file handles when dataset is destroyed."""
        for handle in self._file_handles.values():
            try:
                handle.close()
            except:
                pass


class WorldModelDataset(torch.utils.data.Dataset):
    """Sequential dataset for World Model training with sequences and subsampling.

    Loads chunks of files sequentially and extracts sequences for transformer training.
    Lower subsample rate than VAE since we need consecutive frames.
    """

    def __init__(
        self,
        data_dir: str,
        chunk_files: List[str],
        sequence_length: int = 64,
        subsample_rate: int = 4,
        files_per_chunk: int = 5,
    ):
        """
        Args:
            data_dir: Directory containing HDF5 chunk files
            chunk_files: List of chunk filenames to load from
            sequence_length: Length of sequences to extract
            subsample_rate: Use every Nth frame (default 4 for less data but still sequences)
            files_per_chunk: Number of files to load at once (default 5 = ~6GB)
        """
        self.data_dir = data_dir
        self.chunk_files = chunk_files
        self.sequence_length = sequence_length
        self.subsample_rate = subsample_rate
        self.files_per_chunk = files_per_chunk

        # Chunked loading state
        self.current_chunk_idx = 0
        self.num_chunks = (len(chunk_files) + files_per_chunk - 1) // files_per_chunk

        # Pre-allocate storage (will be populated by _load_chunk)
        self.observations = None
        self.actions = None
        self.rewards = None
        self.dones = None

        print(
            f"World Model Dataset: {len(self.chunk_files)} files → {self.num_chunks} chunks of {self.files_per_chunk} files"
        )
        print(
            f"  Sequence length: {self.sequence_length}, subsample rate: 1/{self.subsample_rate}"
        )
        print(f"  Using lazy loading (sequences indexed per chunk)")

        # Load first chunk (will build index for these files only)
        self._load_chunk(0)

    def _load_chunk(self, chunk_idx: int):
        """Load a chunk of files and extract sequences."""
        # Calculate which files belong to this chunk
        start_file = chunk_idx * self.files_per_chunk
        end_file = min(start_file + self.files_per_chunk, len(self.chunk_files))

        # First pass: count sequences in this chunk
        sequence_info = []  # (file_idx, ep_idx, start_frame)

        for file_idx in range(start_file, end_file):
            filepath = os.path.join(self.data_dir, self.chunk_files[file_idx])
            with h5py.File(filepath, "r") as f:
                ep_names = sorted([k for k in f.keys() if k.startswith("episode_")])
                for ep_idx, ep_name in enumerate(ep_names):
                    num_frames = len(f[ep_name]["observations"])

                    # Can we extract sequences of required length?
                    max_start = num_frames - (
                        self.sequence_length * self.subsample_rate
                    )
                    if max_start >= 0:
                        # Create sequences starting at every subsample_rate frames
                        for start_idx in range(0, max_start + 1, self.subsample_rate):
                            sequence_info.append((file_idx, ep_idx, start_idx))

        num_sequences = len(sequence_info)

        # Pre-allocate arrays for all sequences
        # Use uint8 for observations to save 4x memory (normalize on-the-fly in __getitem__)
        observations = np.empty(
            (num_sequences, self.sequence_length, 64, 64, 3), dtype=np.uint8
        )
        actions = np.empty(
            (num_sequences, self.sequence_length - 1, 3), dtype=np.float32
        )
        rewards = np.empty((num_sequences, self.sequence_length - 1), dtype=np.float32)
        dones = np.empty((num_sequences, self.sequence_length - 1), dtype=np.bool_)

        # Pre-compute frame indices once
        frame_indices = list(
            range(0, self.sequence_length * self.subsample_rate, self.subsample_rate)
        )
        action_indices = frame_indices[:-1]

        # Second pass: load actual data
        seq_idx = 0
        current_file = None
        current_file_handle = None
        episode_names_cache = None

        for file_idx, ep_idx, start_frame in sequence_info:
            # Open new file if needed
            if current_file != file_idx:
                if current_file_handle is not None:
                    current_file_handle.close()
                filepath = os.path.join(self.data_dir, self.chunk_files[file_idx])
                current_file_handle = h5py.File(filepath, "r")
                current_file = file_idx
                # Cache episode names for this file
                episode_names_cache = sorted(
                    [k for k in current_file_handle.keys() if k.startswith("episode_")]
                )

            # Load sequence
            ep_name = episode_names_cache[ep_idx]
            ep_group = current_file_handle[ep_name]

            # Compute actual frame indices for this sequence
            actual_frame_indices = [start_frame + offset for offset in frame_indices]
            actual_action_indices = [start_frame + offset for offset in action_indices]

            # Load observations sequentially instead of fancy indexing
            # This avoids slow HDF5 fancy indexing that can cause blocking I/O
            # Store as uint8 to save memory (normalize on-the-fly in __getitem__)
            obs_dataset = ep_group["observations"]
            for j, frame_idx in enumerate(actual_frame_indices):
                observations[seq_idx, j] = obs_dataset[frame_idx]

            # Load other data sequentially
            actions_dataset = ep_group["actions"]
            rewards_dataset = ep_group["rewards"]
            dones_dataset = ep_group["dones"]
            for j, action_idx in enumerate(actual_action_indices):
                actions[seq_idx, j] = actions_dataset[action_idx]
                rewards[seq_idx, j] = rewards_dataset[action_idx]
                dones[seq_idx, j] = dones_dataset[action_idx]

            seq_idx += 1

        if current_file_handle is not None:
            current_file_handle.close()

        # Store as single arrays instead of list of dicts
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.current_chunk_idx = chunk_idx

        size_mb = (
            observations.nbytes + actions.nbytes + rewards.nbytes + dones.nbytes
        ) / (1024 * 1024)
        print(
            f"Loaded chunk {chunk_idx + 1}/{self.num_chunks}: {num_sequences:,} sequences ({size_mb:.1f} MB)"
        )

    def load_next_chunk(self):
        """Load next chunk of files. Called by rotation callback."""
        next_chunk = (self.current_chunk_idx + 1) % self.num_chunks
        self._load_chunk(next_chunk)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sequence from current chunk - normalize observations on-the-fly."""
        # Normalize uint8 [0, 255] to float32 [0, 1] on-the-fly to save memory
        obs = self.observations[idx].astype(np.float32) / 255.0

        return {
            "observations": torch.from_numpy(obs).permute(
                0, 3, 1, 2
            ),  # (T, C, H, W)
            "actions": torch.from_numpy(self.actions[idx]),
            "rewards": torch.from_numpy(self.rewards[idx]),
            "dones": torch.from_numpy(self.dones[idx]),
        }
