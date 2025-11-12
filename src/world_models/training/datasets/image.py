"""
Image-based datasets for VAE training.
"""

import os
import warnings
from typing import List

import h5py
import numpy as np
import torch

from ...training.data_collection import Episode

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


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
        chunk_group_size: int = 5,
        epochs_per_phase: int = 3,
        images_per_chunk: int = None,
    ):
        """
        Args:
            episodes: Pre-loaded episodes (legacy mode, loads all into memory)
            data_dir: Directory containing chunk files (for lazy/chunked loading)
            chunk_files: List of chunk filenames to load from
            subsample_rate: Only use every Nth image (default 1 = use all)
            chunk_group_size: Number of chunks to load together (for chunked mode)
            epochs_per_phase: How many epochs to use each chunk group before rotating
            images_per_chunk: Target images per base chunk (auto-calculated if None)
        """
        self.data_dir = data_dir
        self.chunk_files = chunk_files
        self.lazy_load = data_dir is not None and chunk_files is not None
        self.subsample_rate = subsample_rate
        self.chunk_group_size = chunk_group_size
        self.epochs_per_phase = epochs_per_phase

        # Cache for HDF5 file handles (process-local for DataLoader workers)
        # Note: initialized as empty dict, no lock needed since each worker has its own process
        self._file_handles = {}

        if self.lazy_load:
            # Build index first
            self._build_lazy_index()

            # Auto-enable chunked mode for large datasets
            should_use_chunking = (
                len(chunk_files) > 50 or len(self.image_indices) > 1_000_000
            )

            if should_use_chunking and images_per_chunk is not None:
                # Enable chunked loading with chunk groups
                self._init_chunked_loading(images_per_chunk)
            else:
                # Use lazy loading (on-demand HDF5 access)
                self.use_chunking = False
        else:
            # Legacy mode: load all images into memory
            self.use_chunking = False
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

    def _init_chunked_loading(self, images_per_chunk: int):
        """Initialize chunked loading with multi-chunk groups.

        Args:
            images_per_chunk: Base chunk size (will be multiplied by chunk_group_size)
        """
        import math

        self.use_chunking = True
        self.base_chunk_size = images_per_chunk

        # Calculate how many base chunks we have total
        self.num_base_chunks = math.ceil(len(self.image_indices) / self.base_chunk_size)

        # Calculate how many chunk groups we have
        self.total_chunk_groups = math.ceil(
            self.num_base_chunks / self.chunk_group_size
        )

        # Track current phase (which chunk group we're on)
        self.current_phase = 0

        # Current chunk group in RAM
        self.current_chunk_images = None
        self.current_chunk_start_idx = 0
        self.current_chunk_end_idx = 0

        # Load first chunk group
        self._load_chunk_group(0)

        print(
            f"Chunked mode: Loading chunk groups of {self.chunk_group_size} base chunks"
        )
        print(
            f"  Total: {self.total_chunk_groups} chunk groups ({self.num_base_chunks} base chunks)"
        )
        print(f"  Rotation: Every {self.epochs_per_phase} epochs")

    def _load_chunk_group(self, phase_num: int):
        """Load a group of chunks into RAM.

        Args:
            phase_num: Which chunk group to load (0-indexed)
        """
        # Calculate which base chunks belong to this group
        start_base_chunk = phase_num * self.chunk_group_size
        end_base_chunk = min(
            start_base_chunk + self.chunk_group_size, self.num_base_chunks
        )

        # Calculate image index range
        start_idx = start_base_chunk * self.base_chunk_size
        end_idx = min(end_base_chunk * self.base_chunk_size, len(self.image_indices))
        chunk_size = end_idx - start_idx

        # Preallocate array for chunk group
        chunk_images = np.empty((chunk_size, 64, 64, 3), dtype=np.uint8)

        # Load images sequentially from HDF5 files
        current_file = None
        current_file_handle = None

        for i, idx in enumerate(range(start_idx, end_idx)):
            file_idx, ep_idx, frame_idx = self.image_indices[idx]

            # Open file if different from last
            if current_file != file_idx:
                if current_file_handle is not None:
                    current_file_handle.close()
                filepath = os.path.join(self.data_dir, self.chunk_files[file_idx])
                current_file_handle = h5py.File(filepath, "r")
                current_file = file_idx

            # Load image
            ep_names = sorted(
                [k for k in current_file_handle.keys() if k.startswith("episode_")]
            )
            ep_name = ep_names[ep_idx]
            chunk_images[i] = current_file_handle[ep_name]["observations"][frame_idx]

        if current_file_handle is not None:
            current_file_handle.close()

        # Store chunk group
        self.current_chunk_images = chunk_images
        self.current_chunk_start_idx = start_idx
        self.current_chunk_end_idx = end_idx

        # Calculate size in MB
        size_mb = chunk_size * 64 * 64 * 3 / 1024 / 1024
        print(
            f"Loaded chunk group {phase_num}/{self.total_chunk_groups}: images {start_idx:,} to {end_idx:,} ({chunk_size:,} images, {size_mb:.1f} MB)"
        )

    def rotate_to_next_chunk_group(self):
        """Rotate to next chunk group. Called by Lightning callback every N epochs."""
        if not self.use_chunking:
            return

        # Advance to next phase (wrap around to 0 after last group)
        self.current_phase = (self.current_phase + 1) % self.total_chunk_groups

        # Load the next chunk group
        self._load_chunk_group(self.current_phase)

        print(
            f"✓ Rotated to chunk group {self.current_phase}/{self.total_chunk_groups}"
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
        if self.use_chunking:
            # In chunked mode, report size as current chunk size (not full dataset)
            # This makes RandomSampler sample from [0, chunk_size) instead of [0, full_size)
            return self.current_chunk_end_idx - self.current_chunk_start_idx
        elif self.lazy_load:
            return len(self.image_indices)
        else:
            return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.use_chunking:
            return self._get_chunked_item(idx)
        elif self.lazy_load:
            return self._get_lazy_item(idx)
        else:
            return self._get_memory_item(idx)

    def _get_memory_item(self, idx: int) -> torch.Tensor:
        """Get item from images in memory."""
        img = self.images[idx]
        # Normalize uint8 [0, 255] to float32 [0, 1]
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).float().permute(2, 0, 1)

    def _get_chunked_item(self, idx: int) -> torch.Tensor:
        """Get item from current chunk group in RAM.

        In chunked mode, __len__() returns chunk_size, so RandomSampler passes
        idx in range [0, chunk_size). We use idx directly as the local index.
        """
        # idx is already local to current chunk (in range [0, chunk_size))
        chunk_size = len(self.current_chunk_images)

        # Sanity check: ensure index is within current chunk
        if idx < 0 or idx >= chunk_size:
            # This shouldn't happen if __len__() is working correctly
            print(f"Warning: Index {idx} outside chunk size {chunk_size}")
            print(f"  This indicates a bug in chunked loading logic")
            # Map to global index and fall back to lazy loading
            global_idx = self.current_chunk_start_idx + idx
            return self._get_lazy_item(global_idx)

        # Get image from RAM using local index
        img = self.current_chunk_images[idx]

        # Normalize uint8 [0, 255] to float32 [0, 1]
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).float().permute(2, 0, 1)

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

    def _get_lazy_item(self, idx: int) -> torch.Tensor:
        """Get item by lazy loading from HDF5 file with caching.

        Cached files (assigned to this worker) are kept open.
        Non-cached files are opened and closed immediately.
        """
        file_idx, ep_idx_in_file, frame_idx = self.image_indices[idx]

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
            img = f[ep_name]["observations"][frame_idx]

            # Normalize uint8 [0, 255] to float32 [0, 1]
            # Note: HDF5 returns numpy arrays, no need for np.array() copy
            img = img.astype(np.float32) / 255.0

            # Convert to tensor and permute to (C, H, W)
            return torch.from_numpy(img).float().permute(2, 0, 1)
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


class VAEDataset(torch.utils.data.Dataset):
    """Sequential dataset for VAE training with subsampling.

    Loads chunks of files sequentially into memory for efficient I/O.
    Subsample rate reduces autocorrelation between training images.
    """

    def __init__(
        self,
        data_dir: str,
        chunk_files: List[str],
        subsample_rate: int = 10,
        files_per_chunk: int = 5,
    ):
        """
        Args:
            data_dir: Directory containing HDF5 chunk files
            chunk_files: List of chunk filenames to load from
            subsample_rate: Use every Nth frame (default 10 for decorrelation)
            files_per_chunk: Number of files to load at once (default 5 = ~6GB)
        """
        self.data_dir = data_dir
        self.chunk_files = chunk_files
        self.subsample_rate = subsample_rate
        self.files_per_chunk = files_per_chunk

        # Chunked loading state (calculate before _build_index for printing)
        self.current_chunk_idx = 0
        self.num_chunks = (len(chunk_files) + files_per_chunk - 1) // files_per_chunk
        self.current_images = None

        # Build index of all images
        self._build_index()

        # Load first chunk
        self._load_chunk(0)

    def _build_index(self):
        """Build index of subsampled images across all files."""
        self.image_indices = []  # List of (file_idx, ep_idx_in_file, frame_idx)

        total_frames = 0
        subsampled_frames = 0

        for file_idx, chunk_file in enumerate(self.chunk_files):
            filepath = os.path.join(self.data_dir, chunk_file)
            with h5py.File(filepath, "r") as f:
                ep_names = sorted([k for k in f.keys() if k.startswith("episode_")])
                for ep_idx_in_file, ep_name in enumerate(ep_names):
                    num_frames = len(f[ep_name]["observations"])
                    for frame_idx in range(0, num_frames, self.subsample_rate):
                        self.image_indices.append((file_idx, ep_idx_in_file, frame_idx))
                        subsampled_frames += 1
                    total_frames += num_frames

        print(
            f"VAE Dataset: {subsampled_frames:,} images from {total_frames:,} total (subsample rate: 1/{self.subsample_rate})"
        )
        print(
            f"  {len(self.chunk_files)} files → {self.num_chunks} chunks of {self.files_per_chunk} files"
        )

    def _load_chunk(self, chunk_idx: int):
        """Load a chunk of files into memory."""
        # Calculate which files belong to this chunk
        start_file = chunk_idx * self.files_per_chunk
        end_file = min(start_file + self.files_per_chunk, len(self.chunk_files))

        # Calculate image indices for this chunk
        start_img_idx = None
        end_img_idx = None
        for i, (file_idx, _, _) in enumerate(self.image_indices):
            if file_idx >= start_file and start_img_idx is None:
                start_img_idx = i
            if file_idx >= end_file:
                end_img_idx = i
                break

        if start_img_idx is None:
            start_img_idx = 0
        if end_img_idx is None:
            end_img_idx = len(self.image_indices)

        chunk_size = end_img_idx - start_img_idx

        # Pre-allocate array
        chunk_images = np.empty((chunk_size, 64, 64, 3), dtype=np.uint8)

        # Load images sequentially
        current_file = None
        current_file_handle = None

        for i, idx in enumerate(range(start_img_idx, end_img_idx)):
            file_idx, ep_idx, frame_idx = self.image_indices[idx]

            # Open new file if needed
            if current_file != file_idx:
                if current_file_handle is not None:
                    current_file_handle.close()
                filepath = os.path.join(self.data_dir, self.chunk_files[file_idx])
                current_file_handle = h5py.File(filepath, "r")
                current_file = file_idx

            # Load image
            ep_names = sorted(
                [k for k in current_file_handle.keys() if k.startswith("episode_")]
            )
            ep_name = ep_names[ep_idx]
            chunk_images[i] = current_file_handle[ep_name]["observations"][frame_idx]

        if current_file_handle is not None:
            current_file_handle.close()

        # Store chunk
        self.current_images = chunk_images
        self.current_chunk_idx = chunk_idx
        self.chunk_start_idx = start_img_idx
        self.chunk_end_idx = end_img_idx

        size_mb = chunk_size * 64 * 64 * 3 / (1024 * 1024)

    def load_next_chunk(self):
        """Load next chunk of files. Called by rotation callback."""
        next_chunk = (self.current_chunk_idx + 1) % self.num_chunks
        self._load_chunk(next_chunk)

    def __len__(self) -> int:
        return len(self.current_images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get image from current chunk."""
        img = self.current_images[idx]
        # Normalize uint8 [0, 255] to float32 [0, 1]
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).float().permute(2, 0, 1)  # (C, H, W)
