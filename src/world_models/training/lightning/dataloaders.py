"""
DataLoader creation utilities for Lightning training.

Provides functions for creating train/val dataloaders with:
- Random sampling from large datasets
- Train/validation splits
- Worker sharding for HDF5 files
- Support for chunked datasets
"""

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from ...utils import get_logger

logger = get_logger("world_models")


def worker_init_fn(worker_id):
    """Initialize worker with file sharding to reduce memory usage.

    Each worker only indexes and accesses a subset of HDF5 chunk files,
    reducing memory from 200 files × 8 workers to ~25 files × 8 workers.
    """
    import torch.utils.data

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Handle Subset wrapper (used for train/val split)
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        # Set worker shard if dataset supports it
        if hasattr(dataset, "set_worker_shard"):
            dataset.set_worker_shard(worker_info.id, worker_info.num_workers)


def create_train_val_dataloaders(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    val_split: float = 0.05,
    train_samples_per_epoch: int = 1000,
    val_samples: int = 500,
    pin_memory: bool = False,
):
    """
    Create train and validation dataloaders with random sampling.

    For chunked datasets: samples from current chunk (no train/val split needed).
    For non-chunked: uses standard train/val split with random sampling.

    Args:
        dataset: Full dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        val_split: Fraction of data to use for validation
        train_samples_per_epoch: Number of batches per training epoch
        val_samples: Number of samples in validation set
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader
    """
    using_chunking = hasattr(dataset, "use_chunking") and dataset.use_chunking

    if using_chunking:
        # Chunked mode: dataset.__len__() returns current chunk size
        # RandomSampler will automatically sample from [0, chunk_size)
        # When chunks rotate, __len__() updates automatically
        chunk_size = len(dataset)

        # No Subset needed - dataset handles indexing internally
        train_dataset = dataset
        val_dataset = dataset

        # Train: random sampling from current chunk
        train_sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=train_samples_per_epoch * batch_size,
        )

        # Validation: sequential sample from current chunk
        val_sampler = SequentialSampler(range(min(val_samples, chunk_size)))

        logger.debug(f"Created chunked dataloaders:")
        logger.debug(f"  Train: sampling from current chunk ({chunk_size:,} images)")
        logger.debug(
            f"  Val: {min(val_samples, chunk_size):,} samples from current chunk"
        )
        logger.debug(
            f"  Note: Chunk rotates every N epochs, dataset.__len__() updates automatically"
        )
    else:
        # Non-chunked mode: standard train/val split
        dataset_size = len(dataset)

        # Create train/val split
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size

        # Create indices for split
        indices = torch.randperm(dataset_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create subset datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        # Create samplers - RandomSampler with replacement for infinite sampling
        train_sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=train_samples_per_epoch * batch_size,
        )

        # Validation uses a fixed subset WITHOUT shuffling (sequential for reproducibility)
        # Use SequentialSampler to avoid the shuffling warning
        if val_samples < len(val_dataset):
            # If we need to subsample, select random indices once (not shuffling each epoch)
            val_indices_subset = torch.randperm(len(val_dataset))[:val_samples].tolist()
            val_dataset = Subset(val_dataset, val_indices_subset)
        val_sampler = SequentialSampler(val_dataset)

        logger.debug(f"Created dataloaders:")
        logger.debug(
            f"  Train: {len(train_dataset):,} samples, {train_samples_per_epoch} batches/epoch"
        )
        logger.debug(
            f"  Val: {len(val_dataset):,} samples ({val_split*100:.1f}% of data)"
        )

    # Create dataloaders with worker sharding
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=(
            worker_init_fn if num_workers > 0 else None
        ),  # Shard files across workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=(
            worker_init_fn if num_workers > 0 else None
        ),  # Shard files across workers
    )

    return train_loader, val_loader


def create_sequence_train_val_dataloaders(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    val_split: float = 0.05,
    train_samples_per_epoch: int = 1000,
    val_samples: int = 500,
    pin_memory: bool = False,
):
    """
    Create train and validation dataloaders for sequence datasets.

    Similar to create_train_val_dataloaders but optimized for sequence data.

    Args:
        dataset: Full sequence dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        val_split: Fraction of data to use for validation
        train_samples_per_epoch: Number of batches per training epoch
        val_samples: Number of samples in validation set
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader
    """
    dataset_size = len(dataset)

    # Create train/val split
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    # Create indices for split
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create samplers - RandomSampler with replacement for infinite sampling
    train_sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=train_samples_per_epoch * batch_size,
    )

    # Validation uses a fixed subset WITHOUT shuffling
    if val_samples < len(val_dataset):
        val_indices_subset = torch.randperm(len(val_dataset))[:val_samples].tolist()
        val_dataset = Subset(val_dataset, val_indices_subset)
    val_sampler = SequentialSampler(val_dataset)

    # Create dataloaders with worker sharding
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=(
            worker_init_fn if num_workers > 0 else None
        ),  # Shard files across workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=(
            worker_init_fn if num_workers > 0 else None
        ),  # Shard files across workers
    )

    logger.debug(f"Created sequence dataloaders:")
    logger.debug(
        f"  Train: {len(train_dataset):,} sequences, {train_samples_per_epoch} batches/epoch"
    )
    logger.debug(
        f"  Val: {len(val_dataset):,} sequences ({val_split*100:.1f}% of data)"
    )

    return train_loader, val_loader
