"""
PyTorch Lightning setup utilities.

Provides reusable functions for creating datasets, callbacks, loggers, and trainers.
"""

import os
from typing import List, Tuple

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from ..config import WorldModelAgentConfig
from .lightning import (
    ChunkRotationCallback,
    create_sequence_train_val_dataloaders,
    create_train_val_dataloaders,
)
from ..utils import get_logger
from .data_collection import DataCollector
from .datasets import ImageDataset, SequenceDataset, VAEDataset, WorldModelDataset

logger = get_logger("world_models")


def create_dataset(model_type: str, config: WorldModelAgentConfig, data_file: str):
    """
    Create appropriate dataset for model type.

    Args:
        model_type: Either 'vae' or 'world_model'
        config: World model configuration
        data_file: Path to data file

    Returns:
        Dataset instance
    """
    print(f"\n[create_dataset] Creating {model_type} dataset:")
    print(f"  Data file: {data_file}")
    print(f"  Config data_dir: {config.data.data_dir}")

    collector = DataCollector(config.data)
    chunk_files = collector.get_chunk_files(data_file)

    print(f"  Chunk files returned: {len(chunk_files)} files")

    if model_type == "vae":
        if chunk_files:
            # Use new VAEDataset with sequential loading and subsampling
            dataset = VAEDataset(
                data_dir=config.data.data_dir,
                chunk_files=chunk_files,
                subsample_rate=config.training.vae_subsample_rate,
                files_per_chunk=config.training.vae_files_per_chunk,
            )
        else:
            # Fallback to loading episodes for backward compatibility
            episodes = collector.load_episodes(data_file)
            dataset = ImageDataset(episodes=episodes)

    elif model_type == "world_model":
        if chunk_files:
            # Use new WorldModelDataset with sequential loading and subsampling
            dataset = WorldModelDataset(
                data_dir=config.data.data_dir,
                chunk_files=chunk_files,
                sequence_length=config.training.world_model_sequence_length,
                subsample_rate=config.training.world_model_subsample_rate,
                files_per_chunk=config.training.world_model_files_per_chunk,
            )
        else:
            # Fallback to loading episodes for backward compatibility
            episodes = collector.load_episodes(data_file)
            dataset = SequenceDataset(episodes, config.world_model.sequence_length)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return dataset


def create_dataloaders(
    model_type: str, dataset, config: WorldModelAgentConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        model_type: Either 'vae' or 'world_model'
        dataset: Dataset instance
        config: World model configuration

    Returns:
        Tuple of (train_loader, val_loader)
    """
    pin_memory = config.training.device == "cuda"
    num_workers = 0  # Use num_workers=0 for sequential chunked loading

    if model_type == "vae":
        train_loader, val_loader = create_train_val_dataloaders(
            dataset=dataset,
            batch_size=config.training.batch_size,
            num_workers=num_workers,
            val_split=config.training.val_split,
            train_samples_per_epoch=config.training.steps_per_epoch,
            val_samples=config.training.val_samples,
            pin_memory=pin_memory,
        )

    elif model_type == "world_model":
        train_loader, val_loader = create_sequence_train_val_dataloaders(
            dataset=dataset,
            batch_size=config.training.world_model_batch_size,
            num_workers=num_workers,
            val_split=config.training.val_split,
            train_samples_per_epoch=config.training.world_model_steps_per_epoch,
            val_samples=config.training.world_model_val_samples,
            pin_memory=pin_memory,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return train_loader, val_loader


def setup_callbacks(
    model_type: str, config: WorldModelAgentConfig, dataset
) -> List[Callback]:
    """
    Create PyTorch Lightning callbacks for training.

    Args:
        model_type: Either 'vae' or 'world_model'
        config: World model configuration
        dataset: Dataset instance

    Returns:
        List of Lightning callbacks
    """
    # Determine checkpoint directory and monitor metric
    checkpoint_dir = os.path.join(config.training.checkpoint_dir, model_type)

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch:02d}-val_loss={val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=config.training.early_stopping_patience,
        mode="min",
        verbose=(model_type == "vae"),  # Only verbose for VAE
    )

    callbacks = [checkpoint_callback, early_stopping]

    # Add chunk rotation callback if using chunked datasets
    if isinstance(dataset, (VAEDataset, WorldModelDataset)):
        callbacks.append(ChunkRotationCallback(epochs_per_phase=1))
        logger.debug("Chunk rotation enabled: rotating every epoch")

    return callbacks


def setup_tensorboard(
    model_type: str, config: WorldModelAgentConfig
) -> TensorBoardLogger:
    """
    Create TensorBoard logger.

    Args:
        model_type: Either 'vae' or 'world_model'
        config: World model configuration

    Returns:
        TensorBoard logger instance
    """
    log_name = f"{model_type}_logs"

    tb_logger = TensorBoardLogger(
        save_dir=config.training.checkpoint_dir,
        name=log_name,
        version=None,  # Auto-increment version
    )

    return tb_logger


def setup_trainer(
    model_type: str,
    config: WorldModelAgentConfig,
    callbacks: List[Callback],
    logger: TensorBoardLogger,
) -> L.Trainer:
    """
    Create PyTorch Lightning trainer.

    Args:
        model_type: Either 'vae' or 'world_model'
        config: World model configuration
        callbacks: List of callbacks
        logger: TensorBoard logger

    Returns:
        Configured Lightning trainer
    """
    if model_type == "vae":
        max_epochs = config.training.train_vae_epochs
        log_every_n_steps = config.training.log_every
        limit_train_batches = None  # Use all batches

    elif model_type == "world_model":
        max_epochs = config.training.train_world_model_epochs
        log_every_n_steps = 50
        limit_train_batches = config.training.world_model_steps_per_epoch

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Mixed precision training (fp16) for 50% memory reduction
    precision = "16-mixed" if config.training.use_mixed_precision else "32-true"

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=1.0,
        enable_progress_bar=True,
        limit_train_batches=limit_train_batches,
    )

    return trainer


def find_checkpoint_to_resume(
    model_type: str, config: WorldModelAgentConfig, resume: bool
) -> str:
    """
    Find checkpoint to resume from.

    Args:
        model_type: Either 'vae' or 'world_model'
        config: World model configuration
        resume: Whether to resume training

    Returns:
        Path to checkpoint, or None if not resuming
    """
    if not resume:
        return None

    checkpoint_dir = os.path.join(config.training.checkpoint_dir, model_type)
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")

    if os.path.exists(last_ckpt):
        logger.info(f"Resuming {model_type} training from {last_ckpt}")
        return last_ckpt

    return None
