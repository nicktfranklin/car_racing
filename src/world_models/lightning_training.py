"""
PyTorch Lightning training modules for World Model components.

Provides efficient training with:
- Fixed-length epochs via limit_train_batches
- Random sampling from large datasets
- Train/validation splits
- Automatic checkpointing
- Early stopping based on validation loss
"""

import torch
import lightning as L
from torch.utils.data import DataLoader, RandomSampler, Subset
from typing import Dict, List, Optional

from .config import WorldModelAgentConfig
from .data_collection import ImageDataset, SequenceDataset
from .models.fsq_vae import FSQVAE
from .models.world_model import WorldModel


class VAELightningModule(L.LightningModule):
    """Lightning module for FSQ-VAE training."""

    def __init__(self, model: FSQVAE, config: WorldModelAgentConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch
        x_recon, z, z_q = self.model(images)
        loss, loss_dict = self.model.compute_loss(images, x_recon, z, z_q)

        # Log metrics
        self.log("train/loss", loss_dict["total_loss"], prog_bar=True)
        self.log("train/recon_loss", loss_dict["recon_loss"])
        self.log("train/commitment_loss", loss_dict["commitment_loss"])

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch
        x_recon, z, z_q = self.model(images)
        loss, loss_dict = self.model.compute_loss(images, x_recon, z, z_q)

        # Log validation metrics
        self.log("val/loss", loss_dict["total_loss"], prog_bar=True)
        self.log("val/recon_loss", loss_dict["recon_loss"])
        self.log("val/commitment_loss", loss_dict["commitment_loss"])

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.fsq_vae.learning_rate
        )
        return optimizer


class WorldModelLightningModule(L.LightningModule):
    """Lightning module for World Model training."""

    def __init__(
        self, world_model: WorldModel, vae: FSQVAE, config: WorldModelAgentConfig
    ):
        super().__init__()
        self.world_model = world_model
        self.vae = vae
        self.vae.eval()  # Keep VAE frozen
        self.config = config
        self.save_hyperparameters(ignore=["world_model", "vae"])

    def training_step(self, batch, batch_idx):
        observations = batch["observations"]  # (B, T+1, C, H, W)
        actions = batch["actions"]  # (B, T, action_dim)
        rewards = batch["rewards"]  # (B, T)
        dones = batch["dones"]  # (B, T)

        batch_size, seq_len_plus_one = observations.shape[:2]

        # Encode observations to state indices
        with torch.no_grad():
            obs_flat = observations.reshape(-1, *observations.shape[2:])
            z_q, indices = self.vae.encode(obs_flat)
            indices = indices.reshape(batch_size, seq_len_plus_one)

        current_states = indices[:, :-1]
        next_states = indices[:, 1:]

        # Forward pass
        loss, loss_dict = self.world_model.compute_loss(
            current_states, actions, next_states, rewards, dones
        )

        # Log metrics
        self.log("train/loss", loss_dict["total_loss"], prog_bar=True)
        self.log("train/state_loss", loss_dict["state_loss"])
        self.log("train/state_accuracy", loss_dict["state_accuracy"], prog_bar=True)
        self.log("train/reward_loss", loss_dict["reward_loss"])
        self.log("train/done_loss", loss_dict["done_loss"])

        return loss

    def validation_step(self, batch, batch_idx):
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        batch_size, seq_len_plus_one = observations.shape[:2]

        with torch.no_grad():
            obs_flat = observations.reshape(-1, *observations.shape[2:])
            z_q, indices = self.vae.encode(obs_flat)
            indices = indices.reshape(batch_size, seq_len_plus_one)

        current_states = indices[:, :-1]
        next_states = indices[:, 1:]

        loss, loss_dict = self.world_model.compute_loss(
            current_states, actions, next_states, rewards, dones
        )

        # Log validation metrics
        self.log("val/loss", loss_dict["total_loss"], prog_bar=True)
        self.log("val/state_loss", loss_dict["state_loss"])
        self.log("val/state_accuracy", loss_dict["state_accuracy"], prog_bar=True)
        self.log("val/reward_loss", loss_dict["reward_loss"])
        self.log("val/done_loss", loss_dict["done_loss"])

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=self.config.world_model.learning_rate
        )
        return optimizer


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
        train_dataset, replacement=True, num_samples=train_samples_per_epoch * batch_size
    )

    # Validation uses a fixed subset
    val_sampler = RandomSampler(
        val_dataset, replacement=False, num_samples=min(val_samples, len(val_dataset))
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset):,} samples, {train_samples_per_epoch} batches/epoch")
    print(f"  Val: {len(val_dataset):,} samples ({val_split*100:.1f}% of data)")

    return train_loader, val_loader
