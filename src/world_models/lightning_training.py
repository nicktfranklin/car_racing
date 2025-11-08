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
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from typing import Dict, List, Optional

from .config import WorldModelAgentConfig, OptimizerConfig
from .data_collection import ImageDataset, SequenceDataset
from .models.fsq_vae import FSQVAE
from .models.world_model import WorldModel


def create_optimizer(parameters, opt_config: OptimizerConfig):
    """Create optimizer from config."""
    if opt_config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            parameters,
            lr=opt_config.learning_rate,
            betas=(opt_config.beta1, opt_config.beta2),
            eps=opt_config.epsilon,
            weight_decay=opt_config.weight_decay,
            amsgrad=opt_config.amsgrad,
        )
    elif opt_config.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            parameters,
            lr=opt_config.learning_rate,
            betas=(opt_config.beta1, opt_config.beta2),
            eps=opt_config.epsilon,
            weight_decay=opt_config.weight_decay,
            amsgrad=opt_config.amsgrad,
        )
    elif opt_config.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            parameters,
            lr=opt_config.learning_rate,
            momentum=opt_config.momentum,
            weight_decay=opt_config.weight_decay,
            nesterov=opt_config.nesterov,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config.optimizer}")

    return optimizer


def create_scheduler(optimizer, opt_config: OptimizerConfig, num_epochs: int):
    """Create learning rate scheduler from config."""
    if not opt_config.use_scheduler:
        return None

    scheduler_type = opt_config.scheduler.lower()

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - opt_config.warmup_epochs,
            eta_min=opt_config.min_lr,
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=opt_config.step_size, gamma=opt_config.gamma
        )
    elif scheduler_type == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt_config.gamma)
    elif scheduler_type == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=opt_config.factor,
            patience=opt_config.patience,
            min_lr=opt_config.min_lr,
        )
        return {
            "scheduler": scheduler,
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1,
        }
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    # Add warmup if needed
    if opt_config.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=opt_config.warmup_epochs,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[opt_config.warmup_epochs],
        )

    return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}


class VAELightningModule(L.LightningModule):
    """Lightning module for FSQ-VAE training."""

    def __init__(self, model: FSQVAE, config: WorldModelAgentConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=["model"])

        # Move perceptual loss to correct device if it exists
        if hasattr(self.model, 'perceptual_loss') and self.model.perceptual_loss is not None:
            self.model.perceptual_loss = self.model.perceptual_loss.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch
        x_recon, z, z_q = self.model(images)

        # Get indices for codebook monitoring
        _, indices = self.model.quantizer(z)

        loss, loss_dict = self.model.compute_loss(images, x_recon, z, z_q, indices)

        # Log metrics
        self.log("train/loss", loss_dict["total_loss"], prog_bar=True)

        # Reconstruction metrics (grouped for easy comparison)
        self.log("train/recon_loss", loss_dict["recon_loss"], prog_bar=True)
        self.log("train/mse_loss", loss_dict["mse_loss"])
        if "perceptual_loss" in loss_dict:
            self.log("train/perceptual_loss", loss_dict["perceptual_loss"])

        # Other losses
        self.log("train/commitment_loss", loss_dict["commitment_loss"])
        if "entropy_loss" in loss_dict:
            self.log("train/entropy_loss", loss_dict["entropy_loss"])

        # Codebook collapse metrics (grouped for monitoring)
        if "codebook_usage" in loss_dict:
            self.log("train/codebook_usage", loss_dict["codebook_usage"])
            self.log("train/unique_codes", loss_dict["unique_codes"])
        if "codebook_perplexity" in loss_dict:
            # Perplexity is the key metric - show in progress bar
            self.log("train/codebook_perplexity", loss_dict["codebook_perplexity"], prog_bar=True)
            self.log("train/perplexity_ratio", loss_dict["perplexity_ratio"])

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch
        x_recon, z, z_q = self.model(images)

        # Get indices for codebook monitoring
        _, indices = self.model.quantizer(z)

        loss, loss_dict = self.model.compute_loss(images, x_recon, z, z_q, indices)

        # Log validation metrics
        self.log("val/loss", loss_dict["total_loss"], prog_bar=True)

        # Reconstruction metrics (grouped for easy comparison)
        self.log("val/recon_loss", loss_dict["recon_loss"], prog_bar=True)
        self.log("val/mse_loss", loss_dict["mse_loss"])
        if "perceptual_loss" in loss_dict:
            self.log("val/perceptual_loss", loss_dict["perceptual_loss"])

        # Other losses
        self.log("val/commitment_loss", loss_dict["commitment_loss"])
        if "entropy_loss" in loss_dict:
            self.log("val/entropy_loss", loss_dict["entropy_loss"])

        # Codebook collapse metrics (grouped for monitoring)
        if "codebook_usage" in loss_dict:
            self.log("val/codebook_usage", loss_dict["codebook_usage"])
            self.log("val/unique_codes", loss_dict["unique_codes"])
        if "codebook_perplexity" in loss_dict:
            # Perplexity is the key metric for collapse detection
            self.log("val/codebook_perplexity", loss_dict["codebook_perplexity"], prog_bar=True)
            self.log("val/perplexity_ratio", loss_dict["perplexity_ratio"])

        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model.parameters(), self.config.fsq_vae.optimizer)

        # Apply gradient clipping if configured
        if self.config.fsq_vae.optimizer.grad_clip_norm is not None:
            self.gradient_clip_val = self.config.fsq_vae.optimizer.grad_clip_norm

        # Create scheduler if configured
        scheduler = create_scheduler(
            optimizer, self.config.fsq_vae.optimizer, self.config.training.train_vae_epochs
        )

        if scheduler is None:
            return optimizer
        else:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}


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
            # indices is now (B*T) with single code per image
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
            # indices is now (B*T) with single code per image
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
        optimizer = create_optimizer(
            self.world_model.parameters(), self.config.world_model.optimizer
        )

        # Apply gradient clipping if configured
        if self.config.world_model.optimizer.grad_clip_norm is not None:
            self.gradient_clip_val = self.config.world_model.optimizer.grad_clip_norm

        # Create scheduler if configured
        scheduler = create_scheduler(
            optimizer,
            self.config.world_model.optimizer,
            self.config.training.train_world_model_epochs,
        )

        if scheduler is None:
            return optimizer
        else:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}


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

    # Validation uses a fixed subset WITHOUT shuffling (sequential for reproducibility)
    # Use SequentialSampler to avoid the shuffling warning
    if val_samples < len(val_dataset):
        # If we need to subsample, select random indices once (not shuffling each epoch)
        val_indices_subset = torch.randperm(len(val_dataset))[:val_samples].tolist()
        val_dataset = Subset(val_dataset, val_indices_subset)
    val_sampler = SequentialSampler(val_dataset)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,  # Increased for better GPU utilization
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,  # Increased for better GPU utilization
    )

    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset):,} samples, {train_samples_per_epoch} batches/epoch")
    print(f"  Val: {len(val_dataset):,} samples ({val_split*100:.1f}% of data)")

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
        train_dataset, replacement=True, num_samples=train_samples_per_epoch * batch_size
    )

    # Validation uses a fixed subset WITHOUT shuffling
    if val_samples < len(val_dataset):
        val_indices_subset = torch.randperm(len(val_dataset))[:val_samples].tolist()
        val_dataset = Subset(val_dataset, val_indices_subset)
    val_sampler = SequentialSampler(val_dataset)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,  # Increased for better GPU utilization
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,  # Increased for better GPU utilization
    )

    print(f"Created sequence dataloaders:")
    print(f"  Train: {len(train_dataset):,} sequences, {train_samples_per_epoch} batches/epoch")
    print(f"  Val: {len(val_dataset):,} sequences ({val_split*100:.1f}% of data)")

    return train_loader, val_loader
