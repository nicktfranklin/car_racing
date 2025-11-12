"""
PyTorch Lightning training modules for World Model components.

Provides Lightning modules for:
- FSQ-VAE training
- World Model (GPT-2) training
"""

import lightning as L
import torch

from ...config import WorldModelAgentConfig
from ...models.fsq_vae import FSQVAE
from ...models.world_model import WorldModel
from ...utils import get_logger, print_model_info
from ...utils.training_utils import create_optimizer, create_scheduler

logger = get_logger("world_models")


class VAELightningModule(L.LightningModule):
    """Lightning module for FSQ-VAE training."""

    def __init__(self, model: FSQVAE, config: WorldModelAgentConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=["model"])

        # Move perceptual loss to correct device if it exists
        if (
            hasattr(self.model, "perceptual_loss")
            and self.model.perceptual_loss is not None
        ):
            self.model.perceptual_loss = self.model.perceptual_loss.to(self.device)

        # Print model architecture and parameters
        print_model_info(self.model, "FSQ-VAE (Lightning)")
        print(f"Codebook size: {self.model.quantizer.codebook_size}")
        print(f"FSQ levels: {self.model.config.fsq_levels}")
        print(f"Latent dimension: {self.model.config.latent_dim}")
        print(f"Optimizer: {config.fsq_vae.optimizer.optimizer}")
        print(f"Learning rate: {config.fsq_vae.optimizer.learning_rate}")
        print(f"Beta (commitment weight): {config.fsq_vae.beta}\n")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch
        x_recon, z, z_q, indices, tokens = self.model(images)

        loss, loss_dict = self.model.compute_loss(images, x_recon, z, z_q, indices)

        # Log metrics
        self.log("train/loss", loss_dict["total_loss"], prog_bar=True)

        # Reconstruction metrics (grouped for easy comparison)
        self.log("train/recon_loss", loss_dict["recon_loss"], prog_bar=False)
        self.log("train/mse_loss", loss_dict["mse_loss"])
        if "perceptual_loss" in loss_dict:
            self.log(
                "train/perceptual_loss", loss_dict["perceptual_loss"], prog_bar=False
            )

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
            self.log(
                "train/codebook_perplexity",
                loss_dict["codebook_perplexity"],
                prog_bar=False,
            )
            self.log("train/perplexity_ratio", loss_dict["perplexity_ratio"])

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch
        x_recon, z, z_q, indices, tokens = self.model(images)

        loss, loss_dict = self.model.compute_loss(images, x_recon, z, z_q, indices)

        # Log validation metrics
        self.log("val/loss", loss_dict["total_loss"], prog_bar=True)

        # Reconstruction metrics (grouped for easy comparison)
        self.log("val/recon_loss", loss_dict["recon_loss"], prog_bar=False)
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
            self.log(
                "val/codebook_perplexity",
                loss_dict["codebook_perplexity"],
                prog_bar=False,
            )
            self.log("val/perplexity_ratio", loss_dict["perplexity_ratio"])

        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(
            self.model.parameters(), self.config.fsq_vae.optimizer
        )

        # Apply gradient clipping if configured
        if self.config.fsq_vae.optimizer.grad_clip_norm is not None:
            self.gradient_clip_val = self.config.fsq_vae.optimizer.grad_clip_norm

        # Create scheduler if configured
        scheduler = create_scheduler(
            optimizer,
            self.config.fsq_vae.optimizer,
            self.config.training.train_vae_epochs,
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

        # Print model architecture and parameters
        print_model_info(self.world_model, "World Model GPT-2 (Lightning)")
        print(f"Vocabulary size: {self.world_model.VOCAB_SIZE}")
        print(f"Tokens per timestep: {self.world_model.tokens_per_timestep}")
        print(f"Hidden size: {config.world_model.hidden_size}")
        print(f"Number of layers: {getattr(config.world_model, 'n_layers', 6)}")
        print(f"Number of heads: {getattr(config.world_model, 'n_heads', 8)}")
        print(f"Optimizer: {config.world_model.optimizer.optimizer}")
        print(f"Learning rate: {config.world_model.optimizer.learning_rate}")
        print(f"Dropout: {config.world_model.dropout}\n")

    def training_step(self, batch, batch_idx):
        observations = batch["observations"]  # (B, T+1, C, H, W)
        actions = batch["actions"]  # (B, T, action_dim)
        rewards = batch["rewards"]  # (B, T)
        dones = batch["dones"]  # (B, T)

        batch_size, seq_len_plus_one = observations.shape[:2]

        # Encode observations to FSQ tokens
        with torch.no_grad():
            obs_flat = observations.reshape(-1, *observations.shape[2:])
            z_q, indices, tokens = self.vae.encode(obs_flat)
            # tokens: (B*(T+1), fsq_dim) with per-dimension discrete tokens
            tokens = tokens.reshape(
                batch_size, seq_len_plus_one, -1
            )  # (B, T+1, fsq_dim)

        current_state_tokens = tokens[:, :-1]  # (B, T, fsq_dim)
        next_state_tokens = tokens[:, 1:]  # (B, T, fsq_dim)

        # Forward pass
        loss, loss_dict = self.world_model.compute_loss(
            current_state_tokens, next_state_tokens, actions, rewards, dones
        )

        # Log metrics
        self.log("train/loss", loss_dict["total_loss"], prog_bar=True)
        self.log("train/token_loss", loss_dict["token_loss"])
        self.log("train/token_accuracy", loss_dict["token_accuracy"], prog_bar=False)
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
            z_q, indices, tokens = self.vae.encode(obs_flat)
            # tokens: (B*(T+1), fsq_dim) with per-dimension discrete tokens
            tokens = tokens.reshape(
                batch_size, seq_len_plus_one, -1
            )  # (B, T+1, fsq_dim)

        current_state_tokens = tokens[:, :-1]  # (B, T, fsq_dim)
        next_state_tokens = tokens[:, 1:]  # (B, T, fsq_dim)

        loss, loss_dict = self.world_model.compute_loss(
            current_state_tokens, next_state_tokens, actions, rewards, dones
        )

        # Log validation metrics
        self.log("val/loss", loss_dict["total_loss"], prog_bar=True)
        self.log("val/token_loss", loss_dict["token_loss"])
        self.log("val/token_accuracy", loss_dict["token_accuracy"], prog_bar=False)
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
