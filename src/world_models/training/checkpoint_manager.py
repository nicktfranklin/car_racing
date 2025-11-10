"""
Checkpoint management for World Model components.

Provides centralized loading/saving of model checkpoints with support for
both Lightning and legacy checkpoint formats.
"""

import os
from typing import Optional

from ..config import WorldModelAgentConfig
from ..models.fsq_vae import FSQVAE
from ..models.world_model import WorldModel
from ..utils import get_logger

logger = get_logger("world_models")


class CheckpointManager:
    """Manages loading and saving of model checkpoints."""

    def __init__(self, config: WorldModelAgentConfig):
        self.config = config
        self.checkpoint_dir = config.training.checkpoint_dir

    def find_latest_checkpoint(self, model_name: str) -> Optional[str]:
        """
        Find the latest checkpoint for a given model.

        Args:
            model_name: Name of the model (e.g., 'vae', 'world_model', 'controller')

        Returns:
            Path to the latest checkpoint, or None if not found.
        """
        # Try Lightning checkpoints first (preferred)
        lightning_dir = os.path.join(self.checkpoint_dir, model_name)

        # Try versioned checkpoint (v1, v2, etc.)
        versioned_ckpt = os.path.join(lightning_dir, "last-v1.ckpt")
        if os.path.exists(versioned_ckpt):
            return versioned_ckpt

        # Try standard last checkpoint
        last_ckpt = os.path.join(lightning_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            return last_ckpt

        # Try legacy checkpoint
        legacy_ckpt = os.path.join(self.checkpoint_dir, f"{model_name}_latest.pth")
        if os.path.exists(legacy_ckpt):
            return legacy_ckpt

        return None

    def load_vae(
        self, use_perceptual_loss: Optional[bool] = None, device: Optional[str] = None
    ) -> FSQVAE:
        """
        Load trained VAE from checkpoint.

        Args:
            use_perceptual_loss: Whether to use perceptual loss. If None, uses config value.
            device: Device to load model on. If None, uses config value.

        Returns:
            Loaded VAE model.
        """
        # Create VAE model
        if use_perceptual_loss is None:
            use_perceptual_loss = self.config.fsq_vae.use_perceptual_loss
        if device is None:
            device = self.config.training.device

        vae = FSQVAE(
            self.config.fsq_vae,
            use_perceptual_loss=use_perceptual_loss,
            device=device,
        )

        # Find checkpoint
        checkpoint_path = self.find_latest_checkpoint("vae")

        if checkpoint_path is None:
            logger.warning(
                "No trained VAE checkpoint found. Using random initialization."
            )
            return vae

        # Load checkpoint
        if checkpoint_path.endswith(".ckpt"):
            # Lightning checkpoint
            logger.info(f"Loading VAE from Lightning checkpoint: {checkpoint_path}")
            from ..lightning_training import VAELightningModule

            vae_module = VAELightningModule.load_from_checkpoint(
                checkpoint_path, model=vae, config=self.config
            )
            vae = vae_module.model
            logger.info("Loaded trained VAE from Lightning checkpoint")
        else:
            # Legacy checkpoint
            logger.info(f"Loading VAE from legacy checkpoint: {checkpoint_path}")
            from .. import VAETrainer

            vae_trainer = VAETrainer(vae, self.config)
            vae_trainer.load_checkpoint(checkpoint_path)
            logger.info("Loaded trained VAE from legacy checkpoint")

        return vae

    def load_world_model(self) -> WorldModel:
        """
        Load trained World Model from checkpoint.

        Returns:
            Loaded World Model.
        """
        # Create world model
        world_model = WorldModel(self.config.world_model)

        # Find checkpoint
        checkpoint_path = self.find_latest_checkpoint("world_model")

        if checkpoint_path is None:
            logger.warning(
                "No trained World Model checkpoint found. Using random initialization."
            )
            return world_model

        # Load checkpoint
        if checkpoint_path.endswith(".ckpt"):
            # Lightning checkpoint
            logger.info(
                f"Loading World Model from Lightning checkpoint: {checkpoint_path}"
            )
            from ..lightning_training import WorldModelLightningModule

            # Need to load VAE first for WorldModelLightningModule
            vae = self.load_vae(use_perceptual_loss=False)

            wm_module = WorldModelLightningModule.load_from_checkpoint(
                checkpoint_path, world_model=world_model, vae=vae, config=self.config
            )
            world_model = wm_module.world_model
            logger.info("Loaded trained World Model from Lightning checkpoint")
        else:
            # Legacy checkpoint
            logger.info(
                f"Loading World Model from legacy checkpoint: {checkpoint_path}"
            )
            from .. import WorldModelTrainer

            # Need VAE for legacy trainer
            vae = self.load_vae(use_perceptual_loss=False)

            wm_trainer = WorldModelTrainer(world_model, vae, self.config)
            wm_trainer.load_checkpoint(checkpoint_path)
            logger.info("Loaded trained World Model from legacy checkpoint")

        return world_model
