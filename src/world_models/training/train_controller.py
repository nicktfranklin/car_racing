"""
Controller training using PPO.
"""

import os

import torch

from ..config import WorldModelAgentConfig
from ..training import ControllerTrainer
from ..utils import get_logger
from .checkpoint_manager import CheckpointManager


def train_controller(config: WorldModelAgentConfig, resume: bool = False):
    """Train the controller."""
    logger = get_logger("world_models")

    # Load trained models using CheckpointManager
    ckpt_manager = CheckpointManager(config)
    vae = ckpt_manager.load_vae(use_perceptual_loss=False)
    world_model = ckpt_manager.load_world_model()

    # Create controller trainer
    trainer = ControllerTrainer(vae, world_model, config)

    # Resume from checkpoint if requested
    controller_checkpoint_path = os.path.join(
        config.training.checkpoint_dir, "controller_latest.pth"
    )
    if resume and os.path.exists(controller_checkpoint_path):
        logger.info(f"Resuming controller training from {controller_checkpoint_path}")
        # TODO: Implement controller resume logic

    # Train
    logger.info(
        f"Training controller for {config.training.train_controller_epochs} generations..."
    )
    history = trainer.train(config.training.train_controller_epochs)

    # Save trained controller
    torch.save(
        trainer.controller.state_dict(),
        os.path.join(config.training.checkpoint_dir, "best_controller.pth"),
    )

    # Save population checkpoint
    logger.info(f"Saving controller checkpoint to {controller_checkpoint_path}")
    trainer.save_checkpoint(controller_checkpoint_path)

    logger.info("Controller training completed!")
    return trainer.controller
