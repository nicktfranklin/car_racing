"""
VAE training using PyTorch Lightning.
"""

from ..config import WorldModelAgentConfig
from ..lightning_training import VAELightningModule
from ..models.fsq_vae import FSQVAE
from ..utils import get_logger
from .datasets import VAEDataset
from .lightning_setup import (
    create_dataloaders,
    create_dataset,
    find_checkpoint_to_resume,
    setup_callbacks,
    setup_tensorboard,
    setup_trainer,
)


def train_vae(config: WorldModelAgentConfig, data_file: str, resume: bool = False):
    """Train the FSQ-VAE using PyTorch Lightning."""
    logger = get_logger("world_models")

    # Create dataset and dataloaders
    dataset = create_dataset("vae", config, data_file)
    train_loader, val_loader = create_dataloaders("vae", dataset, config)

    # Create model and Lightning module
    vae = FSQVAE(
        config.fsq_vae,
        use_perceptual_loss=config.fsq_vae.use_perceptual_loss,
        device=config.training.device,
    )
    lightning_module = VAELightningModule(vae, config)

    # Setup training components
    callbacks = setup_callbacks("vae", config, dataset)
    tb_logger = setup_tensorboard("vae", config)
    trainer = setup_trainer("vae", config, callbacks, tb_logger)

    # Find checkpoint to resume from
    ckpt_path = find_checkpoint_to_resume("vae", config, resume)

    # Log training info
    logger.info(
        f"Training VAE with Lightning (max {config.training.train_vae_epochs} epochs)..."
    )
    logger.info(f"Batches per epoch: {config.training.steps_per_epoch}")
    if isinstance(dataset, VAEDataset):
        logger.info(f"Subsample rate: 1/{config.training.vae_subsample_rate}")
    else:
        logger.info(f"Subsample rate: 1/{config.training.subsample_rate}")
    logger.info(f"Validation split: {config.training.val_split*100:.1f}%")
    logger.info(
        f"Early stopping patience: {config.training.early_stopping_patience} epochs"
    )

    # Train
    trainer.fit(lightning_module, train_loader, val_loader, ckpt_path=ckpt_path)

    # Log completion
    checkpoint_callback = callbacks[0]  # ModelCheckpoint is first callback
    logger.info("VAE training completed!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"TensorBoard logs saved to: {tb_logger.log_dir}")
    logger.info(
        "To view logs, run: tensorboard --logdir={}/vae_logs".format(
            config.training.checkpoint_dir
        )
    )

    return vae
