"""
World Model training using PyTorch Lightning.
"""

from ..config import WorldModelAgentConfig
from ..data_collection import WorldModelDataset
from ..lightning_training import WorldModelLightningModule
from ..models.world_model import WorldModel
from ..utils import get_logger
from .checkpoint_manager import CheckpointManager
from .lightning_setup import (
    create_dataloaders,
    create_dataset,
    find_checkpoint_to_resume,
    setup_callbacks,
    setup_tensorboard,
    setup_trainer,
)


def train_world_model(
    config: WorldModelAgentConfig, data_file: str, resume: bool = False
):
    """Train the world model using Lightning."""
    logger = get_logger("world_models")

    # Create dataset and dataloaders
    dataset = create_dataset("world_model", config, data_file)
    train_loader, val_loader = create_dataloaders("world_model", dataset, config)

    # Load trained VAE using CheckpointManager
    ckpt_manager = CheckpointManager(config)
    vae = ckpt_manager.load_vae(use_perceptual_loss=False)

    # Create world model and Lightning module
    world_model = WorldModel(config.world_model)
    lightning_module = WorldModelLightningModule(world_model, vae, config)

    # Setup training components
    callbacks = setup_callbacks("world_model", config, dataset)
    tb_logger = setup_tensorboard("world_model", config)
    trainer = setup_trainer("world_model", config, callbacks, tb_logger)

    # Find checkpoint to resume from
    ckpt_path = find_checkpoint_to_resume("world_model", config, resume)

    # Log training info
    logger.info(
        "Training World Model with Lightning (max {} epochs)...".format(
            config.training.train_world_model_epochs
        )
    )
    logger.info(f"Batches per epoch: {config.training.world_model_steps_per_epoch}")
    if isinstance(dataset, WorldModelDataset):
        logger.info(f"Sequence length: {config.training.world_model_sequence_length}")
        logger.info(f"Subsample rate: 1/{config.training.world_model_subsample_rate}")
    else:
        logger.info(f"Sequence length: {config.world_model.sequence_length}")
    logger.info(f"Validation split: {config.training.val_split*100:.1f}%")
    logger.info(
        f"Early stopping patience: {config.training.early_stopping_patience} epochs"
    )

    # Train
    trainer.fit(lightning_module, train_loader, val_loader, ckpt_path=ckpt_path)

    # Log completion
    checkpoint_callback = callbacks[0]  # ModelCheckpoint is first callback
    logger.info("World model training completed!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"TensorBoard logs saved to: {tb_logger.log_dir}")
    logger.info(
        "To view logs, run: tensorboard --logdir={}/world_model_logs".format(
            config.training.checkpoint_dir
        )
    )

    return world_model
