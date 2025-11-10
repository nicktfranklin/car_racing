"""
Main training pipeline for World Model agent.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch

from world_models import (
    FSQVAE,
    ControllerTrainer,
    DataCollector,
    ImageDataset,
    SequenceDataset,
    VAEDataset,
    VAELightningModule,
    VAETrainer,
    WorldModel,
    WorldModelAgentConfig,
    WorldModelDataset,
    WorldModelLightningModule,
    WorldModelTrainer,
    create_sequence_train_val_dataloaders,
    create_train_val_dataloaders,
    get_logger,
    setup_logger,
    setup_output_logging,
)


def main():
    parser = argparse.ArgumentParser(description="Train World Model Agent")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["collect", "vae", "world_model", "controller", "all"],
        default="all",
        help="Training stage to run",
    )
    parser.add_argument(
        "--data_file", type=str, default="training_data.h5", help="Data file name"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cpu/cuda/mps/auto)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for data collection (-1 for auto)",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=None,
        help="Number of rollouts to collect (overrides config default)",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=None,
        help="Maximum episode length (overrides config default)",
    )
    parser.add_argument(
        "--fsq-codebook-size",
        type=int,
        default=None,
        help="FSQ codebook size (adjusts FSQ levels automatically)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering for fastest data collection",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint every N episodes during data collection",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional file to log stdout/stderr (overrides config)",
    )
    args = parser.parse_args()

    # Initialize logger
    logger = setup_logger("world_models", level=logging.INFO)

    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    if args.config:
        config = WorldModelAgentConfig.from_yaml(args.config)
        logger.debug(
            f"Config loaded from YAML: max_episode_length={config.data.max_episode_length}, num_workers={config.data.num_workers}"
        )
    else:
        config = WorldModelAgentConfig()
        logger.info("Using default config")

    # Set up logging to file (command line overrides config)
    log_file = args.log_file if args.log_file is not None else config.training.log_file
    if log_file:
        logger.info(f"Logging to file: {log_file}")
    stdout_tee, stderr_tee = setup_output_logging(log_file)

    # Set device
    if args.device is not None:
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        config.training.device = device
    else:
        # Use device from config
        device = config.training.device

    # Override rollout settings if provided
    if args.workers is not None:
        config.data.num_workers = args.workers
    if args.num_rollouts is not None:
        config.data.num_rollouts = args.num_rollouts
    if args.max_episode_length is not None:
        config.data.max_episode_length = args.max_episode_length

    # Override FSQ codebook size if provided
    if args.fsq_codebook_size is not None:
        config.set_fsq_codebook_size(args.fsq_codebook_size)

    # Set render mode for data collection performance
    if args.no_render:
        config.data.render_mode = None

    # Validate configuration
    config.validate_consistency()

    # Create directories
    os.makedirs(config.data.data_dir, exist_ok=True)
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    logger.info("=" * 50)
    logger.info("World Model Training Pipeline")
    logger.info("=" * 50)
    logger.info(f"Device: {device}")
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Max episode length: {config.data.max_episode_length}")
    logger.info(f"Num workers: {config.data.num_workers}")

    if args.stage in ["collect", "all"]:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 1: DATA COLLECTION")
        logger.info("=" * 50)
        collect_data(config, args.data_file, args.checkpoint_every)

    if args.stage in ["vae", "all"]:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 2: VAE TRAINING")
        logger.info("=" * 50)
        train_vae(config, args.data_file, args.resume)

    if args.stage in ["world_model", "all"]:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 3: WORLD MODEL TRAINING")
        logger.info("=" * 50)
        train_world_model(config, args.data_file, args.resume)

    if args.stage in ["controller", "all"]:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 4: CONTROLLER TRAINING")
        logger.info("=" * 50)
        train_controller(config, args.resume)

    logger.info("Training pipeline completed!")

    # Clean up logging
    if stdout_tee is not None:
        sys.stdout = stdout_tee.original
        sys.stderr = stderr_tee.original
        stdout_tee.close()
        stderr_tee.close()


def collect_data(
    config: WorldModelAgentConfig, data_file: str, checkpoint_every: int = 100
):
    """Collect training data with checkpointing."""
    logger = get_logger("world_models")
    collector = DataCollector(config.data)

    logger.info(f"Collecting {config.data.num_rollouts} episodes with checkpointing...")
    collector.collect_random_episodes(
        config.data.num_rollouts, data_file=data_file, checkpoint_every=checkpoint_every
    )

    collector.close()
    logger.info("Data collection completed!")


def train_vae(config: WorldModelAgentConfig, data_file: str, resume: bool = False):
    """Train the FSQ-VAE using PyTorch Lightning."""
    from src.world_models.training import (
        create_dataloaders,
        create_dataset,
        find_checkpoint_to_resume,
        setup_callbacks,
        setup_tensorboard,
        setup_trainer,
    )

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


def train_world_model(
    config: WorldModelAgentConfig, data_file: str, resume: bool = False
):
    """Train the world model using Lightning."""
    from src.world_models.training import (
        CheckpointManager,
        create_dataloaders,
        create_dataset,
        find_checkpoint_to_resume,
        setup_callbacks,
        setup_tensorboard,
        setup_trainer,
    )

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


def train_controller(config: WorldModelAgentConfig, resume: bool = False):
    """Train the controller."""
    from src.world_models.training import CheckpointManager

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

    # Save best controller
    best_controller = trainer.get_best_controller()
    torch.save(
        best_controller.state_dict(),
        os.path.join(config.training.checkpoint_dir, "best_controller.pth"),
    )

    # Save population checkpoint
    logger.info(f"Saving controller checkpoint to {controller_checkpoint_path}")
    trainer.save_checkpoint(controller_checkpoint_path)

    logger.info("Controller training completed!")
    return best_controller


def evaluate_agent(config: WorldModelAgentConfig, num_episodes: int = 10):
    """Evaluate the trained agent in the real environment."""
    import gymnasium as gym

    from src.world_models.training import CheckpointManager
    from world_models import EvolutionaryController

    logger = get_logger("world_models")
    device = config.training.device

    # Load trained VAE using CheckpointManager
    ckpt_manager = CheckpointManager(config)
    vae = ckpt_manager.load_vae(use_perceptual_loss=False)

    # Load controller
    controller = EvolutionaryController(config.controller)
    controller_checkpoint_path = os.path.join(
        config.training.checkpoint_dir, "best_controller.pth"
    )

    if os.path.exists(controller_checkpoint_path):
        controller.load_state_dict(torch.load(controller_checkpoint_path))
        logger.info("Loaded trained controller")
    else:
        logger.error("No trained controller found!")
        return

    # Create environment
    env = gym.make(config.data.env_name, render_mode="human")

    # Use configured device, validate it's available
    device_str = config.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but CUDA is not available on this system"
        )
    elif device_str == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS device requested but MPS is not available on this system"
        )
    device = torch.device(device_str)
    vae.to(device).eval()
    controller.to(device).eval()

    total_returns = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0

        for step in range(config.data.max_episode_length):
            # Preprocess observation (resize from 96x96 to 64x64)
            from skimage.transform import resize

            obs_resized = resize(obs, (64, 64), anti_aliasing=True, preserve_range=True)
            obs_tensor = (
                torch.from_numpy(obs_resized.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            # Encode to state representation
            with torch.no_grad():
                z_q, _ = vae.encode(obs_tensor)
                action = controller(z_q.squeeze(0))
                action_np = action.cpu().numpy()

            # Take action
            obs, reward, terminated, truncated, _ = env.step(action_np)
            episode_return += reward

            if terminated or truncated:
                break

        total_returns.append(episode_return)
        logger.info(f"Episode {episode+1}: Return = {episode_return:.2f}")

    env.close()

    mean_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    logger.info(f"Evaluation Results ({num_episodes} episodes):")
    logger.info(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")

    return mean_return


if __name__ == "__main__":
    main()
