"""
Main training pipeline for World Model agent.
"""

import argparse
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
    VAELightningModule,
    VAETrainer,
    WorldModel,
    WorldModelAgentConfig,
    WorldModelLightningModule,
    WorldModelTrainer,
    create_sequence_train_val_dataloaders,
    create_train_val_dataloaders,
)


class TeeOutput:
    """Write to both file and original stream (stdout/stderr)."""

    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'a', buffering=1)  # Line buffered
        self.original = original_stream

    def write(self, data):
        self.file.write(data)
        self.original.write(data)

    def flush(self):
        self.file.flush()
        self.original.flush()

    def close(self):
        self.file.close()


def setup_logging(log_file):
    """Redirect stdout and stderr to both console and file."""
    if log_file is None:
        return None, None

    # Create log directory if needed
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Add timestamp header to log file
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")

    # Redirect stdout and stderr
    stdout_tee = TeeOutput(log_file, sys.stdout)
    stderr_tee = TeeOutput(log_file, sys.stderr)

    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    print(f"Logging to file: {log_file}")

    return stdout_tee, stderr_tee


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

    # Load configuration
    print(f"Loading config from: {args.config}")
    if args.config:
        config = WorldModelAgentConfig.from_yaml(args.config)
        print(
            f"Config loaded from YAML: max_episode_length={config.data.max_episode_length}, num_workers={config.data.num_workers}"
        )
    else:
        config = WorldModelAgentConfig()
        print(f"Using default config")

    # Set up logging to file (command line overrides config)
    log_file = args.log_file if args.log_file is not None else config.training.log_file
    stdout_tee, stderr_tee = setup_logging(log_file)

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

    print(f"Starting World Model training pipeline...")
    print(f"Device: {device}")
    print(f"Stage: {args.stage}")
    print(f"Max episode length: {config.data.max_episode_length}")
    print(f"Num workers: {config.data.num_workers}")

    if args.stage in ["collect", "all"]:
        print("\n" + "=" * 50)
        print("STAGE 1: DATA COLLECTION")
        print("=" * 50)
        collect_data(config, args.data_file, args.checkpoint_every)

    if args.stage in ["vae", "all"]:
        print("\n" + "=" * 50)
        print("STAGE 2: VAE TRAINING")
        print("=" * 50)
        train_vae(config, args.data_file, args.resume)

    if args.stage in ["world_model", "all"]:
        print("\n" + "=" * 50)
        print("STAGE 3: WORLD MODEL TRAINING")
        print("=" * 50)
        train_world_model(config, args.data_file, args.resume)

    if args.stage in ["controller", "all"]:
        print("\n" + "=" * 50)
        print("STAGE 4: CONTROLLER TRAINING")
        print("=" * 50)
        train_controller(config, args.resume)

    print("\nTraining pipeline completed!")

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
    collector = DataCollector(config.data)

    print(f"Collecting {config.data.num_rollouts} episodes with checkpointing...")
    collector.collect_random_episodes(
        config.data.num_rollouts, data_file=data_file, checkpoint_every=checkpoint_every
    )

    collector.close()
    print("Data collection completed!")


def train_vae(config: WorldModelAgentConfig, data_file: str, resume: bool = False):
    """Train the FSQ-VAE using PyTorch Lightning."""
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    # Create dataset with lazy loading (don't load all episodes into memory)
    collector = DataCollector(config.data)
    chunk_files = collector.get_chunk_files(data_file)

    if chunk_files:
        # Use lazy loading for chunked data with subsampling
        dataset = ImageDataset(
            data_dir=config.data.data_dir,
            chunk_files=chunk_files,
            subsample_rate=config.training.subsample_rate,
        )
    else:
        # Fallback to loading episodes for backward compatibility
        episodes = collector.load_episodes(data_file)
        dataset = ImageDataset(episodes=episodes)

    # Create train/val dataloaders with random sampling
    pin_memory = config.training.device == "cuda"

    train_loader, val_loader = create_train_val_dataloaders(
        dataset=dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_dataloader_workers,
        val_split=config.training.val_split,
        train_samples_per_epoch=config.training.steps_per_epoch,
        val_samples=config.training.val_samples,
        pin_memory=pin_memory,
    )

    # Create model and Lightning module
    vae = FSQVAE(
        config.fsq_vae,
        use_perceptual_loss=config.fsq_vae.use_perceptual_loss,
        device=config.training.device,
    )
    lightning_module = VAELightningModule(vae, config)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.training.checkpoint_dir, "vae"),
        filename="epoch={epoch:02d}-val_loss={val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,  # Don't auto-insert metric name
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=config.training.early_stopping_patience,
        mode="min",
        verbose=True,
    )

    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config.training.checkpoint_dir,
        name="vae_logs",
        version=None,  # Auto-increment version
    )

    # Create trainer
    trainer = L.Trainer(
        max_epochs=config.training.train_vae_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=tb_logger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=config.training.log_every,
        val_check_interval=1.0,  # Validate every epoch
        enable_progress_bar=True,
    )

    # Train
    print(
        f"Training VAE with Lightning (max {config.training.train_vae_epochs} epochs)..."
    )
    print(f"  - {config.training.steps_per_epoch} batches per epoch")
    print(f"  - Subsample rate: 1/{config.training.subsample_rate}")
    print(f"  - Validation split: {config.training.val_split*100:.1f}%")
    print(
        f"  - Early stopping patience: {config.training.early_stopping_patience} epochs"
    )
    print(f"  - Checkpointing best models based on validation loss")

    ckpt_path = None
    if resume:
        last_ckpt = os.path.join(config.training.checkpoint_dir, "vae", "last.ckpt")
        if os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
            print(f"Resuming from {ckpt_path}")

    trainer.fit(lightning_module, train_loader, val_loader, ckpt_path=ckpt_path)

    print("\nVAE training completed!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"\nTensorBoard logs saved to: {tb_logger.log_dir}")
    print(
        "To view logs, run: tensorboard --logdir={}/vae_logs".format(
            config.training.checkpoint_dir
        )
    )

    return vae


def train_world_model(
    config: WorldModelAgentConfig, data_file: str, resume: bool = False
):
    """Train the world model using Lightning."""
    print("\n" + "=" * 50)
    print("STAGE 2: WORLD MODEL TRAINING")
    print("=" * 50)

    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    # Create dataset with lazy loading (don't load all episodes into memory)
    collector = DataCollector(config.data)
    chunk_files = collector.get_chunk_files(data_file)

    if chunk_files:
        # Use lazy loading for chunked data
        dataset = SequenceDataset(
            sequence_length=config.world_model.sequence_length,
            data_dir=config.data.data_dir,
            chunk_files=chunk_files,
        )
    else:
        # Fallback to loading episodes for backward compatibility
        episodes = collector.load_episodes(data_file)
        dataset = SequenceDataset(episodes, config.world_model.sequence_length)

    # Create train/val dataloaders
    pin_memory = config.training.device == "cuda"

    train_loader, val_loader = create_sequence_train_val_dataloaders(
        dataset=dataset,
        batch_size=config.training.world_model_batch_size,
        num_workers=config.training.num_dataloader_workers,
        val_split=config.training.val_split,
        train_samples_per_epoch=config.training.world_model_steps_per_epoch,
        val_samples=config.training.world_model_val_samples,
        pin_memory=pin_memory,
    )

    # Load trained VAE
    vae = FSQVAE(
        config.fsq_vae,
        use_perceptual_loss=config.fsq_vae.use_perceptual_loss,
        device=config.training.device,
    )

    # Try loading from Lightning checkpoint first
    # Try last-v1.ckpt first (new architecture), then last.ckpt (could be old or new)
    vae_lightning_checkpoint_v1 = os.path.join(
        config.training.checkpoint_dir, "vae", "last-v1.ckpt"
    )
    vae_lightning_checkpoint = os.path.join(
        config.training.checkpoint_dir, "vae", "last.ckpt"
    )
    vae_legacy_checkpoint = os.path.join(
        config.training.checkpoint_dir, "vae_latest.pth"
    )

    if os.path.exists(vae_lightning_checkpoint_v1):
        print(
            f"Loading VAE from Lightning checkpoint (v1): {vae_lightning_checkpoint_v1}"
        )
        vae_module = VAELightningModule.load_from_checkpoint(
            vae_lightning_checkpoint_v1, model=vae, config=config
        )
        vae = vae_module.model
        print("Loaded trained VAE from Lightning checkpoint (v1)")
    elif os.path.exists(vae_lightning_checkpoint):
        print(f"Loading VAE from Lightning checkpoint: {vae_lightning_checkpoint}")
        vae_module = VAELightningModule.load_from_checkpoint(
            vae_lightning_checkpoint, model=vae, config=config
        )
        vae = vae_module.model
        print("Loaded trained VAE from Lightning checkpoint")
    elif os.path.exists(vae_legacy_checkpoint):
        print(f"Loading VAE from legacy checkpoint: {vae_legacy_checkpoint}")
        from src.world_models import VAETrainer

        vae_trainer = VAETrainer(vae, config)
        vae_trainer.load_checkpoint(vae_legacy_checkpoint)
        print("Loaded trained VAE from legacy checkpoint")
    else:
        print("Warning: No trained VAE found. Training world model with random VAE.")

    # Create world model and Lightning module
    world_model = WorldModel(config.world_model)
    lightning_module = WorldModelLightningModule(world_model, vae, config)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.training.checkpoint_dir, "world_model"),
        filename="epoch={epoch:02d}-val_loss={val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,  # Don't auto-insert metric name
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=config.training.early_stopping_patience,
        mode="min",
    )

    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config.training.checkpoint_dir,
        name="world_model_logs",
        version=None,  # Auto-increment version
    )

    # Create trainer
    trainer = L.Trainer(
        max_epochs=config.training.train_world_model_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=tb_logger,
        limit_train_batches=config.training.world_model_steps_per_epoch,
        val_check_interval=1.0,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    # Determine checkpoint path for resuming
    ckpt_path = None
    if resume:
        last_ckpt = os.path.join(
            config.training.checkpoint_dir, "world_model", "last.ckpt"
        )
        if os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
            print(f"Resuming world model training from {ckpt_path}")

    # Train
    print(
        "Training World Model with Lightning (max {} epochs)...".format(
            config.training.train_world_model_epochs
        )
    )
    print(f"  - {config.training.world_model_steps_per_epoch} batches per epoch")
    print(f"  - Sequence length: {config.world_model.sequence_length}")
    print(f"  - Validation split: {config.training.val_split*100:.1f}%")
    print(
        f"  - Early stopping patience: {config.training.early_stopping_patience} epochs"
    )
    print(f"  - Checkpointing best models based on validation loss")

    trainer.fit(lightning_module, train_loader, val_loader, ckpt_path=ckpt_path)

    print("\nWorld model training completed!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"\nTensorBoard logs saved to: {tb_logger.log_dir}")
    print(
        "To view logs, run: tensorboard --logdir={}/world_model_logs".format(
            config.training.checkpoint_dir
        )
    )

    return world_model


def train_controller(config: WorldModelAgentConfig, resume: bool = False):
    """Train the controller."""
    device = config.training.device
    # Load trained models
    vae = FSQVAE(
        config.fsq_vae, use_perceptual_loss=False, device=device
    )  # No perceptual loss needed for inference
    world_model = WorldModel(config.world_model)

    vae_checkpoint_path = os.path.join(config.training.checkpoint_dir, "vae_latest.pth")
    wm_checkpoint_path = os.path.join(
        config.training.checkpoint_dir, "world_model_latest.pth"
    )

    if os.path.exists(vae_checkpoint_path):
        vae_trainer = VAETrainer(vae, config)
        vae_trainer.load_checkpoint(vae_checkpoint_path)
        print("Loaded trained VAE")
    else:
        print("Warning: No trained VAE found.")

    if os.path.exists(wm_checkpoint_path):
        wm_trainer = WorldModelTrainer(world_model, vae, config)
        wm_trainer.load_checkpoint(wm_checkpoint_path)
        print("Loaded trained world model")
    else:
        print("Warning: No trained world model found.")

    # Create controller trainer
    trainer = ControllerTrainer(vae, world_model, config)

    # Resume from checkpoint if requested
    controller_checkpoint_path = os.path.join(
        config.training.checkpoint_dir, "controller_latest.pth"
    )
    if resume and os.path.exists(controller_checkpoint_path):
        print(f"Resuming controller training from {controller_checkpoint_path}")
        # TODO: Implement controller resume logic

    # Train
    print(
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
    print(f"Saving controller checkpoint to {controller_checkpoint_path}")
    trainer.save_checkpoint(controller_checkpoint_path)

    print("Controller training completed!")
    return best_controller


def evaluate_agent(config: WorldModelAgentConfig, num_episodes: int = 10):
    """Evaluate the trained agent in the real environment."""
    import gymnasium as gym

    from world_models import EvolutionaryController

    device = config.training.device
    # Load trained models
    vae = FSQVAE(
        config.fsq_vae, use_perceptual_loss=False, device=device
    )  # No perceptual loss needed for inference
    controller = EvolutionaryController(config.controller)

    # Load checkpoints
    vae_checkpoint_path = os.path.join(config.training.checkpoint_dir, "vae_latest.pth")
    controller_checkpoint_path = os.path.join(
        config.training.checkpoint_dir, "best_controller.pth"
    )

    if os.path.exists(vae_checkpoint_path):
        vae_trainer = VAETrainer(vae, config)
        vae_trainer.load_checkpoint(vae_checkpoint_path)
        print("Loaded trained VAE")
    else:
        print("No trained VAE found!")
        return

    if os.path.exists(controller_checkpoint_path):
        controller.load_state_dict(torch.load(controller_checkpoint_path))
        print("Loaded trained controller")
    else:
        print("No trained controller found!")
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
        print(f"Episode {episode+1}: Return = {episode_return:.2f}")

    env.close()

    mean_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")

    return mean_return


if __name__ == "__main__":
    main()
