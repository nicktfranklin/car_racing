"""
Main training pipeline for World Model agent.
"""

import argparse
import os

import numpy as np
import torch

from world_models import (
    FSQVAE,
    ControllerTrainer,
    DataCollector,
    ImageDataset,
    SequenceDataset,
    VAETrainer,
    WorldModel,
    WorldModelAgentConfig,
    WorldModelTrainer,
)


def main():
    parser = argparse.ArgumentParser(description="Train World Model Agent")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
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
    """Train the FSQ-VAE."""
    # Create dataset with lazy loading (don't load all episodes into memory)
    collector = DataCollector(config.data)
    chunk_files = collector.get_chunk_files(data_file)

    if chunk_files:
        # Use lazy loading for chunked data
        dataset = ImageDataset(data_dir=config.data.data_dir, chunk_files=chunk_files)
    else:
        # Fallback to loading episodes for backward compatibility
        episodes = collector.load_episodes(data_file)
        dataset = ImageDataset(episodes=episodes)

    # Create model and trainer
    vae = FSQVAE(config.fsq_vae)
    trainer = VAETrainer(vae, config)

    # Resume from checkpoint if requested
    vae_checkpoint_path = os.path.join(config.training.checkpoint_dir, "vae_latest.pth")
    if resume and os.path.exists(vae_checkpoint_path):
        print(f"Resuming VAE training from {vae_checkpoint_path}")
        trainer.load_checkpoint(vae_checkpoint_path)

    # Train
    print(f"Training VAE for {config.training.train_vae_epochs} epochs...")
    history = trainer.train(dataset, config.training.train_vae_epochs)

    # Save checkpoint
    print(f"Saving VAE checkpoint to {vae_checkpoint_path}")
    trainer.save_checkpoint(vae_checkpoint_path)

    print("VAE training completed!")
    return vae


def train_world_model(
    config: WorldModelAgentConfig, data_file: str, resume: bool = False
):
    """Train the world model."""
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

    # Load trained VAE
    vae = FSQVAE(config.fsq_vae)
    vae_checkpoint_path = os.path.join(config.training.checkpoint_dir, "vae_latest.pth")
    if os.path.exists(vae_checkpoint_path):
        vae_trainer = VAETrainer(vae, config)
        vae_trainer.load_checkpoint(vae_checkpoint_path)
        print("Loaded trained VAE")
    else:
        print("Warning: No trained VAE found. Training world model with random VAE.")

    # Create world model and trainer
    world_model = WorldModel(config.world_model)
    trainer = WorldModelTrainer(world_model, vae, config)

    # Resume from checkpoint if requested
    wm_checkpoint_path = os.path.join(
        config.training.checkpoint_dir, "world_model_latest.pth"
    )
    if resume and os.path.exists(wm_checkpoint_path):
        print(f"Resuming world model training from {wm_checkpoint_path}")
        trainer.load_checkpoint(wm_checkpoint_path)

    # Train
    print(
        f"Training world model for {config.training.train_world_model_epochs} epochs..."
    )
    history = trainer.train(dataset, config.training.train_world_model_epochs)

    # Save checkpoint
    print(f"Saving world model checkpoint to {wm_checkpoint_path}")
    trainer.save_checkpoint(wm_checkpoint_path)

    print("World model training completed!")
    return world_model


def train_controller(config: WorldModelAgentConfig, resume: bool = False):
    """Train the controller."""
    # Load trained models
    vae = FSQVAE(config.fsq_vae)
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

    # Load trained models
    vae = FSQVAE(config.fsq_vae)
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
        print("Warning: CUDA requested but not available, falling back to CPU")
        device_str = "cpu"
    elif device_str == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but not available, falling back to CPU")
        device_str = "cpu"
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
