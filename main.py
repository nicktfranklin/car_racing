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
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
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
        "--device", type=str, default="auto", help="Device to use (cpu/cuda/auto)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of parallel workers for data collection (-1 for auto)",
    )
    args = parser.parse_args()

    # Load configuration
    if args.config:
        # TODO: Implement config loading from file
        config = WorldModelAgentConfig()
    else:
        config = WorldModelAgentConfig()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    config.training.device = device

    # Set parallel workers
    config.data.num_workers = args.workers

    # Validate configuration
    config.validate_consistency()

    # Create directories
    os.makedirs(config.data.data_dir, exist_ok=True)
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    print(f"Starting World Model training pipeline...")
    print(f"Device: {device}")
    print(f"Stage: {args.stage}")

    if args.stage in ["collect", "all"]:
        print("\n" + "=" * 50)
        print("STAGE 1: DATA COLLECTION")
        print("=" * 50)
        collect_data(config, args.data_file)

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


def collect_data(config: WorldModelAgentConfig, data_file: str):
    """Collect training data."""
    collector = DataCollector(config.data)

    # Check if data already exists
    data_path = os.path.join(config.data.data_dir, data_file)
    if os.path.exists(data_path):
        print(f"Data file {data_path} already exists. Skipping collection.")
        return

    print(f"Collecting {config.data.num_rollouts} episodes...")
    episodes = collector.collect_random_episodes(config.data.num_rollouts)

    print("Saving data...")
    collector.save_episodes(episodes, data_file)

    collector.close()
    print("Data collection completed!")


def train_vae(config: WorldModelAgentConfig, data_file: str, resume: bool = False):
    """Train the FSQ-VAE."""
    # Load data
    collector = DataCollector(config.data)
    episodes = collector.load_episodes(data_file)
    dataset = ImageDataset(episodes)

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
    # Load data
    collector = DataCollector(config.data)
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

    device = torch.device(
        config.training.device if torch.cuda.is_available() else "cpu"
    )
    vae.to(device).eval()
    controller.to(device).eval()

    total_returns = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0

        for step in range(config.data.max_episode_length):
            # Preprocess observation
            obs_tensor = (
                torch.from_numpy(obs.astype(np.float32) / 255.0)
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
