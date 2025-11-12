"""
Agent evaluation in real environment.
"""

import os

import gymnasium as gym
import numpy as np
import torch
from skimage.transform import resize

from ..config import WorldModelAgentConfig
from ..models.controller import EvolutionaryController
from ..utils import get_logger
from .checkpoint_manager import CheckpointManager


def evaluate_agent(config: WorldModelAgentConfig, num_episodes: int = 10):
    """Evaluate the trained agent in the real environment."""
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
