"""
Dream environment evaluation - pure imagination rollouts using world model.
"""

import os
from datetime import datetime
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from ..config import WorldModelAgentConfig
from .metrics import MetricsCollector
from .video_utils import VideoRecorder


def get_initial_observations(
    num_episodes: int,
    mode: str = "random",
    env_name: str = "CarRacing-v3",
    device: str = "cpu"
) -> List[torch.Tensor]:
    """
    Get initial observations for dream rollouts.

    Args:
        num_episodes: Number of initial observations to generate
        mode: 'random' for random noise, 'real' for real environment resets
        env_name: Gym environment name
        device: Device to put tensors on

    Returns:
        List of initial observation tensors (1, 3, 64, 64)
    """
    initial_obs = []

    if mode == "random":
        # Generate random noise
        for _ in range(num_episodes):
            obs = torch.randn(1, 3, 64, 64, device=device)
            initial_obs.append(obs)

    elif mode == "real":
        # Get real initial states from environment
        env = gym.make(env_name)
        for _ in range(num_episodes):
            obs, _ = env.reset()

            # Preprocess observation
            from skimage.transform import resize
            obs = resize(obs, (64, 64), anti_aliasing=True)
            obs = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0)
            obs = obs / 255.0
            obs = obs.to(device)

            initial_obs.append(obs)
        env.close()

    else:
        raise ValueError(f"Unknown initial state mode: {mode}")

    return initial_obs


def rollout_in_dream(
    vae,
    world_model,
    controller,
    initial_state: torch.Tensor,
    max_steps: int,
    temperature: float = 1.0,
    video_recorder: Optional[VideoRecorder] = None,
    device: str = "cpu",
    fsq_levels: list = [8, 8, 8, 4]
) -> tuple:
    """
    Perform a single rollout entirely in the world model (dream).

    Args:
        vae: Trained VAE model
        world_model: Trained world model
        controller: Trained controller
        initial_state: Initial observation tensor (1, 3, 64, 64)
        max_steps: Maximum number of steps
        temperature: Sampling temperature for world model
        video_recorder: Optional video recorder
        device: Device to run on

    Returns:
        (episode_return, episode_length)
    """
    vae.eval()
    world_model.eval()
    controller.eval()

    # Move initial state to device
    obs = initial_state.to(device)

    # Encode initial observation to get latent state
    with torch.no_grad():
        z_q, indices, state_tokens = vae.encode(obs)
        # state_tokens is (batch, fsq_dim), need (batch, 1, fsq_dim) for world model
        state_tokens = state_tokens.unsqueeze(1)

    episode_return = 0.0
    episode_length = 0

    # Add initial frame to video if recorder provided
    if video_recorder is not None:
        # Decode to get visible frame
        with torch.no_grad():
            reconstructed = vae.decode(z_q)
        frame = reconstructed[0].permute(1, 2, 0).cpu().numpy()
        video_recorder.add_frame(frame)

    # Rollout loop
    for step in range(max_steps):
        with torch.no_grad():
            # Controller predicts action from latent state
            action = controller(z_q)  # z_q is (1, 4), output is (1, 3)

            # World model predicts next state (needs action shape: (batch, 1, 3))
            next_state_tokens, reward, done, cache = world_model.sample_next_state(
                state_tokens, action.unsqueeze(1), temperature=temperature
            )

            # Convert tokens back to continuous representation
            # Tokens are discrete [0, level-1], convert to continuous [-1, 1]
            levels_tensor = torch.tensor(fsq_levels, dtype=torch.float32, device=device)
            z_q = (next_state_tokens.float() * 2.0 / (levels_tensor - 1)) - 1.0

            # Update state tokens for next iteration
            state_tokens = next_state_tokens

            # Accumulate reward
            episode_return += reward.item()
            episode_length += 1

            # Add frame to video if recorder provided
            if video_recorder is not None:
                reconstructed = vae.decode(z_q)
                frame = reconstructed[0].permute(1, 2, 0).cpu().numpy()
                video_recorder.add_frame(frame)

            # Check if episode is done
            if done.item():
                break

    return episode_return, episode_length


def evaluate_dream_environment(
    vae,
    world_model,
    controller,
    config: WorldModelAgentConfig
) -> dict:
    """
    Evaluate agent in dream environment (pure imagination).

    Args:
        vae: Trained VAE model
        world_model: Trained world model
        controller: Trained controller
        config: Configuration object

    Returns:
        Dictionary with evaluation metrics
    """
    eval_config = config.evaluation
    device = config.training.device

    # Set seed if specified
    if eval_config.seed is not None:
        torch.manual_seed(eval_config.seed)
        np.random.seed(eval_config.seed)

    # Move models to device
    vae = vae.to(device)
    world_model = world_model.to(device)
    controller = controller.to(device)

    # Get initial observations
    print(f"\nGenerating initial observations (mode: {eval_config.dream_initial_state})...")
    initial_states = get_initial_observations(
        num_episodes=eval_config.num_episodes,
        mode=eval_config.dream_initial_state,
        env_name=config.data.env_name,
        device=device
    )

    # Setup video recorder if needed
    video_recorder = None
    if eval_config.save_video:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_path = os.path.join(
            eval_config.output_dir,
            f"dream_{timestamp}.mp4"
        )
        video_recorder = VideoRecorder(
            output_path=video_path,
            fps=eval_config.video_fps,
            resolution=(64, 64)
        )

    # Run episodes
    print(f"Evaluating in DREAM environment ({eval_config.num_episodes} episodes)...")
    metrics = MetricsCollector()

    for ep in tqdm(range(eval_config.num_episodes), desc="Dream episodes"):
        episode_return, episode_length = rollout_in_dream(
            vae=vae,
            world_model=world_model,
            controller=controller,
            initial_state=initial_states[ep],
            max_steps=eval_config.max_episode_length,
            temperature=eval_config.dream_temperature,
            video_recorder=video_recorder if eval_config.save_video else None,
            device=device,
            fsq_levels=config.fsq_vae.fsq_levels
        )

        metrics.add_episode(episode_return, episode_length)

        # Print progress
        if (ep + 1) % max(1, eval_config.num_episodes // 5) == 0:
            summary = metrics.get_summary()
            print(
                f"  Episodes {ep+1}/{eval_config.num_episodes}: "
                f"mean return = {summary['return']['mean']:.1f} ± {summary['return']['std']:.1f}"
            )

    # Save video if recorder was used
    if video_recorder is not None:
        video_recorder.save()

    # Get final summary
    summary = metrics.get_summary()

    # Print results
    print("\nDream Environment Results:")
    print(f"  Mean return: {summary['return']['mean']:.2f} ± {summary['return']['std']:.2f}")
    print(f"  Min/Max return: {summary['return']['min']:.2f} / {summary['return']['max']:.2f}")
    print(f"  Mean length: {summary['length']['mean']:.1f} ± {summary['length']['std']:.1f}")

    return summary
