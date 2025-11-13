"""
Agent evaluation in real and dream environments.
"""

import os
from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from skimage.transform import resize
from tqdm import tqdm

from ..config import WorldModelAgentConfig
from ..models.controller import EvolutionaryController
from ..training.checkpoint_manager import CheckpointManager
from ..utils import get_logger
from .dream_evaluation import evaluate_dream_environment
from .metrics import MetricsCollector, save_metrics_to_json
from .video_utils import VideoRecorder


def run_evaluation(config: WorldModelAgentConfig):
    """
    Main evaluation function using config parameters.

    Args:
        config: Configuration object with evaluation settings

    Returns:
        Dictionary with evaluation results
    """
    logger = get_logger("world_models")
    eval_config = config.evaluation

    # Create output directory
    os.makedirs(eval_config.output_dir, exist_ok=True)

    # Load models
    logger.info("Loading models...")
    ckpt_manager = CheckpointManager(config)

    vae = ckpt_manager.load_vae(use_perceptual_loss=False)
    controller = EvolutionaryController(config.controller)

    controller_path = os.path.join(config.training.checkpoint_dir, "best_controller.pth")
    if os.path.exists(controller_path):
        controller.load_state_dict(torch.load(controller_path, map_location=config.training.device))
        logger.info(f"  Controller: {controller_path}")
    else:
        logger.error(f"Controller checkpoint not found: {controller_path}")
        raise FileNotFoundError(f"Controller checkpoint not found: {controller_path}")

    # Load world model if needed for dream evaluation
    world_model = None
    if eval_config.eval_dream:
        world_model = ckpt_manager.load_world_model()
        logger.info("  World model loaded for dream evaluation")

    results = {}
    checkpoint_info = {
        "vae": ckpt_manager.find_latest_checkpoint("vae"),
        "controller": controller_path
    }

    # Real environment evaluation
    if eval_config.eval_real:
        logger.info(f"\n{'='*50}")
        logger.info("REAL ENVIRONMENT EVALUATION")
        logger.info('='*50)
        results["real"] = evaluate_real_environment(vae, controller, config)
        checkpoint_info["world_model"] = "not_used"

    # Dream environment evaluation
    if eval_config.eval_dream:
        logger.info(f"\n{'='*50}")
        logger.info("DREAM ENVIRONMENT EVALUATION")
        logger.info('='*50)
        results["dream"] = evaluate_dream_environment(vae, world_model, controller, config)
        checkpoint_info["world_model"] = ckpt_manager.find_latest_checkpoint("world_model")

    # Save combined metrics
    if eval_config.save_metrics:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metrics_path = os.path.join(
            eval_config.output_dir,
            f"metrics_{timestamp}.json"
        )
        save_metrics_to_json(
            metrics=results,
            filepath=metrics_path,
            config_path=None,  # Could pass config path here
            checkpoint_info=checkpoint_info
        )

    # Print summary
    print_evaluation_summary(results, config)

    return results


def evaluate_real_environment(
    vae,
    controller,
    config: WorldModelAgentConfig
) -> dict:
    """
    Evaluate agent in real CarRacing environment.

    Args:
        vae: Trained VAE model
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
    vae = vae.to(device).eval()
    controller = controller.to(device).eval()

    # Create environment
    env = gym.make(
        config.data.env_name,
        render_mode=eval_config.render_mode,
        max_episode_steps=eval_config.max_episode_length
    )

    # Setup video recorder if needed
    video_recorder = None
    if eval_config.save_video:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_path = os.path.join(
            eval_config.output_dir,
            f"real_{timestamp}.mp4"
        )
        video_recorder = VideoRecorder(
            output_path=video_path,
            fps=eval_config.video_fps,
            resolution=(96, 96)  # Keep original CarRacing resolution
        )

    # Run episodes
    print(f"Evaluating in REAL environment ({eval_config.num_episodes} episodes)...")
    metrics = MetricsCollector()

    for ep in tqdm(range(eval_config.num_episodes), desc="Real episodes"):
        obs, _ = env.reset()
        episode_return = 0.0
        episode_length = 0

        for step in range(eval_config.max_episode_length):
            # Add frame to video before preprocessing
            if video_recorder is not None:
                if eval_config.video_comparison:
                    # Get VAE reconstruction for comparison
                    obs_tensor = preprocess_observation(obs, device)
                    with torch.no_grad():
                        z_q, _, _ = vae.encode(obs_tensor)
                        reconstruction = vae.decode(z_q)
                    recon_np = reconstruction[0].permute(1, 2, 0).cpu().numpy()
                    video_recorder.add_comparison_frame(obs, recon_np)
                else:
                    video_recorder.add_frame(obs)

            # Preprocess observation
            obs_tensor = preprocess_observation(obs, device)

            # Get action from controller
            with torch.no_grad():
                z_q, _, _ = vae.encode(obs_tensor)
                action = controller(z_q)  # z_q is (1, 4), controller expects batch dim
                action_np = action.squeeze(0).cpu().numpy()  # Remove batch dim for env

            # Take action
            obs, reward, terminated, truncated, _ = env.step(action_np)
            episode_return += reward
            episode_length += 1

            if terminated or truncated:
                break

        metrics.add_episode(episode_return, episode_length)

        # Print progress
        if (ep + 1) % max(1, eval_config.num_episodes // 5) == 0:
            summary = metrics.get_summary()
            print(
                f"  Episodes {ep+1}/{eval_config.num_episodes}: "
                f"mean return = {summary['return']['mean']:.1f} ± {summary['return']['std']:.1f}"
            )

    env.close()

    # Save video if recorder was used
    if video_recorder is not None:
        video_recorder.save()

    # Get final summary
    summary = metrics.get_summary()

    # Print results
    print("\nReal Environment Results:")
    print(f"  Mean return: {summary['return']['mean']:.2f} ± {summary['return']['std']:.2f}")
    print(f"  Min/Max return: {summary['return']['min']:.2f} / {summary['return']['max']:.2f}")
    print(f"  Mean length: {summary['length']['mean']:.1f} ± {summary['length']['std']:.1f}")

    return summary


def preprocess_observation(obs: np.ndarray, device: str) -> torch.Tensor:
    """
    Preprocess observation for VAE.

    Args:
        obs: Raw observation from environment (96, 96, 3)
        device: Device to put tensor on

    Returns:
        Preprocessed tensor (1, 3, 64, 64)
    """
    # Resize from 96x96 to 64x64
    obs_resized = resize(obs, (64, 64), anti_aliasing=True, preserve_range=True)

    # Convert to tensor and normalize
    obs_tensor = (
        torch.from_numpy(obs_resized.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    return obs_tensor


def print_evaluation_summary(results: dict, config: WorldModelAgentConfig):
    """
    Print summary of evaluation results.

    Args:
        results: Dictionary with evaluation results
        config: Configuration object
    """
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print('='*50)

    if "real" in results:
        real_results = results["real"]
        print(f"\nReal Environment:")
        print(f"  Episodes: {real_results['num_episodes']}")
        print(f"  Mean return: {real_results['return']['mean']:.2f} ± {real_results['return']['std']:.2f}")
        print(f"  Mean length: {real_results['length']['mean']:.1f}")

    if "dream" in results:
        dream_results = results["dream"]
        print(f"\nDream Environment:")
        print(f"  Episodes: {dream_results['num_episodes']}")
        print(f"  Mean return: {dream_results['return']['mean']:.2f} ± {dream_results['return']['std']:.2f}")
        print(f"  Mean length: {dream_results['length']['mean']:.1f}")

    if "real" in results and "dream" in results:
        real_mean = results["real"]["return"]["mean"]
        dream_mean = results["dream"]["return"]["mean"]
        gap = real_mean - dream_mean
        print(f"\nReal-Dream Gap: {gap:.2f} (Real: {real_mean:.2f}, Dream: {dream_mean:.2f})")

    print(f"\nOutput directory: {config.evaluation.output_dir}")
    print('='*50)


# Backward compatibility - keep old function signature
def evaluate_agent(config: WorldModelAgentConfig, num_episodes: int = 10):
    """
    Evaluate the trained agent in the real environment.

    Legacy function for backward compatibility. Use run_evaluation() for new code.

    Args:
        config: Configuration object
        num_episodes: Number of episodes to evaluate

    Returns:
        Mean episode return
    """
    logger = get_logger("world_models")

    # Temporarily override config
    original_num_episodes = config.evaluation.num_episodes
    original_eval_real = config.evaluation.eval_real
    original_eval_dream = config.evaluation.eval_dream
    original_save_video = config.evaluation.save_video

    config.evaluation.num_episodes = num_episodes
    config.evaluation.eval_real = True
    config.evaluation.eval_dream = False
    config.evaluation.save_video = False

    # Run evaluation
    results = run_evaluation(config)

    # Restore config
    config.evaluation.num_episodes = original_num_episodes
    config.evaluation.eval_real = original_eval_real
    config.evaluation.eval_dream = original_eval_dream
    config.evaluation.save_video = original_save_video

    return results["real"]["return"]["mean"]
