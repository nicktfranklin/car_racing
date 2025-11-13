#!/usr/bin/env python
"""
Evaluation script for World Model agent.

Uses config file for defaults with command-line override support.
Evaluates trained agents in real and/or dream (imagined) environments.
"""

import argparse
import logging

from src.world_models.inference import run_evaluation
from world_models import WorldModelAgentConfig, setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate World Model Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all config defaults
  python eval.py --config configs/config.yaml

  # Evaluate in both real and dream environments
  python eval.py --config configs/config.yaml --mode both

  # Quick test with rendering
  python eval.py --config configs/test_config.yaml --mode real --render

  # Generate videos with custom number of episodes
  python eval.py --config configs/config.yaml --num-episodes 20 --video

  # Dream evaluation only with seed
  python eval.py --config configs/config.yaml --mode dream --seed 42
        """
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file (default: configs/config.yaml)"
    )

    # Evaluation mode overrides
    parser.add_argument(
        "--mode",
        type=str,
        choices=["real", "dream", "both"],
        default=None,
        help="Evaluation mode: 'real' (CarRacing env), 'dream' (world model), or 'both' (overrides config)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to evaluate (overrides config)"
    )

    # Video recording overrides
    parser.add_argument(
        "--video",
        action="store_true",
        default=None,
        help="Enable video recording (overrides config)"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording (overrides config)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=None,
        help="Enable real-time rendering (sets render_mode='human', overrides config)"
    )

    # Path overrides
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for videos and metrics (overrides config)"
    )

    # Other overrides
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default=None,
        help="Device to use for inference (overrides config)"
    )

    args = parser.parse_args()

    # Initialize logger
    logger = setup_logger("world_models", level=logging.INFO)

    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = WorldModelAgentConfig.from_yaml(args.config)

    # Apply command-line overrides
    logger.info("Applying command-line overrides...")

    if args.mode is not None:
        config.evaluation.eval_real = args.mode in ["real", "both"]
        config.evaluation.eval_dream = args.mode in ["dream", "both"]
        logger.info(f"  Mode: real={config.evaluation.eval_real}, dream={config.evaluation.eval_dream}")

    if args.num_episodes is not None:
        config.evaluation.num_episodes = args.num_episodes
        logger.info(f"  Episodes: {config.evaluation.num_episodes}")

    if args.video:
        config.evaluation.save_video = True
        logger.info("  Video: enabled")
    elif args.no_video:
        config.evaluation.save_video = False
        logger.info("  Video: disabled")

    if args.render:
        config.evaluation.render_mode = "human"
        logger.info("  Render mode: human")

    if args.output_dir is not None:
        config.evaluation.output_dir = args.output_dir
        logger.info(f"  Output directory: {config.evaluation.output_dir}")

    if args.seed is not None:
        config.evaluation.seed = args.seed
        logger.info(f"  Seed: {config.evaluation.seed}")

    if args.device is not None:
        if args.device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        config.training.device = device
        logger.info(f"  Device: {config.training.device}")

    # Print evaluation settings
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SETTINGS")
    logger.info("="*50)
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {config.training.device}")
    logger.info(f"Episodes: {config.evaluation.num_episodes}")
    logger.info(f"Max episode length: {config.evaluation.max_episode_length}")
    logger.info(f"Real environment: {config.evaluation.eval_real}")
    logger.info(f"Dream environment: {config.evaluation.eval_dream}")
    logger.info(f"Save video: {config.evaluation.save_video}")
    logger.info(f"Render mode: {config.evaluation.render_mode}")
    logger.info(f"Output directory: {config.evaluation.output_dir}")
    if config.evaluation.seed is not None:
        logger.info(f"Seed: {config.evaluation.seed}")
    logger.info("="*50 + "\n")

    # Run evaluation
    try:
        results = run_evaluation(config)
        logger.info("\nEvaluation completed successfully!")
        return results
    except Exception as e:
        logger.error(f"\nEvaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
