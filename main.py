"""
Main training pipeline for World Model agent.
"""

import argparse
import logging
import os
import sys

from src.world_models.training import (
    collect_data,
    train_controller,
    train_vae,
    train_world_model,
)
from world_models import WorldModelAgentConfig, setup_logger, setup_output_logging


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


if __name__ == "__main__":
    main()
