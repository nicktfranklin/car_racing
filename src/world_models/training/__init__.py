"""
Training utilities and helpers for World Model components.

Import Architecture:
====================
This module provides a clean interface for training functions while maintaining
backward compatibility with legacy trainer classes.

Structure:
----------
- New modular training functions are in separate files:
  * data_collection.py - Data collection
  * train_vae.py - VAE training with Lightning
  * train_world_model.py - World Model training with Lightning
  * train_controller.py - Controller training with PPO
  * evaluation.py - Agent evaluation

- Legacy trainer classes (VAETrainer, WorldModelTrainer, ControllerTrainer) are
  imported from the parent training.py module for backward compatibility.

Import Policy:
--------------
All imports are at the top of files (module-level). No imports are allowed
inside functions or methods. This ensures clean, predictable import behavior
and makes dependencies explicit.
"""

# Legacy trainers (for backward compatibility)
import sys
from pathlib import Path

# Import from parent training.py module
training_module_path = Path(__file__).parent.parent / "training.py"
if training_module_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("world_models.training_legacy", training_module_path)
    training_legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training_legacy)

    ControllerTrainer = training_legacy.ControllerTrainer
    VAETrainer = training_legacy.VAETrainer
    WorldModelTrainer = training_legacy.WorldModelTrainer
else:
    # Fallback if training.py doesn't exist
    ControllerTrainer = None
    VAETrainer = None
    WorldModelTrainer = None

from .checkpoint_manager import CheckpointManager
from .data_collection import collect_data
from .evaluation import evaluate_agent
from .lightning_setup import (
    create_dataloaders,
    create_dataset,
    find_checkpoint_to_resume,
    setup_callbacks,
    setup_tensorboard,
    setup_trainer,
)
from .train_controller import train_controller
from .train_vae import train_vae
from .train_world_model import train_world_model

__all__ = [
    "CheckpointManager",
    "create_dataset",
    "create_dataloaders",
    "setup_callbacks",
    "setup_tensorboard",
    "setup_trainer",
    "find_checkpoint_to_resume",
    "ControllerTrainer",
    "VAETrainer",
    "WorldModelTrainer",
    "collect_data",
    "train_vae",
    "train_world_model",
    "train_controller",
    "evaluate_agent",
]
