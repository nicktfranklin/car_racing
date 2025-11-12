"""
Training utilities and helpers for World Model components.

Import Architecture:
====================
This module provides a clean interface for training functions.

Structure:
----------
- Modular training functions in separate files:
  * data_collection.py - Data collection
  * train_vae.py - VAE training with Lightning
  * train_world_model.py - World Model training with Lightning
  * train_controller.py - Controller training with PPO
  * evaluation.py - Agent evaluation

- ControllerTrainer is imported from the parent training.py module
  (VAE and World Model use Lightning modules instead of trainer classes)

Import Policy:
--------------
All imports are at the top of files (module-level). No imports are allowed
inside functions or methods. This ensures clean, predictable import behavior
and makes dependencies explicit.
"""

from .checkpoint_manager import CheckpointManager
from .controller_trainer import ControllerTrainer
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
    "collect_data",
    "train_vae",
    "train_world_model",
    "train_controller",
    "evaluate_agent",
]
