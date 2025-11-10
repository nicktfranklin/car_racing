"""
Training utilities and helpers for World Model components.
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
from .lightning_setup import (
    create_dataloaders,
    create_dataset,
    find_checkpoint_to_resume,
    setup_callbacks,
    setup_tensorboard,
    setup_trainer,
)

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
]
