"""
World Models: FSQ-VAE + LSTM implementation for CarRacing environment.

A complete implementation of the World Models architecture with FSQ-VAE
and LSTM-based world modeling.
"""

__version__ = "0.1.0"

from .config import WorldModelAgentConfig
from .data_collection import DataCollector, ImageDataset, SequenceDataset
from .models import FSQVAE, Controller, EvolutionaryController, WorldModel
from .training import ControllerTrainer, VAETrainer, WorldModelTrainer

__all__ = [
    "WorldModelAgentConfig",
    "FSQVAE",
    "WorldModel",
    "Controller",
    "EvolutionaryController",
    "DataCollector",
    "ImageDataset",
    "SequenceDataset",
    "VAETrainer",
    "WorldModelTrainer",
    "ControllerTrainer",
]
