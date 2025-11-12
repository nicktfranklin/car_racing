"""
World Models: FSQ-VAE + LSTM implementation for CarRacing environment.

A complete implementation of the World Models architecture with FSQ-VAE
and LSTM-based world modeling.
"""

__version__ = "0.1.0"

from .agents import Agent, HumanAgent, RandomAgent, WorldModelAgent, create_agent
from .config import WorldModelAgentConfig
from .lightning_training import (
    ChunkRotationCallback,
    VAELightningModule,
    WorldModelLightningModule,
    create_sequence_train_val_dataloaders,
    create_train_val_dataloaders,
)
from .models import FSQVAE, Controller, EvolutionaryController, WorldModel
from .training.controller_trainer import ControllerTrainer
from .training.data_collection import DataCollector
from .training.datasets import (
    ImageDataset,
    SequenceDataset,
    VAEDataset,
    WorldModelDataset,
)
from .utils import get_logger, setup_logger, setup_output_logging

__all__ = [
    "Agent",
    "RandomAgent",
    "HumanAgent",
    "WorldModelAgent",
    "create_agent",
    "WorldModelAgentConfig",
    "FSQVAE",
    "WorldModel",
    "Controller",
    "EvolutionaryController",
    "DataCollector",
    "ImageDataset",
    "SequenceDataset",
    "VAEDataset",
    "WorldModelDataset",
    "ControllerTrainer",
    "VAELightningModule",
    "WorldModelLightningModule",
    "ChunkRotationCallback",
    "create_train_val_dataloaders",
    "create_sequence_train_val_dataloaders",
    "setup_logger",
    "get_logger",
    "setup_output_logging",
]
