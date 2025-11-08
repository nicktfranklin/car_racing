"""
World Models: FSQ-VAE + LSTM implementation for CarRacing environment.

A complete implementation of the World Models architecture with FSQ-VAE
and LSTM-based world modeling.
"""

__version__ = "0.1.0"

from .agents import Agent, HumanAgent, RandomAgent, WorldModelAgent, create_agent
from .config import WorldModelAgentConfig
from .data_collection import DataCollector, ImageDataset, SequenceDataset
from .models import FSQVAE, Controller, EvolutionaryController, WorldModel
from .training import ControllerTrainer, VAETrainer, WorldModelTrainer
from .lightning_training import (
    VAELightningModule,
    WorldModelLightningModule,
    ChunkRotationCallback,
    create_train_val_dataloaders,
    create_sequence_train_val_dataloaders,
)

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
    "VAETrainer",
    "WorldModelTrainer",
    "ControllerTrainer",
    "VAELightningModule",
    "WorldModelLightningModule",
    "ChunkRotationCallback",
    "create_train_val_dataloaders",
    "create_sequence_train_val_dataloaders",
]
