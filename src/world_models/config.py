"""
Configuration models for World Model agent with FSQ-VAE and LSTM.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FSQVAEConfig(BaseModel):
    """Configuration for Finite Scalar Quantization VAE."""

    model_config = ConfigDict(extra="forbid")

    # Image dimensions
    input_channels: int = Field(default=3, description="Number of input channels")
    input_height: int = Field(default=64, description="Input image height")
    input_width: int = Field(default=64, description="Input image width")

    # Encoder architecture
    encoder_channels: List[int] = Field(
        default=[32, 64, 128, 256], description="Channel sizes for encoder layers"
    )
    encoder_strides: List[int] = Field(
        default=[2, 2, 2, 2], description="Stride for each encoder layer"
    )

    # FSQ quantization parameters
    fsq_levels: List[int] = Field(
        default=[8, 5, 5, 5],
        description="Quantization levels for each dimension (e.g., [8,5,5,5] = 1000 codes)",
    )
    latent_dim: int = Field(
        default=32, description="Latent dimension before FSQ quantization"
    )

    # Decoder architecture
    decoder_channels: List[int] = Field(
        default=[256, 128, 64, 32], description="Channel sizes for decoder layers"
    )
    decoder_strides: List[int] = Field(
        default=[2, 2, 2, 2], description="Stride for each decoder layer"
    )

    # Training parameters
    learning_rate: float = Field(default=1e-3, description="Learning rate")
    beta: float = Field(default=1.0, description="KL divergence weight")


class WorldModelConfig(BaseModel):
    """Configuration for LSTM-based world model."""

    model_config = ConfigDict(extra="forbid")

    # Architecture
    hidden_size: int = Field(default=256, description="LSTM hidden size")
    num_layers: int = Field(default=1, description="Number of LSTM layers")
    dropout: float = Field(default=0.0, description="Dropout rate")

    # Input/Output dimensions
    state_dim: int = Field(
        default=4, description="State representation dimension (FSQ dimensions)"
    )
    action_dim: int = Field(default=3, description="Action dimension")

    # FSQ parameters (must match FSQVAEConfig)
    fsq_levels: List[int] = Field(
        default=[8, 5, 5, 5], description="FSQ levels for state prediction"
    )

    # Training parameters
    learning_rate: float = Field(default=1e-3, description="Learning rate")
    sequence_length: int = Field(default=50, description="Training sequence length")


class ControllerConfig(BaseModel):
    """Configuration for the controller network."""

    model_config = ConfigDict(extra="forbid")

    # Architecture
    hidden_sizes: List[int] = Field(
        default=[256, 256], description="Hidden layer sizes"
    )
    activation: str = Field(default="tanh", description="Activation function")

    # Input/Output dimensions
    state_dim: int = Field(
        default=4, description="State representation dimension (FSQ dimensions)"
    )
    action_dim: int = Field(default=3, description="Action dimension")

    # Training parameters
    learning_rate: float = Field(default=1e-3, description="Learning rate")


class DataConfig(BaseModel):
    """Configuration for data collection and processing."""

    model_config = ConfigDict(extra="forbid")

    # Environment
    env_name: str = Field(
        default="CarRacing-v3", description="Gymnasium environment name"
    )
    render_mode: str = Field(default="rgb_array", description="Rendering mode")

    # Data collection
    num_rollouts: int = Field(
        default=10000, description="Number of rollouts to collect"
    )
    max_episode_length: int = Field(default=1000, description="Maximum episode length")

    # Parallel collection
    num_workers: int = Field(
        default=-1, description="Number of parallel workers (-1 for auto)"
    )
    batch_size: int = Field(default=100, description="Episodes per worker batch")

    # Data processing
    frame_skip: int = Field(default=4, description="Frame skip for data collection")
    frame_stack: int = Field(default=1, description="Number of frames to stack")

    # Storage
    data_dir: str = Field(default="./data", description="Directory to save data")


class TrainingConfig(BaseModel):
    """Configuration for training pipeline."""

    model_config = ConfigDict(extra="forbid")

    # General training
    device: str = Field(default="cuda", description="Training device")
    batch_size: int = Field(default=32, description="Batch size")
    num_epochs: int = Field(default=100, description="Number of training epochs")

    # Stage-wise training
    train_vae_epochs: int = Field(default=50, description="VAE training epochs")
    train_world_model_epochs: int = Field(
        default=50, description="World model training epochs"
    )
    train_controller_epochs: int = Field(
        default=100, description="Controller training epochs"
    )

    # Evaluation
    eval_every: int = Field(default=10, description="Evaluate every N epochs")
    save_every: int = Field(default=25, description="Save checkpoint every N epochs")

    # Logging
    log_every: int = Field(default=100, description="Log every N steps")

    # Paths
    checkpoint_dir: str = Field(
        default="./checkpoints", description="Checkpoint directory"
    )


class WorldModelAgentConfig(BaseModel):
    """Main configuration combining all components."""

    model_config = ConfigDict(extra="forbid")

    fsq_vae: FSQVAEConfig = Field(default_factory=FSQVAEConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    def validate_consistency(self) -> None:
        """Validate consistency between component configurations."""
        # Check FSQ levels consistency
        if self.fsq_vae.fsq_levels != self.world_model.fsq_levels:
            raise ValueError("FSQ levels must match between VAE and World Model")

        # Check state dimensions
        if len(self.fsq_vae.fsq_levels) != self.world_model.state_dim:
            raise ValueError("Number of FSQ levels must match world_model state_dim")

        if self.world_model.state_dim != self.controller.state_dim:
            raise ValueError("World model state_dim must match controller state_dim")

        # Check action dimensions
        if self.world_model.action_dim != self.controller.action_dim:
            raise ValueError(
                "Action dimensions must match between world model and controller"
            )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "WorldModelAgentConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)

    def set_fsq_codebook_size(self, codebook_size: int) -> None:
        """Set FSQ codebook size by adjusting levels to achieve target size."""
        # Find optimal level distribution for target codebook size
        levels = self._compute_fsq_levels(codebook_size)
        self.fsq_vae.fsq_levels = levels
        self.world_model.fsq_levels = levels
        # Update dimensions
        self.world_model.state_dim = len(levels)
        self.controller.state_dim = len(levels)

    def _compute_fsq_levels(self, target_size: int) -> List[int]:
        """Compute FSQ levels to achieve approximately target codebook size."""
        if target_size <= 0:
            raise ValueError("Codebook size must be positive")

        # Common FSQ level configurations that work well
        if target_size <= 64:
            return [8, 8]  # 64 codes
        elif target_size <= 125:
            return [5, 5, 5]  # 125 codes
        elif target_size <= 256:
            return [8, 8, 4]  # 256 codes
        elif target_size <= 512:
            return [8, 8, 8]  # 512 codes
        elif target_size <= 1000:
            return [8, 5, 5, 5]  # 1000 codes
        elif target_size <= 2048:
            return [8, 8, 8, 4]  # 2048 codes
        elif target_size <= 4096:
            return [8, 8, 8, 8]  # 4096 codes
        else:
            # For larger sizes, use 5 dimensions with high levels
            import math

            level = int(math.ceil(target_size ** (1 / 5)))
            return [level] * 5


if __name__ == "__main__":
    # Test configuration creation and validation
    config = WorldModelAgentConfig()
    config.validate_consistency()
    print("Configuration validation passed!")
    print(f"FSQ codebook size: {config.fsq_vae.fsq_levels}")
    print(f"Total codes: {sum(config.fsq_vae.fsq_levels)}")
