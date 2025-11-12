"""
Configuration models for World Model agent with FSQ-VAE and LSTM.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


class OptimizerConfig(BaseModel):
    """Configuration for optimizer and learning rate scheduling."""

    model_config = ConfigDict(extra="forbid")

    # Optimizer type
    optimizer: str = Field(
        default="adam", description="Optimizer type (adam, adamw, sgd)"
    )

    # Learning rate
    learning_rate: float = Field(default=1e-3, description="Initial learning rate")

    # Adam/AdamW parameters
    beta1: float = Field(default=0.9, description="Adam beta1 parameter")
    beta2: float = Field(default=0.999, description="Adam beta2 parameter")
    epsilon: float = Field(default=1e-8, description="Adam epsilon parameter")
    weight_decay: float = Field(
        default=0.0, description="Weight decay (L2 regularization)"
    )
    amsgrad: bool = Field(
        default=False, description="Whether to use AMSGrad variant of Adam"
    )

    # SGD parameters
    momentum: float = Field(default=0.9, description="SGD momentum")
    nesterov: bool = Field(
        default=False, description="Whether to use Nesterov momentum"
    )

    # Learning rate scheduler
    use_scheduler: bool = Field(
        default=False, description="Whether to use LR scheduler"
    )
    scheduler: str = Field(
        default="cosine",
        description="LR scheduler type (cosine, step, exponential, reduce_on_plateau)",
    )

    # Scheduler parameters
    warmup_epochs: int = Field(default=0, description="Number of warmup epochs")
    min_lr: float = Field(
        default=1e-6, description="Minimum learning rate for schedulers"
    )

    # Step scheduler
    step_size: int = Field(default=30, description="Step size for StepLR")
    gamma: float = Field(
        default=0.1, description="Multiplicative factor for StepLR/ExponentialLR"
    )

    # ReduceLROnPlateau
    patience: int = Field(default=10, description="Patience for ReduceLROnPlateau")
    factor: float = Field(default=0.5, description="Factor for ReduceLROnPlateau")

    # Gradient clipping
    grad_clip_norm: Optional[float] = Field(
        default=None, description="Max norm for gradient clipping (None = no clipping)"
    )


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
    learning_rate: float = Field(
        default=1e-3, description="Learning rate (deprecated, use optimizer config)"
    )
    beta: float = Field(default=1.0, description="Commitment loss weight")

    # Perceptual loss
    use_perceptual_loss: bool = Field(
        default=True, description="Whether to use perceptual loss (VGG-based)"
    )
    perceptual_weight: float = Field(
        default=1.0, description="Weight for perceptual loss"
    )
    mse_weight: float = Field(
        default=0.1, description="Weight for MSE loss when using perceptual loss"
    )

    # Codebook diversity (anti-collapse)
    entropy_weight: float = Field(
        default=0.1,
        description="Weight for entropy regularization to encourage diverse code usage",
    )

    # Optimizer configuration
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


class WorldModelConfig(BaseModel):
    """Configuration for transformer-based world model."""

    model_config = ConfigDict(extra="forbid")

    # Architecture
    hidden_size: int = Field(default=256, description="Embedding dimension")
    dropout: float = Field(default=0.1, description="Dropout rate")

    # Transformer parameters
    n_layers: int = Field(default=6, description="Number of transformer layers")
    n_heads: int = Field(default=8, description="Number of attention heads")
    num_layers: int = Field(
        default=6, description="Alias for n_layers (for compatibility)"
    )

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
    learning_rate: float = Field(
        default=3e-4, description="Learning rate (deprecated, use optimizer config)"
    )
    sequence_length: int = Field(default=64, description="Training sequence length")

    # Optimizer configuration
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


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
    learning_rate: float = Field(
        default=1e-3, description="Learning rate (deprecated, use optimizer config)"
    )

    # Optimizer configuration
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


class DataConfig(BaseModel):
    """Configuration for data collection and processing."""

    model_config = ConfigDict(extra="forbid")

    # Environment
    env_name: str = Field(
        default="CarRacing-v3", description="Gymnasium environment name"
    )
    render_mode: Optional[str] = Field(
        default=None, description="Rendering mode (None for fastest collection)"
    )

    # Data collection
    num_rollouts: int = Field(
        default=10000, description="Number of rollouts to collect"
    )
    max_episode_length: int = Field(default=1000, description="Maximum episode length")

    # Parallel collection
    num_workers: int = Field(
        default=-1, description="Number of parallel workers (-1 for auto)"
    )
    batch_size: int = Field(
        default=100, description="Episodes per worker batch (deprecated)"
    )
    episodes_per_batch: int = Field(
        default=10,
        description="Number of episodes each worker collects before returning (higher = less overhead)",
    )

    # Checkpointing
    checkpoint_every: int = Field(
        default=100, description="Save checkpoint every N episodes"
    )

    # Data processing
    frame_skip: int = Field(default=4, description="Frame skip for data collection")
    frame_stack: int = Field(default=1, description="Number of frames to stack")

    # Storage
    data_dir: str = Field(default="./data", description="Directory to save data")


class TrainingConfig(BaseModel):
    """Configuration for training pipeline."""

    model_config = ConfigDict(extra="forbid")

    # General training
    device: str = Field(default="mps", description="Training device")
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

    # Lightning-specific training parameters
    steps_per_epoch: int = Field(
        default=1000, description="Number of batches per epoch (for random sampling)"
    )
    val_samples: int = Field(default=500, description="Number of validation samples")
    val_split: float = Field(
        default=0.05, description="Fraction of data to use for validation"
    )
    early_stopping_patience: int = Field(
        default=10, description="Early stopping patience in epochs"
    )
    num_dataloader_workers: int = Field(
        default=4, description="Number of dataloader workers"
    )
    subsample_rate: int = Field(
        default=1, description="Subsample rate for image dataset (use every Nth frame)"
    )

    # VAE dataset parameters
    vae_subsample_rate: int = Field(
        default=10,
        description="Subsample rate for VAE (use every Nth frame for decorrelation)",
    )
    vae_files_per_chunk: int = Field(
        default=5, description="Number of files to load per chunk for VAE training"
    )

    # World Model dataset parameters
    world_model_subsample_rate: int = Field(
        default=4, description="Subsample rate for World Model (use every Nth frame)"
    )
    world_model_files_per_chunk: int = Field(
        default=5,
        description="Number of files to load per chunk for World Model training",
    )
    world_model_sequence_length: int = Field(
        default=64, description="Sequence length for World Model training"
    )

    # World Model specific parameters
    world_model_batch_size: int = Field(
        default=32,
        description="Batch size for world model training (lower than VAE due to sequences)",
    )
    world_model_steps_per_epoch: int = Field(
        default=200, description="Number of batches per epoch for world model training"
    )
    world_model_val_samples: int = Field(
        default=200, description="Number of validation samples for world model"
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
    log_file: Optional[str] = Field(
        default=None,
        description="Optional file to log stdout/stderr (None = no file logging)",
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


