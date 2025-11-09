# World Model Agent for CarRacing

A complete implementation of the World Models architecture with FSQ-VAE and GPT-2 transformer for the Gymnasium Car Racing environment.

## 🏗️ Architecture

This implementation is losely based on the [World Models paper](https://arxiv.org/abs/1803.10122) with these key modifications:

1. **FSQ-VAE** instead of standard VAE (from [Finite Scalar Quantization paper](https://arxiv.org/abs/2309.15505))
2. **GPT-2 Transformer** with autoregressive FSQ token prediction instead of MDN-RNN

### Components

- **FSQ-VAE**: Encodes 64×64 RGB images into discrete 4D representations using Finite Scalar Quantization
  - Spatial attention pooling for global feature aggregation
  - Perceptual loss (VGG-based) for better reconstruction quality
  - Entropy regularization to prevent codebook collapse

- **World Model**: GPT-2 transformer that autoregressively predicts next state tokens, rewards, and done flags
  - Each FSQ dimension predicted separately in sequence
  - Captures dependencies between FSQ dimensions
  - Sequence: `[s_0^0, s_0^1, s_0^2, s_0^3, a_0, s_1^0, s_1^1, s_1^2, s_1^3, a_1, ...]`

- **Controller**: Neural network that maps state representations to actions

## 🚀 Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Quick Installation

```bash
# Clone the repository
git clone <repository-url>
cd CarRacing

# Install the package in development mode with all dependencies
uv sync

# For Box2D on macOS (if needed)
export CPPFLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1"
uv sync --reinstall-package box2d-py
```

### From Source

```bash
# Install in development mode
uv pip install -e .

# Or install from the source directory
uv pip install .
```

### Dependencies

- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: Training framework with TensorBoard integration
- **Transformers (HuggingFace)**: GPT-2 model implementation
- **Gymnasium**: RL environments including Car Racing
- **H5PY**: Efficient data storage for large datasets
- **Pydantic**: Type-safe configuration management
- **Box2D & Pygame**: Physics and rendering

## 📊 Usage

### Training Pipeline

The complete training pipeline consists of 4 stages:

```bash
# 1. Collect random rollout data (parallel workers for speed)
uv run python main.py --stage collect --num-rollouts 10000 --workers 8

# 2. Train FSQ-VAE on collected images
uv run python main.py --stage vae

# 3. Train world model on sequences
uv run python main.py --stage world_model

# 4. Train controller using evolutionary strategy
uv run python main.py --stage controller

# 5. Or run the complete pipeline
uv run python main.py
```

### Data Collection Options

```bash
# Fast collection with parallel workers
uv run python main.py --stage collect --num-rollouts 10000 --workers 8 --no-render

# Resume from checkpoint (automatic detection)
uv run python main.py --stage collect --num-rollouts 10000 --checkpoint-every 100

# Custom episode length
uv run python main.py --stage collect --max-episode-length 2000
```

### Configuration

All configurations are managed through YAML files with Pydantic validation:

```bash
# Use custom configuration
uv run python main.py --config configs/config.yaml --stage all

# Override specific parameters
uv run python main.py --num-rollouts 1000 --fsq-codebook-size 256

# Resume training from checkpoint
uv run python main.py --stage vae --resume

# Use specific device
uv run python main.py --device cuda --stage all
```

### TensorBoard Monitoring

Training metrics are automatically logged to TensorBoard:

```bash
# Start TensorBoard server
tensorboard --logdir ./checkpoints

# View metrics at http://localhost:6006
# - VAE: reconstruction loss, perceptual loss, codebook usage, perplexity
# - World Model: FSQ accuracy per dimension, reward/done losses
```

### Python API Usage

```python
from world_models import (
    WorldModelAgentConfig,
    FSQVAE,
    WorldModel,
    VAEDataset,
    WorldModelDataset,
    VAELightningModule,
    WorldModelLightningModule,
)
import lightning as L

# Load configuration
config = WorldModelAgentConfig.from_yaml("configs/config.yaml")

# Create models
vae = FSQVAE(config.fsq_vae, use_perceptual_loss=True, device="cuda")
world_model = WorldModel(config.world_model)

# Create Lightning modules
vae_module = VAELightningModule(vae, config)
world_model_module = WorldModelLightningModule(world_model, vae, config)

# Create datasets with sequential chunk loading
vae_dataset = VAEDataset(
    data_dir="./data",
    chunk_files=chunk_files,
    subsample_rate=16,  # Decorrelation
    files_per_chunk=5,   # Memory management
)

world_model_dataset = WorldModelDataset(
    data_dir="./data",
    chunk_files=chunk_files,
    sequence_length=16,
    subsample_rate=4,
    files_per_chunk=5,
)
```

## 🏛️ Architecture Details

### FSQ-VAE

**Encoder Architecture**:
- 5-layer CNN: `[32, 64, 128, 256, 512]` channels
- Stride-2 convolutions: 64×64 → 2×2 feature maps
- Spatial attention pooling: aggregates 2×2×512 → 128D vector
- Linear projection: 128D → 4D FSQ representation

**FSQ Quantization**:
- Levels: `[8, 8, 8, 4]` → 2048 discrete codes
- Each dimension quantized independently to `{-1, -1+2/L, ..., 1}`
- Straight-through estimator for gradient flow

**Decoder Architecture**:
- Linear projection: 4D → 2×2×512 spatial features
- 5-layer transpose CNN: mirrors encoder
- Sigmoid output: reconstructed 64×64×3 RGB image

**Training**:
- Perceptual loss (VGG-based) + MSE reconstruction
- Commitment loss: encourages encoder to commit to quantized values
- Entropy regularization: prevents codebook collapse

### World Model (GPT-2 Transformer)

**Architecture**:
- 6 transformer layers, 8 attention heads
- Hidden size: 256, FFN dimension: 1024
- Position embeddings: up to 160 tokens (16 timesteps × 10 tokens)

**Token Sequence Structure**:
```
[s_0^0, s_0^1, s_0^2, s_0^3, a_0, s_1^0, s_1^1, s_1^2, s_1^3, a_1, ...]
```

Where:
- `s_t^i`: FSQ dimension i at timestep t (4 dimensions per state)
- `a_t`: Action at timestep t (continuous, embedded)
- Each timestep = 5 tokens (4 FSQ + 1 action)

**Autoregressive Prediction**:
- Each FSQ dimension has its own embedding and output head
- `s_t^0` predicted after `a_{t-1}` (depends on history)
- `s_t^1` predicted after `s_t^0` (depends on history + dim 0)
- `s_t^2` predicted after `s_t^1` (depends on history + dims 0-1)
- `s_t^3` predicted after `s_t^2` (depends on history + dims 0-2)
- Captures dependencies between FSQ dimensions

**Training**:
- Cross-entropy loss per FSQ dimension (4 separate losses)
- MSE loss for rewards
- Binary cross-entropy for done flags
- Total loss: average FSQ loss + reward loss + done loss

**Metrics Logged**:
- Per-dimension accuracy and loss (4 dimensions)
- Overall FSQ accuracy (average)
- Reward and done prediction losses

### Data Loading Strategy

**VAE Training**:
- Sequential chunk loading: 5 files at a time (~6GB)
- Subsampling: use every 16th frame for decorrelation
- Effective dataset: 20M → 1.25M images
- Chunk rotation every epoch for diversity

**World Model Training**:
- Sequential chunk loading: 5 files at a time (~6GB)
- Subsampling: use every 4th frame (need consecutive frames)
- Sequence length: 16 timesteps
- Sequences extracted within episode boundaries

**Benefits**:
- Memory efficient: only 6GB RAM regardless of dataset size
- I/O optimized: pure sequential HDF5 reads
- Scalable: works with 20M+ image datasets

## 📁 Project Structure

```
├── src/world_models/              # Main package
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Pydantic configuration models
│   ├── data_collection.py        # Datasets and data collection
│   ├── lightning_training.py     # Lightning modules
│   ├── training.py               # Legacy trainers
│   ├── agents.py                 # Agent implementations
│   └── models/                   # Neural network models
│       ├── __init__.py
│       ├── fsq_vae.py           # FSQ-VAE with attention pooling
│       ├── world_model.py       # GPT-2 transformer
│       └── controller.py        # Controller networks
├── configs/
│   └── config.yaml              # Training configuration
├── main.py                       # Main training pipeline
├── pyproject.toml               # Package configuration
└── README.md                    # This file
```

## 🎯 Key Features

- **Autoregressive FSQ Prediction**: Captures dependencies between codebook dimensions
- **Memory-Efficient Training**: Sequential chunk loading handles 20M+ images
- **Lightning Framework**: Clean training with TensorBoard logging and checkpointing
- **Parallel Data Collection**: Multi-process rollout collection
- **Modular Design**: Each component can be trained independently
- **Type-Safe Configuration**: Pydantic models with YAML support
- **Automatic Checkpointing**: Resume training from interruptions

## 📈 Training Process

### 1. Data Collection (10k episodes, ~30 minutes with 8 workers)

```bash
uv run python main.py --stage collect --num-rollouts 10000 --workers 8 --no-render
```

- Parallel random agent rollouts in CarRacing-v3
- Stored as chunked HDF5 files (100 episodes per file)
- Total: ~20M images (240GB if loaded all at once, stored as 2.4GB compressed)
- Checkpointing: saves progress every 100 episodes

### 2. VAE Training (50 epochs, ~2 hours on GPU)

```bash
uv run python main.py --stage vae
```

- Sequential chunk loading (5 files = 500K images = 6GB RAM)
- Subsample rate: 16 (effective 1.25M training images)
- Perceptual loss + MSE for better reconstruction
- Entropy regularization prevents codebook collapse
- Metrics: reconstruction loss, codebook perplexity, usage statistics

### 3. World Model Training (50 epochs, ~3 hours on GPU)

```bash
uv run python main.py --stage world_model
```

- Sequential chunk loading (5 files = ~93K sequences = 6GB RAM)
- Sequence length: 16 timesteps (80 tokens after interleaving)
- Subsample rate: 4 (consecutive frames with some temporal skip)
- Autoregressive FSQ prediction (4 dimensions per timestep)
- Metrics: per-dimension accuracy, FSQ loss, reward/done losses

### 4. Controller Training (100 generations, ~1 hour)

```bash
uv run python main.py --stage controller
```

- Evolutionary strategy on controller parameters
- Evaluate in learned world model environment
- Population-based optimization

## 🔍 Model Statistics

- **VAE**: ~2.5M parameters
  - Encoder: 5-layer CNN with attention pooling
  - Decoder: 5-layer transpose CNN

- **World Model**: ~4.2M parameters
  - 6-layer transformer (256D hidden, 8 heads)
  - 4 separate FSQ embeddings (levels: 8, 8, 8, 4)
  - 4 separate FSQ output heads

- **FSQ Codebook**: 2048 discrete states (8×8×8×4)
- **State Representation**: 4 dimensions, predicted autoregressively

## 🚗 Environment Details

**CarRacing-v3**:
- **Observation**: 96×96×3 RGB → resized to 64×64×3
- **Action Space**: [steering, gas, brake] ∈ ℝ³
  - Steering: -1 (left) to +1 (right)
  - Gas: 0 to 1
  - Brake: 0 to 1
- **Episode Length**: Up to 2000 steps (configurable)
- **Reward**: Speed-based with penalties for leaving track

## 📊 Configuration Options

### Key Config Parameters

```yaml
fsq_vae:
  fsq_levels: [8, 8, 8, 4]          # 2048 codebook size
  latent_dim: 128                    # Before FSQ projection
  encoder_channels: [32, 64, 128, 256, 512]
  use_perceptual_loss: true
  perceptual_weight: 1.0
  mse_weight: 0.05
  entropy_weight: 0.1                # Anti-collapse

world_model:
  hidden_size: 256
  n_layers: 6
  n_heads: 8
  fsq_levels: [8, 8, 8, 4]          # Must match VAE

training:
  device: cuda
  batch_size: 256                    # VAE
  world_model_batch_size: 32         # Smaller for sequences

  # VAE dataset
  vae_subsample_rate: 16             # Every 16th frame
  vae_files_per_chunk: 5             # 5 files = ~6GB

  # World Model dataset
  world_model_subsample_rate: 4      # Every 4th frame
  world_model_files_per_chunk: 5
  world_model_sequence_length: 16    # 16 timesteps = 80 tokens

  # Training
  train_vae_epochs: 1000
  train_world_model_epochs: 1000
  early_stopping_patience: 10
```

## 🧪 Testing

```bash
# Test package installation
uv run python -c "import world_models; print('✅ Package installed!')"

# Test data collection
uv run python main.py --stage collect --num-rollouts 10 --workers 2

# Test VAE training (small dataset)
uv run python main.py --stage vae --num-epochs 2
```

## 📚 References

1. [World Models](https://arxiv.org/abs/1803.10122) - Ha & Schmidhuber, 2018
2. [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505) - Mentzer et al., 2023
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
4. [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/) - Gymnasium docs

## 📄 License

MIT License - see LICENSE file for details.
