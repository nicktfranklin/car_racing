# World Model Agent for CarRacing

A complete implementation of the World Models architecture with FSQ-VAE and LSTM-based world modeling for the Gymnasium Car Racing environment.

## 🏗️ Architecture

This implementation is based on the [World Models paper](https://arxiv.org/abs/1803.10122) with these key modifications:

1. **FSQ-VAE** instead of standard VAE (from [Finite Scalar Quantization paper](https://arxiv.org/abs/2309.15505))
2. **LSTM with softmax** over state tokens instead of MDN-RNN
3. **Pydantic BaseModels** for all configurations

### Components

- **FSQ-VAE**: Encodes 64×64 RGB images into discrete 4D representations using Finite Scalar Quantization
- **World Model**: LSTM that predicts next state tokens, rewards, and done flags
- **Controller**: Neural network that maps state representations to actions

## 🚀 Installation

This project is now an installable Python package using [uv](https://docs.astral.sh/uv/) for dependency management.

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

### Package Installation Options

```bash
# Install base package
uv add world-models

# Install with development tools
uv add "world-models[dev]"

# Install with Jupyter notebook support
uv add "world-models[jupyter]"

# Install with documentation tools
uv add "world-models[docs]"

# Install everything
uv add "world-models[dev,jupyter,docs]"
```

### From Source

```bash
# Install in development mode
uv pip install -e .

# Or install from the source directory
uv pip install .
```

### Dependencies

- **Gymnasium**: RL environments including Car Racing
- **PyTorch**: Deep learning framework
- **FSQ-VAE**: Finite Scalar Quantization implementation
- **Pydantic**: Configuration management
- **H5PY**: Efficient data storage
- **Box2D & Pygame**: Physics and rendering

## 📊 Usage

### Quick Test

```bash
# Test all components
uv run python test_models.py

# Run example demonstration
uv run python example.py
```

### Training Pipeline

```bash
# 1. Collect random rollout data
uv run python main.py --stage collect

# 2. Train FSQ-VAE on collected images
uv run python main.py --stage vae

# 3. Train world model on sequences
uv run python main.py --stage world_model

# 4. Train controller using evolutionary strategy
uv run python main.py --stage controller

# 5. Or run the complete pipeline
uv run python main.py --stage all
```

### Configuration

All configurations are managed through Pydantic models:

```python
from world_models import WorldModelAgentConfig

config = WorldModelAgentConfig()
config.fsq_vae.fsq_levels = [8, 5, 5, 5]  # 1000 discrete codes
config.world_model.hidden_size = 256
config.controller.state_dim = 4  # FSQ dimensions
config.training.batch_size = 32
```

### Python API Usage

```python
from world_models import (
    WorldModelAgentConfig,
    FSQVAE,
    WorldModel,
    EvolutionaryController,
    DataCollector,
    VAETrainer,
    WorldModelTrainer,
    ControllerTrainer,
)

# Load configuration
config = WorldModelAgentConfig()

# Create models
vae = FSQVAE(config.fsq_vae)
world_model = WorldModel(config.world_model)
controller = EvolutionaryController(config.controller)

# Create trainers
vae_trainer = VAETrainer(vae, config)
world_model_trainer = WorldModelTrainer(world_model, vae, config)
controller_trainer = ControllerTrainer(vae, world_model, config)
```

## 🏛️ Architecture Details

### FSQ-VAE
- **Encoder**: CNN that compresses 64×64×3 images to 32D latents
- **FSQ Quantizer**: Maps 32D → 4D discrete representation with levels [8,5,5,5]
- **Decoder**: Reconstructs images from 4D quantized codes
- **Codebook Size**: 8×5×5×5 = 1000 discrete states

### World Model
- **Input**: Current state token + action
- **LSTM**: 256 hidden units, predicts next state distribution
- **Outputs**: Next state logits (1000-way softmax), reward, done flag
- **Training**: Cross-entropy loss on state prediction + MSE on rewards/dones

### Controller
- **Evolutionary**: Simple linear network optimized with evolutionary strategy
- **Standard**: Multi-layer perceptron (for gradient-based training)
- **Actions**: Steering [-1,1], Gas [0,1], Brake [0,1]

## 📁 Project Structure

```
├── src/world_models/          # Main package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Pydantic configuration models
│   ├── data_collection.py    # Environment data collection
│   ├── training.py           # Training loops for each component
│   └── models/               # Neural network models
│       ├── __init__.py
│       ├── fsq_vae.py       # FSQ-VAE implementation
│       ├── world_model.py   # LSTM world model
│       └── controller.py    # Controller networks
├── main.py                   # Main training pipeline script
├── test_models.py           # Component testing
├── example.py               # Usage demonstration
├── hello.py                 # Simple CarRacing demo
├── pyproject.toml           # Package configuration
├── LICENSE                  # MIT license
└── README.md               # This file
```

## 🎯 Key Features

- **Modular Design**: Each component can be trained independently
- **Configuration Management**: Type-safe configs with Pydantic
- **Efficient Storage**: HDF5 for large datasets
- **Resumable Training**: Checkpoint saving/loading
- **Multiple Controllers**: Both evolutionary and gradient-based

## 📈 Training Process

1. **Data Collection** (10k episodes):
   - Random agent rollouts in CarRacing-v3
   - Stored as HDF5 files with observations, actions, rewards

2. **VAE Training** (50 epochs):
   - Learn to encode/decode images with FSQ quantization
   - Reconstruct 64×64 RGB images with discrete latent codes

3. **World Model Training** (50 epochs):
   - Predict next state tokens from current state + action
   - Learn reward and termination prediction

4. **Controller Training** (100 generations):
   - Evolutionary strategy on controller parameters
   - Evaluate in learned world model environment

## 🔍 Model Statistics

- **VAE**: ~1.4M parameters
- **World Model**: ~0.9M parameters
- **Controller**: 15 parameters (evolutionary)
- **FSQ Codebook**: 1000 discrete states
- **State Representation**: 4 dimensions

## 🚗 Environment Details

**CarRacing-v3**:
- **Observation**: 96×96×3 RGB → resized to 64×64×3
- **Action Space**: [steering, gas, brake] ∈ ℝ³
- **Episode Length**: Up to 1000 steps
- **Reward**: Speed-based with penalties for leaving track

## 📊 Monitoring Training

Training progress is logged to the console with detailed metrics:
- **VAE**: Reconstruction loss, commitment loss, total loss
- **World Model**: State prediction accuracy, reward/done losses
- **Controller**: Population fitness statistics (best/mean)

All metrics are printed every epoch/generation for easy monitoring.

## 🧪 Testing

```bash
# Test all components work together
uv run python test_models.py

# Test the example demonstration
uv run python example.py

# Test package installation
uv run python -c "import world_models; print('✅ Package installed!')"

# Test original CarRacing demo
uv run python test_installation.py

# Test Hello World demo
uv run python hello.py
```

## 📦 Package Details

The project is now a proper Python package that can be installed via uv/pip:

- **Package Name**: `world-models`
- **Version**: `0.1.0`
- **Source Layout**: `src/world_models/`
- **Build System**: Hatchling
- **Dependencies**: Fully specified in `pyproject.toml`

### Package Installation Commands

```bash
# Development installation (editable)
uv pip install -e .

# Production installation
uv pip install .

# Install from PyPI (when published)
uv add world-models
```

## 📚 References

1. [World Models](https://arxiv.org/abs/1803.10122) - Original paper
2. [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505) - FSQ-VAE paper
3. [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/) - Environment docs

## 📄 License

MIT License - see LICENSE file for details.