# CarRacing

A Python project for working with the Gymnasium Car Racing environment using PyTorch.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Installation

1. Install dependencies:
```bash
# Install core dependencies
uv sync

# Install Box2D with proper C++ headers (macOS specific)
export CPPFLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1"
uv add box2d-py

# Add pygame for rendering
uv add pygame
```

### Dependencies

- **Gymnasium**: OpenAI Gym environments including Car Racing
- **PyTorch**: Deep learning framework
- **Jupyter**: Interactive notebooks
- **Box2D**: Physics engine for Car Racing environment
- **Pygame**: Rendering engine
- **Matplotlib**: Plotting and visualization

## Usage

### Python Script
```bash
uv run python hello.py
```

### Jupyter Notebook
```bash
uv run jupyter lab
```

Then open `car_racing_demo.ipynb` for an interactive demonstration.

### Test Installation
```bash
uv run python test_installation.py
```

## Environment Details

The Car Racing environment (`CarRacing-v3`) features:
- **Observation**: 96x96x3 RGB image
- **Action Space**: 3D continuous actions
  - Steering: -1.0 (left) to 1.0 (right)
  - Gas: 0.0 to 1.0
  - Brake: 0.0 to 1.0

## Files

- `hello.py`: Simple demonstration script
- `car_racing_demo.ipynb`: Interactive Jupyter notebook
- `test_installation.py`: Installation verification script