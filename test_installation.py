#!/usr/bin/env python3
"""Test script to verify installations."""

import sys

def test_imports():
    """Test importing core packages."""
    try:
        import gymnasium as gym
        print("✅ Gymnasium imported successfully")

        # List available environments
        print("\nAvailable environments that might include Car Racing:")
        all_envs = list(gym.envs.registry.env_specs.keys()) if hasattr(gym.envs.registry, 'env_specs') else list(gym.envs.registry.keys())
        car_envs = [env for env in all_envs if 'car' in env.lower() or 'racing' in env.lower()]
        for env in car_envs:
            print(f"  - {env}")

    except ImportError as e:
        print(f"❌ Failed to import gymnasium: {e}")

    try:
        import torch
        import torchvision
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        print(f"✅ TorchVision {torchvision.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")

    try:
        import jupyter
        print("✅ Jupyter imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Jupyter: {e}")

def test_car_racing():
    """Test if Car Racing environment is available."""
    try:
        import gymnasium as gym
        env = gym.make('CarRacing-v3')
        print("✅ CarRacing-v3 environment created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        env.close()
    except Exception as e:
        print(f"❌ Failed to create CarRacing environment: {e}")
        print("   This likely means Box2D is not installed properly.")

if __name__ == "__main__":
    print("Testing installation...")
    print("=" * 50)
    test_imports()
    print()
    test_car_racing()