"""
Metrics collection and saving for evaluation.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class MetricsCollector:
    """Collect and compute metrics during evaluation."""

    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []
        self.returns: List[float] = []
        self.lengths: List[int] = []

    def add_episode(
        self,
        episode_return: float,
        episode_length: int,
        actions: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Add an episode's metrics.

        Args:
            episode_return: Total reward for the episode
            episode_length: Number of steps in the episode
            actions: Optional array of actions taken
            **kwargs: Additional metrics to track
        """
        self.returns.append(episode_return)
        self.lengths.append(episode_length)

        episode_data = {
            "episode": len(self.episodes),
            "return": float(episode_return),
            "length": int(episode_length),
        }

        # Add any additional metrics
        for key, value in kwargs.items():
            episode_data[key] = value

        self.episodes.append(episode_data)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with mean, std, min, max for returns and lengths
        """
        if len(self.returns) == 0:
            return {
                "return": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "length": {"mean": 0, "std": 0, "min": 0, "max": 0},
                "num_episodes": 0,
                "episodes": []
            }

        returns_array = np.array(self.returns)
        lengths_array = np.array(self.lengths)

        return {
            "return": {
                "mean": float(np.mean(returns_array)),
                "std": float(np.std(returns_array)),
                "min": float(np.min(returns_array)),
                "max": float(np.max(returns_array)),
            },
            "length": {
                "mean": float(np.mean(lengths_array)),
                "std": float(np.std(lengths_array)),
                "min": int(np.min(lengths_array)),
                "max": int(np.max(lengths_array)),
            },
            "num_episodes": len(self.episodes),
            "episodes": self.episodes
        }

    def reset(self):
        """Reset all metrics."""
        self.episodes = []
        self.returns = []
        self.lengths = []


def compute_episode_metrics(
    returns: List[float],
    lengths: List[int]
) -> Dict[str, Any]:
    """
    Compute summary statistics from episode data.

    Args:
        returns: List of episode returns
        lengths: List of episode lengths

    Returns:
        Dictionary with statistics
    """
    if len(returns) == 0:
        return {
            "return": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "length": {"mean": 0, "std": 0, "min": 0, "max": 0},
        }

    returns_array = np.array(returns)
    lengths_array = np.array(lengths)

    return {
        "return": {
            "mean": float(np.mean(returns_array)),
            "std": float(np.std(returns_array)),
            "min": float(np.min(returns_array)),
            "max": float(np.max(returns_array)),
        },
        "length": {
            "mean": float(np.mean(lengths_array)),
            "std": float(np.std(lengths_array)),
            "min": int(np.min(lengths_array)),
            "max": int(np.max(lengths_array)),
        },
    }


def save_metrics_to_json(
    metrics: Dict[str, Any],
    filepath: str,
    config_path: Optional[str] = None,
    checkpoint_info: Optional[Dict[str, str]] = None
):
    """
    Save metrics to JSON file with metadata.

    Args:
        metrics: Metrics dictionary to save
        filepath: Path to output JSON file
        config_path: Optional path to config file used
        checkpoint_info: Optional dict of checkpoint paths used
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Build output structure
    output = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }

    if config_path is not None:
        output["config"] = config_path

    if checkpoint_info is not None:
        output["checkpoints"] = checkpoint_info

    # Save to file
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Metrics saved to: {filepath}")


def load_metrics_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Metrics dictionary
    """
    with open(filepath, "r") as f:
        return json.load(f)
