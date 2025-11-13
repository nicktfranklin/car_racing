"""Inference and evaluation modules."""

from .dream_evaluation import evaluate_dream_environment
from .evaluation import evaluate_agent, evaluate_real_environment, run_evaluation
from .metrics import MetricsCollector, save_metrics_to_json
from .video_utils import VideoRecorder

__all__ = [
    "run_evaluation",
    "evaluate_real_environment",
    "evaluate_dream_environment",
    "evaluate_agent",  # Legacy
    "VideoRecorder",
    "MetricsCollector",
    "save_metrics_to_json",
]
