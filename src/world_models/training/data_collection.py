"""
Data collection for World Model training.
"""

from ..config import WorldModelAgentConfig
from ..data_collection import DataCollector
from ..utils import get_logger


def collect_data(
    config: WorldModelAgentConfig, data_file: str, checkpoint_every: int = 100
):
    """Collect training data with checkpointing."""
    logger = get_logger("world_models")
    collector = DataCollector(config.data)

    logger.info(f"Collecting {config.data.num_rollouts} episodes with checkpointing...")
    collector.collect_random_episodes(
        config.data.num_rollouts, data_file=data_file, checkpoint_every=checkpoint_every
    )

    collector.close()
    logger.info("Data collection completed!")
