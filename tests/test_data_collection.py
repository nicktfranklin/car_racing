"""
Tests for data collection functionality.

Tests the DataCollector, Episode storage, and dataset implementations.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from world_models.config import DataConfig
from world_models.training.data_collection import DataCollector, Episode
from world_models.training.datasets import ImageDataset, SequenceDataset


@pytest.fixture
def config():
    """Create test configuration."""
    config = DataConfig()
    config.num_rollouts = 5
    config.max_episode_length = 100
    return config


@pytest.fixture
def collector(config):
    """Create DataCollector."""
    collector = DataCollector(config)
    yield collector
    collector.close()


@pytest.fixture
def sample_episodes(collector, config):
    """Collect sample episodes for testing."""
    episodes = collector.collect_random_episodes(config.num_rollouts)
    return episodes


class TestDataCollector:
    """Tests for DataCollector class."""

    def test_random_episode_collection(self, collector, config):
        """Test collecting random episodes."""
        episodes = collector.collect_random_episodes(config.num_rollouts)

        assert len(episodes) == config.num_rollouts
        for episode in episodes:
            assert isinstance(episode, Episode)
            assert len(episode.observations) > 0
            assert len(episode.actions) > 0
            assert len(episode.rewards) > 0
            assert len(episode.dones) > 0

    def test_parallel_vs_sequential(self):
        """Test that parallel collection works."""
        # Sequential
        config_seq = DataConfig()
        config_seq.num_rollouts = 5
        config_seq.max_episode_length = 50
        config_seq.num_workers = 1

        collector_seq = DataCollector(config_seq)
        episodes_seq = collector_seq.collect_random_episodes(config_seq.num_rollouts)
        collector_seq.close()

        # Parallel
        config_par = DataConfig()
        config_par.num_rollouts = 5
        config_par.max_episode_length = 50
        config_par.num_workers = 2

        collector_par = DataCollector(config_par)
        episodes_par = collector_par.collect_random_episodes(config_par.num_rollouts)
        collector_par.close()

        # Both should collect requested episodes
        assert len(episodes_seq) == config_seq.num_rollouts
        assert len(episodes_par) == config_par.num_rollouts

    def test_save_load_episodes(self, collector, sample_episodes):
        """Test saving and loading episodes."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        try:
            # Save episodes
            collector.save_episodes(sample_episodes, filepath)
            assert os.path.exists(filepath)

            # Load episodes
            loaded_episodes = collector.load_episodes(filepath)
            assert len(loaded_episodes) == len(sample_episodes)

            # Check episode content matches
            for orig, loaded in zip(sample_episodes, loaded_episodes):
                assert len(orig.observations) == len(loaded.observations)
                assert len(orig.actions) == len(loaded.actions)
                assert len(orig.rewards) == len(loaded.rewards)
                assert len(orig.dones) == len(loaded.dones)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestEpisode:
    """Tests for Episode class."""

    def test_episode_properties(self, sample_episodes):
        """Test Episode properties."""
        for episode in sample_episodes:
            # All lists should have same length
            length = len(episode.observations)
            assert len(episode.actions) == length
            assert len(episode.rewards) == length
            assert len(episode.dones) == length

            # Check data types - Episode stores as lists
            assert isinstance(episode.observations, list)
            assert isinstance(episode.actions, list)
            assert isinstance(episode.rewards, list)
            assert isinstance(episode.dones, list)

            # Check that to_arrays() works
            obs_arr, act_arr, rew_arr, done_arr = episode.to_arrays()
            assert obs_arr.shape[0] == length
            assert act_arr.shape[0] == length
            assert rew_arr.shape[0] == length
            assert done_arr.shape[0] == length

    def test_episode_statistics(self, sample_episodes):
        """Test episode statistics."""
        lengths = [len(ep.observations) for ep in sample_episodes]
        returns = [sum(ep.rewards) for ep in sample_episodes]

        # Episodes should have reasonable lengths
        assert all(length > 0 for length in lengths)
        assert np.mean(lengths) > 0

        # Returns should be computed
        assert len(returns) == len(sample_episodes)


class TestSequenceDataset:
    """Tests for SequenceDataset."""

    def test_sequence_dataset_creation(self, sample_episodes):
        """Test creating sequence dataset."""
        seq_length = 10
        dataset = SequenceDataset(sample_episodes, sequence_length=seq_length)

        assert len(dataset) >= 0
        if len(dataset) > 0:
            sample = dataset[0]
            assert isinstance(sample, dict)
            assert "observations" in sample
            assert "actions" in sample
            assert "rewards" in sample
            assert "dones" in sample

    def test_sequence_shapes(self, sample_episodes):
        """Test sequence dataset output shapes."""
        seq_length = 10
        dataset = SequenceDataset(sample_episodes, sequence_length=seq_length)

        if len(dataset) > 0:
            sample = dataset[0]

            # Check sequence length dimension
            # Observations include next state for prediction (seq_length + 1)
            # Actions, rewards, dones have seq_length elements
            assert sample["observations"].shape[0] == seq_length + 1
            assert sample["actions"].shape[0] == seq_length
            assert sample["rewards"].shape[0] == seq_length
            assert sample["dones"].shape[0] == seq_length

            # Check tensor types
            assert isinstance(sample["observations"], torch.Tensor)
            assert isinstance(sample["actions"], torch.Tensor)
            assert isinstance(sample["rewards"], torch.Tensor)
            assert isinstance(sample["dones"], torch.Tensor)


class TestImageDataset:
    """Tests for ImageDataset."""

    def test_image_dataset_creation(self, sample_episodes):
        """Test creating image dataset."""
        dataset = ImageDataset(sample_episodes)

        # Should have one image per timestep across all episodes
        total_steps = sum(len(ep.observations) for ep in sample_episodes)
        assert len(dataset) == total_steps

    def test_image_shapes(self, sample_episodes):
        """Test image dataset output shapes."""
        dataset = ImageDataset(sample_episodes)

        if len(dataset) > 0:
            sample = dataset[0]

            # Check image shape (C, H, W)
            assert len(sample.shape) == 3
            assert sample.shape[0] == 3  # RGB channels
            assert sample.shape[1] == 64  # Height
            assert sample.shape[2] == 64  # Width

            # Check tensor type
            assert isinstance(sample, torch.Tensor)

    def test_image_values(self, sample_episodes):
        """Test image pixel values are in valid range."""
        dataset = ImageDataset(sample_episodes)

        if len(dataset) > 0:
            sample = dataset[0]

            # Pixel values should be normalized
            assert sample.min() >= -1.0
            assert sample.max() <= 1.0
