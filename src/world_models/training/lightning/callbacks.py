"""
Lightning callbacks for training.

Provides callbacks for:
- Chunk rotation for large dataset training
"""

import lightning as L


class ChunkRotationCallback(L.Callback):
    """Rotates chunk groups every N epochs for multi-chunk random sampling.

    This callback enables efficient random sampling from large datasets by:
    1. Loading multiple chunks into RAM (e.g., 5 chunks = 640K images)
    2. Using RandomSampler to sample from this in-memory pool for N epochs
    3. Rotating to the next chunk group every N epochs

    The dataset's __len__() method returns the current chunk size, so
    RandomSampler automatically adapts when chunks rotate. No dataloader
    recreation needed!
    """

    def __init__(self, epochs_per_phase: int = 3):
        """
        Args:
            epochs_per_phase: How many epochs to use each chunk group before rotating
        """
        super().__init__()
        self.epochs_per_phase = epochs_per_phase

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        # Check if it's time to rotate (every epochs_per_phase epochs)
        if (trainer.current_epoch + 1) % self.epochs_per_phase == 0:
            # Access the dataset (accounting for Subset wrapper from train/val split)
            train_dataloader = trainer.train_dataloader
            if hasattr(train_dataloader, "dataset"):
                dataset = train_dataloader.dataset
                # Unwrap Subset to get the underlying dataset (for non-chunked mode)
                if hasattr(dataset, "dataset"):
                    dataset = dataset.dataset
                # Call rotation method if available (supports both old and new datasets)
                if hasattr(dataset, "rotate_to_next_chunk_group"):
                    dataset.rotate_to_next_chunk_group()
                elif hasattr(dataset, "load_next_chunk"):
                    dataset.load_next_chunk()
