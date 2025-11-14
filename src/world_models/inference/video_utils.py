"""
Video recording utilities for evaluation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np


class VideoRecorder:
    """Record videos during evaluation."""

    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        resolution: Tuple[int, int] = (64, 64)
    ):
        """
        Initialize video recorder.

        Args:
            output_path: Path to save video file (should end in .mp4)
            fps: Frames per second
            resolution: (height, width) of video frames
        """
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.frames: List[np.ndarray] = []

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def add_frame(self, frame: np.ndarray):
        """
        Add a single frame to the video.

        Args:
            frame: RGB image array of shape (H, W, 3) with values in [0, 255] or [0, 1]
        """
        # Normalize to uint8 [0, 255]
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # Resize if needed
        if frame.shape[:2] != self.resolution:
            from skimage.transform import resize
            frame = resize(
                frame,
                self.resolution,
                anti_aliasing=True,
                preserve_range=True
            ).astype(np.uint8)

        self.frames.append(frame)

    def add_comparison_frame(
        self,
        real: np.ndarray,
        reconstruction: Optional[np.ndarray] = None,
        prediction: Optional[np.ndarray] = None
    ):
        """
        Add a frame with side-by-side comparison.

        Args:
            real: Real observation (H, W, 3)
            reconstruction: VAE reconstruction (H, W, 3), optional
            prediction: World model prediction (H, W, 3), optional
        """
        frames_to_stack = [real]

        if reconstruction is not None:
            frames_to_stack.append(reconstruction)

        if prediction is not None:
            frames_to_stack.append(prediction)

        # Create horizontal concatenation
        comparison = create_comparison_frame({
            "Real": real,
            "Reconstruction": reconstruction,
            "Prediction": prediction
        } if prediction is not None else {
            "Real": real,
            "Reconstruction": reconstruction
        } if reconstruction is not None else {
            "Real": real
        })

        self.add_frame(comparison)

    def save(self):
        """Save recorded frames to video file."""
        if len(self.frames) == 0:
            print(f"Warning: No frames to save for {self.output_path}")
            return

        try:
            # Save as MP4 using imageio
            imageio.mimsave(
                self.output_path,
                self.frames,
                fps=self.fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )
            print(f"Video saved: {self.output_path} ({len(self.frames)} frames, {self.fps} fps)")
        except Exception as e:
            print(f"Error saving video: {e}")
            # Fallback: save as GIF
            gif_path = self.output_path.replace('.mp4', '.gif')
            try:
                imageio.mimsave(gif_path, self.frames, fps=self.fps)
                print(f"Saved as GIF instead: {gif_path}")
            except Exception as e2:
                print(f"Error saving GIF: {e2}")

    def clear(self):
        """Clear all recorded frames."""
        self.frames = []


def create_comparison_frame(frames_dict: Dict[str, Optional[np.ndarray]]) -> np.ndarray:
    """
    Create a side-by-side comparison frame with labels.

    Args:
        frames_dict: Dictionary mapping labels to frames (H, W, 3)
                    Frames can be None to skip

    Returns:
        Horizontal concatenation of frames with labels
    """
    # Filter out None frames
    valid_frames = {k: v for k, v in frames_dict.items() if v is not None}

    if len(valid_frames) == 0:
        # Return black frame
        return np.zeros((64, 64, 3), dtype=np.uint8)

    # Normalize all frames to uint8 [0, 255]
    normalized_frames = []
    labels = []

    for label, frame in valid_frames.items():
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        normalized_frames.append(frame)
        labels.append(label)

    # Get dimensions (assume all same size)
    h, w = normalized_frames[0].shape[:2]

    # Add text labels (simple version - add white bar on top with black text)
    label_height = 20
    labeled_frames = []

    for frame, label in zip(normalized_frames, labels):
        # Create white bar for label
        label_bar = np.ones((label_height, w, 3), dtype=np.uint8) * 255

        # Note: For simplicity, we're not adding text rendering here
        # If needed, use PIL or cv2 to add text
        # For now, just stack the white bar

        labeled_frame = np.vstack([label_bar, frame])
        labeled_frames.append(labeled_frame)

    # Concatenate horizontally
    comparison = np.hstack(labeled_frames)

    return comparison


def save_frame_as_image(frame: np.ndarray, filepath: str):
    """
    Save a single frame as an image file.

    Args:
        frame: RGB image array (H, W, 3)
        filepath: Path to save image (e.g., .png, .jpg)
    """
    # Normalize to uint8 [0, 255]
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save image
    imageio.imwrite(filepath, frame)
