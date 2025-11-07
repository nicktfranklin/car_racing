"""
Visualize VAE latent space and reconstructions.

This script provides validation of the trained VAE by:
1. Sampling random latents and decoding them
2. Comparing reconstructions with original images
3. Interpolating between latents
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.world_models import (
    FSQVAE,
    WorldModelAgentConfig,
    DataCollector,
    ImageDataset,
    VAELightningModule,
)


def load_vae(config: WorldModelAgentConfig, checkpoint_dir: str = "./checkpoints", device="cpu"):
    """Load trained VAE from checkpoint."""
    # Load with perceptual loss to match checkpoint, but we won't use it for inference
    vae = FSQVAE(config.fsq_vae, use_perceptual_loss=config.fsq_vae.use_perceptual_loss, device=device)

    # Try different checkpoint locations (newest first)
    # 1. Try last-v2.ckpt first (newest single-code architecture)
    last_v2_ckpt = os.path.join(checkpoint_dir, "vae", "last-v2.ckpt")
    # 2. Try last-v1.ckpt (spatial architecture)
    last_v1_ckpt = os.path.join(checkpoint_dir, "vae", "last-v1.ckpt")
    # 3. Try last.ckpt (oldest)
    last_ckpt = os.path.join(checkpoint_dir, "vae", "last.ckpt")
    # 4. Find best checkpoint in vae directory
    vae_dir = os.path.join(checkpoint_dir, "vae")

    checkpoint_path = None

    if os.path.exists(last_v2_ckpt):
        checkpoint_path = last_v2_ckpt
        print(f"Loading VAE from last checkpoint (v2): {checkpoint_path}")
    elif os.path.exists(last_v1_ckpt):
        checkpoint_path = last_v1_ckpt
        print(f"Loading VAE from last checkpoint (v1): {checkpoint_path}")
    elif os.path.exists(last_ckpt):
        checkpoint_path = last_ckpt
        print(f"Loading VAE from last checkpoint: {checkpoint_path}")
    elif os.path.exists(vae_dir):
        # Find best checkpoint based on filename
        import glob
        ckpt_files = glob.glob(os.path.join(vae_dir, "epoch=*.ckpt"))
        if ckpt_files:
            # Sort by validation loss in filename
            ckpt_files.sort()
            checkpoint_path = ckpt_files[0]  # First one should have lowest loss
            print(f"Loading VAE from best checkpoint: {checkpoint_path}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        vae_module = VAELightningModule.load_from_checkpoint(
            checkpoint_path,
            model=vae,
            config=config,
            map_location=device
        )
        vae = vae_module.model
    else:
        raise FileNotFoundError(
            f"No VAE checkpoint found. Tried:\n"
            f"  - {last_ckpt}\n"
            f"  - {vae_dir}/epoch=*.ckpt"
        )

    vae.eval()
    return vae


def sample_random_latents(vae: FSQVAE, num_samples: int = 16, device="cpu"):
    """Sample random latents from the FSQ codebook and decode them."""
    vae = vae.to(device)

    # FSQ uses discrete levels, so we sample from the valid quantization levels
    samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Sample random FSQ latent (single code per image)
            # Shape: (1, fsq_dim)
            z_q = torch.zeros(1, len(vae.config.fsq_levels), device=device)

            for i, level in enumerate(vae.config.fsq_levels):
                # Sample uniformly from quantized values in [-1, 1]
                if level == 1:
                    z_q[:, i] = 0.0
                else:
                    # Quantized values: -1, -1+2/(L-1), ..., 1
                    idx = np.random.randint(0, level)
                    z_q[:, i] = -1.0 + 2.0 * idx / (level - 1)

            # Decode
            img = vae.decode(z_q)
            samples.append(img.cpu())

    return torch.cat(samples, dim=0)


def get_reconstructions(vae: FSQVAE, dataset: ImageDataset, num_samples: int = 8, device="cpu"):
    """Get original images and their reconstructions."""
    vae = vae.to(device)

    # Sample random images from dataset
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    originals = []
    reconstructions = []

    with torch.no_grad():
        for idx in indices:
            img = dataset[idx].unsqueeze(0).to(device)
            recon, _, _ = vae(img)

            originals.append(img.cpu())
            reconstructions.append(recon.cpu())

    return torch.cat(originals, dim=0), torch.cat(reconstructions, dim=0)


def interpolate_latents(vae: FSQVAE, dataset: ImageDataset, num_steps: int = 8, device="cpu"):
    """Interpolate between two images in latent space."""
    vae = vae.to(device)

    # Get two random images
    idx1, idx2 = np.random.choice(len(dataset), 2, replace=False)
    img1 = dataset[idx1].unsqueeze(0).to(device)
    img2 = dataset[idx2].unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode both images
        z1 = vae.encoder(img1)
        z2 = vae.encoder(img2)

        # Interpolate in continuous latent space (before quantization)
        interpolations = []
        for alpha in np.linspace(0, 1, num_steps):
            z_interp = (1 - alpha) * z1 + alpha * z2

            # Quantize and decode
            z_q, _ = vae.quantizer(z_interp)
            img_interp = vae.decode(z_q)
            interpolations.append(img_interp.cpu())

    return torch.cat([img1.cpu()] + interpolations + [img2.cpu()], dim=0)


def plot_images(images: torch.Tensor, title: str, save_path: str = None, nrow: int = 8):
    """Plot a grid of images."""
    # Convert from (N, C, H, W) to (N, H, W, C)
    images = images.permute(0, 2, 3, 1).numpy()
    images = np.clip(images, 0, 1)

    n_images = len(images)
    ncol = min(nrow, n_images)
    nrow_actual = (n_images + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 2, nrow_actual * 2))
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis('off')

    # Hide extra axes
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_reconstruction_comparison(originals: torch.Tensor, reconstructions: torch.Tensor,
                                   save_path: str = None):
    """Plot original vs reconstructed images side by side."""
    # Convert from (N, C, H, W) to (N, H, W, C)
    originals = originals.permute(0, 2, 3, 1).numpy()
    reconstructions = reconstructions.permute(0, 2, 3, 1).numpy()

    originals = np.clip(originals, 0, 1)
    reconstructions = np.clip(reconstructions, 0, 1)

    n_images = len(originals)

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 4))

    for idx in range(n_images):
        axes[0, idx].imshow(originals[idx])
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_title('Original', fontsize=10)

        axes[1, idx].imshow(reconstructions[idx])
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].set_title('Reconstructed', fontsize=10)

    plt.suptitle('VAE Reconstructions', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize VAE latent space")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--data-file", type=str, default="training_data.h5", help="Data file for reconstructions")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--num-reconstructions", type=int, default=8, help="Number of reconstructions to show")
    parser.add_argument("--num-interpolations", type=int, default=8, help="Number of interpolation steps")
    parser.add_argument("--output-dir", type=str, default="./visualizations", help="Output directory for images")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")

    args = parser.parse_args()

    # Load config
    config = WorldModelAgentConfig.from_yaml(args.config)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.training.device)

    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE
    print("Loading VAE...")
    vae = load_vae(config, args.checkpoint_dir, device=device)
    print("VAE loaded successfully!")

    # 1. Sample random latents
    print(f"\nGenerating {args.num_samples} random samples from latent space...")
    random_samples = sample_random_latents(vae, args.num_samples, device=device)
    plot_images(
        random_samples,
        "Random Samples from Latent Space",
        save_path=os.path.join(args.output_dir, "random_samples.png")
    )

    # 2. Get reconstructions
    print(f"\nGenerating {args.num_reconstructions} reconstructions...")
    collector = DataCollector(config.data)
    chunk_files = collector.get_chunk_files(args.data_file)

    if chunk_files:
        dataset = ImageDataset(
            data_dir=config.data.data_dir,
            chunk_files=chunk_files,
            subsample_rate=config.training.subsample_rate,
        )
    else:
        print("Warning: No chunked data found, loading all episodes into memory...")
        episodes = collector.load_episodes(args.data_file)
        dataset = ImageDataset(episodes=episodes)

    originals, reconstructions = get_reconstructions(
        vae, dataset, args.num_reconstructions, device=device
    )
    plot_reconstruction_comparison(
        originals,
        reconstructions,
        save_path=os.path.join(args.output_dir, "reconstructions.png")
    )

    # 3. Latent interpolation
    print(f"\nGenerating latent space interpolation ({args.num_interpolations} steps)...")
    interpolated = interpolate_latents(vae, dataset, args.num_interpolations, device=device)
    plot_images(
        interpolated,
        "Latent Space Interpolation",
        save_path=os.path.join(args.output_dir, "interpolation.png"),
        nrow=args.num_interpolations + 2
    )

    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
