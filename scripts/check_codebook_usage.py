"""
Check codebook usage of trained VAE.
"""

import numpy as np
import torch
from tqdm import tqdm

from world_models import FSQVAE, ImageDataset, VAELightningModule, WorldModelAgentConfig
from world_models.training.data_collection import DataCollector


def main():
    # Load config and VAE
    config = WorldModelAgentConfig.from_yaml("config.yaml")
    device = torch.device(config.training.device)

    print("Loading VAE...")
    vae = FSQVAE(
        config.fsq_vae,
        use_perceptual_loss=config.fsq_vae.use_perceptual_loss,
        device=device,
    )

    # Load checkpoint
    checkpoint_path = "./checkpoints/vae/last-v2.ckpt"
    vae_module = VAELightningModule.load_from_checkpoint(
        checkpoint_path, model=vae, config=config, map_location=device
    )
    vae = vae_module.model
    vae.eval()
    vae.to(device)

    print(f"VAE loaded. Codebook size: {vae.quantizer.codebook_size}")

    # Load dataset
    print("\nLoading dataset...")
    collector = DataCollector(config.data)
    chunk_files = collector.get_chunk_files("training_data.h5")

    dataset = ImageDataset(
        data_dir=config.data.data_dir,
        chunk_files=chunk_files[:20],  # Use subset for speed
        subsample_rate=config.training.subsample_rate,
    )

    print(f"Dataset loaded: {len(dataset):,} images")

    # Collect all codes used
    print("\nAnalyzing codebook usage...")
    all_indices = []
    num_batches = min(100, len(dataset) // 256)  # Check ~25k images

    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            start_idx = i * 256
            end_idx = min(start_idx + 256, len(dataset))

            # Get batch
            batch = []
            for j in range(start_idx, end_idx):
                batch.append(dataset[j])
            batch = torch.stack(batch).to(device)

            # Encode
            z_q, indices = vae.encode(batch)
            all_indices.extend(indices.cpu().numpy().tolist())

    # Analyze
    all_indices = np.array(all_indices)
    unique_codes = np.unique(all_indices)

    print(f"\n{'='*60}")
    print(f"CODEBOOK USAGE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total codebook size:     {vae.quantizer.codebook_size}")
    print(f"Unique codes used:       {len(unique_codes)}")
    print(
        f"Codebook utilization:    {len(unique_codes)/vae.quantizer.codebook_size*100:.1f}%"
    )
    print(f"Images analyzed:         {len(all_indices):,}")
    print(f"\nCode usage distribution:")

    # Show most/least common codes
    from collections import Counter

    code_counts = Counter(all_indices)
    most_common = code_counts.most_common(10)

    print(f"\nTop 10 most frequent codes:")
    for code, count in most_common:
        print(f"  Code {code:4d}: {count:6d} times ({count/len(all_indices)*100:.2f}%)")

    print(f"\nCode frequency statistics:")
    frequencies = np.array(list(code_counts.values()))
    print(f"  Mean:   {frequencies.mean():.1f}")
    print(f"  Median: {np.median(frequencies):.1f}")
    print(f"  Std:    {frequencies.std():.1f}")
    print(f"  Min:    {frequencies.min()}")
    print(f"  Max:    {frequencies.max()}")

    # Check if codes are clustered
    print(f"\nCode distribution:")
    print(f"  Min code index: {unique_codes.min()}")
    print(f"  Max code index: {unique_codes.max()}")
    print(f"  Code range:     {unique_codes.max() - unique_codes.min()}")

    # Histogram of code usage
    if len(unique_codes) < vae.quantizer.codebook_size:
        unused_codes = vae.quantizer.codebook_size - len(unique_codes)
        print(f"\n⚠️  WARNING: {unused_codes} codes are completely unused!")
        print(f"   This indicates potential codebook collapse.")
    else:
        print(f"\n✓ All codes in codebook are being used!")


if __name__ == "__main__":
    main()
