"""
Finite Scalar Quantization VAE (FSQ-VAE) implementation.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FSQVAEConfig


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantization module."""

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)

        # Create quantization bounds for each dimension
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))

        # Compute implicit codebook size
        self.codebook_size = int(np.prod(levels))

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize the input tensor using FSQ."""
        # z shape: (batch, dim) where dim = len(levels)
        quantized = torch.zeros_like(z)

        for i, level in enumerate(self.levels):
            # Map to [-1, 1] then quantize to discrete levels
            # Create discrete levels: -1, -1+2/L, -1+4/L, ..., 1
            if level == 1:
                quantized[..., i] = 0
            else:
                # Quantize to level discrete values in [-1, 1]
                quantized[..., i] = (
                    torch.round(z[..., i] * (level - 1) / 2) * 2 / (level - 1)
                )
                quantized[..., i] = torch.clamp(quantized[..., i], -1, 1)

        return quantized

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with straight-through estimator."""
        z_quantized = self.quantize(z)

        # Straight-through estimator: use quantized values in forward pass
        # but gradients flow through the continuous values
        z_quantized = z + (z_quantized - z).detach()

        # Compute indices for each quantized vector
        indices = self._get_indices(z_quantized)

        return z_quantized, indices

    def _get_indices(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """Convert quantized values to codebook indices."""
        batch_size = z_quantized.shape[0]
        indices = torch.zeros(batch_size, dtype=torch.long, device=z_quantized.device)

        for i, level in enumerate(self.levels):
            if level > 1:
                # Convert from [-1, 1] to [0, level-1]
                level_indices = (
                    ((z_quantized[..., i] + 1) * (level - 1) / 2).round().long()
                )
                level_indices = torch.clamp(level_indices, 0, level - 1)

                # Accumulate index (treating as mixed radix)
                if i == 0:
                    indices = level_indices
                else:
                    indices = indices * level + level_indices

        return indices


class ConvEncoder(nn.Module):
    """Convolutional encoder for FSQ-VAE."""

    def __init__(self, config: FSQVAEConfig):
        super().__init__()
        self.config = config

        layers = []
        in_channels = config.input_channels

        for i, (out_channels, stride) in enumerate(
            zip(config.encoder_channels, config.encoder_strides)
        ):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        # Add final convolution to get to latent dimension
        layers.append(nn.Conv2d(in_channels, config.latent_dim, 1, stride=1, padding=0))
        self.encoder = nn.Sequential(*layers)

        # Project from latent_dim to FSQ dimensions
        self.fsq_projection = nn.Linear(config.latent_dim, len(config.fsq_levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images to latent space."""
        z = self.encoder(x)
        # Global average pooling to get fixed-size representation
        z = F.adaptive_avg_pool2d(z, 1).squeeze(-1).squeeze(-1)
        # Project to FSQ dimensions
        z = self.fsq_projection(z)
        # Normalize to [-1, 1] for FSQ
        z = torch.tanh(z)
        return z


class ConvDecoder(nn.Module):
    """Convolutional decoder for FSQ-VAE."""

    def __init__(self, config: FSQVAEConfig):
        super().__init__()
        self.config = config

        # Calculate initial spatial size after encoding
        h, w = config.input_height, config.input_width
        for stride in config.encoder_strides:
            h, w = h // stride, w // stride

        self.initial_h, self.initial_w = h, w
        self.initial_channels = config.decoder_channels[0]

        # Project latent to initial decoder size
        self.projection = nn.Linear(
            len(config.fsq_levels),  # FSQ quantized representation
            self.initial_channels * self.initial_h * self.initial_w,
        )

        # Decoder layers
        layers = []
        in_channels = self.initial_channels

        for i, (out_channels, stride) in enumerate(
            zip(
                config.decoder_channels[1:] + [config.input_channels],
                config.decoder_strides,
            )
        ):
            if i < len(config.decoder_channels) - 1:
                layers.extend(
                    [
                        nn.ConvTranspose2d(
                            in_channels, out_channels, 4, stride=stride, padding=1
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
            else:
                # Final layer - no batch norm or activation
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels, out_channels, 4, stride=stride, padding=1
                    )
                )
            in_channels = out_channels

        self.decoder = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to images."""
        batch_size = z_q.shape[0]

        # Project to decoder initial size
        h = self.projection(z_q)
        h = h.view(batch_size, self.initial_channels, self.initial_h, self.initial_w)

        # Decode
        x_recon = self.decoder(h)
        x_recon = torch.sigmoid(x_recon)  # Output in [0, 1]

        return x_recon


class FSQVAE(nn.Module):
    """Complete FSQ-VAE model."""

    def __init__(self, config: FSQVAEConfig):
        super().__init__()
        self.config = config

        self.encoder = ConvEncoder(config)
        self.quantizer = FSQQuantizer(config.fsq_levels)
        self.decoder = ConvDecoder(config)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode and quantize input."""
        z = self.encoder(x)
        z_q, indices = self.quantizer(z)
        return z_q, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode from quantized representation."""
        return self.decoder(z_q)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        z = self.encoder(x)
        z_q, indices = self.quantizer(z)
        x_recon = self.decoder(z_q)

        return x_recon, z, z_q

    def compute_loss(
        self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor, z_q: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute FSQ-VAE loss."""
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # Commitment loss - encourage encoder to commit to quantized values
        commitment_loss = F.mse_loss(z, z_q.detach(), reduction="mean")

        # Total loss
        total_loss = recon_loss + self.config.beta * commitment_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "commitment_loss": commitment_loss.item(),
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    # Test the FSQ-VAE implementation
    config = FSQVAEConfig()
    model = FSQVAE(config)

    # Test forward pass
    batch_size = 4
    x = torch.randn(
        batch_size, config.input_channels, config.input_height, config.input_width
    )

    with torch.no_grad():
        x_recon, z, z_q = model(x)
        loss, loss_dict = model.compute_loss(x, x_recon, z, z_q)

    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Quantized latent shape: {z_q.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Codebook size: {model.quantizer.codebook_size}")
    print(f"Loss: {loss.item():.4f}")
    print("FSQ-VAE test passed!")
