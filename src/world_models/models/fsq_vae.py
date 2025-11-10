"""
Finite Scalar Quantization VAE (FSQ-VAE) implementation.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ..config import FSQVAEConfig


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features."""

    def __init__(self, layers=None, device="cpu"):
        super().__init__()
        if layers is None:
            # Use early layers for perceptual similarity
            layers = ["relu1_2", "relu2_2", "relu3_3"]

        self.layers = layers

        # Load pretrained VGG16
        try:
            # Try newer API first
            from torchvision.models import VGG16_Weights

            vgg = (
                models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
                .features.eval()
                .to(device)
            )
        except (ImportError, AttributeError):
            # Fall back to older API
            vgg = models.vgg16(pretrained=True).features.eval().to(device)

        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False

        # Split VGG into blocks for feature extraction
        self.blocks = nn.ModuleList()
        self.layer_names = []

        # VGG16 layer mapping
        layer_mapping = {
            "relu1_1": 1,
            "relu1_2": 3,
            "relu2_1": 6,
            "relu2_2": 8,
            "relu3_1": 11,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu4_1": 18,
            "relu4_2": 20,
            "relu4_3": 22,
        }

        prev_idx = 0
        for layer_name in layers:
            idx = layer_mapping[layer_name]
            self.blocks.append(vgg[prev_idx : idx + 1])
            self.layer_names.append(layer_name)
            prev_idx = idx + 1

        # Normalization for VGG (ImageNet stats) - must specify device for buffers
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    def normalize(self, x):
        """Normalize images for VGG."""
        return (x - self.mean) / self.std

    def forward(self, x, y):
        """Compute perceptual loss between x and y.

        Args:
            x: predicted image (batch, 3, H, W) in [0, 1]
            y: target image (batch, 3, H, W) in [0, 1]
        """
        # Normalize inputs
        x = self.normalize(x)
        y = self.normalize(y)

        # Extract features with no_grad on VGG (weights frozen)
        # But keep grad enabled on inputs (x, y) for backprop to VAE
        loss = 0.0
        for block in self.blocks:
            # Forward through frozen VGG blocks
            with torch.set_grad_enabled(x.requires_grad or y.requires_grad):
                x = block(x)
                y = block(y)
            # Compute MSE loss (needs grad for backprop)
            loss += F.mse_loss(x, y)

        return loss / len(self.blocks)


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
        """Quantize the input tensor using FSQ.

        Args:
            z: shape (batch, dim, H, W) or (batch, dim) where dim = len(levels)
        """
        quantized = torch.zeros_like(z)

        for i, level in enumerate(self.levels):
            # Map to [-1, 1] then quantize to discrete levels
            # Create discrete levels: -1, -1+2/L, -1+4/L, ..., 1
            if level == 1:
                if z.dim() == 4:
                    quantized[:, i] = 0
                else:
                    quantized[..., i] = 0
            else:
                # Quantize to level discrete values in [-1, 1]
                if z.dim() == 4:
                    quantized[:, i] = (
                        torch.round(z[:, i] * (level - 1) / 2) * 2 / (level - 1)
                    )
                    quantized[:, i] = torch.clamp(quantized[:, i], -1, 1)
                else:
                    quantized[..., i] = (
                        torch.round(z[..., i] * (level - 1) / 2) * 2 / (level - 1)
                    )
                    quantized[..., i] = torch.clamp(quantized[..., i], -1, 1)

        return quantized

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with straight-through estimator.

        Returns:
            z_quantized: (batch, fsq_dim) quantized continuous values
            indices: (batch,) flat codebook indices
            tokens: (batch, fsq_dim) per-dimension discrete tokens
        """
        z_quantized = self.quantize(z)

        # Straight-through estimator: use quantized values in forward pass
        # but gradients flow through the continuous values
        z_quantized = z + (z_quantized - z).detach()

        # Compute indices and tokens for each quantized vector
        indices, tokens = self._get_indices(z_quantized)

        return z_quantized, indices, tokens

    def _get_indices(
        self, z_quantized: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert quantized values to codebook indices and per-dimension tokens.

        Returns:
            indices: (batch,) flat codebook indices [0, codebook_size-1]
            tokens: (batch, fsq_dim) per-dimension tokens, where tokens[:, i] is in [0, levels[i]-1]
        """
        batch_size = z_quantized.shape[0]
        indices = torch.zeros(batch_size, dtype=torch.long, device=z_quantized.device)
        tokens = torch.zeros(
            batch_size, self.dim, dtype=torch.long, device=z_quantized.device
        )

        for i, level in enumerate(self.levels):
            if level > 1:
                # Convert from [-1, 1] to [0, level-1]
                level_indices = (
                    ((z_quantized[..., i] + 1) * (level - 1) / 2).round().long()
                )
                level_indices = torch.clamp(level_indices, 0, level - 1)

                # Store per-dimension token
                tokens[:, i] = level_indices

                # Accumulate flat index (treating as mixed radix)
                if i == 0:
                    indices = level_indices
                else:
                    indices = indices * level + level_indices

        return indices, tokens


class SpatialAttentionPool(nn.Module):
    """Learned attention-based spatial pooling."""

    def __init__(self, channels):
        super().__init__()
        # Learn which spatial locations are important
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, H, W)
        Returns:
            pooled: (batch, channels)
        """
        # Compute attention weights over spatial dimensions
        attn = self.attention(x)  # (batch, 1, H, W)
        attn = attn.flatten(2)  # (batch, 1, H*W)
        attn_weights = F.softmax(attn, dim=2)  # (batch, 1, H*W)

        # Apply attention weights
        x_flat = x.flatten(2)  # (batch, channels, H*W)
        pooled = (x_flat * attn_weights).sum(dim=2)  # (batch, channels)

        return pooled


class ConvEncoder(nn.Module):
    """Convolutional encoder for FSQ-VAE with attention pooling."""

    def __init__(self, config: FSQVAEConfig):
        super().__init__()
        self.config = config

        encoder_layers = []
        in_channels = config.input_channels
        self.skip_channels = []  # Track channels for skip connections

        for i, (out_channels, stride) in enumerate(
            zip(config.encoder_channels, config.encoder_strides)
        ):
            encoder_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1),
                    nn.GroupNorm(
                        min(8, out_channels // 4), out_channels
                    ),  # Adaptive group size
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
            self.skip_channels.append(out_channels)

        # Add final convolution to get to latent dimension
        encoder_layers.append(
            nn.Conv2d(in_channels, config.latent_dim, 1, stride=1, padding=0)
        )
        encoder_layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.ModuleList(encoder_layers)

        # Attention pooling to aggregate spatial information
        self.attention_pool = SpatialAttentionPool(config.latent_dim)

        # Project from latent_dim to FSQ dimensions
        self.fsq_projection = nn.Linear(config.latent_dim, len(config.fsq_levels))

    def forward(self, x: torch.Tensor, return_skips: bool = False):
        """Encode input images to latent representation.

        Args:
            x: (batch, 3, H, W) input images
            return_skips: If True, return skip connections for U-Net

        Returns:
            z: (batch, fsq_dim) latent codes
            skips: List of skip connection features (if return_skips=True)
        """
        skips = []

        # Pass through encoder, collecting skip connections
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            # Store skip connections after each conv block (every 3 layers: conv, norm, relu)
            if return_skips and (i + 1) % 3 == 0 and i < len(self.encoder) - 2:
                skips.append(x)

        # x is now (batch, latent_dim, H, W)
        # Apply attention pooling
        z = self.attention_pool(x)  # (batch, latent_dim)

        # Project to FSQ dimensions
        z = self.fsq_projection(z)  # (batch, fsq_dim)

        # Normalize to [-1, 1] for FSQ
        z = torch.tanh(z)

        if return_skips:
            return z, skips
        return z


class ConvDecoder(nn.Module):
    """Convolutional decoder for FSQ-VAE with U-Net skip connections."""

    def __init__(self, config: FSQVAEConfig):
        super().__init__()
        self.config = config

        # Calculate initial spatial size after encoding
        h, w = config.input_height, config.input_width
        for stride in config.encoder_strides:
            h, w = h // stride, w // stride

        self.initial_h, self.initial_w = h, w
        self.initial_channels = config.decoder_channels[0]

        # Project FSQ quantized representation to initial spatial size
        # Input: (batch, fsq_dim), Output: (batch, decoder_channels[0], H, W)
        self.projection = nn.Linear(
            len(config.fsq_levels),
            self.initial_channels * self.initial_h * self.initial_w,
        )

        # Decoder layers (no skip connections - forces bottleneck usage)
        decoder_layers = []
        in_channels = self.initial_channels

        for i, (out_channels, stride) in enumerate(
            zip(
                config.decoder_channels[1:] + [config.input_channels],
                config.decoder_strides,
            )
        ):
            if i < len(config.decoder_channels) - 1:
                decoder_layers.extend(
                    [
                        nn.ConvTranspose2d(
                            in_channels, out_channels, 4, stride=stride, padding=1
                        ),
                        nn.GroupNorm(min(8, out_channels // 4), out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
            else:
                # Final layer - no norm or activation
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_channels, out_channels, 4, stride=stride, padding=1
                    )
                )
            in_channels = out_channels

        self.decoder = nn.ModuleList(decoder_layers)

    def forward(self, z_q: torch.Tensor, skips: list = None) -> torch.Tensor:
        """Decode quantized latents to images.

        Args:
            z_q: (batch, fsq_dim) quantized latent representation
            skips: Ignored - no skip connections to force bottleneck usage

        Returns:
            x_recon: (batch, 3, H, W) reconstructed image
        """
        batch_size = z_q.shape[0]

        # Project to initial spatial size
        h = self.projection(z_q)
        h = h.view(batch_size, self.initial_channels, self.initial_h, self.initial_w)

        # Decode WITHOUT skip connections - forces use of bottleneck
        for layer in self.decoder:
            h = layer(h)

        x_recon = torch.sigmoid(h)  # Output in [0, 1]

        return x_recon


class FSQVAE(nn.Module):
    """Complete FSQ-VAE model."""

    def __init__(self, config: FSQVAEConfig, use_perceptual_loss=True, device="cpu"):
        super().__init__()
        self.config = config
        self.use_perceptual_loss = use_perceptual_loss

        self.encoder = ConvEncoder(config)
        self.quantizer = FSQQuantizer(config.fsq_levels)
        self.decoder = ConvDecoder(config)

        # Add perceptual loss
        if use_perceptual_loss:
            self.perceptual_loss = PerceptualLoss(device=device)
        else:
            self.perceptual_loss = None

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and quantize input.

        Returns:
            z_q: (batch, fsq_dim) quantized continuous representation
            indices: (batch,) flat codebook indices
            tokens: (batch, fsq_dim) per-dimension discrete tokens
        """
        z = self.encoder(x, return_skips=False)
        z_q, indices, tokens = self.quantizer(z)
        return z_q, indices, tokens

    def decode(self, z_q: torch.Tensor, skips: list = None) -> torch.Tensor:
        """Decode from quantized representation."""
        return self.decoder(z_q, skips)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass without skip connections (forces bottleneck usage).

        Returns:
            x_recon: (batch, 3, H, W) reconstructed image
            z: (batch, fsq_dim) continuous latent
            z_q: (batch, fsq_dim) quantized latent
            indices: (batch,) flat codebook indices
            tokens: (batch, fsq_dim) per-dimension discrete tokens
        """
        z = self.encoder(
            x, return_skips=False
        )  # Returns only z when return_skips=False
        z_q, indices, tokens = self.quantizer(z)
        x_recon = self.decoder(z_q, skips=None)  # No skip connections

        return x_recon, z, z_q, indices, tokens

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        z_q: torch.Tensor,
        indices: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute FSQ-VAE loss with perceptual loss, codebook monitoring, and diversity regularization."""
        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(x_recon, x, reduction="mean")

        # Perceptual loss
        if self.use_perceptual_loss and self.perceptual_loss is not None:
            perceptual_loss_value = self.perceptual_loss(x_recon, x)
            # Use config weights
            recon_loss = (
                self.config.mse_weight * mse_loss
                + self.config.perceptual_weight * perceptual_loss_value
            )
        else:
            recon_loss = mse_loss
            perceptual_loss_value = torch.tensor(0.0)

        # Commitment loss - encourage encoder to commit to quantized values
        commitment_loss = F.mse_loss(z, z_q.detach(), reduction="mean")

        # Entropy regularization - encourage diverse code usage
        entropy_loss = torch.tensor(0.0, device=x.device)
        if (
            indices is not None
            and hasattr(self.config, "entropy_weight")
            and self.config.entropy_weight > 0
        ):
            # Compute code distribution in batch
            batch_size = indices.shape[0]
            code_counts = torch.bincount(
                indices, minlength=self.quantizer.codebook_size
            ).float()
            code_probs = code_counts / batch_size

            # Entropy of code distribution (higher = more diverse)
            # We want to MAXIMIZE entropy, so we MINIMIZE negative entropy
            epsilon = 1e-10
            code_probs = code_probs + epsilon  # Avoid log(0)
            entropy = -(code_probs * torch.log(code_probs)).sum()

            # Target: uniform distribution has entropy = log(codebook_size)
            max_entropy = torch.log(torch.tensor(float(self.quantizer.codebook_size)))

            # Loss: penalize low entropy (encourage high entropy/diversity)
            entropy_loss = (
                -entropy
            )  # Negative because we minimize loss but want to maximize entropy

        # Total loss
        total_loss = (
            recon_loss
            + self.config.beta * commitment_loss
            + (
                self.config.entropy_weight
                if hasattr(self.config, "entropy_weight")
                else 0.0
            )
            * entropy_loss
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "mse_loss": mse_loss.item(),
            "perceptual_loss": (
                perceptual_loss_value.item()
                if isinstance(perceptual_loss_value, torch.Tensor)
                else 0.0
            ),
            "commitment_loss": commitment_loss.item(),
            "entropy_loss": (
                entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else 0.0
            ),
        }

        # Codebook usage monitoring
        if indices is not None:
            unique_codes = torch.unique(indices).numel()
            total_codes = self.quantizer.codebook_size
            codebook_usage = unique_codes / total_codes

            # Compute codebook perplexity (key metric for collapse detection)
            # Perplexity measures the effective number of codes being used
            # Perfect usage: perplexity = codebook_size
            # Collapsed: perplexity ≈ 1
            batch_size = indices.shape[0]
            code_counts = torch.bincount(indices, minlength=total_codes).float()
            code_probs = code_counts / batch_size
            epsilon = 1e-10
            code_probs = code_probs + epsilon
            entropy = -(code_probs * torch.log(code_probs)).sum()
            perplexity = torch.exp(entropy)

            # Normalized perplexity (0 to 1, where 1 = perfect uniform usage)
            max_perplexity = total_codes
            perplexity_ratio = perplexity / max_perplexity

            loss_dict["codebook_usage"] = (
                codebook_usage  # Fraction of codes used at least once
            )
            loss_dict["unique_codes"] = unique_codes
            loss_dict["codebook_perplexity"] = (
                perplexity.item()
            )  # Effective number of codes
            loss_dict["perplexity_ratio"] = perplexity_ratio.item()  # Normalized (0-1)

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
