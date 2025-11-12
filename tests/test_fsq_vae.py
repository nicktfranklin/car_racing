"""
Tests for FSQ-VAE implementation.

Tests the FSQVAE model, FSQ quantizer, and related functionality.
"""

import pytest
import torch

from world_models import FSQVAE
from world_models.config import FSQVAEConfig


@pytest.fixture
def config():
    """Create test configuration."""
    return FSQVAEConfig()


@pytest.fixture
def vae(config):
    """Create FSQ-VAE model."""
    return FSQVAE(config)


class TestFSQVAE:
    """Tests for FSQVAE model."""

    def test_forward_pass(self, vae, config):
        """Test FSQ-VAE forward pass."""
        batch_size = 4
        x = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        with torch.no_grad():
            x_recon, z, z_q = vae(x)

        # Check output shapes
        assert x_recon.shape == x.shape
        assert z.shape[0] == batch_size
        assert z_q.shape[0] == batch_size
        assert z_q.shape == z.shape

    def test_loss_computation(self, vae, config):
        """Test loss computation."""
        batch_size = 4
        x = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        with torch.no_grad():
            x_recon, z, z_q = vae(x)
            loss, loss_dict = vae.compute_loss(x, x_recon, z, z_q)

        # Check loss is valid
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Check loss dict contains expected keys
        assert "total_loss" in loss_dict
        assert "recon_loss" in loss_dict

    def test_encoding(self, vae, config):
        """Test encoding observations to quantized latents."""
        batch_size = 4
        x = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        with torch.no_grad():
            z_q, indices = vae.encode(x)

        # Check shapes
        assert z_q.shape[0] == batch_size
        assert indices.shape[0] == batch_size

        # Indices should be valid integers
        assert indices.dtype in [torch.int32, torch.int64, torch.long]

    def test_decoding(self, vae, config):
        """Test decoding quantized latents to observations."""
        batch_size = 4
        x = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        with torch.no_grad():
            z_q, _ = vae.encode(x)
            x_recon = vae.decode(z_q)

        # Reconstruction should match input shape
        assert x_recon.shape == x.shape

    def test_encode_decode_consistency(self, vae, config):
        """Test that encode -> decode produces consistent reconstructions."""
        batch_size = 4
        x = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        with torch.no_grad():
            # Full forward pass
            x_recon1, z, z_q = vae(x)

            # Encode then decode
            z_q_enc, _ = vae.encode(x)
            x_recon2 = vae.decode(z_q_enc)

        # Reconstructions should be identical
        assert torch.allclose(x_recon1, x_recon2, rtol=1e-5)

    def test_codebook_size(self, vae):
        """Test that codebook size is set correctly."""
        codebook_size = vae.quantizer.codebook_size

        assert codebook_size > 0
        assert isinstance(codebook_size, int)

    def test_quantization_deterministic(self, vae, config):
        """Test that quantization is deterministic."""
        batch_size = 4
        x = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        with torch.no_grad():
            z_q1, indices1 = vae.encode(x)
            z_q2, indices2 = vae.encode(x)

        # Same input should produce same quantization
        assert torch.equal(indices1, indices2)
        assert torch.allclose(z_q1, z_q2)

    def test_different_inputs_different_codes(self, vae, config):
        """Test that different inputs produce different codes."""
        batch_size = 4
        x1 = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )
        x2 = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        with torch.no_grad():
            _, indices1 = vae.encode(x1)
            _, indices2 = vae.encode(x2)

        # Different inputs should (almost certainly) produce different codes
        assert not torch.equal(indices1, indices2)


class TestFSQQuantizer:
    """Tests for FSQ quantizer."""

    def test_quantizer_forward(self, vae, config):
        """Test FSQ quantizer forward pass."""
        batch_size = 4
        x = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        with torch.no_grad():
            # Get continuous latent
            z = vae.encoder(x)

            # Quantize
            z_q, indices = vae.quantizer(z)

        # Check shapes
        assert z_q.shape == z.shape
        assert indices.shape[0] == batch_size

    def test_quantizer_levels(self, vae):
        """Test that quantizer uses specified levels."""
        levels = vae.quantizer.levels

        # Levels should be configured
        assert levels is not None
        assert len(levels) > 0
        assert all(isinstance(l, int) and l > 1 for l in levels)

    def test_quantization_bounds(self, vae, config):
        """Test that quantized values are within bounds."""
        batch_size = 4
        x = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        with torch.no_grad():
            _, _, z_q = vae(x)

        # Quantized values should be bounded
        # Exact bounds depend on FSQ levels, but should not be extreme
        assert not torch.isnan(z_q).any()
        assert not torch.isinf(z_q).any()
        assert z_q.abs().max() < 100  # Reasonable bound


class TestVAETraining:
    """Tests for VAE training mode."""

    def test_training_mode(self, vae, config):
        """Test that model can be set to training mode."""
        vae.train()
        assert vae.training

        batch_size = 2
        x = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        # Should work in training mode
        x_recon, z, z_q = vae(x)
        loss, loss_dict = vae.compute_loss(x, x_recon, z, z_q)

        assert loss.requires_grad

    def test_eval_mode(self, vae, config):
        """Test that model can be set to eval mode."""
        vae.eval()
        assert not vae.training

        batch_size = 2
        x = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        with torch.no_grad():
            x_recon, z, z_q = vae(x)
            loss, loss_dict = vae.compute_loss(x, x_recon, z, z_q)

        # In eval mode with no_grad, shouldn't require gradients
        assert not loss.requires_grad

    def test_gradient_flow(self, vae, config):
        """Test that gradients flow through the model."""
        vae.train()

        batch_size = 2
        x = torch.randn(
            batch_size, config.input_channels, config.input_height, config.input_width
        )

        # Forward pass
        x_recon, z, z_q = vae(x)
        loss, _ = vae.compute_loss(x, x_recon, z, z_q)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        has_gradients = False
        for param in vae.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        assert has_gradients, "No gradients computed"


class TestVAEArchitecture:
    """Tests for VAE architecture components."""

    def test_encoder_exists(self, vae):
        """Test that encoder exists and is callable."""
        assert hasattr(vae, "encoder")
        assert callable(vae.encoder)

    def test_decoder_exists(self, vae):
        """Test that decoder exists and is callable."""
        assert hasattr(vae, "decoder")
        assert callable(vae.decoder)

    def test_quantizer_exists(self, vae):
        """Test that quantizer exists and is callable."""
        assert hasattr(vae, "quantizer")
        assert callable(vae.quantizer)

    def test_parameter_count(self, vae):
        """Test that model has trainable parameters."""
        total_params = sum(p.numel() for p in vae.parameters())
        trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0
