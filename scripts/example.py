#!/usr/bin/env python3
"""
Example script demonstrating World Model usage.
"""

import numpy as np
import torch

from world_models import (
    FSQVAE,
    EvolutionaryController,
    WorldModel,
    WorldModelAgentConfig,
)


def main():
    print("🏎️ World Model Agent Example")
    print("=" * 50)

    # Load configuration
    config = WorldModelAgentConfig()
    config.validate_consistency()

    print(f"FSQ levels: {config.fsq_vae.fsq_levels}")
    print(f"Codebook size: {np.prod(config.fsq_vae.fsq_levels)}")
    print(f"State dimension: {config.controller.state_dim}")
    print(f"Action dimension: {config.controller.action_dim}")

    # Create models
    vae = FSQVAE(config.fsq_vae)
    world_model = WorldModel(config.world_model)
    controller = EvolutionaryController(config.controller)

    print(f"\n📊 Model Statistics:")
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(
        f"World Model parameters: {sum(p.numel() for p in world_model.parameters()):,}"
    )
    print(f"Controller parameters: {sum(p.numel() for p in controller.parameters()):,}")

    # Demonstrate forward passes
    print(f"\n🔄 Testing Forward Passes:")

    # 1. VAE encode/decode
    batch_size = 4
    images = torch.randn(batch_size, 3, 64, 64)

    with torch.no_grad():
        # Encode to state representation
        z_q, state_indices = vae.encode(images)
        reconstructed = vae.decode(z_q)

        print(f"✓ VAE: {images.shape} -> {z_q.shape} -> {reconstructed.shape}")

        # 2. Controller action selection
        actions = controller(z_q)
        print(f"✓ Controller: {z_q.shape} -> {actions.shape}")

        # 3. World model prediction
        actions_seq = actions.unsqueeze(1)  # Add time dimension
        state_indices_seq = state_indices.unsqueeze(1)

        next_state_logits, rewards, dones, _ = world_model(
            state_indices_seq, actions_seq
        )
        print(f"✓ World Model: {state_indices_seq.shape} -> {next_state_logits.shape}")

        # Sample next state
        next_state_probs = torch.softmax(next_state_logits, dim=-1)
        next_state_indices = torch.multinomial(next_state_probs.squeeze(1), 1)

        # Convert back to FSQ representation
        from world_models.models.world_model import indices_to_fsq

        next_z_q = indices_to_fsq(
            next_state_indices.squeeze(-1), config.fsq_vae.fsq_levels
        )

        # Decode next state
        next_images = vae.decode(next_z_q)
        print(f"✓ Complete cycle: {images.shape} -> ... -> {next_images.shape}")

    print(f"\n🎯 Action Ranges (for CarRacing):")
    print(
        f"Steering: [{actions[:, 0].min():.3f}, {actions[:, 0].max():.3f}] (should be [-1, 1])"
    )
    print(
        f"Gas: [{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}] (should be [0, 1])"
    )
    print(
        f"Brake: [{actions[:, 2].min():.3f}, {actions[:, 2].max():.3f}] (should be [0, 1])"
    )

    print(f"\n📈 Loss Computation Example:")
    with torch.no_grad():
        # VAE loss
        x_recon, z, z_q = vae(images)
        vae_loss, vae_loss_dict = vae.compute_loss(images, x_recon, z, z_q)

        # World model loss
        seq_len = 5
        current_states = torch.randint(
            0, world_model.num_state_tokens, (batch_size, seq_len)
        )
        actions_seq = torch.randn(batch_size, seq_len, 3)
        next_states = torch.randint(
            0, world_model.num_state_tokens, (batch_size, seq_len)
        )
        rewards_seq = torch.randn(batch_size, seq_len)
        dones_seq = torch.randint(0, 2, (batch_size, seq_len))

        wm_loss, wm_loss_dict = world_model.compute_loss(
            current_states, actions_seq, next_states, rewards_seq, dones_seq
        )

        print(f"VAE Loss: {vae_loss:.4f}")
        print(f"  - Reconstruction: {vae_loss_dict['recon_loss']:.4f}")
        print(f"  - Commitment: {vae_loss_dict['commitment_loss']:.4f}")

        print(f"World Model Loss: {wm_loss:.4f}")
        print(f"  - State Loss: {wm_loss_dict['state_loss']:.4f}")
        print(f"  - Reward Loss: {wm_loss_dict['reward_loss']:.4f}")
        print(f"  - Done Loss: {wm_loss_dict['done_loss']:.4f}")
        print(f"  - State Accuracy: {wm_loss_dict['state_accuracy']:.3f}")

    print(f"\n🚀 Ready for Training!")
    print("Run the following commands:")
    print("1. python main.py --stage collect    # Collect training data")
    print("2. python main.py --stage vae        # Train FSQ-VAE")
    print("3. python main.py --stage world_model # Train World Model")
    print("4. python main.py --stage controller # Train Controller")
    print("5. python main.py --stage all        # Run full pipeline")


if __name__ == "__main__":
    main()
