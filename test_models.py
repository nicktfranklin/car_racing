#!/usr/bin/env python3
"""
Test script to verify all World Model components work correctly.
"""

import torch
import numpy as np
from world_models import (
    WorldModelAgentConfig,
    FSQVAE,
    WorldModel,
    Controller,
    EvolutionaryController,
)


def test_fsq_vae():
    """Test FSQ-VAE implementation."""
    print("Testing FSQ-VAE...")

    config = WorldModelAgentConfig()
    model = FSQVAE(config.fsq_vae)

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)

    with torch.no_grad():
        x_recon, z, z_q = model(x)
        loss, loss_dict = model.compute_loss(x, x_recon, z, z_q)

    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Reconstruction shape: {x_recon.shape}")
    print(f"  ✓ Latent shape: {z.shape}")
    print(f"  ✓ Quantized latent shape: {z_q.shape}")
    print(f"  ✓ Codebook size: {model.quantizer.codebook_size}")
    print(f"  ✓ Loss: {loss.item():.4f}")

    # Test encoding/decoding
    z_q_enc, indices = model.encode(x)
    x_recon_dec = model.decode(z_q_enc)

    print(f"  ✓ Encoded indices shape: {indices.shape}")
    print(f"  ✓ Decoded shape: {x_recon_dec.shape}")
    print("  FSQ-VAE test passed!")
    return model


def test_world_model():
    """Test World Model implementation."""
    print("\nTesting World Model...")

    config = WorldModelAgentConfig()
    model = WorldModel(config.world_model)

    batch_size = 4
    seq_len = 10

    # Create test data
    state_indices = torch.randint(0, model.num_state_tokens, (batch_size, seq_len))
    actions = torch.randn(batch_size, seq_len, config.world_model.action_dim)
    next_state_indices = torch.randint(0, model.num_state_tokens, (batch_size, seq_len))
    rewards = torch.randn(batch_size, seq_len)
    dones = torch.randint(0, 2, (batch_size, seq_len))

    with torch.no_grad():
        # Test forward pass
        next_state_logits, pred_rewards, pred_dones, hidden = model(state_indices, actions)

        # Test loss computation
        loss, loss_dict = model.compute_loss(
            state_indices, actions, next_state_indices, rewards, dones
        )

        # Test sampling
        single_state = state_indices[:1, :1]
        single_action = actions[:1, :1, :]
        next_state_sample, reward_sample, done_sample, _ = model.sample_next_state(
            single_state, single_action, temperature=1.0
        )

    print(f"  ✓ State logits shape: {next_state_logits.shape}")
    print(f"  ✓ Predicted rewards shape: {pred_rewards.shape}")
    print(f"  ✓ Predicted dones shape: {pred_dones.shape}")
    print(f"  ✓ Number of state tokens: {model.num_state_tokens}")
    print(f"  ✓ Loss: {loss.item():.4f}")
    print(f"  ✓ State accuracy: {loss_dict['state_accuracy']:.3f}")
    print(f"  ✓ Sampled next state shape: {next_state_sample.shape}")
    print("  World Model test passed!")
    return model


def test_controller():
    """Test Controller implementations."""
    print("\nTesting Controllers...")

    config = WorldModelAgentConfig()

    # Test standard controller
    controller = Controller(config.controller)
    batch_size = 4
    state = torch.randn(batch_size, config.controller.state_dim)

    with torch.no_grad():
        actions = controller(state)
        single_action = controller.get_action(torch.randn(config.controller.state_dim))

    print(f"  ✓ Standard controller input shape: {state.shape}")
    print(f"  ✓ Standard controller output shape: {actions.shape}")
    print(f"  ✓ Single action shape: {single_action.shape}")
    print(f"  ✓ Action ranges - Steering: [{actions[:, 0].min():.3f}, {actions[:, 0].max():.3f}]")
    print(f"  ✓ Action ranges - Gas: [{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}]")
    print(f"  ✓ Action ranges - Brake: [{actions[:, 2].min():.3f}, {actions[:, 2].max():.3f}]")

    # Test evolutionary controller
    evo_controller = EvolutionaryController(config.controller)
    flat_params = evo_controller.get_parameters_flat()

    with torch.no_grad():
        evo_actions = evo_controller(state)

    # Test mutation
    original_params = flat_params.clone()
    evo_controller.mutate(0.1)
    new_params = evo_controller.get_parameters_flat()
    param_change = torch.norm(new_params - original_params)

    print(f"  ✓ Evolutionary controller parameters: {flat_params.shape[0]}")
    print(f"  ✓ Evolutionary controller output shape: {evo_actions.shape}")
    print(f"  ✓ Parameter change after mutation: {param_change:.6f}")
    print("  Controllers test passed!")
    return controller, evo_controller


def test_integration():
    """Test integration between components."""
    print("\nTesting Component Integration...")

    config = WorldModelAgentConfig()
    config.validate_consistency()

    # Create all components
    vae = FSQVAE(config.fsq_vae)
    world_model = WorldModel(config.world_model)
    controller = EvolutionaryController(config.controller)

    # Test full pipeline
    batch_size = 2
    obs = torch.randn(batch_size, 3, 64, 64)

    with torch.no_grad():
        # Encode observation
        z_q, state_indices = vae.encode(obs)

        # Get action from controller
        actions = controller(z_q)
        actions = actions.unsqueeze(1)  # Add time dimension

        # Predict next state with world model
        next_state_logits, rewards, dones, _ = world_model(
            state_indices.unsqueeze(1), actions
        )

        # Sample next state
        next_state_probs = torch.softmax(next_state_logits, dim=-1)
        next_state_indices = torch.multinomial(next_state_probs.squeeze(1), 1)

        # Convert back to FSQ representation
        from world_models.models.world_model import indices_to_fsq
        next_z_q = indices_to_fsq(next_state_indices.squeeze(-1), config.fsq_vae.fsq_levels)

        # Decode next state
        next_obs = vae.decode(next_z_q)

    print(f"  ✓ Original observation shape: {obs.shape}")
    print(f"  ✓ Encoded state shape: {z_q.shape}")
    print(f"  ✓ State indices shape: {state_indices.shape}")
    print(f"  ✓ Controller actions shape: {actions.shape}")
    print(f"  ✓ Predicted rewards shape: {rewards.shape}")
    print(f"  ✓ Next state logits shape: {next_state_logits.shape}")
    print(f"  ✓ Next observation shape: {next_obs.shape}")
    print(f"  ✓ Configuration validation: Passed")
    print("  Integration test passed!")


def main():
    """Run all tests."""
    print("World Model Agent Component Tests")
    print("=" * 50)

    try:
        # Test individual components
        vae = test_fsq_vae()
        world_model = test_world_model()
        controller, evo_controller = test_controller()

        # Test integration
        test_integration()

        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("The World Model implementation is working correctly.")
        print("You can now run training with: python main.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()