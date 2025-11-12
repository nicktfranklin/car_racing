import torch.nn as nn


def print_model_info(model: nn.Module, model_name: str):
    """Print model architecture and parameter count."""
    print(f"\n{'=' * 60}")
    print(f"{model_name} Architecture")
    print(f"{'=' * 60}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")

    # Print model structure
    print(f"\nModel structure:")
    print(model)
    print(f"{'=' * 60}\n")
