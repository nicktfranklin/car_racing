"""
Training utilities for optimizers and schedulers.

Provides reusable utilities for creating optimizers and learning rate schedulers
from configuration objects.
"""

import torch.optim as optim

from ..config import OptimizerConfig


def create_optimizer(parameters, opt_config: OptimizerConfig):
    """Create optimizer from config."""
    if opt_config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            parameters,
            lr=opt_config.learning_rate,
            betas=(opt_config.beta1, opt_config.beta2),
            eps=opt_config.epsilon,
            weight_decay=opt_config.weight_decay,
            amsgrad=opt_config.amsgrad,
        )
    elif opt_config.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            parameters,
            lr=opt_config.learning_rate,
            betas=(opt_config.beta1, opt_config.beta2),
            eps=opt_config.epsilon,
            weight_decay=opt_config.weight_decay,
            amsgrad=opt_config.amsgrad,
        )
    elif opt_config.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            parameters,
            lr=opt_config.learning_rate,
            momentum=opt_config.momentum,
            weight_decay=opt_config.weight_decay,
            nesterov=opt_config.nesterov,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config.optimizer}")

    return optimizer


def create_scheduler(optimizer, opt_config: OptimizerConfig, num_epochs: int):
    """Create learning rate scheduler from config."""
    if not opt_config.use_scheduler:
        return None

    scheduler_type = opt_config.scheduler.lower()

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - opt_config.warmup_epochs,
            eta_min=opt_config.min_lr,
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=opt_config.step_size, gamma=opt_config.gamma
        )
    elif scheduler_type == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt_config.gamma)
    elif scheduler_type == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=opt_config.factor,
            patience=opt_config.patience,
            min_lr=opt_config.min_lr,
        )
        return {
            "scheduler": scheduler,
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1,
        }
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    # Add warmup if needed
    if opt_config.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=opt_config.warmup_epochs,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[opt_config.warmup_epochs],
        )

    return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
