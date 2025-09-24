"""
Training loops for World Model components.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import WorldModelAgentConfig
from .data_collection import DataCollector, ImageDataset, SequenceDataset
from .models.controller import Controller, EvolutionaryController, evaluate_controller
from .models.fsq_vae import FSQVAE
from .models.world_model import WorldModel, fsq_to_indices


class VAETrainer:
    """Trainer for FSQ-VAE."""

    def __init__(self, model: FSQVAE, config: WorldModelAgentConfig):
        self.model = model
        self.config = config
        self.device = torch.device(
            config.training.device if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.fsq_vae.learning_rate
        )

    def train(self, dataset: ImageDataset, num_epochs: int) -> Dict[str, List[float]]:
        """Train the VAE."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.model.train()
        history = {"loss": [], "recon_loss": [], "commitment_loss": []}

        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_recon_losses = []
            epoch_commitment_losses = []

            pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{num_epochs}")
            for batch_idx, images in enumerate(pbar):
                images = images.to(self.device)

                # Forward pass
                x_recon, z, z_q = self.model(images)
                loss, loss_dict = self.model.compute_loss(images, x_recon, z, z_q)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track losses
                epoch_losses.append(loss_dict["total_loss"])
                epoch_recon_losses.append(loss_dict["recon_loss"])
                epoch_commitment_losses.append(loss_dict["commitment_loss"])

                # Update progress bar
                pbar.set_postfix(
                    {
                        "Loss": f"{loss_dict['total_loss']:.4f}",
                        "Recon": f"{loss_dict['recon_loss']:.4f}",
                        "Commit": f"{loss_dict['commitment_loss']:.4f}",
                    }
                )


            # Record epoch averages
            history["loss"].append(np.mean(epoch_losses))
            history["recon_loss"].append(np.mean(epoch_recon_losses))
            history["commitment_loss"].append(np.mean(epoch_commitment_losses))

            print(f"Epoch {epoch+1}: Loss = {history['loss'][-1]:.4f}")

        return history

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.fsq_vae,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class WorldModelTrainer:
    """Trainer for LSTM world model."""

    def __init__(
        self, world_model: WorldModel, vae: FSQVAE, config: WorldModelAgentConfig
    ):
        self.world_model = world_model
        self.vae = vae
        self.config = config
        self.device = torch.device(
            config.training.device if torch.cuda.is_available() else "cpu"
        )

        self.world_model.to(self.device)
        self.vae.to(self.device)
        self.vae.eval()  # Keep VAE in eval mode

        self.optimizer = optim.Adam(
            self.world_model.parameters(), lr=config.world_model.learning_rate
        )

    def train(
        self, dataset: SequenceDataset, num_epochs: int
    ) -> Dict[str, List[float]]:
        """Train the world model."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.world_model.train()
        history = {
            "loss": [],
            "state_loss": [],
            "reward_loss": [],
            "done_loss": [],
            "state_accuracy": [],
        }

        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_state_losses = []
            epoch_reward_losses = []
            epoch_done_losses = []
            epoch_accuracies = []

            pbar = tqdm(dataloader, desc=f"WorldModel Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                observations = batch["observations"].to(
                    self.device
                )  # (B, T+1, C, H, W)
                actions = batch["actions"].to(self.device)  # (B, T, action_dim)
                rewards = batch["rewards"].to(self.device)  # (B, T)
                dones = batch["dones"].to(self.device)  # (B, T)

                batch_size, seq_len_plus_one = observations.shape[:2]
                seq_len = seq_len_plus_one - 1

                # Encode observations to state indices
                with torch.no_grad():
                    # Reshape observations for VAE
                    obs_flat = observations.reshape(-1, *observations.shape[2:])
                    z_q, indices = self.vae.encode(obs_flat)

                    # Reshape back to sequences
                    indices = indices.reshape(batch_size, seq_len_plus_one)

                # Get current and next state indices
                current_states = indices[:, :-1]  # (B, T)
                next_states = indices[:, 1:]  # (B, T)

                # Forward pass through world model
                loss, loss_dict = self.world_model.compute_loss(
                    current_states, actions, next_states, rewards, dones
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.world_model.parameters(), max_norm=1.0
                )
                self.optimizer.step()

                # Track losses
                epoch_losses.append(loss_dict["total_loss"])
                epoch_state_losses.append(loss_dict["state_loss"])
                epoch_reward_losses.append(loss_dict["reward_loss"])
                epoch_done_losses.append(loss_dict["done_loss"])
                epoch_accuracies.append(loss_dict["state_accuracy"])

                # Update progress bar
                pbar.set_postfix(
                    {
                        "Loss": f"{loss_dict['total_loss']:.4f}",
                        "StateAcc": f"{loss_dict['state_accuracy']:.3f}",
                        "StateLoss": f"{loss_dict['state_loss']:.4f}",
                    }
                )


            # Record epoch averages
            history["loss"].append(np.mean(epoch_losses))
            history["state_loss"].append(np.mean(epoch_state_losses))
            history["reward_loss"].append(np.mean(epoch_reward_losses))
            history["done_loss"].append(np.mean(epoch_done_losses))
            history["state_accuracy"].append(np.mean(epoch_accuracies))

            print(
                f"Epoch {epoch+1}: Loss = {history['loss'][-1]:.4f}, "
                f"State Accuracy = {history['state_accuracy'][-1]:.3f}"
            )

        return history

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.world_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.world_model,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.world_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class ControllerTrainer:
    """Trainer for controller using evolutionary strategy."""

    def __init__(
        self, vae: FSQVAE, world_model: WorldModel, config: WorldModelAgentConfig
    ):
        self.vae = vae
        self.world_model = world_model
        self.config = config
        self.device = torch.device(
            config.training.device if torch.cuda.is_available() else "cpu"
        )

        # Move models to device and set to eval mode
        self.vae.to(self.device).eval()
        self.world_model.to(self.device).eval()

        # Create population of controllers
        self.population_size = 16
        self.num_elite = 4
        self.mutation_strength = 0.02

        self.population = [
            EvolutionaryController(config.controller).to(self.device)
            for _ in range(self.population_size)
        ]

    def train(self, num_generations: int) -> Dict[str, List[float]]:
        """Train controllers using evolutionary strategy."""
        history = {"best_fitness": [], "mean_fitness": []}

        for generation in range(num_generations):
            # Evaluate population
            fitness_scores = []

            pbar = tqdm(
                self.population, desc=f"Generation {generation+1}/{num_generations}"
            )
            for controller in pbar:
                fitness = self._evaluate_controller(controller)
                fitness_scores.append(fitness)
                pbar.set_postfix({"Fitness": f"{fitness:.2f}"})

            # Sort population by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order

            # Track statistics
            best_fitness = fitness_scores[sorted_indices[0]]
            mean_fitness = np.mean(fitness_scores)
            history["best_fitness"].append(best_fitness)
            history["mean_fitness"].append(mean_fitness)

            print(
                f"Generation {generation+1}: Best = {best_fitness:.2f}, "
                f"Mean = {mean_fitness:.2f}"
            )


            # Create next generation
            if generation < num_generations - 1:
                new_population = []

                # Keep elite
                for i in range(self.num_elite):
                    elite_idx = sorted_indices[i]
                    new_population.append(self.population[elite_idx])

                # Generate offspring
                while len(new_population) < self.population_size:
                    # Select parent from elite
                    parent_idx = np.random.choice(self.num_elite)
                    parent = self.population[sorted_indices[parent_idx]]

                    # Create offspring
                    offspring = EvolutionaryController(self.config.controller).to(
                        self.device
                    )
                    offspring.load_state_dict(parent.state_dict())
                    offspring.mutate(self.mutation_strength)

                    new_population.append(offspring)

                self.population = new_population

        return history

    def _evaluate_controller(self, controller: EvolutionaryController) -> float:
        """Evaluate a controller using the world model."""
        controller.eval()
        total_return = 0.0
        num_episodes = 3
        max_steps = 200

        with torch.no_grad():
            for _ in range(num_episodes):
                # Initialize environment state (random initial observation)
                obs = torch.randn(1, 3, 64, 64, device=self.device)

                # Encode initial state
                z_q, state_indices = self.vae.encode(obs)

                # Initialize world model hidden state
                hidden = self.world_model.init_hidden(1, self.device)

                episode_return = 0.0
                done = False

                for step in range(max_steps):
                    if done:
                        break

                    # Get action from controller
                    action = controller(z_q.squeeze(0))  # Remove batch dimension
                    action = action.unsqueeze(0).unsqueeze(
                        0
                    )  # Add batch and time dimensions

                    # Predict next state using world model
                    next_state_logits, reward, done_logit, hidden = self.world_model(
                        state_indices, action, hidden
                    )

                    # Sample next state
                    next_state_probs = torch.softmax(next_state_logits, dim=-1)
                    next_state_indices = torch.multinomial(
                        next_state_probs.squeeze(1), 1
                    )

                    # Convert back to FSQ representation
                    from models.world_model import indices_to_fsq

                    z_q = indices_to_fsq(
                        next_state_indices.squeeze(-1), self.config.fsq_vae.fsq_levels
                    )
                    z_q = z_q.unsqueeze(0)  # Add batch dimension

                    # Update episode return
                    episode_return += reward.item()

                    # Check if done
                    done = torch.sigmoid(done_logit).item() > 0.5

                    # Update state
                    state_indices = next_state_indices

                total_return += episode_return

        return total_return / num_episodes

    def get_best_controller(self) -> EvolutionaryController:
        """Get the best controller from the population."""
        fitness_scores = []
        for controller in self.population:
            fitness = self._evaluate_controller(controller)
            fitness_scores.append(fitness)

        best_idx = np.argmax(fitness_scores)
        return self.population[best_idx]

    def save_checkpoint(self, filepath: str):
        """Save population checkpoint."""
        checkpoint = {
            "population": [controller.state_dict() for controller in self.population],
            "config": self.config.controller,
        }
        torch.save(checkpoint, filepath)


if __name__ == "__main__":
    # Test training components
    from .config import WorldModelAgentConfig

    config = WorldModelAgentConfig()
    config.validate_consistency()

    # Create dummy data for testing
    print("Creating test models...")
    vae = FSQVAE(config.fsq_vae)
    world_model = WorldModel(config.world_model)

    print("Models created successfully!")
    print(f"VAE codebook size: {vae.quantizer.codebook_size}")
    print(f"World model state tokens: {world_model.num_state_tokens}")

    # Test can create trainers
    vae_trainer = VAETrainer(vae, config)
    world_model_trainer = WorldModelTrainer(world_model, vae, config)
    controller_trainer = ControllerTrainer(vae, world_model, config)

    print("All trainers created successfully!")
