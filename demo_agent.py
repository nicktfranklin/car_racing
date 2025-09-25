"""
Real-time agent demonstration for CarRacing environment.
"""

import argparse
import os
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
import torch

from src.world_models import WorldModelAgentConfig, FSQVAE, EvolutionaryController
from src.world_models.agents import Agent, create_agent
from src.world_models.training import VAETrainer


def handle_pygame_events(human_agent):
    """Handle pygame events for human control."""
    keys = pygame.key.get_pressed()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

    # Update human agent actions based on keys
    if hasattr(human_agent, 'update_action'):
        steering = 0.0
        gas = 0.0
        brake = 0.0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering = 1.0

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            gas = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            brake = 1.0

        human_agent.update_action(steering, gas, brake)

    return True


def load_world_model_agent(config: WorldModelAgentConfig, device: str) -> Optional[Agent]:
    """Load trained World Model agent."""
    try:
        # Load VAE
        vae = FSQVAE(config.fsq_vae)
        vae_checkpoint_path = os.path.join(config.training.checkpoint_dir, "vae_latest.pth")

        if os.path.exists(vae_checkpoint_path):
            vae_trainer = VAETrainer(vae, config)
            vae_trainer.load_checkpoint(vae_checkpoint_path)
            print(f"✅ Loaded VAE from {vae_checkpoint_path}")
        else:
            print(f"❌ VAE checkpoint not found: {vae_checkpoint_path}")
            return None

        # Load Controller
        controller = EvolutionaryController(config.controller)
        controller_checkpoint_path = os.path.join(config.training.checkpoint_dir, "best_controller.pth")

        if os.path.exists(controller_checkpoint_path):
            controller.load_state_dict(torch.load(controller_checkpoint_path))
            print(f"✅ Loaded controller from {controller_checkpoint_path}")
        else:
            print(f"❌ Controller checkpoint not found: {controller_checkpoint_path}")
            return None

        from src.world_models.agents import WorldModelAgent
        return WorldModelAgent(vae, controller, device)

    except Exception as e:
        print(f"❌ Failed to load World Model agent: {e}")
        return None


def run_demo(agent: Agent, env_name: str = "CarRacing-v3", max_steps: int = 1000):
    """Run real-time agent demonstration."""
    # Create environment
    env = gym.make(env_name, render_mode="human")

    print(f"🚗 Starting real-time demo with {agent.name}")
    print("Controls (for Human agent): Arrow keys or WASD to control, ESC to quit")
    print("For other agents: Press ESC to quit")

    try:
        episode = 0

        while True:
            episode += 1
            obs, _ = env.reset()
            agent.reset()

            total_reward = 0.0
            step = 0
            start_time = time.time()

            print(f"\n🏁 Episode {episode} started")

            while step < max_steps:
                # Handle pygame events (for human control and quit)
                if hasattr(agent, 'update_action'):
                    if not handle_pygame_events(agent):
                        return
                else:
                    # Check for quit even with non-human agents
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                return

                # Get action from agent
                action = agent.get_action(obs)

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1

                # Small delay for visualization (adjust as needed)
                time.sleep(0.02)  # ~50 FPS

                if terminated or truncated:
                    break

            elapsed_time = time.time() - start_time
            print(f"🏆 Episode {episode} completed:")
            print(f"   Steps: {step}")
            print(f"   Total reward: {total_reward:.2f}")
            print(f"   Duration: {elapsed_time:.1f}s")

            # Brief pause between episodes
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    finally:
        env.close()
        print("👋 Demo ended")


def main():
    parser = argparse.ArgumentParser(description="Real-time agent demonstration")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["random", "human", "world_model", "constant"],
        default="random",
        help="Agent type to use"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path (required for world_model agent)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CarRacing-v3",
        help="Environment name"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for World Model agent (cpu/cuda/mps/auto)"
    )
    parser.add_argument(
        "--constant-action",
        type=float,
        nargs=3,
        default=[0.0, 0.5, 0.0],
        help="Action for constant agent [steering, gas, brake]"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"🖥️  Using device: {device}")

    # Create environment to get action space
    temp_env = gym.make(args.env)
    action_space = temp_env.action_space
    temp_env.close()

    # Create agent
    agent = None

    if args.agent == "world_model":
        if args.config is None:
            print("❌ Config file required for world_model agent")
            return

        try:
            config = WorldModelAgentConfig.from_yaml(args.config)
            agent = load_world_model_agent(config, device)
            if agent is None:
                print("🔄 Falling back to random agent")
                agent = create_agent("random", action_space=action_space)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            print("🔄 Falling back to random agent")
            agent = create_agent("random", action_space=action_space)

    else:
        # Create other agent types
        kwargs = {"action_space": action_space}
        if args.agent == "constant":
            kwargs["action"] = args.constant_action

        agent = create_agent(args.agent, **kwargs)

    if agent is None:
        print("❌ Failed to create agent")
        return

    # Initialize pygame for human control
    if args.agent == "human":
        pygame.init()
        pygame.display.set_mode((1, 1))  # Minimal window for event handling

    # Run demonstration
    run_demo(agent, args.env, args.max_steps)


if __name__ == "__main__":
    main()