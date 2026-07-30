import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def print_environment_setup():
    print("Task 1: Environment Setup")
    print(f"Gymnasium version: {gym.__version__}")
    print("Imported libraries: gymnasium, numpy, matplotlib")
    print()


def create_environment():
    print("Task 2: Create and Initialize an RL Environment")
    env = gym.make("CartPole-v1")
    observation, info = env.reset()

    print("Environment: CartPole-v1")
    print(f"Initial observation: {observation}")
    print(f"Environment info: {info}")
    print()

    return env, observation, info


def explore_spaces(env):
    print("Task 3: Explore Observation and Action Spaces")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Type of observation space: {type(env.observation_space)}")
    print(f"Number of possible actions: {env.action_space.n}")
    print()


def run_random_agent(env):
    print("Task 4: Execute a Random Agent")
    observation, info = env.reset()
    terminated = False
    truncated = False
    step_number = 0
    cumulative_reward = 0
    rewards = []

    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        step_number += 1
        cumulative_reward += reward
        rewards.append(cumulative_reward)

        episode_status = "terminated" if terminated else "truncated" if truncated else "running"
        print(
            f"Step: {step_number}, "
            f"Action: {action}, "
            f"Observation: {observation}, "
            f"Reward: {reward}, "
            f"Episode status: {episode_status}"
        )

    print()
    print(f"Total number of steps: {step_number}")
    print(f"Cumulative reward: {cumulative_reward}")
    print()

    return rewards


def plot_episode_rewards(rewards):
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(rewards) + 1), rewards, marker="o")
    plt.title("CartPole-v1 Random Agent Cumulative Reward")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    print_environment_setup()
    env, observation, info = create_environment()
    explore_spaces(env)
    rewards = run_random_agent(env)
    env.close()

    plot_episode_rewards(rewards)


if __name__ == "__main__":
    main()
