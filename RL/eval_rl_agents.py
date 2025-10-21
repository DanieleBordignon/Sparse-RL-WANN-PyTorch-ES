import os

import numpy as np
from stable_baselines3 import PPO, A2C, DQN, SAC
from domain.make_env import make_env


def evaluate_agent(args):
    algo = args.algo
    saved_dir = args.saved
    episodes = 10
    visualize = True

    # Initialize the environment
    env = make_env(args.task)

    # Load the model
    model_path = os.path.join(saved_dir, f"{args.task}_{args.algo}")
    if algo == 'PPO':
        model = PPO.load(model_path)
    elif algo == 'A2C':
        model = A2C.load(model_path)
    elif algo == 'DQN':
        model = DQN.load(model_path)
    else:
        raise ValueError(f"Algorithm {algo} not supported")

    if True:
        # Print details about the model for debugging
        print("Model Architecture:")
        print(model.policy)
        print("Model Hyperparameters:")
        print(model.__dict__)
        print("Optimizer Details:")
        print(model.policy.optimizer)

    total_rewards = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0 # Initialize step counter

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1 # Increment step counter

            if visualize:
                env.render()

        total_rewards.append(episode_reward)
        termination_reason = "Goal Reached" if terminated else "Max Steps Reached"
        print(f"Episode {episode + 1}/{episodes}: Reward = {episode_reward:.2f}, Steps = {episode_steps}, Reason = {termination_reason}")

    env.close()

    # Print and log results
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

