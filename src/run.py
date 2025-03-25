import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import gym_tetris
from Agent import DQNAgent




def recorder(episode_id):
    return True
    # return episode_id % 1 == 0


if __name__ == '__main__':
   
    episodes=500
    batch_size=32
    target_update=10
    env = gym.make("LunarLander-v3", render_mode="human")  
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.train(batch_size)

        if episode % target_update == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    env.close()