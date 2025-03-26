import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from Agent import DQNAgent
import ale_py.roms as roms
from preprocess import FrameProcessor



def recorder(episode_id):
    return True
    # return episode_id % 1 == 0


if __name__ == '__main__':
    print(roms.get_all_rom_ids())
    episodes = 1000
    batch_size = 32
    target_update = 50
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array") 

    frameproc = FrameProcessor()
    state_dim = 4
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    for episode in range(episodes):
        state, _ = env.reset()
        state = frameproc.process(state)
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = frameproc.process(next_state)
            
            agent.add_experience(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.train(batch_size)

        agent.print_avg_qvalue()
        if episode % target_update == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    env.close()