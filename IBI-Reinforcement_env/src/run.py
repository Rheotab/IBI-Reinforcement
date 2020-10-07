import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
import random
import gym
from gym import wrappers

from Agent import Agent
import numpy as np
import torch
import time

from preprocess import Preprocess


def recorder(episode_id):
    return True
    # return episode_id % 1 == 0


if __name__ == '__main__':
    env_id = 'BreakoutNoFrameskip-v4'

    env = gym.make(env_id)

    outdir = '/tmp/BreakoutNoFrameskip-v4'

    batch_size = 3

    max_frames = 1000000

    env = wrappers.Monitor(env, video_callable=recorder, directory=outdir, force=True)
    env = Preprocess(env, train=False)

    env.seed(0)
    agent = Agent(env=env)

    i = 0
    results = []
    observation, _, _, _ = env.reset()
    score = 0
    current_frame = 0
    for frame in range(max_frames):
        current_frame += 1
        action = agent.get_action(observation)
        prec_ob = observation
        observation, reward, done, _ = env.step(action)
        interaction = (prec_ob, action, observation, reward, done)
        # print(interaction)
        agent.memorise(interaction)
        score += reward
        if len(agent.memory.buffer) > batch_size:
            agent.update()
        if done:
            results.append(score)
            print("Frame :" + str(frame) + ", score : " + str(score))
            score = 0
            current_frame = 0
            observation = env.reset()

        results.append(score)
