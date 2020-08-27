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

    buffer_size = 500000
    epsilon = 0.95
    batch_size = 32
    gamma = 0.95
    eta = 0.001
    update_target = 10000


    debug = True
    epoch = 10000000 # One epoch is forward and backward in a neural net. We can see this as nb observation
    #episodes = 1000000 # Number of "games"

    env = wrappers.Monitor(env, video_callable=recorder, directory=outdir, force=True)
    env = Preprocess(env, train=False)

    env.seed(0)
    agent = Agent(gamma=gamma,
                  lr=eta,
                  buffer_size=buffer_size,
                  update_target=update_target,
                  epsilon=epsilon,
                  action_space=env.action_space,
                  batch_size=batch_size,
                  pretrained_path=None)

    print("Action Space : " + str(env.action_space))
    print("Meanings : " + str(env.get_action_meanings()))
    print("BUFFER SIZE : " + str(buffer_size))
    print("EPSILON : " + str(epsilon))
    print("Batch_size : " + str(batch_size))
    print("Gamma : " + str(gamma))
    print("LR : " + str(eta))
    print("update target net : " + str(update_target))


    i = 0
    results = []

    while i < epoch:
        ob, reward, done, _ = env.reset()
        nb_iter = 0
        score = 0
        while not done:
            action = agent.get_action(ob)
            prec_ob = ob
            ob, reward, done, _ = env.step(action)
            interaction = (prec_ob, action, ob, reward, done)
            # print(interaction)
            agent.memorise(interaction)
            agent.learn()
            nb_iter += 1
            score += reward
        i += nb_iter
        print("TRAIN")
        print("SCORE : " + str(score))
        print("Iteration : " + str(nb_iter))

        results.append(score)

