import matplotlib.pyplot as plt
import torch
import gym
from gym import wrappers, logger
from DQN import DQN
import math
import numpy as np
import random as rand

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, buffer_size=10000):
        self.action_space = action_space
        self.buffer_size = buffer_size
        self.buffer = []

        self.qlearning_nn = DQN(128)

    def act(self, observation, reward, done):
        qvalues = self.qlearning_nn(torch.Tensor(observation).reshape(1, 4))
        self.politique_boltzmann(qvalues, 0.1)

        return self.action_space.sample()

    def memorise(self, interaction):
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        self.buffer.append(interaction)

    def politique_boltzmann(self, qval, tau):
        qval_np = qval.detach().numpy()
        s = 0
        prob = np.array([])
        for i in qval_np[0]:
            s += np.exp(i/tau)
        for a in qval_np:
            p_a = np.exp(a/tau)
            prob = np.append(prob, (p_a/s))
            r = random.uniform(0,1)
            if r < prob[0]:
                return self.action_space[0]
            else:
                return self.action_space[1]



           



if __name__ == '__main__':

    #HYPERPARAMETERS
    episode_count = 1000
    epsilon = 0.3

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env_id = 'CartPole-v0'

    env = gym.make(env_id)


    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)


    reward = 0
    done = False

    results = []

    for i in range(episode_count):
        ob = env.reset()



        nb_iter = 0
        while True:
            action = agent.act(ob, reward, done)
            prec_ob = ob
            ob, reward, done, _ = env.step(action)

            interaction = (prec_ob, action, ob, reward, done)
            #print(interaction)
            agent.memorise(interaction)

            nb_iter+=1
            if done:
                break

        results.append(nb_iter)

    plt.plot(results)
    plt.ylabel('number of iterations')
    plt.show()
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    # env.close()