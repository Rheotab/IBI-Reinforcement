import matplotlib.pyplot as plt
import gym
import random
from gym import wrappers, logger
from Agent1 import Agent
import numpy as np
import time


if __name__ == '__main__':

    #HYPERPARAMETERS
    episode_count = 400
    epsilon = 0.3
    batch_size = 200
    gamma = 0.05

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env_id = 'CartPole-v0'

    env = gym.make(env_id)


    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/CartPole-v0'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = Agent(env.action_space, buffer_size=100000, epsilon=0.3, batch_size=100, gamma=0.01, eta=0.001, N=10000)


    reward = 0
    done = False

    results = []

    for i in range(episode_count):
        print("EP " + str(i))
        ob = env.reset()
        nb_iter = 0
        done = False
        while not done:
            action = agent.act(ob, reward, done)
            prec_ob = ob
            ob, reward, done, _ = env.step(action)
            interaction = (prec_ob, action, ob, reward, done)
            #print(interaction)
            agent.memorise(interaction)
            agent.learn()
            nb_iter+=1
        results.append(nb_iter)
    agent.show_loss()
    plt.plot(results)
    plt.ylabel('number of iterations')
    plt.show()
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    # env.close()