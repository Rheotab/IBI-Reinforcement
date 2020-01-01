import matplotlib.pyplot as plt
import gym
import random
from gym import wrappers, logger
import numpy as np
import time
from Agent_Breakout import Agent
from tqdm import tqdm
# from atari_preprocess import AtariPreprocessing
from preprocess import Preprocess

if __name__ == '__main__':

    # HYPERPARAMETERS
    episode_count = 500

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env_id = 'BreakoutNoFrameskip-v4'

    env = gym.make(env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().

    outdir = '/tmp/BreakoutNoFrameskip-v4'

    def recorder(episode_id):
        return episode_id % 10 == 0


    env = Preprocess(env)
    env = wrappers.Monitor(env, video_callable=recorder, directory=outdir, force=True)
    env.seed(0)
    agent = Agent(nb_ep=episode_count, action_space=env.action_space, buffer_size=2000, epsilon=0.5, batch_size=64,
                  gamma=0.8, eta=0.0005, N=500)

    reward = 0
    done = False

    results = []

    for i in tqdm(range(episode_count)):
        ob, reward, done, _ = env.reset()
        nb_iter = 0
        done = False
        while not done:
            action = agent.act(ob, reward, done)
            prec_ob = ob
            ob, reward, done, _ = env.step(action)
            interaction = (prec_ob, action, ob, reward, done)
            # print(interaction)
            agent.memorise(interaction)
            agent.learn()
            nb_iter += 1
        print("EP " + str(i) + " - score " + str(nb_iter))
        results.append(nb_iter)
    env.close()
    agent.show_mean_loss_ep()
    plt.plot(results)
    plt.ylabel('number of iterations')
    plt.show()
    # Note there's no env.render() here. But the environment still can open window and
    # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
    # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    # env.close()
