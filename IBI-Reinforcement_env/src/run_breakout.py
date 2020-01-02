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
        return True
        #return episode_id % 1 == 0


    buffer_size = 200000
    epsilon = 0.2
    batch_size = 32
    gamma = 0.95
    eta = 0.001
    N = 500


    env = wrappers.Monitor(env, video_callable=recorder, directory=outdir, force=True)
    env = Preprocess(env)
    env.seed(0)
    agent = Agent(nb_ep=episode_count, action_space=env.action_space, buffer_size=buffer_size, epsilon=epsilon, batch_size=batch_size,
                  gamma=gamma, eta=eta, N=N)


    reward = 0
    done = False
    debug = True
    if debug:
        print("NB EP : " + str(episode_count))
        print("Action Space : " + str(env.action_space))
        print("Meanings : " + str(env.get_action_meanings()))
        print("BUFFER SIZE : " + str(buffer_size))
        print("EPSILON : " + str(epsilon))
        print("Batch_size : " + str(batch_size))
        print("Gamma : " + str(gamma))
        print("LR : " + str(eta))
        print("update target net : " + str(N))

    results = []

    for i in tqdm(range(episode_count)):
        ob, reward, done, _ = env.reset()
        nb_iter = 0
        done = False
        score = 0
        while not done:
            action = agent.act(ob, reward, done)
            prec_ob = ob
            ob, reward, done, _ = env.step(action)
            interaction = (prec_ob, action, ob, reward, done)
            # print(interaction)
            agent.memorise(interaction)
            agent.learn()
            nb_iter += 1
            score += reward
        print("EP " + str(i) + " - score " + str(reward))
        print("EP " + str(i) + " - itertion " + str(nb_iter))
        print("I saw " + agent.how_many_did_u_see() + " interaction so far")

        results.append(nb_iter)
    env.close()
    plt.plot(results)
    plt.ylabel('number of iterations')
    plt.xlabel('score')
    plt.show()
    # Note there's no env.render() here. But the environment still can open window and
    # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
    # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    # env.close()
