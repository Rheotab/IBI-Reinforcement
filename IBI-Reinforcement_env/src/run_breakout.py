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
    episode_count = 100

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env_id = 'BreakoutNoFrameskip-v4'

    env_test = gym.make(env_id)
    env_train = gym.make(env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().

    outdir = '/tmp/BreakoutNoFrameskip-v4'


    def recorder(episode_id):
        return True
        # return episode_id % 1 == 0


    buffer_size = 200000
    epsilon = 0.2
    batch_size = 32
    gamma = 0.95
    eta = 0.00025
    N = 10000
    populate = True
    nb_pop = 20000

    debug = True

    env_test = wrappers.Monitor(env_test, video_callable=recorder, directory=outdir, force=True)
    env_test = Preprocess(env_test, train=False)

    env_train = Preprocess(env_train, train=True)
    env_train.seed(0)
    env_test.seed(0)
    agent = Agent(nb_ep=episode_count, action_space=env_train.action_space, buffer_size=buffer_size, epsilon=epsilon,
                  batch_size=batch_size,
                  gamma=gamma, eta=eta, N=N)

    reward = 0
    done = False

    if debug:
        print("NB EP : " + str(episode_count))
        print("Action Space : " + str(env_train.action_space))
        print("Meanings : " + str(env_train.get_action_meanings()))
        print("BUFFER SIZE : " + str(buffer_size))
        print("EPSILON : " + str(epsilon))
        print("Batch_size : " + str(batch_size))
        print("Gamma : " + str(gamma))
        print("LR : " + str(eta))
        print("update target net : " + str(N))

    results_train = []
    results_test = []
    if populate:
        env_pop = gym.make(env_id)
        env_pop = Preprocess(env_pop, train=True)
        for i in tqdm(range(nb_pop)):
            done = False
            ob, reward, done, _ = env_pop.reset()
            while not done:
                action = agent.random_act()
                prec_ob = ob
                ob, reward, done, _ = env_pop.step(action)
                interaction = (prec_ob, action, ob, reward, done)
                agent.memorise(interaction)
            agent.reset_noop()

    for i in tqdm(range(episode_count)):
        ob, reward, done, _ = env_train.reset()
        nb_iter = 0
        done = False
        score = 0
        while not done:
            action = agent.act(ob, reward, done)
            prec_ob = ob
            ob, reward, done, _ = env_train.step(action)
            interaction = (prec_ob, action, ob, reward, done)
            # print(interaction)
            agent.memorise(interaction)
            agent.learn()
            nb_iter += 1
            score += reward
        agent.reset_noop()
        print("TRAIN")
        print("EP " + str(i) + " - score " + str(score))
        print("EP " + str(i) + " - iteration " + str(nb_iter))
        print("I saw " + agent.how_many_did_u_see() + " interaction so far")
        done = False
        score = 0
        nb_iter = 0
        results_train.append(score)
        if i % 4 == 0:
            ob, reward, done, _ = env_test.reset()
            while not done:
                action = agent.act(ob, reward, done)
                prec_ob = ob
                ob, reward, done, _ = env_test.step(action)
                interaction = (prec_ob, action, ob, reward, done)
                # print(interaction)
                agent.memorise(interaction)
                # agent.learn()
                nb_iter += 1
                score += reward
            agent.reset_noop()
            print("TEST")
            print("EP " + str(i) + " - score " + str(score))
            print("EP " + str(i) + " - iteration " + str(nb_iter))
            print("I saw " + agent.how_many_did_u_see() + " interaction so far")
        results_test.append(score)
    env_train.close()
    env_test.close()
    #agent.save_model()

    # Note there's no env.render() here. But the environment still can open window and
    # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
    # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    # env.close()
