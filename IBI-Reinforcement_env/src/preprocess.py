import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
'''
Inspired by
https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
[Mnih et al., 2015] Human- level control through deep reinforcement learning. Nature, 518(7540) :529.

TODO : 
- DONE = Number of Lives or TRUE (game end)
'''


class Preprocess(gym.Wrapper):

    def __init__(self, env,train, noop_max=30, size_wrap=4, skip_frame=4, screen_size=84, norm=False):
        super(Preprocess, self).__init__(env)
        self.size_wrap = size_wrap
        self.skip_frame = skip_frame
        self.norm = norm
        self.train = train
        self.screen_size = screen_size
        self.ale = env.unwrapped.ale
        self.noop_max = noop_max
        self.lives = self.ale.lives()
        # pd = self.env.unwrapped.get_action_meanings()
        # print("iuha")

    def step(self, action):
        R = 0.0
        res = None
        done = False

        n_lives = self.ale.lives()
        for t in range(self.size_wrap):
            ob, reward, done, info = self.env.step(action)
            if n_lives != self.lives and self.train:
                done = True
            R += reward
            # Max R,G,B
            ob = np.max(ob, axis=2)
            # Shape
            ob = cv2.resize(ob, (self.screen_size, self.screen_size))
            if res is None:
                res = np.array([ob])
            else:
                res = np.append(res, [ob], axis=0)
            if done:
                break
        self.lives = n_lives
        i, _, _ = res.shape
        if done and i != self.skip_frame:
            for k in range(i - 1, self.skip_frame - 1):
                res = np.append(res, [res[k]], axis=0)
        np.maximum(res[self.size_wrap - 2], res[self.size_wrap - 1], out=res[self.size_wrap - 1])
        if self.norm:
            res = np.asarray(res, dtype=np.float32) / 255.0
        else:
            res = np.asarray(res, dtype=np.uint8)
        res = res.reshape((1, 4, 84, 84))
        return res, R, done, None

    def reset(self, **kwargs):  # NoopReset
        self.env.reset(**kwargs)
        return self.step(0)  ## 0 is noop
