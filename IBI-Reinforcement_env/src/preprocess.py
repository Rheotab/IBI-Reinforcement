import gym
import numpy as np
import cv2

'''
Inspired by
https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
[Mnih et al., 2015] Human- level control through deep reinforcement learning. Nature, 518(7540) :529.


Working directly with raw Atari 2600 frames, which are 2103 160
pixel images with a 128-colour palette.
- First, to encode a singleframe we take the maximum valuefor each pixel colour
value over the frame being encoded and the previous frame. 
- Second, we then extract the Y channel, also known as luminance, from the RGB frame and rescale it to
84 3 84. 
'''


class Preprocess(gym.Wrapper):

    def __init__(self, env, noop_max=30, size_wrap=4, skip_frame=4, screen_size=84, norm=False):
        super(Preprocess, self).__init__(env)
        self.size_wrap = 4
        self.skip_frame = 4
        self.norm = norm
        self.screen_size = screen_size
        self.ale = env.unwrapped.ale
        self.noop_max = noop_max
        #pd = self.env.unwrapped.get_action_meanings()
        #print("iuha")


    def step(self, action):
        R = 0.0
        res = None
        done = False
        for t in range(self.size_wrap):
            ob, reward, done, info = self.env.step(action)
            R += reward
            # Shape
            ob = cv2.resize(ob, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
            # Max R,G,B
            ob = np.max(ob, axis=2)
            if res is None:
                res = np.array([ob])
            else:
                res = np.append(res, [ob], axis=0)
            if done:
                break
        if not done:
            np.maximum(res[self.size_wrap - 2], res[self.size_wrap - 1], out=res[self.size_wrap - 1])
        if self.norm:
            res = np.asarray(res, dtype=np.float32) / 255.0
        else:
            res = np.asarray(res, dtype=np.uint8)
        return res, R, done, None


    def reset(self, **kwargs):  # NoopReset
        self.env.reset(**kwargs)
        return self.step(0) ## FIXME WE HAVE TO MAKE SURE THIS IS NOOP
