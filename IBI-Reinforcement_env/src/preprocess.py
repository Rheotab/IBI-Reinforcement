import gym
import numpy as np

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

    def __init__(self, env, size_wrap=4, skip_frame=4, screen_size=84, norm=False):
        super(Preprocess, self).__init__(env)
        self.size_wrap = 4
        self.skip_frame = 4
        self.norm = norm
        self.screen_size = screen_size
        self.obs_buffer = [np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                           np.empty(env.observation_space.shape[:2], dtype=np.uint8)]
        self.ale = env.unwrapped.ale

    def step(self, action):
        R = 0.0
        res = []
        done = False
        for t in range(self.size_wrap):
            ob, reward, done, info = self.env.step(action)
            if done:
                break
            R += reward
            # FIXME RESHAPE OB
            res.append(ob)
        if not done:
            np.maximum(self.ob[self.size_wrap - 2], self.obs_buffer[self.size_wrap - 1], out=self.obs_buffer[self.size_wrap - 1])

        if self.norm:
            res = np.asarray(res, dtype=np.float32) / 255.0
        else:
            res = np.asarray(res, dtype=np.uint8)
        return res, R, done, None

    def reset(self, **kwargs):  # NoopReset
        self.env.reset(**kwargs)
        noops = self.env.unwrapped.np_random.randint(1, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset(**kwargs)
        self.ale.getScreenGrayscale(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)
        return self._get_obs()

    def _get_obs(self):
        import cv2
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        obs = cv2.resize(self.obs_buffer[0], (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        if self.norm:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)
        return obs
