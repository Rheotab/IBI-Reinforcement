import cv2
import numpy as np
import torch
import gymnasium as gym
from collections import deque

class FrameProcessor:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)  

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)  
        frame = frame / 255.0  
        return frame

    def process(self, state):
        processed_frame = self.preprocess(state)
        self.frames.append(processed_frame)

        while len(self.frames) < self.stack_size:
            self.frames.append(processed_frame)

        return np.array(self.frames, dtype=np.float32)  # Shape: (4, 84, 84)