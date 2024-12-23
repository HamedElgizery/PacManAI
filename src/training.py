#!/usr/bin/env python

import sys
import os
import torch
import gymnasium as gym
import ale_py

CHECK_POINTS_PATH = os.path.join("..", "checkpoints")
class DQLearn:
    def __init__(self, check_point=None):

        if check_point is not None:
            load_checkpoint()

    def save_checkpoint(self, name):
        pass

    def load_checkpoint(self, name):
        pass
    
    

env = gym.make("ALE/MsPacman-v5", render_mode="human")
env.reset()

for _ in range(1000000):
    action = 0 
    '''
    if keyboard.is_pressed('left'):
        action = 3
    if keyboard.is_pressed('right'):
        action = 2
    if keyboard.is_pressed('up'):
        action = 1
    if keyboard.is_pressed('down'):
        action = 4
    '''
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

